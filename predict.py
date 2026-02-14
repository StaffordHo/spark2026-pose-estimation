"""
Inference script for generating submission.csv from trained pose estimation model.
Supports Test-Time Augmentation (TTA) for improved accuracy.
"""

import os
import sys
import argparse
import csv

import h5py
import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.event_representations import events_to_voxel_grid_fast
from src.models import PoseNet
from src.dataset import SPARK2026Dataset


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    model_cfg = config["model"]
    model = PoseNet(
        backbone=model_cfg.get("backbone", "resnet"),
        in_channels=model_cfg["in_channels"],
        feature_dim=model_cfg["feature_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        base_channels=model_cfg.get("base_channels", 32),
        dropout=model_cfg.get("dropout", 0.3),
        with_uncertainty=model_cfg.get("with_uncertainty", False),
        pretrained=False,
        rotation_repr=model_cfg.get("rotation_repr", "quaternion"),
        freeze_backbone=False
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, config


def load_multiple_models(checkpoint_paths, device):
    """Load multiple models for ensemble inference."""
    models = []
    config = None
    for path in checkpoint_paths:
        model, cfg = load_model(path, device)
        models.append(model)
        if config is None:
            config = cfg
    return models, config


def build_voxel_for_window(data, pose_idx, num_bins, height, width, target_size=None):
    """Build a voxel grid for a single event window."""
    t_end = data["timestamp"][pose_idx]
    t_start = data["timestamp"][pose_idx - 1]
    
    event_mask = (data["ts"] >= t_start) & (data["ts"] < t_end)
    
    xs = data["xs"][event_mask]
    ys = data["ys"][event_mask]
    ts = data["ts"][event_mask]
    ps = data["ps"][event_mask]
    
    if target_size is not None:
        target_h, target_w = target_size
        xs_scaled = (xs.astype(np.float32) * target_w / width)
        ys_scaled = (ys.astype(np.float32) * target_h / height)
        voxel = events_to_voxel_grid_fast(
            xs_scaled, ys_scaled, ts, ps,
            num_bins=num_bins, height=target_h, width=target_w
        )
    else:
        voxel = events_to_voxel_grid_fast(
            xs, ys, ts, ps,
            num_bins=num_bins, height=height, width=width
        )
    
    return voxel


def apply_tta_augmentations(voxel_tensor):
    """Generate TTA variants of a batch of voxel grids.
    
    Augmentations:
    1. Original
    2. Horizontal flip
    3. Vertical flip  
    4. Horizontal + Vertical flip
    
    Returns list of augmented tensors and their corresponding 
    transform functions for un-flipping the translation predictions.
    """
    augmented = [
        voxel_tensor,                                          # Original
        torch.flip(voxel_tensor, dims=[-1]),                   # Horizontal flip
        torch.flip(voxel_tensor, dims=[-2]),                   # Vertical flip
        torch.flip(voxel_tensor, dims=[-2, -1]),               # Both flips
    ]
    return augmented


def tta_merge_predictions(pred_list):
    """Merge TTA predictions by averaging translations and quaternions.
    
    For translations:
    - Horizontal flip: negate Tx
    - Vertical flip: negate Ty
    - Both: negate Tx and Ty
    
    For quaternions: average and re-normalize (simple average works 
    well when predictions are close, which they should be under TTA).
    """
    # pred_list[0] = original, [1] = h-flip, [2] = v-flip, [3] = both
    translations = []
    quaternions = []
    
    for i, pred in enumerate(pred_list):
        t = pred["translation"].float()
        q = pred["quaternion"].float()
        
        # Un-flip the translation predictions
        if i == 1:  # Horizontal flip → negate Tx
            t = t.clone()
            t[:, 0] = -t[:, 0]
        elif i == 2:  # Vertical flip → negate Ty
            t = t.clone()
            t[:, 1] = -t[:, 1]
        elif i == 3:  # Both flips → negate Tx and Ty
            t = t.clone()
            t[:, 0] = -t[:, 0]
            t[:, 1] = -t[:, 1]
        
        translations.append(t)
        quaternions.append(q)
    
    # Average translations
    avg_translation = torch.stack(translations).mean(dim=0)
    
    # Average quaternions — ensure consistent sign (q and -q are the same rotation)
    q_ref = quaternions[0]
    for i in range(1, len(quaternions)):
        # If dot product is negative, flip the quaternion to same hemisphere
        dot = (quaternions[i] * q_ref).sum(dim=-1, keepdim=True)
        quaternions[i] = torch.where(dot < 0, -quaternions[i], quaternions[i])
    
    avg_quaternion = torch.stack(quaternions).mean(dim=0)
    # Re-normalize
    avg_quaternion = avg_quaternion / (avg_quaternion.norm(dim=-1, keepdim=True) + 1e-8)
    
    return avg_translation, avg_quaternion


def main():
    parser = argparse.ArgumentParser(description="Generate submission.csv from trained model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint (or comma-separated for ensemble)")
    parser.add_argument("--test_dir", type=str, default="test", help="Directory containing test H5 files")
    parser.add_argument("--template", type=str, default="test/template.csv", help="Path to template CSV")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (4x slower, ~5-15%% better)")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP (use float32)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model(s) — support comma-separated paths for ensemble
    checkpoint_paths = [p.strip() for p in args.checkpoint.split(",")]
    
    if len(checkpoint_paths) > 1:
        print(f"Ensemble mode: loading {len(checkpoint_paths)} models...")
        models, config = load_multiple_models(checkpoint_paths, device)
        print(f"Loaded {len(models)} models for ensemble")
    else:
        print(f"Loading model from {checkpoint_paths[0]}...")
        model, config = load_model(checkpoint_paths[0], device)
        models = [model]
        print(f"Model loaded successfully")
    
    # Get data config
    data_cfg = config["data"]
    num_bins = data_cfg["num_bins"]
    height = data_cfg["height"]
    width = data_cfg["width"]
    target_size = tuple(data_cfg["target_size"]) if "target_size" in data_cfg else None
    normalize_translation = data_cfg.get("normalize_translation", False)
    use_amp = not args.no_amp and config["training"].get("amp", True)
    
    print(f"Config: num_bins={num_bins}, resolution={height}x{width}, target_size={target_size}")
    print(f"Translation normalization: {normalize_translation}")
    print(f"TTA: {'ENABLED (4 augmentations)' if args.tta else 'disabled'}")
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")
    
    # Parse template
    print(f"\nReading template: {args.template}")
    template_rows = []
    with open(args.template, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[0]:
                template_rows.append(row[0])
    
    print(f"Total predictions needed: {len(template_rows)}")
    
    # Group by sequence
    seq_to_indices = {}
    for row_id in template_rows:
        parts = row_id.rsplit("_", 1)
        seq_name = parts[0]
        pose_idx = int(parts[1])
        if seq_name not in seq_to_indices:
            seq_to_indices[seq_name] = []
        seq_to_indices[seq_name].append((row_id, pose_idx))
    
    print(f"Sequences to process: {sorted(seq_to_indices.keys())}")
    
    # Translation denormalization constants
    if normalize_translation:
        trans_mean = SPARK2026Dataset.TRANS_MEAN.to(device)
        trans_std = SPARK2026Dataset.TRANS_STD.to(device)
    
    # Run inference
    predictions = {}
    
    for seq_name in sorted(seq_to_indices.keys()):
        h5_path = os.path.join(args.test_dir, f"{seq_name}.h5")
        if not os.path.exists(h5_path):
            print(f"  WARNING: {h5_path} not found, skipping sequence {seq_name}")
            continue
        
        indices = seq_to_indices[seq_name]
        print(f"\nProcessing {seq_name}: {len(indices)} poses from {h5_path}")
        
        with h5py.File(h5_path, "r") as f:
            data = {
                "xs": f["events"]["xs"][()],
                "ys": f["events"]["ys"][()],
                "ts": f["events"]["ts"][()],
                "ps": f["events"]["ps"][()],
                "timestamp": f["labels"]["data"]["timestamp"][()],
            }
        
        batch_voxels = []
        batch_ids = []
        
        for row_id, pose_idx in indices:
            voxel = build_voxel_for_window(
                data, pose_idx, num_bins, height, width, target_size
            )
            batch_voxels.append(voxel)
            batch_ids.append(row_id)
            
            if len(batch_voxels) == args.batch_size:
                _run_batch(models, batch_voxels, batch_ids, predictions,
                          device, use_amp, normalize_translation,
                          trans_mean if normalize_translation else None,
                          trans_std if normalize_translation else None,
                          tta=args.tta)
                batch_voxels = []
                batch_ids = []
        
        if batch_voxels:
            _run_batch(models, batch_voxels, batch_ids, predictions,
                      device, use_amp, normalize_translation,
                      trans_mean if normalize_translation else None,
                      trans_std if normalize_translation else None,
                      tta=args.tta)
        
        print(f"  Done: {len(indices)} predictions generated")
    
    # Write submission CSV
    print(f"\nWriting submission to {args.output}...")
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for row_id in template_rows:
            if row_id in predictions:
                tx, ty, tz, qx, qy, qz, qw = predictions[row_id]
                writer.writerow([row_id, tx, ty, tz, qx, qy, qz, qw])
            else:
                print(f"  WARNING: Missing prediction for {row_id}")
                writer.writerow([row_id, 0, 0, 5.68, 0, 0, 0, 1])
    
    print(f"\nSubmission complete! {len(predictions)}/{len(template_rows)} predictions written.")


@torch.no_grad()
def _run_batch(models, batch_voxels, batch_ids, predictions,
               device, use_amp, normalize_translation,
               trans_mean, trans_std, tta=False):
    """Run model inference on a batch, with optional TTA and ensemble."""
    voxel_tensor = torch.stack(batch_voxels).to(device)
    
    all_translations = []
    all_quaternions = []
    
    for model in models:
        if tta:
            # Generate augmented versions
            aug_inputs = apply_tta_augmentations(voxel_tensor)
            pred_list = []
            for aug_voxel in aug_inputs:
                with autocast('cuda', enabled=use_amp):
                    pred = model(aug_voxel)
                pred_list.append(pred)
            
            translation, quaternion = tta_merge_predictions(pred_list)
        else:
            with autocast('cuda', enabled=use_amp):
                pred = model(voxel_tensor)
            translation = pred["translation"].float()
            quaternion = pred["quaternion"].float()
        
        all_translations.append(translation)
        all_quaternions.append(quaternion)
    
    # Ensemble: average across models
    if len(models) > 1:
        translation = torch.stack(all_translations).mean(dim=0)
        
        # Average quaternions with sign alignment
        q_ref = all_quaternions[0]
        for i in range(1, len(all_quaternions)):
            dot = (all_quaternions[i] * q_ref).sum(dim=-1, keepdim=True)
            all_quaternions[i] = torch.where(dot < 0, -all_quaternions[i], all_quaternions[i])
        quaternion = torch.stack(all_quaternions).mean(dim=0)
        quaternion = quaternion / (quaternion.norm(dim=-1, keepdim=True) + 1e-8)
    else:
        translation = all_translations[0]
        quaternion = all_quaternions[0]
    
    # Denormalize translation
    if normalize_translation and trans_mean is not None:
        translation = translation * trans_std + trans_mean
    
    # Store predictions
    for i, row_id in enumerate(batch_ids):
        t = translation[i].cpu().numpy()
        q = quaternion[i].cpu().numpy()
        q = q / (np.linalg.norm(q) + 1e-8)
        predictions[row_id] = (
            f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}",
            f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}"
        )


if __name__ == "__main__":
    main()
