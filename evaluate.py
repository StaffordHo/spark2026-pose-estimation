"""
Evaluation script for trained pose estimation models.
"""

import os
import sys
import argparse
import json

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SPARK2026Dataset
from src.models import PoseNet
from src.metrics import compute_metrics, MetricTracker


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
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
        pretrained=False,  # Don't download weights for eval
        rotation_repr=model_cfg.get("rotation_repr", "quaternion"),
        freeze_backbone=False  # Don't freeze for eval
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, config


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> dict:
    """Evaluate model on a dataset."""
    tracker = MetricTracker()
    
    all_preds = {"translation": [], "quaternion": []}
    all_targets = {"translation": [], "quaternion": []}
    seq_errors = {}
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        voxel = batch["voxel"].to(device)
        target = {
            "translation": batch["translation"].to(device),
            "quaternion": batch["quaternion"].to(device)
        }
        
        pred = model(voxel)
        tracker.update(pred, target)
        
        # Store per-sequence results
        for i, seq_name in enumerate(batch["seq_name"]):
            if seq_name not in seq_errors:
                seq_errors[seq_name] = {"pos": [], "rot": []}
            
            pos_err = torch.norm(pred["translation"][i] - target["translation"][i]).item()
            
            q_pred = pred["quaternion"][i]
            q_gt = target["quaternion"][i]
            dot = torch.abs(torch.dot(q_pred, q_gt))
            rot_err = 2.0 * torch.acos(torch.clamp(dot, -1.0, 1.0)) * 180 / np.pi
            
            seq_errors[seq_name]["pos"].append(pos_err)
            seq_errors[seq_name]["rot"].append(rot_err.item())
        
        # Store raw predictions
        all_preds["translation"].append(pred["translation"].cpu())
        all_preds["quaternion"].append(pred["quaternion"].cpu())
        all_targets["translation"].append(target["translation"].cpu())
        all_targets["quaternion"].append(target["quaternion"].cpu())
    
    # Aggregate metrics
    metrics = tracker.compute()
    
    # Per-sequence summary
    seq_summary = {}
    for seq_name, errors in seq_errors.items():
        seq_summary[seq_name] = {
            "pos_mean": np.mean(errors["pos"]),
            "pos_std": np.std(errors["pos"]),
            "rot_mean": np.mean(errors["rot"]),
            "rot_std": np.std(errors["rot"])
        }
    
    return {
        "overall": metrics,
        "per_sequence": seq_summary
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pose estimation model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data_dir", type=str, default=".", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    print(f"Model loaded: {model.backbone_name} backbone")
    
    # Create dataset
    data_cfg = config["data"]
    target_size = tuple(data_cfg["target_size"]) if "target_size" in data_cfg else None
    normalize_translation = data_cfg.get("normalize_translation", False)
    
    print(f"\nLoading {args.split} dataset...")
    dataset = SPARK2026Dataset(
        data_dir=args.data_dir,
        split=args.split,
        num_bins=data_cfg["num_bins"],
        height=data_cfg["height"],
        width=data_cfg["width"],
        target_size=target_size,
        normalize_translation=normalize_translation
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate(model, dataloader, device)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    overall = results["overall"]
    print(f"\nPosition Error:")
    print(f"  Mean:   {overall['pos_error_mean']:.4f} m")
    print(f"  Median: {overall['pos_error_median']:.4f} m")
    print(f"  Std:    {overall['pos_error_std']:.4f} m")
    
    print(f"\nRotation Error:")
    print(f"  Mean:   {overall['rot_error_mean']:.2f}°")
    print(f"  Median: {overall['rot_error_median']:.2f}°")
    print(f"  Std:    {overall['rot_error_std']:.2f}°")
    
    # Show worst sequences
    per_seq = results["per_sequence"]
    worst_pos = sorted(per_seq.items(), key=lambda x: x[1]["pos_mean"], reverse=True)[:5]
    worst_rot = sorted(per_seq.items(), key=lambda x: x[1]["rot_mean"], reverse=True)[:5]
    
    print("\nWorst 5 sequences by position error:")
    for seq, err in worst_pos:
        print(f"  {seq}: {err['pos_mean']:.4f}m ± {err['pos_std']:.4f}")
    
    print("\nWorst 5 sequences by rotation error:")
    for seq, err in worst_rot:
        print(f"  {seq}: {err['rot_mean']:.2f}° ± {err['rot_std']:.2f}")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
