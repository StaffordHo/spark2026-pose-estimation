"""
Local evaluation script — simulates CodaBench scoring on the validation set.

Evaluates checkpoints and ranks them by estimated Pose Error.
Uses batched inference for speed (~2 min per checkpoint).

Usage:
    python evaluate.py --config config/rtx4090_sota.yaml --tta
    python evaluate.py --checkpoint checkpoints/epoch_1.pth --tta
"""

import os
import sys
import argparse
import glob
import yaml
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SPARK2026Dataset, get_dataloaders
from src.models import PoseNet


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(config: dict) -> nn.Module:
    """Create model from config."""
    model_cfg = config["model"]
    model = PoseNet(
        backbone=model_cfg.get("backbone", "voxel_cnn"),
        in_channels=model_cfg.get("in_channels", 10),
        feature_dim=model_cfg.get("feature_dim", 512),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        base_channels=model_cfg.get("base_channels", 32),
        dropout=model_cfg.get("dropout", 0.1),
        with_uncertainty=model_cfg.get("with_uncertainty", False),
        rotation_repr=model_cfg.get("rotation_repr", "quaternion"),
        pretrained=False,
        freeze_backbone=False
    )
    return model


def evaluate_checkpoint(checkpoint_path, val_loader, device, tta=False):
    """Evaluate a single checkpoint on the validation set using batched inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", -1) + 1
    normalize_translation = config["data"].get("normalize_translation", True)
    use_amp = config["training"].get("amp", False)

    # Translation normalization stats
    TRANS_MEAN = SPARK2026Dataset.TRANS_MEAN.to(device)
    TRANS_STD = SPARK2026Dataset.TRANS_STD.to(device)

    all_trans_errors = []
    all_rot_errors = []
    num_batches = len(val_loader)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i % 50 == 0:
                print(f"  Batch {i}/{num_batches}", flush=True)
            voxel = batch["voxel"].to(device)
            gt_trans_raw = batch["translation"].to(device)
            gt_quat_raw = batch["quaternion"].to(device)

            if normalize_translation:
                gt_trans_m = gt_trans_raw * TRANS_STD + TRANS_MEAN
            else:
                gt_trans_m = gt_trans_raw

            def get_m_coords(v):
                with autocast('cuda', enabled=use_amp):
                    p = model(v)
                t = p["translation"].float()
                if normalize_translation:
                    t = t * TRANS_STD + TRANS_MEAN
                q = p["quaternion"].float()
                return t, q

            if tta:
                # 4 augmentations: Original, HFlip, VFlip, Both
                t0, q0 = get_m_coords(voxel)
                
                t1, q1 = get_m_coords(torch.flip(voxel, dims=[3])) # HFlip
                t1[:, 0] = -t1[:, 0]
                q1[:, 1] = -q1[:, 1]; q1[:, 2] = -q1[:, 2]
                
                t2, q2 = get_m_coords(torch.flip(voxel, dims=[2])) # VFlip
                t2[:, 1] = -t2[:, 1]
                q2[:, 0] = -q2[:, 0]; q2[:, 2] = -q2[:, 2]
                
                t3, q3 = get_m_coords(torch.flip(voxel, dims=[2, 3])) # Both
                t3[:, 0] = -t3[:, 0]; t3[:, 1] = -t3[:, 1]
                q3[:, 0] = -q3[:, 0]; q3[:, 1] = -q3[:, 1]

                pred_trans_m = (t0 + t1 + t2 + t3) / 4.0
                
                # Quaternion average with sign alignment
                for qi in [q1, q2, q3]:
                    dot = (qi * q0).sum(dim=-1, keepdim=True)
                    qi[:] = torch.where(dot < 0, -qi, qi)
                pred_quat = (q0 + q1 + q2 + q3) / 4.0
                pred_quat = pred_quat / (pred_quat.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                pred_trans_m, pred_quat = get_m_coords(voxel)

            # Errors
            trans_err = torch.norm(pred_trans_m - gt_trans_m, dim=1)
            all_trans_errors.append(trans_err.cpu())

            dot = torch.abs(torch.sum(pred_quat * gt_quat_raw, dim=1))
            dot = torch.clamp(dot, 0.0, 1.0)
            rot_err = 2.0 * torch.acos(dot) * 180.0 / np.pi # degrees
            all_rot_errors.append(rot_err.cpu())

    trans_errors = torch.cat(all_trans_errors).numpy()
    rot_errors = torch.cat(all_rot_errors).numpy()

    return {
        "epoch": epoch,
        "checkpoint": os.path.basename(checkpoint_path),
        "trans_mean": trans_errors.mean(),
        "trans_median": np.median(trans_errors),
        "orient_mean": rot_errors.mean(),    # degrees
        "orient_median": np.median(rot_errors), # degrees
        "pose_error": trans_errors.mean() + (rot_errors.mean() * np.pi / 180.0), # radians-based pose score
        "n_samples": len(trans_errors),
    }


def main():
    parser = argparse.ArgumentParser(description="Local evaluation (simulates CodaBench)")
    parser.add_argument("--config", type=str, default="config/rtx4090_sota.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Single checkpoint path, or comma-separated list")
    parser.add_argument("--tta", action="store_true", help="Test-time augmentation")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine checkpoints to evaluate
    if args.checkpoint:
        checkpoints = [c.strip() for c in args.checkpoint.split(",")]
    else:
        checkpoint_dir = config["logging"]["checkpoint_dir"]
        checkpoints = sorted(
            glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth")),
            key=lambda x: int(os.path.basename(x).replace("epoch_", "").replace(".pth", ""))
        )
        best_path = os.path.join(checkpoint_dir, "best.pth")
        if os.path.exists(best_path):
            checkpoints.append(best_path)

    if not checkpoints:
        print("No checkpoints found!")
        return

    # Create val dataloader (loaded from first checkpoint's config for consistency)
    print("Loading validation data...", flush=True)
    first_ckpt = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
    ckpt_config = first_ckpt["config"]
    
    target_size_cfg = ckpt_config["data"].get("target_size", [256, 256])
    target_size = tuple(target_size_cfg)
    num_bins = ckpt_config["data"].get("num_bins", 5)
    normalize_translation = ckpt_config["data"].get("normalize_translation", True)

    val_dataset = SPARK2026Dataset(
        ckpt_config["data"]["data_dir"], split="val",
        num_bins=num_bins,
        target_size=target_size, augmentation=False,
        normalize_translation=normalize_translation
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=ckpt_config["data"].get("batch_size", 32),
        shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"\nEvaluating {len(checkpoints)} checkpoint(s) on {len(val_dataset)} val samples")
    print(f"Target Size: {target_size} | Num Bins: {num_bins}")
    print(f"TTA: {'enabled' if args.tta else 'disabled'}")

    print(f"\nEvaluating {len(checkpoints)} checkpoint(s) on {len(val_dataset)} val samples")
    print(f"TTA: {'enabled' if args.tta else 'disabled'}")
    print("=" * 75)

    all_results = []

    for ckpt_path in checkpoints:
        name = os.path.basename(ckpt_path)
        print(f"Evaluating {name}...", end=" ", flush=True)
        t0 = time.time()
        results = evaluate_checkpoint(ckpt_path, val_loader, device, tta=args.tta)
        dt = time.time() - t0
        print(f"{dt:.1f}s | Trans: {results['trans_mean']:.4f}m | Orient: {results['orient_mean']:.4f} deg | Pose: {results['pose_error']:.4f}")
        all_results.append(results)

    # Rank by pose error
    print("\n" + "=" * 75)
    print("RANKING (sorted by Pose Error, lower = better)")
    print("=" * 75)
    print(f"{'Rank':<5} {'Checkpoint':<20} {'Trans':>8} {'Orient':>8} {'Pose':>10}")
    print("-" * 55)

    all_results.sort(key=lambda x: x["pose_error"])
    for rank, r in enumerate(all_results, 1):
        marker = " ★" if rank == 1 else ""
        print(f"{rank:<5} {r['checkpoint']:<20} {r['trans_mean']:>8.4f} {r['orient_mean']:>8.4f} {r['pose_error']:>10.4f}{marker}")

    best = all_results[0]
    print(f"\n✅ Best: {best['checkpoint']} (Pose Error: {best['pose_error']:.4f})")
    print(f"   → python predict.py --checkpoint checkpoints/{best['checkpoint']} --output submission.csv --tta")


if __name__ == "__main__":
    main()
