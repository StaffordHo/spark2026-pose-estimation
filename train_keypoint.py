"""
Training script for Keypoint + PnP pose estimation.

Trains the FPN + Keypoint Heatmap architecture. At training time, generates
2D keypoint heatmap supervision from pose labels + camera intrinsics.

Usage:
    python train_keypoint.py --config config/rtx4090_keypoint.yaml
    python train_keypoint.py --config config/rtx4090_keypoint.yaml --fast_dev_run
"""

import os
import sys
import argparse
import time
import yaml
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SPARK2026Dataset
from src.models.keypoint_posenet import KeypointPoseNet
from src.spacecraft_model import (
    PROBA2_KEYPOINTS_3D, CAMERA_MATRIX, IMAGE_HEIGHT, IMAGE_WIDTH,
    N_KEYPOINTS, project_keypoints, generate_heatmaps
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class KeypointDataset(torch.utils.data.Dataset):
    """
    Wrapper around SPARK2026Dataset that adds keypoint heatmap targets.
    
    Generates 2D keypoint supervision by projecting 3D Proba-2 keypoints
    using the pose labels and camera intrinsics.
    """
    
    def __init__(self, base_dataset, heatmap_size=64, sigma=2.0):
        self.base = base_dataset
        self.heatmap_size = heatmap_size
        self.sigma = sigma
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        sample = self.base[idx]
        
        # Get raw (unnormalized) pose for projection
        seq_name, pose_idx = self.base.samples[idx]
        data = self.base._seq_data[seq_name]
        
        translation = np.array([
            data["Tx"][pose_idx],
            data["Ty"][pose_idx],
            data["Tz"][pose_idx]
        ], dtype=np.float64)
        
        quaternion = np.array([
            data["Qx"][pose_idx],
            data["Qy"][pose_idx],
            data["Qz"][pose_idx],
            data["Qw"][pose_idx]
        ], dtype=np.float64)
        quaternion = quaternion / np.linalg.norm(quaternion)
        
        # Project 3D keypoints to 2D
        kp_2d, visibility = project_keypoints(
            PROBA2_KEYPOINTS_3D, quaternion, translation
        )
        
        # Generate heatmaps
        heatmaps = generate_heatmaps(
            kp_2d, visibility,
            self.heatmap_size,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            sigma=self.sigma
        )
        
        # Normalized keypoint coordinates [0, 1] for coordinate loss
        kp_norm = np.zeros((N_KEYPOINTS, 2), dtype=np.float32)
        vis_mask = np.zeros(N_KEYPOINTS, dtype=np.float32)
        for i in range(N_KEYPOINTS):
            if visibility[i]:
                kp_norm[i, 0] = kp_2d[i, 0] / IMAGE_WIDTH
                kp_norm[i, 1] = kp_2d[i, 1] / IMAGE_HEIGHT
                vis_mask[i] = 1.0
        
        sample["heatmap_target"] = torch.from_numpy(heatmaps)
        sample["keypoints_norm_target"] = torch.from_numpy(kp_norm)
        sample["visibility_mask"] = torch.from_numpy(vis_mask)
        
        return sample


def train_epoch(model, loader, optimizer, scaler, device, config, epoch, writer):
    model.train()
    total_loss = 0
    total_heatmap_loss = 0
    total_coord_loss = 0
    n_batches = 0
    
    coord_weight = config["loss"].get("coord_loss_weight", 5.0)
    log_interval = config["logging"]["log_interval"]
    
    for batch_idx, batch in enumerate(loader):
        voxel = batch["voxel"].to(device)
        heatmap_target = batch["heatmap_target"].to(device)
        kp_target = batch["keypoints_norm_target"].to(device)
        vis_mask = batch["visibility_mask"].to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=config["training"]["amp"]):
            output = model(voxel)
            
            # Compute losses in float32 for stability
            heatmaps_f32 = output["heatmaps"].float()
            target_f32 = heatmap_target.float()
            heatmap_loss = F.mse_loss(heatmaps_f32, target_f32)
            
            # Coordinate L1 loss (only for visible keypoints)
            kp_pred = output["keypoints_norm"].float()  # (B, N, 2)
            kp_tgt = kp_target.float()
            vis_expanded = vis_mask.unsqueeze(2).float()  # (B, N, 1)
            coord_loss = (torch.abs(kp_pred - kp_tgt) * vis_expanded).sum() / (vis_expanded.sum() + 1e-6)
            
            loss = heatmap_loss + coord_weight * coord_loss
        
        # NaN guard: skip bad batches
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠ Batch [{batch_idx}] NaN/Inf loss — skipping", flush=True)
            optimizer.zero_grad()
            continue
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config["training"].get("grad_clip", 0) > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_heatmap_loss += heatmap_loss.item()
        total_coord_loss += coord_loss.item()
        n_batches += 1
        
        if batch_idx % log_interval == 0:
            print(f"  Batch [{batch_idx}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Heatmap: {heatmap_loss.item():.4f} "
                  f"Coord: {coord_loss.item():.4f}", flush=True)
    
    avg_loss = total_loss / n_batches
    avg_heatmap = total_heatmap_loss / n_batches
    avg_coord = total_coord_loss / n_batches
    
    if writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/heatmap_loss", avg_heatmap, epoch)
        writer.add_scalar("train/coord_loss", avg_coord, epoch)
    
    return avg_loss


def validate(model, loader, device, config):
    """Validate: compute heatmap loss and keypoint detection accuracy."""
    model.eval()
    total_heatmap_loss = 0
    total_coord_error = 0
    n_batches = 0
    n_visible = 0
    
    with torch.no_grad():
        for batch in loader:
            voxel = batch["voxel"].to(device)
            heatmap_target = batch["heatmap_target"].to(device)
            kp_target = batch["keypoints_norm_target"].to(device)
            vis_mask = batch["visibility_mask"].to(device)
            
            with autocast('cuda', enabled=config["training"]["amp"]):
                output = model(voxel)
            
            heatmap_loss = F.mse_loss(output["heatmaps"], heatmap_target)
            total_heatmap_loss += heatmap_loss.item()
            
            # Mean coordinate error (in normalized coords) for visible keypoints
            vis_expanded = vis_mask.unsqueeze(2)
            coord_err = (torch.abs(output["keypoints_norm"] - kp_target) * vis_expanded).sum()
            n_vis = vis_expanded.sum()
            total_coord_error += coord_err.item()
            n_visible += n_vis.item()
            
            n_batches += 1
    
    avg_heatmap = total_heatmap_loss / n_batches
    avg_coord_err = total_coord_error / max(n_visible, 1)
    
    # Convert normalized error to pixel error (approximate)
    pixel_err = avg_coord_err * max(IMAGE_WIDTH, IMAGE_HEIGHT)
    
    return {
        "heatmap_loss": avg_heatmap,
        "coord_error_norm": avg_coord_err,
        "coord_error_px": pixel_err,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Keypoint + PnP model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Parse data config
    target_size_cfg = config["data"].get("target_size", None)
    target_size = tuple(target_size_cfg) if target_size_cfg else None
    
    # Create base datasets
    print("\nLoading data...", flush=True)
    train_base = SPARK2026Dataset(
        config["data"]["data_dir"], split="train",
        num_bins=config["data"]["num_bins"],
        target_size=target_size,
        augmentation=config["data"].get("augmentation", False),
        normalize_translation=False  # Don't normalize — we need raw poses for projection
    )
    val_base = SPARK2026Dataset(
        config["data"]["data_dir"], split="val",
        num_bins=config["data"]["num_bins"],
        target_size=target_size,
        augmentation=False,
        normalize_translation=False
    )
    
    # Wrap with keypoint supervision
    heatmap_size = config["model"].get("heatmap_size", 64)
    sigma = config["model"].get("heatmap_sigma", 2.0)
    
    train_dataset = KeypointDataset(train_base, heatmap_size=heatmap_size, sigma=sigma)
    val_dataset = KeypointDataset(val_base, heatmap_size=heatmap_size, sigma=sigma)
    
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Create model
    model = KeypointPoseNet(
        in_channels=config["model"]["in_channels"],
        n_keypoints=config["model"]["n_keypoints"],
        heatmap_size=heatmap_size,
        fpn_channels=config["model"]["fpn_channels"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"]
    )
    model = model.to(device)
    
    trainable = model.get_num_params()
    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total:,} total params, {trainable:,} trainable")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config["training"].get("T_0", 10),
        T_mult=config["training"].get("T_mult", 2)
    )
    
    scaler = GradScaler('cuda', enabled=config["training"]["amp"])
    
    # Checkpoint dir
    ckpt_dir = config["logging"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config["logging"]["log_dir"], "keypoint"))
    
    num_epochs = args.epochs or config["training"]["epochs"]
    
    if args.fast_dev_run:
        num_epochs = 1
        print("\n=== FAST DEV RUN ===", flush=True)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...", flush=True)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 50, flush=True)
        
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, config, epoch, writer)
        dt = time.time() - t0
        print(f"Train Loss: {train_loss:.4f} ({dt:.1f}s)", flush=True)
        
        scheduler.step()
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, device, config)
        print(f"Val Heatmap Loss: {val_metrics['heatmap_loss']:.4f}")
        print(f"Val Coord Error: {val_metrics['coord_error_norm']:.4f} ({val_metrics['coord_error_px']:.1f} px)")
        
        writer.add_scalar("val/heatmap_loss", val_metrics["heatmap_loss"], epoch)
        writer.add_scalar("val/coord_error_px", val_metrics["coord_error_px"], epoch)
        
        # Save best
        val_loss = val_metrics["heatmap_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "config": config
            }, os.path.join(ckpt_dir, "best.pth"))
            print(f"Saved best model (val loss: {val_loss:.4f})")
        
        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config
        }, os.path.join(ckpt_dir, "latest.pth"))
        
        # Per-epoch checkpoint
        save_every = 1 if config["training"].get("save_every_epoch", False) else 10
        if (epoch + 1) % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"))
            print(f"Saved epoch checkpoint: epoch_{epoch+1}.pth")
    
    writer.close()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
