"""
Training script for spacecraft pose estimation.
"""

import os
import sys
import argparse
import yaml
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SPARK2026Dataset, get_dataloaders
from src.models import PoseNet
from src.losses import PoseLoss, PoseLossUncertainty
from src.metrics import MetricTracker


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> nn.Module:
    """Create model from config."""
    model_cfg = config["model"]
    model = PoseNet(
        backbone=model_cfg["backbone"],
        in_channels=model_cfg["in_channels"],
        feature_dim=model_cfg["feature_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        base_channels=model_cfg["base_channels"],
        dropout=model_cfg["dropout"],
        with_uncertainty=model_cfg["with_uncertainty"]
    )
    return model


def create_scheduler(optimizer, config: dict, num_epochs: int):
    """Create learning rate scheduler."""
    sched_type = config["training"].get("scheduler", "cosine")
    warmup = config["training"].get("warmup_epochs", 5)
    
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    
    return scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: dict,
    epoch: int,
    writer: SummaryWriter
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    log_interval = config["logging"]["log_interval"]
    use_amp = config["training"]["amp"]
    grad_clip = config["training"].get("grad_clip", 1.0)
    
    for batch_idx, batch in enumerate(train_loader):
        voxel = batch["voxel"].to(device)
        target = {
            "translation": batch["translation"].to(device),
            "quaternion": batch["quaternion"].to(device)
        }
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=use_amp):
            pred = model(voxel)
            loss, loss_dict = criterion(pred, target)
        
        scaler.scale(loss).backward()
        
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Logging
        global_step = epoch * num_batches + batch_idx
        if batch_idx % log_interval == 0:
            print(f"  Batch [{batch_idx}/{num_batches}] "
                  f"Loss: {loss.item():.4f} "
                  f"Trans: {loss_dict['loss_translation'].item():.4f} "
                  f"Rot: {loss_dict['loss_rotation'].item():.4f}", flush=True)
            
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/loss_translation", loss_dict["loss_translation"].item(), global_step)
            writer.add_scalar("train/loss_rotation", loss_dict["loss_rotation"].item(), global_step)
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict
) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    metric_tracker = MetricTracker()
    use_amp = config["training"]["amp"]
    
    for batch in val_loader:
        voxel = batch["voxel"].to(device)
        target = {
            "translation": batch["translation"].to(device),
            "quaternion": batch["quaternion"].to(device)
        }
        
        with autocast('cuda', enabled=use_amp):
            pred = model(voxel)
            loss, _ = criterion(pred, target)
        
        total_loss += loss.item()
        metric_tracker.update(pred, target)
    
    metrics = metric_tracker.compute()
    metrics["loss"] = total_loss / len(val_loader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train spacecraft pose estimation model")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--fast_dev_run", action="store_true", help="Quick test run with 2 batches")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    
    # Set seed
    set_seed(config["experiment"]["seed"])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    
    # Create directories
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    os.makedirs(config["logging"]["checkpoint_dir"], exist_ok=True)
    
    # Create dataloaders
    print("\nLoading data...", flush=True)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        num_bins=config["data"]["num_bins"],
        height=config["data"]["height"],
        width=config["data"]["width"]
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)
    
    # Create model
    print("\nCreating model...", flush=True)
    model = create_model(config)
    model = model.to(device)
    print(f"Model parameters: {model.get_num_params():,}", flush=True)
    
    # Loss function
    if config["model"]["with_uncertainty"]:
        criterion = PoseLossUncertainty()
    else:
        criterion = PoseLoss(
            trans_weight=config["loss"]["trans_weight"],
            rot_weight=config["loss"]["rot_weight"],
            trans_loss_type=config["loss"]["trans_loss_type"],
            rot_loss_type=config["loss"]["rot_loss_type"]
        )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Scheduler
    num_epochs = config["training"]["epochs"]
    scheduler = create_scheduler(optimizer, config, num_epochs)
    
    # AMP scaler
    scaler = GradScaler('cuda', enabled=config["training"]["amp"])
    
    # TensorBoard writer
    exp_name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config["logging"]["log_dir"], f"{exp_name}_{timestamp}")
    writer = SummaryWriter(log_path)
    print(f"Logging to: {log_path}", flush=True)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    
    # Fast dev run
    if args.fast_dev_run:
        num_epochs = 2
        config["logging"]["log_interval"] = 1
        print("\n=== FAST DEV RUN ===", flush=True)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...", flush=True)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]", flush=True)
        print("-" * 50, flush=True)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, config, epoch, writer
        )
        print(f"Train Loss: {train_loss:.4f}", flush=True)
        
        # Update scheduler
        scheduler.step()
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        
        # Validate
        if (epoch + 1) % config["logging"]["val_interval"] == 0:
            val_metrics = validate(model, val_loader, criterion, device, config)
            
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Pos Error: {val_metrics['pos_error_mean']:.3f}m (median: {val_metrics['pos_error_median']:.3f}m)")
            print(f"Val Rot Error: {val_metrics['rot_error_mean']:.2f}° (median: {val_metrics['rot_error_median']:.2f}°)")
            
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/pos_error_mean", val_metrics["pos_error_mean"], epoch)
            writer.add_scalar("val/rot_error_mean", val_metrics["rot_error_mean"], epoch)
            
            # Save best model
            if config["logging"]["save_best"] and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint_path = os.path.join(
                    config["logging"]["checkpoint_dir"], "best.pth"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "best_val_loss": best_val_loss,
                    "config": config
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                config["logging"]["checkpoint_dir"], f"epoch_{epoch+1}.pth"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config
            }, checkpoint_path)
    
    writer.close()
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
