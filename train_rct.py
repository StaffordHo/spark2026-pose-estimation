"""
Unsupervised Domain Adaptation via Rotation Consistency Training (RCT).

This script fine-tunes a trained model directly on the Unlabeled Test Set
to adapt to the "simulation-to-real" domain gap. It uses a self-supervised
consistency loss: the network's prediction on a horizontally flipped
event grid must geometrically match its prediction on the original grid.
"""

import os
import sys
import argparse
import time
import yaml
import glob
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.pose_net import PoseNet
from src.event_representations import events_to_voxel_grid_fast
import h5py


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RCTDataset(Dataset):
    """
    Loads unlabeled test sequences (RT901 - RT928).
    Returns raw event coordinates so we can apply precise flipping
    BEFORE voxelizing, ensuring authentic data corruption/augmentation.
    """
    def __init__(self, data_dir=".", height=448, width=448, num_bins=5, sequence_length=1):
        self.height = height
        self.width = width
        self.num_bins = num_bins
        # 100ms windows
        self.window_size = sequence_length * 1000
        
        # Load Test Sequence paths
        self.h5_files = sorted(glob.glob(os.path.join(data_dir, "test", "RT*.h5")))
        print(f"Loading {len(self.h5_files)} test sequences for Domain Adaptation...")
        
        self.samples = []
        self._seq_data = {}
        
        for h5_path in self.h5_files:
            seq_name = os.path.basename(h5_path).replace(".h5", "")
            with h5py.File(h5_path, "r") as f:
                # Read all events into memory (Test sequences are small)
                xs = f["events"]["xs"][()]
                ys = f["events"]["ys"][()]
                ts = f["events"]["ts"][()]
                ps = f["events"]["ps"][()]
                
                self._seq_data[seq_name] = {
                    "xs": xs, "ys": ys, "ts": ts, "ps": ps
                }
                
                # We don't have pose timestamps in the test set.
                # Generate synthetic pose evaluation points every 100ms (1000 units),
                # spanning from the first event to the last event.
                start_t = ts[0]
                end_t = ts[-1]
                
                # Start index 1 so we have at least 1 history window
                pose_timestamps = np.arange(start_t + self.window_size, end_t, self.window_size)
                
                for t_end in pose_timestamps:
                    self.samples.append((seq_name, t_end))
                    
        print(f"RCT Dataset: {len(self.samples)} unlabeled test samples ready.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_name, t_end = self.samples[idx]
        data = self._seq_data[seq_name]
        
        t_start = t_end - self.window_size
        
        # Binary search for events
        ts_all = data["ts"]
        i_start = np.searchsorted(ts_all, t_start, side='left')
        i_end = np.searchsorted(ts_all, t_end, side='left')
        
        xs = data["xs"][i_start:i_end]
        ys = data["ys"][i_start:i_end]
        ts = ts_all[i_start:i_end]
        ps = data["ps"][i_start:i_end]

        # 1. Generate the standard ORIGINAL Voxel Grid
        voxel_orig = events_to_voxel_grid_fast(
            xs, ys, ts, ps,
            num_bins=self.num_bins,
            height=self.height, width=self.width
        )
        
        # 2. Generate the FLIPPED Voxel Grid
        # Flip the X coordinates explicitly
        xs_flipped = (1280 - 1) - xs
        
        voxel_flipped = events_to_voxel_grid_fast(
            xs_flipped, ys, ts, ps,
            num_bins=self.num_bins,
            height=self.height, width=self.width
        )
        
        return {
            "voxel_orig": voxel_orig,
            "voxel_flipped": voxel_flipped
        }


def consistency_loss(pred_q_orig, pred_q_flipped):
    """
    Computes rotation consistency loss.
    pred_q_orig: (B, 4) or (B, 6)
    pred_q_flipped: (B, 4) or (B, 6)
    
    If the network outputs quaternions directly, we enforce that
    pred_q_flipped MUST equal the mathematically flipped pred_q_orig.
    Since V8 (our target) uses 6D representations, we must enforce consistency
    on the 6D vectors.
    """
    # For a horizontal mirror flip with 6D continuous rotation (V8 uses 6D):
    # R_flip = F @ R @ F
    # The first column (X-axis) gets its Y and Z components flipped.
    # The second column (Y-axis) gets its X component flipped.
    
    # Let's extract original axes
    v1_o = pred_q_orig[:, :3]  # (B, 3) 
    v2_o = pred_q_orig[:, 3:]  # (B, 3)
    
    # Expected flipped representation:
    # F = diag([-1, 1, 1])
    # x_new = F * x
    expected_v1 = v1_o.clone()
    expected_v1[:, 1] = -v1_o[:, 1]
    expected_v1[:, 2] = -v1_o[:, 2]
    
    expected_v2 = v2_o.clone()
    expected_v2[:, 0] = -v2_o[:, 0]
    
    expected_pred = torch.cat([expected_v1, expected_v2], dim=1)
    
    loss = F.mse_loss(pred_q_flipped, expected_pred)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    # We will load the V8 config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Domain Adaptation (RCT) on {device}")
    
    # Load V8 checkpoint
    print(f"Loading base model: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model = PoseNet(
        in_channels=config["model"]["in_channels"],
        rotation_repr=config["model"].get("rotation_repr", "quat"),
        backbone=config["model"]["backbone"],
        feature_dim=config["model"].get("feature_dim", 1024),
        hidden_dim=config["model"].get("hidden_dim", 256),
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # Freeze Backbone (Robust feature extractors)
    print("Freezing CNN backbone. Fine-tuning heads ONLY.")
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Setup RCT Dataset
    dataset = RCTDataset(
        data_dir=".",
        height=config["data"]["height"],
        width=config["data"]["width"],
        num_bins=config["data"]["num_bins"],
        sequence_length=config["data"].get("sequence_length", 1)
    )
    
    loader = DataLoader(
        dataset, batch_size=config["data"]["batch_size"], 
        shuffle=True, num_workers=2, drop_last=True
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    
    scaler = GradScaler('cuda')
    
    # Checkpoint dir
    ckpt_dir = "checkpoints_rct"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print("\nStarting Self-Supervised Domain Adaptation (RCT)...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        n_batches = 0
        t0 = time.time()
        
        for batch_idx, batch in enumerate(loader):
            voxel_orig = batch["voxel_orig"].to(device)
            voxel_flipped = batch["voxel_flipped"].to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # Forward pass on original
                out_orig = model(voxel_orig)
                q_orig = out_orig["quaternion"]
                
                # Forward pass on flipped
                out_flipped = model(voxel_flipped)
                q_flipped = out_flipped["quaternion"]
                
                # Consistency Loss
                loss = consistency_loss(q_orig, q_flipped)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} | RCT Loss: {loss.item():.6f}")
                
        dt = time.time() - t0
        avg_loss = total_loss / n_batches
        print(f"Epoch [{epoch+1}/{args.epochs}] Complete | Avg Loss: {avg_loss:.6f} | Time: {dt:.1f}s")
        
        # Save epoch checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config
        }, os.path.join(ckpt_dir, f"rct_epoch_{epoch+1}.pth"))
        
    print(f"Domain Adaptation complete. Checkpoints saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
