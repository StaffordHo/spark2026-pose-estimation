"""
Pseudo-Label Domain Adaptation Training.

This script fine-tunes a trained model directly on the Unlabeled Test Set
using the best past predictions (e.g., V8) as "Pseudo-Labels".
Heavy augmentations are applied to the input events to force the network
to learn robust, noise-invariant features for the Zero-G domain.
"""

import os
import sys
import argparse
import time
import yaml
import glob
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.pose_net import PoseNet
from src.event_representations import events_to_voxel_grid_fast
from src.dataset import SPARK2026Dataset
import h5py

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_pseudo_labels(csv_path):
    """Loads pseudo-labels from a CodaBench submission CSV."""
    # Submission CSV usually doesn't have a header, but let's be careful.
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'timestamp' or row[0] == 'Tx': # skip header if exists
                continue
            # Output from predict.py is usually:
            # timestamp, Tx, Ty, Tz, Qx, Qy, Qz, Qw
            # We want the 7 floats
            labels.append([float(x) for x in row[1:]])
    return np.array(labels, dtype=np.float32)

class PseudoDataset(Dataset):
    """
    Loads unlabeled test sequences (RT901 - RT928) and pairs them
    with pseudo-labels (from a known good submission CSV like V8).
    Applies heavy domain-corruption augmentations to force adaptation.
    """
    def __init__(self, data_dir=".", csv_path="submission.csv", height=448, width=448, num_bins=5, sequence_length=1):
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.window_size = sequence_length * 1000
        
        # Load Pseudo Labels
        self.pseudo_labels = load_pseudo_labels(csv_path)
        print(f"Loaded {len(self.pseudo_labels)} pseudo-labels from {csv_path}")
        
        # Load Test Sequence paths
        self.h5_files = sorted(glob.glob(os.path.join(data_dir, "test", "RT*.h5")))
        print(f"Loading {len(self.h5_files)} test sequences for Pseudo-Labeling...")
        
        self.samples = []
        self._seq_data = {}
        
        # We need to extract exactly the 5316 poses evaluated by CodaBench at 100ms intervals.
        # Let's read the template file to know exactly how many poses per sequence.
        template_counts = {}
        with open(os.path.join(data_dir, "test", "template.csv"), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'timestamp': continue
                seq = row[0][:5]
                template_counts[seq] = template_counts.get(seq, 0) + 1
        
        total_poses = 0
        for h5_path in self.h5_files:
            seq_name = os.path.basename(h5_path).replace(".h5", "")
            with h5py.File(h5_path, "r") as f:
                xs = f["events"]["xs"][()]
                ys = f["events"]["ys"][()]
                ts = f["events"]["ts"][()]
                ps = f["events"]["ps"][()]
                
                self._seq_data[seq_name] = {
                    "xs": xs, "ys": ys, "ts": ts, "ps": ps
                }
                
                start_t = ts[0]
                num_poses = template_counts.get(seq_name, 0)
                
                for i in range(num_poses):
                    # Evaluated strictly at end of 100ms (100,000us) macro-windows
                    t_end = start_t + (i + 1) * 100000
                    
                    self.samples.append({
                        "seq_name": seq_name, 
                        "t_end": t_end,
                        "label_idx": total_poses
                    })
                    total_poses += 1
                    
        assert total_poses == len(self.pseudo_labels), f"Poses {total_poses} != Labels {len(self.pseudo_labels)}"
        print(f"PseudoDataset: {len(self.samples)} test samples paired with labels.")

    def __len__(self):
        return len(self.samples)

    def apply_strong_augmentations(self, xs, ys, ts, ps):
        """
        Strong domain corruption to prevent the network from just memorizing
        the V8 labels perfectly on clean data. We want it to learn to recover
        the V8 labels even when the sensor data is heavily degraded.
        """
        num_events = len(xs)
        if num_events < 50:
            return xs, ys, ts, ps
            
        # 1. Random Event Dropout (Simulating sensor failure / varying density)
        drop_ratio = np.random.uniform(0.1, 0.5) # Drop 10-50%
        n_keep = int(num_events * (1.0 - drop_ratio))
        indices = np.random.choice(num_events, n_keep, replace=False)
        indices = np.sort(indices) # keep temporal order
        
        xs = xs[indices]
        ys = ys[indices]
        ts = ts[indices]
        ps = ps[indices]
        
        # 2. Random Rectangular Erasing (Simulating occlusions / glare)
        if np.random.rand() < 0.5:
            # Erase a random chunk of the sensor
            er_w = np.random.randint(50, 300)
            er_h = np.random.randint(50, 200)
            ex = np.random.randint(0, 1280 - er_w)
            ey = np.random.randint(0, 720 - er_h)
            
            mask = ~((xs >= ex) & (xs < ex + er_w) & (ys >= ey) & (ys < ey + er_h))
            xs = xs[mask]
            ys = ys[mask]
            ts = ts[mask]
            ps = ps[mask]
            
        return xs, ys, ts, ps

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_name = sample["seq_name"]
        t_end = sample["t_end"]
        label_idx = sample["label_idx"]
        
        data = self._seq_data[seq_name]
        t_start = t_end - self.window_size
        
        ts_all = data["ts"]
        i_start = np.searchsorted(ts_all, t_start, side='left')
        i_end = np.searchsorted(ts_all, t_end, side='left')
        
        xs = data["xs"][i_start:i_end]
        ys = data["ys"][i_start:i_end]
        ts = ts_all[i_start:i_end]
        ps = data["ps"][i_start:i_end]

        # Apply domain corruptions
        xs, ys, ts, ps = self.apply_strong_augmentations(xs, ys, ts, ps)

        voxel = events_to_voxel_grid_fast(
            xs, ys, ts, ps,
            num_bins=self.num_bins,
            height=self.height, width=self.width
        )
        
        # Denormed translation output expected by model
        lbl = self.pseudo_labels[label_idx]
        T = torch.tensor(lbl[0:3], dtype=torch.float32)
        Q = torch.tensor(lbl[3:7], dtype=torch.float32)
        
        # Normalize translation to match network targets
        T_norm = (T - SPARK2026Dataset.TRANS_MEAN) / SPARK2026Dataset.TRANS_STD
        
        return {
            "voxel": voxel,
            "translation": T_norm,
            "quaternion": Q
        }


def translation_loss(pred_t, gt_t):
    return F.mse_loss(pred_t, gt_t)

def rotation_loss(pred_v, gt_q):
    from src.losses import geodesic_loss
    return geodesic_loss(pred_v, gt_q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pseudo_csv", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Domain Adaptation (Pseudo-Labels) on {device}")
    
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

    # Let's unfreeze the entire network to let the CNN features fully adapt
    # to the new zero-G domain noise, but keep learning rate *very* low (1e-5).
    print("Fine-tuning entire network at low learning rate.")
    for param in model.parameters():
        param.requires_grad = True

    dataset = PseudoDataset(
        data_dir=".",
        csv_path=args.pseudo_csv,
        height=config["data"]["height"],
        width=config["data"]["width"],
        num_bins=config["data"]["num_bins"],
        sequence_length=config["data"].get("sequence_length", 1)
    )
    
    loader = DataLoader(
        dataset, batch_size=16, 
        shuffle=True, num_workers=2, drop_last=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    
    ckpt_dir = "checkpoints_pseudo"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print("\nStarting Pseudo-Label Domain Adaptation...")
    trans_weight = config["loss"].get("trans_weight", 1.0)
    rot_weight = config["loss"].get("rot_weight", 5.0)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_t_loss = 0
        total_r_loss = 0
        n_batches = 0
        t0 = time.time()
        
        for batch_idx, batch in enumerate(loader):
            voxel = batch["voxel"].to(device)
            gt_t = batch["translation"].to(device)
            gt_q = batch["quaternion"].to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                out = model(voxel)
                pred_t = out["translation"]
                pred_v = out["quaternion"] # 6D
                
                t_loss = translation_loss(pred_t, gt_t) * trans_weight
                r_loss = rotation_loss(pred_v, gt_q) * rot_weight
                loss = t_loss + r_loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_t_loss += t_loss.item()
            total_r_loss += r_loss.item()
            n_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} (T: {t_loss.item():.4f}, R: {r_loss.item():.4f})")
                
        dt = time.time() - t0
        avg_loss = total_loss / n_batches
        print(f"Epoch [{epoch+1}/{args.epochs}] Complete | Avg Loss: {avg_loss:.4f} | "
              f"T: {total_t_loss/n_batches:.4f} R: {total_r_loss/n_batches:.4f} | Time: {dt:.1f}s")
        
        # Save epoch checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config
        }, os.path.join(ckpt_dir, f"pseudo_epoch_{epoch+1}.pth"))
        
    print(f"Domain Adaptation complete. Checkpoints saved to {ckpt_dir}/")

if __name__ == "__main__":
    main()
