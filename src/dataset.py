"""
SPARK2026 Dataset for spacecraft pose estimation from event cameras.
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from .event_representations import events_to_voxel_grid_fast


class SPARK2026Dataset(Dataset):
    """
    Dataset class for SPARK2026 synthetic event camera data.
    
    Each sample consists of events within a time window and the corresponding pose.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_bins: int = 5,
        height: int = 720,
        width: int = 1280,
        window_size: int = 1000,  # 100ms in 100µs units
        sequences: Optional[List[str]] = None,
        transform=None
    ):
        """
        Args:
            data_dir: Path to directory containing h5 files
            split: One of "train", "val", "test"
            num_bins: Number of temporal bins for voxel grid
            height, width: Image dimensions
            window_size: Event window size in timestamp units (100µs)
            sequences: Optional list of specific sequences to use
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.split = split
        self.num_bins = num_bins
        self.height = height
        self.width = width
        self.window_size = window_size
        self.transform = transform
        
        # Get all h5 files
        h5_files = sorted(glob.glob(os.path.join(data_dir, "h5", "RT*.h5")))
        
        if sequences is not None:
            h5_files = [f for f in h5_files if os.path.basename(f).replace(".h5", "") in sequences]
        else:
            # Default split: 240 train, 30 val, 30 test
            if split == "train":
                h5_files = h5_files[:240]
            elif split == "val":
                h5_files = h5_files[240:270]
            elif split == "test":
                h5_files = h5_files[270:300]
        
        # Build index of all samples (sequence, pose_idx)
        self.samples = []
        self.file_cache: Dict[str, dict] = {}
        
        print(f"Loading {split} split with {len(h5_files)} sequences...", flush=True)
        
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                num_poses = len(f["labels"]["data"]["timestamp"])
                seq_name = os.path.basename(h5_path).replace(".h5", "")
                
                # Each pose (except the first) is a valid sample
                for pose_idx in range(1, num_poses):
                    self.samples.append((h5_path, seq_name, pose_idx))
        
        print(f"Total samples: {len(self.samples)}", flush=True)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_sequence(self, h5_path: str) -> dict:
        """Load and cache a sequence."""
        if h5_path not in self.file_cache:
            with h5py.File(h5_path, "r") as f:
                self.file_cache[h5_path] = {
                    "xs": f["events"]["xs"][()],
                    "ys": f["events"]["ys"][()],
                    "ts": f["events"]["ts"][()],
                    "ps": f["events"]["ps"][()],
                    "Tx": f["labels"]["data"]["Tx"][()],
                    "Ty": f["labels"]["data"]["Ty"][()],
                    "Tz": f["labels"]["data"]["Tz"][()],
                    "Qx": f["labels"]["data"]["Qx"][()],
                    "Qy": f["labels"]["data"]["Qy"][()],
                    "Qz": f["labels"]["data"]["Qz"][()],
                    "Qw": f["labels"]["data"]["Qw"][()],
                    "timestamp": f["labels"]["data"]["timestamp"][()],
                }
            
            # Limit cache size to avoid memory issues
            if len(self.file_cache) > 50:
                oldest_key = next(iter(self.file_cache))
                del self.file_cache[oldest_key]
        
        return self.file_cache[h5_path]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h5_path, seq_name, pose_idx = self.samples[idx]
        
        data = self._load_sequence(h5_path)
        
        # Get timestamps for this and previous pose
        t_end = data["timestamp"][pose_idx]
        t_start = data["timestamp"][pose_idx - 1]
        
        # Get events in this window
        event_mask = (data["ts"] >= t_start) & (data["ts"] < t_end)
        
        xs = data["xs"][event_mask]
        ys = data["ys"][event_mask]
        ts = data["ts"][event_mask]
        ps = data["ps"][event_mask]
        
        # Convert to voxel grid
        voxel = events_to_voxel_grid_fast(
            xs, ys, ts, ps,
            num_bins=self.num_bins,
            height=self.height,
            width=self.width
        )
        
        # Get pose labels
        translation = torch.tensor([
            data["Tx"][pose_idx],
            data["Ty"][pose_idx],
            data["Tz"][pose_idx]
        ], dtype=torch.float32)
        
        quaternion = torch.tensor([
            data["Qx"][pose_idx],
            data["Qy"][pose_idx],
            data["Qz"][pose_idx],
            data["Qw"][pose_idx]
        ], dtype=torch.float32)
        
        # Normalize quaternion
        quaternion = quaternion / quaternion.norm()
        
        sample = {
            "voxel": voxel,
            "translation": translation,
            "quaternion": quaternion,
            "seq_name": seq_name,
            "pose_idx": pose_idx
        }
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    num_bins: int = 5,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, val, test dataloaders.
    """
    train_dataset = SPARK2026Dataset(data_dir, split="train", num_bins=num_bins, **kwargs)
    val_dataset = SPARK2026Dataset(data_dir, split="val", num_bins=num_bins, **kwargs)
    test_dataset = SPARK2026Dataset(data_dir, split="test", num_bins=num_bins, **kwargs)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
