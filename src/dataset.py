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
    
    # Translation normalization stats
    # X and Y means set to 0 to ensure flipping augmentation doesn't introduce bias
    TRANS_MEAN = torch.tensor([0.0, 0.0, 5.68], dtype=torch.float32)
    TRANS_STD = torch.tensor([0.93, 0.50, 1.40], dtype=torch.float32)
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_bins: int = 5,
        height: int = 720,
        width: int = 1280,
        window_size: int = 1000,  # 100ms in 100µs units
        sequences: Optional[List[str]] = None,
        transform=None,
        target_size: Optional[Tuple[int, int]] = None,  # (H, W) to resize to
        augmentation: bool = False,
        normalize_translation: bool = False,
        density_augmentation: bool = False,
        robust_augmentation: bool = False,
        sequence_length: int = 1,
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
            target_size: Optional (H, W) tuple to resize voxel grids
            augmentation: Whether to apply data augmentation (train only)
            density_augmentation: Randomly subsample events (domain robustness)
            robust_augmentation: Voxel jittering and erasing (V11+)
        """
        self.data_dir = data_dir
        self.split = split
        self.num_bins = num_bins
        self.height = height
        self.width = width
        self.window_size = window_size
        self.transform = transform
        self.target_size = target_size
        self.augmentation = augmentation and (split == "train")
        self.normalize_translation = normalize_translation
        self.density_augmentation = density_augmentation and (split == "train")
        self.robust_augmentation = robust_augmentation and (split == "train")
        self.sequence_length = sequence_length
        
        # Get all h5 files
        h5_files = sorted(glob.glob(os.path.join(data_dir, "h5", "RT*.h5")))
        
        if sequences is not None:
            h5_files = [f for f in h5_files if os.path.basename(f).replace(".h5", "") in sequences]
        else:
            # Strict split: 
            # Train: 000-239 (240 sequences)
            # Val:   240-269 (30 sequences)
            # Test:  270-299 (30 sequences)
            if split == "train":
                h5_files = h5_files[:240]
            elif split == "val":
                h5_files = h5_files[240:270]
            elif split == "test":
                h5_files = h5_files[270:300]
            elif split == "train_all":
                # Use for final model AFTER hyperparameter tuning
                h5_files = h5_files[:270]
            else:
                raise ValueError(f"Unknown split: {split}")
        
        # Pre-load ALL sequences into RAM for fast access
        # ~13 MB per sequence × 240 sequences ≈ 3 GB total — fits easily
        self.samples = []
        self._seq_data: Dict[str, dict] = {}
        
        print(f"Loading {split} split with {len(h5_files)} sequences into RAM...", flush=True)
        
        for h5_path in h5_files:
            seq_name = os.path.basename(h5_path).replace(".h5", "")
            with h5py.File(h5_path, "r") as f:
                self._seq_data[seq_name] = {
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
                num_poses = len(self._seq_data[seq_name]["timestamp"])
                
                # Each pose (except the first) is a valid sample
                for pose_idx in range(1, num_poses):
                    self.samples.append((seq_name, pose_idx))
        
        print(f"Total samples: {len(self.samples)}", flush=True)
        if self.target_size:
            print(f"Resizing voxel grids to {self.target_size}", flush=True)
        if self.augmentation:
            print(f"Data augmentation enabled", flush=True)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_name, pose_idx = self.samples[idx]
        
        data = self._seq_data[seq_name]
        
        # Get timestamps for this window (rigidly sequence_length * 100ms)
        # 1 time unit = 100µs, so 1000 units = 100ms.
        t_end = data["timestamp"][pose_idx]
        t_start = t_end - (self.sequence_length * 1000)
        
        # Use binary search for O(log N) event windowing
        # Timestamps are sorted, so searchsorted finds the window instantly
        ts_all = data["ts"]
        i_start = np.searchsorted(ts_all, t_start, side='left')
        i_end = np.searchsorted(ts_all, t_end, side='left')
        
        xs = data["xs"][i_start:i_end]
        ys = data["ys"][i_start:i_end]
        ts = ts_all[i_start:i_end]
        ps = data["ps"][i_start:i_end]
        
        # Random event density augmentation — bridges the 6x density gap
        # between synthetic training and real test data
        if self.density_augmentation and len(xs) > 10:
            keep_ratio = np.random.uniform(0.2, 1.0)
            n_keep = max(10, int(len(xs) * keep_ratio))
            indices = np.sort(np.random.choice(len(xs), n_keep, replace=False))
            xs = xs[indices]
            ys = ys[indices]
            ts = ts[indices]
            ps = ps[indices]
        
        # Build voxel grid directly at target resolution if specified
        # This is ~14x faster than building at full res and resizing
        if self.target_size is not None:
            target_h, target_w = self.target_size
            # Scale event coordinates to target resolution
            xs_scaled = (xs.astype(np.float32) * target_w / self.width)
            ys_scaled = (ys.astype(np.float32) * target_h / self.height)
            voxel = events_to_voxel_grid_fast(
                xs_scaled, ys_scaled, ts, ps,
                num_bins=self.num_bins,
                height=target_h,
                width=target_w
            )
        else:
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
        
        # Normalize translation to ~N(0,1) for stable training
        if self.normalize_translation:
            translation = (translation - self.TRANS_MEAN) / self.TRANS_STD
        
        # --- Data Augmentation ---
        if self.augmentation:
            # Random horizontal flip (50%)
            if torch.rand(1).item() > 0.5:
                voxel = torch.flip(voxel, dims=[2])  # Flip width (X)
                translation[0] = -translation[0]     # Negate Tx (since mean is 0)
                # Flip quaternion: negate Qy and Qz for horizontal flip (reflection across YZ)
                quaternion[1] = -quaternion[1]  # Qy
                quaternion[2] = -quaternion[2]  # Qz
            
            # Random vertical flip (50%)
            if torch.rand(1).item() > 0.5:
                voxel = torch.flip(voxel, dims=[1])  # Flip height (Y)
                translation[1] = -translation[1]     # Negate Ty (since mean is 0)
                # Flip quaternion: negate Qx and Qz for vertical flip (reflection across XZ)
                quaternion[0] = -quaternion[0]  # Qx
                quaternion[2] = -quaternion[2]  # Qz
            
            # Random polarity flip (30%) - swap ON and OFF channels
            if torch.rand(1).item() > 0.7:
                # ON channels: 0, 2, 4, ... OFF channels: 1, 3, 5, ...
                num_ch = voxel.shape[0]
                perm = torch.zeros(num_ch, dtype=torch.long)
                for c in range(0, num_ch, 2):
                    perm[c] = c + 1    # ON -> OFF position
                    perm[c + 1] = c    # OFF -> ON position
                voxel = voxel[perm]
            
            # Random additive noise (20%)
            if torch.rand(1).item() > 0.8:
                noise = torch.rand_like(voxel) * 0.05
                voxel = voxel + noise
            
            # --- Voxel Jittering & Erasing (Gated for V13) ---
            if getattr(self, 'robust_augmentation', False):
                # Random Translation (Shift ±5%)
                if torch.rand(1).item() > 0.5:
                    # shift in px
                    h, w = voxel.shape[1], voxel.shape[2]
                    dy = int(h * 0.05 * (torch.rand(1).item() * 2 - 1))
                    dx = int(w * 0.05 * (torch.rand(1).item() * 2 - 1))
                    voxel = torch.roll(voxel, shifts=(dy, dx), dims=(1, 2))
                
                # Random Erasing (Cutout)
                if torch.rand(1).item() > 0.5:
                    h, w = voxel.shape[1], voxel.shape[2]
                    erase_h = int(h * 0.2)
                    erase_w = int(w * 0.2)
                    y = torch.randint(0, h - erase_h, (1,)).item()
                    x = torch.randint(0, w - erase_w, (1,)).item()
                    voxel[:, y:y+erase_h, x:x+erase_w] = 0.0
        
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
    target_size: Optional[Tuple[int, int]] = None,
    augmentation: bool = False,
    normalize_translation: bool = False,
    density_augmentation: bool = False,
    robust_augmentation: bool = False,
    sequence_length: int = 1,
    train_split: str = "train",
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, val, test dataloaders.
    train_split: "train" (240 seqs) or "train_all" (270 seqs, train+val combined)
    """
    train_dataset = SPARK2026Dataset(
        data_dir, split=train_split, num_bins=num_bins,
        target_size=target_size, augmentation=augmentation,
        normalize_translation=normalize_translation,
        density_augmentation=density_augmentation, 
        robust_augmentation=robust_augmentation, 
        sequence_length=sequence_length, **kwargs
    )
    val_dataset = SPARK2026Dataset(
        data_dir, split="val", num_bins=num_bins,
        target_size=target_size, augmentation=False,
        normalize_translation=normalize_translation,
        sequence_length=sequence_length, **kwargs
    )
    test_dataset = SPARK2026Dataset(
        data_dir, split="test", num_bins=num_bins,
        target_size=target_size, augmentation=False,
        normalize_translation=normalize_translation,
        sequence_length=sequence_length, **kwargs
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader
