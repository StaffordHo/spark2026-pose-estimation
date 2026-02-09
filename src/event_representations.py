"""
Event camera data representations for deep learning.
Converts raw event streams to tensor formats suitable for CNNs.
"""

import numpy as np
import torch


def events_to_voxel_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    ps: np.ndarray,
    num_bins: int = 5,
    height: int = 720,
    width: int = 1280,
    normalize: bool = True
) -> torch.Tensor:
    """
    Convert events to a spatio-temporal voxel grid.
    
    Args:
        xs: X-coordinates of events
        ys: Y-coordinates of events  
        ts: Timestamps of events (in 100µs units)
        ps: Polarities of events (0 or 1)
        num_bins: Number of temporal bins
        height: Image height
        width: Image width
        normalize: Whether to normalize the voxel grid
        
    Returns:
        Voxel grid tensor of shape (2*num_bins, H, W) - 2 channels per bin for ON/OFF
    """
    if len(xs) == 0:
        return torch.zeros((2 * num_bins, height, width), dtype=torch.float32)
    
    # Normalize timestamps to [0, num_bins-1]
    ts = ts.astype(np.float64)
    t_min, t_max = ts.min(), ts.max()
    
    if t_max - t_min > 0:
        ts_norm = (ts - t_min) / (t_max - t_min) * (num_bins - 1)
    else:
        ts_norm = np.zeros_like(ts)
    
    # Create voxel grid: 2 channels per bin (ON and OFF events)
    voxel_grid = np.zeros((2 * num_bins, height, width), dtype=np.float32)
    
    # Bilinear interpolation in time
    t_floor = np.floor(ts_norm).astype(np.int32)
    t_ceil = np.clip(t_floor + 1, 0, num_bins - 1)
    t_floor = np.clip(t_floor, 0, num_bins - 1)
    
    weight_ceil = ts_norm - t_floor
    weight_floor = 1.0 - weight_ceil
    
    # Clip coordinates to valid range
    xs = np.clip(xs, 0, width - 1).astype(np.int32)
    ys = np.clip(ys, 0, height - 1).astype(np.int32)
    
    # Separate ON and OFF events
    on_mask = ps == 1
    off_mask = ps == 0
    
    # Accumulate events into voxel grid
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        t_f, t_c = t_floor[i], t_ceil[i]
        w_f, w_c = weight_floor[i], weight_ceil[i]
        
        if on_mask[i]:
            voxel_grid[t_f * 2, y, x] += w_f
            voxel_grid[t_c * 2, y, x] += w_c
        else:
            voxel_grid[t_f * 2 + 1, y, x] += w_f
            voxel_grid[t_c * 2 + 1, y, x] += w_c
    
    if normalize:
        # Normalize per-channel
        for c in range(voxel_grid.shape[0]):
            if voxel_grid[c].max() > 0:
                voxel_grid[c] = voxel_grid[c] / voxel_grid[c].max()
    
    return torch.from_numpy(voxel_grid)


def events_to_voxel_grid_fast(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    ps: np.ndarray,
    num_bins: int = 5,
    height: int = 720,
    width: int = 1280,
    normalize: bool = True
) -> torch.Tensor:
    """
    Fast vectorized version of voxel grid conversion.
    Uses numpy advanced indexing for speed.
    """
    if len(xs) == 0:
        return torch.zeros((2 * num_bins, height, width), dtype=torch.float32)
    
    # Normalize timestamps
    ts = ts.astype(np.float64)
    t_min, t_max = ts.min(), ts.max()
    
    if t_max - t_min > 0:
        ts_norm = (ts - t_min) / (t_max - t_min) * (num_bins - 1)
    else:
        ts_norm = np.zeros_like(ts)
    
    # Compute bin indices and weights
    t_floor = np.floor(ts_norm).astype(np.int32)
    t_ceil = np.clip(t_floor + 1, 0, num_bins - 1)
    t_floor = np.clip(t_floor, 0, num_bins - 1)
    
    weight_ceil = (ts_norm - t_floor).astype(np.float32)
    weight_floor = (1.0 - weight_ceil).astype(np.float32)
    
    # Clip coordinates
    xs = np.clip(xs, 0, width - 1).astype(np.int32)
    ys = np.clip(ys, 0, height - 1).astype(np.int32)
    
    # Channel indices: ON events go to even channels, OFF to odd
    channel_offset = 1 - ps  # ON(1)->0, OFF(0)->1
    
    # Create voxel grid using np.add.at for accumulation
    voxel_grid = np.zeros((2 * num_bins, height, width), dtype=np.float32)
    
    # Floor contribution
    ch_floor = t_floor * 2 + channel_offset
    np.add.at(voxel_grid, (ch_floor, ys, xs), weight_floor)
    
    # Ceil contribution  
    ch_ceil = t_ceil * 2 + channel_offset
    np.add.at(voxel_grid, (ch_ceil, ys, xs), weight_ceil)
    
    if normalize:
        max_vals = voxel_grid.max(axis=(1, 2), keepdims=True)
        max_vals[max_vals == 0] = 1.0
        voxel_grid = voxel_grid / max_vals
    
    return torch.from_numpy(voxel_grid)


def events_to_frame(
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    height: int = 720,
    width: int = 1280
) -> torch.Tensor:
    """
    Simple event frame representation (2 channels: ON and OFF).
    
    Args:
        xs, ys: Event coordinates
        ps: Polarities
        height, width: Frame dimensions
        
    Returns:
        Event frame tensor of shape (2, H, W)
    """
    frame = np.zeros((2, height, width), dtype=np.float32)
    
    if len(xs) == 0:
        return torch.from_numpy(frame)
    
    xs = np.clip(xs, 0, width - 1).astype(np.int32)
    ys = np.clip(ys, 0, height - 1).astype(np.int32)
    
    # ON events
    on_mask = ps == 1
    np.add.at(frame[0], (ys[on_mask], xs[on_mask]), 1.0)
    
    # OFF events
    off_mask = ps == 0
    np.add.at(frame[1], (ys[off_mask], xs[off_mask]), 1.0)
    
    # Normalize
    for c in range(2):
        if frame[c].max() > 0:
            frame[c] = frame[c] / frame[c].max()
    
    return torch.from_numpy(frame)
