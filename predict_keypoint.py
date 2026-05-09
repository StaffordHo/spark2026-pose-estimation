"""
Prediction script for Keypoint + PnP pose estimation.

Loads a keypoint model checkpoint, detects 2D keypoints on test data,
and recovers 6DoF pose via EPnP solving.

Usage:
    python predict_keypoint.py --checkpoint checkpoints_kp/best.pth --output submission.csv
"""

import os
import sys
import argparse
import glob
import time

import numpy as np
import h5py
import torch
import pandas as pd
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.keypoint_posenet import KeypointPoseNet, solve_pnp
from src.event_representations import events_to_voxel_grid_fast
from src.spacecraft_model import (
    PROBA2_KEYPOINTS_3D, CAMERA_MATRIX, IMAGE_HEIGHT, IMAGE_WIDTH, N_KEYPOINTS
)


def load_model(checkpoint_path, device):
    """Load keypoint model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    model = KeypointPoseNet(
        in_channels=config["model"]["in_channels"],
        n_keypoints=config["model"]["n_keypoints"],
        heatmap_size=config["model"].get("heatmap_size", 64),
        fpn_channels=config["model"]["fpn_channels"],
        pretrained=False,  # Don't reload ImageNet weights
        freeze_backbone=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    
    return model, config


def predict_sequence(model, h5_path, pose_indices, config, device):
    """Run keypoint prediction + PnP on a test sequence."""
    num_bins = config["data"]["num_bins"]
    height = config["data"]["height"]
    width = config["data"]["width"]
    target_size_cfg = config["data"].get("target_size", None)
    target_size = tuple(target_size_cfg) if target_size_cfg else None
    
    with h5py.File(h5_path, "r") as f:
        xs = f["events"]["xs"][()]
        ys = f["events"]["ys"][()]
        ts = f["events"]["ts"][()]
        ps = f["events"]["ps"][()]
    
    # Generate synthetic timestamps (test files don't have labels)
    max_idx = max(pose_indices)
    timestamps = np.arange(0, max_idx + 1) * 1000
    
    predictions = {}
    
    for pose_idx in pose_indices:
        if pose_idx < 1:
            continue
        t_start = timestamps[pose_idx - 1]
        t_end = timestamps[pose_idx]
        
        # Binary search for event window
        i_start = np.searchsorted(ts, t_start, side='left')
        i_end = np.searchsorted(ts, t_end, side='left')
        
        ev_xs = xs[i_start:i_end]
        ev_ys = ys[i_start:i_end]
        ev_ts = ts[i_start:i_end]
        ev_ps = ps[i_start:i_end]
        
        # Build voxel grid
        if target_size is not None:
            target_h, target_w = target_size
            xs_s = ev_xs.astype(np.float32) * target_w / width
            ys_s = ev_ys.astype(np.float32) * target_h / height
            voxel = events_to_voxel_grid_fast(
                xs_s, ys_s, ev_ts, ev_ps,
                num_bins=num_bins, height=target_h, width=target_w
            )
        else:
            voxel = events_to_voxel_grid_fast(
                ev_xs, ev_ys, ev_ts, ev_ps,
                num_bins=num_bins, height=height, width=width
            )
        
        voxel_tensor = voxel.unsqueeze(0).to(device)
        
        # Predict keypoints
        with torch.no_grad():
            output = model(voxel_tensor)
        
        kp_norm = output["keypoints_norm"][0].cpu().numpy()  # (N, 2) in [0, 1]
        
        # Convert normalized coords to pixel coords
        kp_px = np.zeros_like(kp_norm)
        kp_px[:, 0] = kp_norm[:, 0] * IMAGE_WIDTH
        kp_px[:, 1] = kp_norm[:, 1] * IMAGE_HEIGHT
        
        # PnP solve
        success, translation, quaternion = solve_pnp(
            kp_px, PROBA2_KEYPOINTS_3D, CAMERA_MATRIX
        )
        
        if not success:
            # Fallback: use average pose from training set
            translation = np.array([-0.24, 0.03, 5.68])
            quaternion = np.array([0, 0, 0, 1.0])
        
        predictions[pose_idx] = {
            "Tx": translation[0],
            "Ty": translation[1],
            "Tz": translation[2],
            "Qx": quaternion[0],
            "Qy": quaternion[1],
            "Qz": quaternion[2],
            "Qw": quaternion[3],
        }
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Keypoint + PnP prediction")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="submission_kp.csv")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, config = load_model(args.checkpoint, device)
    print("Model loaded successfully")
    
    # Read template
    template_path = os.path.join("test", "template.csv")
    template = pd.read_csv(template_path)
    
    # Group template by sequence
    seq_to_indices = {}
    for idx, row in template.iterrows():
        filename = row["timestamp"]
        parts = filename.rsplit("_", 1)
        seq_name = parts[0]
        pose_idx = int(parts[1])
        if seq_name not in seq_to_indices:
            seq_to_indices[seq_name] = []
        seq_to_indices[seq_name].append(pose_idx)
    
    print(f"\nSequences to process: {sorted(seq_to_indices.keys())}")
    
    all_preds = {}
    for seq_name in sorted(seq_to_indices.keys()):
        h5_path = os.path.join("test", f"{seq_name}.h5")
        if not os.path.exists(h5_path):
            print(f"WARNING: {h5_path} not found, skipping")
            continue
        
        pose_indices = seq_to_indices[seq_name]
        print(f"\nProcessing {seq_name} ({len(pose_indices)} poses)...", end=" ", flush=True)
        
        t0 = time.time()
        preds = predict_sequence(model, h5_path, pose_indices, config, device)
        dt = time.time() - t0
        print(f"done ({dt:.1f}s)")
        
        for pose_idx, p in preds.items():
            # Use zero-padded format to match template (e.g. RT901_001)
            key = f"{seq_name}_{pose_idx:03d}"
            all_preds[key] = p
    
    # Fill template
    for idx, row in template.iterrows():
        filename = row["timestamp"]
        key = filename
        if key in all_preds:
            p = all_preds[key]
            template.at[idx, "Tx"] = p["Tx"]
            template.at[idx, "Ty"] = p["Ty"]
            template.at[idx, "Tz"] = p["Tz"]
            template.at[idx, "Qx"] = p["Qx"]
            template.at[idx, "Qy"] = p["Qy"]
            template.at[idx, "Qz"] = p["Qz"]
            template.at[idx, "Qw"] = p["Qw"]
    
    # Save
    template.to_csv(args.output, index=False)
    print(f"\nSaved predictions to {args.output}")
    
    # Zip
    import zipfile
    zip_path = args.output.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output, "submission.csv")
    print(f"Zipped to {zip_path}")


if __name__ == "__main__":
    main()
