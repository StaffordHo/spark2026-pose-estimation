"""
Evaluate the keypoint model on the validation set.
Computes translation and rotation errors using the ground truth labels.
"""
import os
import sys
import argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SPARK2026Dataset
from src.models.keypoint_posenet import KeypointPoseNet, solve_pnp
from src.spacecraft_model import PROBA2_KEYPOINTS_3D, CAMERA_MATRIX, IMAGE_WIDTH, IMAGE_HEIGHT


def compute_metrics(pred_t, pred_q, gt_t, gt_q):
    """Compute translation error (m) and rotation error (rad)."""
    # Translation error (L2 distance)
    t_err = np.linalg.norm(pred_t - gt_t)
    
    # Rotation error
    q1 = pred_q / np.linalg.norm(pred_q)
    q2 = gt_q / np.linalg.norm(gt_q)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    # The angle between two quaternions
    theta = 2 * np.arccos(dot)
    return t_err, theta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_kp/latest.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    
    model = KeypointPoseNet(
        in_channels=config["model"]["in_channels"],
        n_keypoints=config["model"]["n_keypoints"],
        heatmap_size=config["model"].get("heatmap_size", 64),
        fpn_channels=config["model"]["fpn_channels"],
        pretrained=False,
        freeze_backbone=False
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    
    # Load Validation Data
    val_dataset = SPARK2026Dataset(
        data_dir=".",
        split="val",
        num_bins=config["data"]["num_bins"],
        target_size=tuple(config["data"].get("target_size", [256, 256])),
        augmentation=False,
        normalize_translation=False
    )
    
    print(f"Evaluating on {len(val_dataset)} validation samples...")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2
    )
    
    trans_errs = []
    rot_errs = []
    pnp_fails = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating keypoint model"):
            voxel = batch["voxel"].to(device)
            # Ground truth pose
            gt_translation = batch["translation"][0].numpy()
            gt_quaternion = batch["quaternion"][0].numpy()
            
            output = model(voxel)
            kp_norm = output["keypoints_norm"][0].cpu().numpy()  # (N, 2)
            
            kp_px = np.zeros_like(kp_norm)
            kp_px[:, 0] = kp_norm[:, 0] * IMAGE_WIDTH
            kp_px[:, 1] = kp_norm[:, 1] * IMAGE_HEIGHT
            
            success, pred_t, pred_q = solve_pnp(kp_px, PROBA2_KEYPOINTS_3D, CAMERA_MATRIX)
            
            if not success:
                pnp_fails += 1
                pred_t = np.array([-0.24, 0.03, 5.68])
                pred_q = np.array([0, 0, 0, 1.0])
            
            t_err, r_err = compute_metrics(pred_t, pred_q, gt_translation, gt_quaternion)
            trans_errs.append(t_err)
            rot_errs.append(r_err)
    
    print("\n--- Validation Results ---")
    print(f"Mean Translation Error: {np.mean(trans_errs):.4f} m")
    print(f"Mean Rotation Error:    {np.mean(rot_errs):.4f} rad ({np.degrees(np.mean(rot_errs)):.2f}°)")
    print(f"PnP Failures (fallback): {pnp_fails} / {len(val_dataset)}")
    
    # Also print medians for robustness
    print(f"Median Translation Err: {np.median(trans_errs):.4f} m")
    print(f"Median Rotation Err:    {np.median(rot_errs):.4f} rad ({np.degrees(np.median(rot_errs)):.2f}°)")


if __name__ == "__main__":
    main()
