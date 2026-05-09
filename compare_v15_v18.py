"""Compare V15 vs V18 on validation set to predict ensemble improvement."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate_checkpoint, load_config
from src.dataset import SPARK2026Dataset
import numpy as np

device = torch.device('cuda')

# Create val loader using V15 config
ckpt = torch.load('checkpoints_v15/epoch_90.pth', map_location='cpu', weights_only=False)
ckpt_config = ckpt['config']
target_size = tuple(ckpt_config['data'].get('target_size', [256, 256]))
num_bins = ckpt_config['data'].get('num_bins', 5)
normalize_translation = ckpt_config['data'].get('normalize_translation', True)

val_dataset = SPARK2026Dataset(
    ckpt_config['data']['data_dir'], split='val',
    num_bins=num_bins, target_size=target_size,
    augmentation=False, normalize_translation=normalize_translation
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True
)

print(f"Evaluating on {len(val_dataset)} val samples with TTA...")

# Evaluate V15
print("\n=== V15 (Rotation Expert) ===")
r15 = evaluate_checkpoint('checkpoints_v15/epoch_90.pth', val_loader, device, tta=True)
v15_trans = r15["trans_mean"]
v15_rot = r15["orient_mean"]
v15_pose = r15["pose_error"]
print(f"Trans: {v15_trans:.4f}m | Orient: {v15_rot:.4f} deg | Pose: {v15_pose:.4f}")

# Evaluate V18
print("\n=== V18 (Translation Expert) ===")
r18 = evaluate_checkpoint('checkpoints_v18/best.pth', val_loader, device, tta=True)
v18_trans = r18["trans_mean"]
v18_rot = r18["orient_mean"]
v18_pose = r18["pose_error"]
print(f"Trans: {v18_trans:.4f}m | Orient: {v18_rot:.4f} deg | Pose: {v18_pose:.4f}")

# Compare
print("\n" + "=" * 60)
print("ENSEMBLE COMPARISON (V18 Translation + V15 Rotation)")
print("=" * 60)
print(f"V15 Translation: {v15_trans:.4f}m")
print(f"V18 Translation: {v18_trans:.4f}m")
trans_diff = v15_trans - v18_trans
if v18_trans < v15_trans:
    print(f"  -> V18 is BETTER by {trans_diff:.4f}m ({trans_diff/v15_trans*100:.1f}%)")
else:
    print(f"  -> V18 is WORSE by {-trans_diff:.4f}m ({-trans_diff/v15_trans*100:.1f}%)")

print(f"\nV15 Rotation: {v15_rot:.4f} deg (used in ensemble)")
print(f"V18 Rotation: {v18_rot:.4f} deg (NOT used in ensemble)")

ensemble_pose = v18_trans + v15_rot * np.pi / 180.0
print(f"\nEnsemble Pose Score: {ensemble_pose:.4f}")
print(f"V15 Standalone Pose Score: {v15_pose:.4f}")
pose_diff = v15_pose - ensemble_pose
if ensemble_pose < v15_pose:
    print(f"  -> ENSEMBLE IS BETTER by {pose_diff:.4f} ({pose_diff/v15_pose*100:.1f}%)")
else:
    print(f"  -> ENSEMBLE IS WORSE by {-pose_diff:.4f} ({-pose_diff/v15_pose*100:.1f}%)")
