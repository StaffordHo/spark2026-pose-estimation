import torch
import os

checkpoint_path = "checkpoints_v17/latest.pth"
if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found at {checkpoint_path}")
else:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
    if 'val_metrics' in checkpoint:
        metrics = checkpoint['val_metrics']
        print(f"Val Translation Error (m): {metrics.get('trans_error_m', 'N/A')}")
        print(f"Val Rotation Error (deg): {metrics.get('rot_error_deg', 'N/A')}")
        print(f"Val Pose Error: {metrics.get('pose_error', 'N/A')}")
    else:
        print("No val_metrics found in checkpoint.")
