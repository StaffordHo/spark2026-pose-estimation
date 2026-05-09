import torch
import os

checkpoint_path = r"c:\Users\kk\Desktop\spark2026-pose-estimation\checkpoints_v10\best.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Best Epoch: {checkpoint['epoch']}")
    print(f"Best Val Loss: {checkpoint['val_loss']:.4f}")
    
    # Check if config is present to verify resolution
    if 'config' in checkpoint:
        print(f"Resolution: {checkpoint['config']['data']['target_size']}")
        print(f"Rotation Repr: {checkpoint['config']['model']['rotation_repr']}")
else:
    print("Checkpoint not found")

checkpoint_latest = r"c:\Users\kk\Desktop\spark2026-pose-estimation\checkpoints_v10\epoch_30.pth"
if os.path.exists(checkpoint_latest):
    checkpoint = torch.load(checkpoint_latest, map_location="cpu", weights_only=False)
    print(f"\nEpoch 30 Checkpoint:")
    print(f"Best Val Loss in checkpoint: {checkpoint.get('best_val_loss', 'N/A')}")
