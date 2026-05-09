import torch
import os

def check_checkpoint(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    print(f"--- Checking {os.path.basename(path)} ---")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Epoch: {checkpoint.get('epoch')}")
    print(f"Val Loss: {checkpoint.get('val_loss')}")
    
    # In V11, we are looking for improved orientation.
    # The training script logs Val Pos Error and Val Rot Error to stdout.
    # Let's see if there are any metric dictionaries in the checkpoint.
    if 'val_metrics' in checkpoint:
        print(f"Val Metrics: {checkpoint['val_metrics']}")
    
    if 'config' in checkpoint:
        print(f"Backbone: {checkpoint['config']['model']['backbone']}")
        print(f"Target Size: {checkpoint['config']['data']['target_size']}")

if __name__ == "__main__":
    check_checkpoint("checkpoints_v13/latest.pth")
    check_checkpoint("checkpoints_v7/best.pth")
