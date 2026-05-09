import torch
import os
import sys
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.getcwd())

from src.dataset import SPARK2026Dataset
from src.models.pose_net import build_model
from src.metrics import MetricTracker

def eval_split(checkpoint_path, split_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {split_name} using {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    # Force test/val settings
    config["data"]["augmentation"] = False
    config["data"]["density_augmentation"] = False
    
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    dataset = SPARK2026Dataset(
        data_dir=config["data"]["data_dir"],
        num_bins=config["data"]["num_bins"],
        height=config["data"]["height"],
        width=config["data"]["width"],
        target_size=config["data"]["target_size"],
        split=split_name,
        augmentation=False,
        normalize_translation=config["data"]["normalize_translation"]
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    tracker = MetricTracker()
    
    with torch.no_grad():
        for batch in loader:
            voxel = batch["voxel"].to(device)
            target = {
                "translation": batch["translation"].to(device),
                "quaternion": batch["quaternion"].to(device)
            }
            pred = model(voxel)
            tracker.update(pred, target)
    
    metrics = tracker.compute()
    print(f"\nResults for {split_name}:")
    print(f"  Pos Error: {metrics['pos_error_mean']:.3f}m")
    print(f"  Rot Error: {metrics['rot_error_mean']:.2f} deg")

if __name__ == "__main__":
    v7_cp = r"checkpoints_v7\best.pth"
    v13_cp = r"checkpoints_v13\latest.pth"
        
    print("\n=== V7 EVALUATION (Baseline) ===")
    if os.path.exists(v7_cp):
        eval_split(v7_cp, "val")
    else:
        print("V7 Checkpoint not found")

    print("\n=== V13 EVALUATION (Latest Fine-tune) ===")
    if os.path.exists(v13_cp):
        eval_split(v13_cp, "val")
    else:
        print("V13 Checkpoint not found")
