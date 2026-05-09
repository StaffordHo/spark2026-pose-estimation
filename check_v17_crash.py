import yaml
import torch
from src.dataset import SPARK2026Dataset
from src.models import PoseNet
from train import create_model

# Load config
with open("config/rtx4090_v17.yaml", "r") as f:
    config = yaml.safe_load(f)

print("Initializing V17 Model...")
model = create_model(config)
model = model.cuda()
model.train()

print("Initializing 30-channel Dataset...")
dataset = SPARK2026Dataset(
    data_dir=".", split="val", num_bins=15, 
    target_size=(320, 320), augmentation=False, 
    normalize_translation=True, density_augmentation=False, 
    sequence_length=3
)

print(f"Dataset size: {len(dataset)}")
print("Running Forward Pass Stress Test...")

total = len(dataset)
for i in range(total):
    sample = dataset[i]
    voxel = sample["voxel"].unsqueeze(0).cuda()
    
    # Fast-fail dimensionality check before the model
    if voxel.shape[1] != 30:
        print(f"\nCRITICAL SHAPE FAILURE BATCH {i}:")
        print(f"File: {sample['seq_name']}, Pose Index: {sample['pose_idx']}")
        print(f"Generated Voxel Shape: {voxel.shape}")
        break
        print(f"\nCRITICAL CRASH ON BATCH {i}:")
        import traceback
        traceback.print_exc()
        break
        
print("V17 Architecture Stress Test Complete.")
