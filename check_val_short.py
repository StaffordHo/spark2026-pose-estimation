import torch
from evaluate import evaluate_checkpoint
from src.dataset import SPARK2026Dataset
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoints_v6/best.pth"

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
ckpt_config = checkpoint["config"]

dataset = SPARK2026Dataset(
    ckpt_config["data"]["data_dir"], split="val",
    num_bins=ckpt_config["data"]["num_bins"],
    target_size=tuple(ckpt_config["data"]["target_size"]),
    normalize_translation=ckpt_config["data"].get("normalize_translation", True)
)
# Evaluate only first 100 samples
subset_dataset = Subset(dataset, range(min(100, len(dataset))))
val_loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)

print(f"Evaluating first 100 samples of V6 best.pth WITHOUT TTA...")
results = evaluate_checkpoint(checkpoint_path, val_loader, device, tta=False)

print("\nRESULTS (First 100 samples):")
print(f"  Trans Error: {results['trans_mean']:.4f}m (median: {results['trans_median']:.4f}m)")
print(f"  Orient Error: {results['orient_mean']:.4f} deg (median: {results['orient_median']:.4f} deg)")
print(f"  Pose Error: {results['pose_error']:.4f}")
