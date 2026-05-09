import torch
from evaluate import evaluate_checkpoint
from src.dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoints_v17/latest.pth"

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
ckpt_config = checkpoint["config"]

if __name__ == '__main__':
    _, val_loader, _ = get_dataloaders(
        data_dir=ckpt_config["data"]["data_dir"],
        batch_size=8,
        num_workers=0,
        num_bins=ckpt_config["data"]["num_bins"],
        target_size=tuple(ckpt_config["data"]["target_size"]),
        augmentation=False,
        normalize_translation=ckpt_config["data"].get("normalize_translation", True),
        density_augmentation=False,
        robust_augmentation=False,
        sequence_length=ckpt_config["data"].get("sequence_length", 3),
    )

    print(f"Evaluating ALL {len(val_loader.dataset)} samples of V17 latest.pth...")
    results = evaluate_checkpoint(checkpoint_path, val_loader, device, tta=False)

    print("\nRESULTS (ALL VAL SAMPLES):")
    print(f"  Trans Error: {results['trans_mean']:.4f}m (median: {results['trans_median']:.4f}m)")
    print(f"  Orient Error: {results['orient_mean']:.4f} deg (median: {results['orient_median']:.4f} deg)")
    print(f"  Pose Error: {results['pose_error']:.4f}")
