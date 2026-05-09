import torch, os

# Check old checkpoints
print("=== OLD checkpoints/ (256x256) ===")
for ep in range(1, 31):
    path = f"checkpoints/epoch_{ep}.pth"
    if os.path.exists(path):
        c = torch.load(path, map_location="cpu", weights_only=False)
        vl = c.get("best_val_loss", 0)
        print(f"  Epoch {ep:2d}: best_val_loss={vl:.4f}")

print("\n=== NEW checkpoints_v2/ (480x480) ===")
for ep in [1, 5, 10, 15, 20, 23, 25, 30, 33]:
    path = f"checkpoints_v2/epoch_{ep}.pth"
    if os.path.exists(path):
        c = torch.load(path, map_location="cpu", weights_only=False)
        vl = c.get("best_val_loss", 0)
        print(f"  Epoch {ep:2d}: best_val_loss={vl:.4f}")
