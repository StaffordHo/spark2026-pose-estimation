import torch, os

print("=== V4 checkpoints ===")
for ep in range(1, 20):
    path = f"checkpoints_v4/epoch_{ep}.pth"
    if os.path.exists(path):
        c = torch.load(path, map_location="cpu", weights_only=False)
        vl = c.get("best_val_loss", 0)
        print(f"  Epoch {ep:2d}: best_val_loss={vl:.4f}")

print("\nBest checkpoint info:")
c = torch.load("checkpoints_v4/best.pth", map_location="cpu", weights_only=False)
print(f"  Best epoch: {c['epoch']+1}")
print(f"  Best val loss: {c.get('best_val_loss', 'N/A')}")
