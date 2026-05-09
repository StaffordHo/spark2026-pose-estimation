import torch, os

print("=== V5 checkpoints ===")
best_vl = float('inf')
best_ep = -1

for ep in range(1, 81):
    path = f"checkpoints_v5/epoch_{ep}.pth"
    if os.path.exists(path):
        try:
            c = torch.load(path, map_location="cpu", weights_only=False)
            vl = c.get("best_val_loss", 0)
            print(f"  Epoch {ep:2d}: best_val_loss={vl:.4f}")
            if vl < best_vl and vl > 0:
                best_vl = vl
                best_ep = ep
        except Exception as e:
            print(f"  Epoch {ep:2d}: error loading ({e})")

print("\nBest checkpoint info:")
if os.path.exists("checkpoints_v5/best.pth"):
    c = torch.load("checkpoints_v5/best.pth", map_location="cpu", weights_only=False)
    print(f"  Best epoch: {c['epoch']+1}")
    print(f"  Best val loss: {c.get('best_val_loss', 'N/A')}")
else:
    print(f"  Best epoch found in list: {best_ep} (Loss: {best_vl:.4f})")
