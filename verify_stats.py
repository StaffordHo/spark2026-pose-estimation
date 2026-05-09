import h5py
import numpy as np
import glob
import os

# Check translation distribution in training data
h5_files = sorted(glob.glob("h5/RT*.h5"))[:30]
all_tx = []
all_ty = []
all_tz = []

print(f"Checking {len(h5_files)} H5 files...")
for fpath in h5_files:
    try:
        with h5py.File(fpath, 'r') as f:
            all_tx.extend(f['labels']['data']['Tx'][()])
            all_ty.extend(f['labels']['data']['Ty'][()])
            all_tz.extend(f['labels']['data']['Tz'][()])
    except Exception as e:
        pass

tx = np.array(all_tx)
ty = np.array(all_ty)
tz = np.array(all_tz)

print("\nTranslation Statistics (Meters):")
print(f"  Tx: mean={tx.mean():.4f}, std={tx.std():.4f}, range=[{tx.min():.2f}, {tx.max():.2f}]")
print(f"  Ty: mean={ty.mean():.4f}, std={ty.std():.4f}, range=[{ty.min():.2f}, {ty.max():.2f}]")
print(f"  Tz: mean={tz.mean():.4f}, std={tz.std():.4f}, range=[{tz.min():.2f}, {tz.max():.2f}]")

print("\nSanity Check: If we negate Tx with non-zero mean:")
mean_m = tx.mean()
std_m = tx.std()
t_val = 1.0
t_norm = (t_val - mean_m) / std_m
t_flipped_norm = -t_norm
t_reconstructed = t_flipped_norm * std_m + mean_m
print(f"  Original: {t_val:.4f}")
print(f"  Negated then Reconstructed: {t_reconstructed:.4f} (Error: {abs(t_reconstructed - (-t_val)):.4f})")
