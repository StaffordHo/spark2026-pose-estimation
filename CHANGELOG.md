# Project Changelog — SPARK2026 Pose Estimation

> Living document tracking all changes, their rationale, and research origins.
> Append new entries at the top. Never remove previous entries.

---

## [v0.2.3] — 2026-02-13: Training Speed Optimization

### Summary
Eliminated the main data loading bottleneck — voxel grids were being built at full 1280×720 resolution then resized to 256×256. Now builds directly at target resolution by scaling event coordinates.

### Speedups
1. **Direct-resolution voxel construction** — Scale event (x,y) coordinates to target size before `np.add.at`. Eliminates both full-res grid allocation (921K pixels → 65K) and the `F.interpolate` resize step. **~14× faster per sample.**
2. **Unlimited file cache** — Removed the 50-sequence cache limit. All 240 training sequences stay in RAM after first load. Eliminates HDF5 re-reads.
3. **Persistent workers** — Added `persistent_workers=True` to all DataLoaders. Workers survive between epochs, keeping their file caches warm.
4. **Prefetch** — Added `prefetch_factor=4` to keep GPU fed while CPU prepares next batches.
5. **More workers** — `num_workers: 4 → 8`

### Files modified
- `src/dataset.py` — Rewrote `__getitem__` voxel construction, removed cache limit, updated DataLoader settings
- `config/rtx4090_sota.yaml` — `num_workers: 8`

---

## [v0.2.2] — 2026-02-13: Robust Training Loop

### Summary
Training crashed around batch 1900 (no error logs). Made the training loop crash-proof so no single bad batch can kill a 60-epoch run.

### Fixes
1. **Per-batch try/except** — Wraps each batch in error handling:
   - CUDA OOM → clears GPU cache, skips batch, continues training
   - Any RuntimeError / Exception → logs warning, skips batch, continues
2. **`latest.pth` checkpoint every epoch** — Always have a recent checkpoint to resume from
3. **Scheduler + scaler state in checkpoints** — Resume continues with correct LR schedule
4. **Resume restored** — `--resume` now loads scheduler and scaler state too

### Files modified
- `train.py` — `train_epoch()` error handling, checkpoint saving, resume logic

---

## [v0.2.1] — 2026-02-12: Fix Training Divergence (NaN)

### Summary
Fixed gradient explosion causing NaN loss at batch ~240. Translation loss was diverging (2.9 → 10.7 → NaN) while rotation was stable.

### Root Cause
- Translation targets range Tz=[3.0, 8.25]m. Initial model predicts ~0, so errors are 3-8m → `smooth_l1` operates in L1 regime (constant gradient).
- Combined with LR=0.001, this causes runaway gradient accumulation.
- `torch.acos` in `geodesic_loss_matrix` produces NaN under AMP float16 when trace values hit ±1.0 exactly (float16 precision too low for acos boundary).

### Fixes
1. **Translation normalization** — Subtract mean, divide by std → targets become ~N(0,1)
   - Stats computed over 30 sequences: mean=[-0.24, 0.03, 5.68], std=[0.93, 0.50, 1.40]
   - `src/dataset.py` — Added `TRANS_MEAN/TRANS_STD` constants, `normalize_translation` flag
   - Config key: `data.normalize_translation: true`
2. **Lower LR** — 0.001 → 0.0003 (conservative for stability)
3. **Tighter grad clip** — 1.0 → 0.5
4. **NaN guard** — `train.py` checks for NaN **before** `backward()` to avoid GradScaler state corruption (`unscale_()` double-call error)
5. **AMP float16 acos fix** — `src/losses.py` `geodesic_loss_matrix`: cast trace to float32 before acos, clamp to [-1+1e-7, 1-1e-7]

---

## [v0.2.0] — 2026-02-11: SOTA Methods Implementation

### Summary
Implemented 5 research-backed improvements to maximize validation scores, targeting the rotation error bottleneck identified from baseline training. All changes are backward-compatible — the original configs and code paths still work.

### Changes

#### 1. 6D Continuous Rotation Representation
- **Paper**: Zhou et al., *"On the Continuity of Rotation Representations in Neural Networks"*, CVPR 2019
- **Rationale**: Quaternions are mathematically discontinuous for regression — no continuous mapping from SO(3) to ℝ⁴ exists. This causes the optimizer to struggle with rotation prediction. The 6D representation (first two columns of the rotation matrix) is continuous and provably easier to regress.
- **Evidence**: Baseline training showed rotation loss = ~75% of total loss, converging much slower than translation. This is a classic symptom of quaternion discontinuity.
- **Expected impact**: 20-40% reduction in rotation error.
- **Files modified**:
  - `src/losses.py` — Added `rotation_6d_to_matrix()`, `matrix_to_rotation_6d()`, `matrix_to_quaternion()`, `quaternion_to_matrix()`, `geodesic_loss_matrix()`, `PoseLoss6D` class
  - `src/models/pose_head.py` — Added `PoseHead6D` class (outputs 6D rotation + converted quaternion for eval)
  - `src/models/pose_net.py` — Added `rotation_repr="6d"` option
  - `src/models/__init__.py` — Exported new classes
- **Config key**: `model.rotation_repr: "6d"`

#### 2. Pretrained EfficientNet-B0 Backbone
- **Paper**: EfficientPose (Park et al.), SPADES baseline evaluations (Rathinam et al., 2023)
- **Rationale**: The original `EventResNet` was custom-built and trained from scratch. A pretrained backbone provides strong low-level feature priors (edges, textures, shapes) that transfer well even to non-RGB inputs. EfficientNet-B0 (5.3M params, compound scaling, squeeze-excitation) outperforms similarly-sized ResNets.
- **Implementation detail**: The first conv layer was adapted from 3 channels to 10 channels. Pretrained weights are initialized by averaging the 3-channel weights and repeating across 10 channels. This preserves the learned filter structure.
- **Expected impact**: 15-25% improvement from pretrained features + faster convergence.
- **Files modified**:
  - `src/models/backbone.py` — Added `PretrainedEfficientNet` class with `freeze_backbone()` / `unfreeze_backbone()` methods
  - `src/models/pose_net.py` — Added `"efficientnet"` backbone option with `pretrained` and `freeze_backbone` flags
  - `src/models/__init__.py` — Exported `PretrainedEfficientNet`
- **Config keys**: `model.backbone: "efficientnet"`, `model.pretrained: true`, `model.freeze_backbone: true`

#### 3. Input Resize (1280×720 → 256×256)
- **Rationale**: The spacecraft occupies a small portion of the 1280×720 frame — most pixels are empty black space. Resizing to 256×256 reduces memory by ~14×, enabling batch_size=64 (up from 32). Larger batches provide more stable gradient estimates, improving convergence. The bilinear interpolation preserves the spatial event distribution.
- **Expected impact**: 5-10% improvement via larger batch sizes and faster iteration.
- **Files modified**:
  - `src/dataset.py` — Added `target_size` parameter; applied `F.interpolate(bilinear)` after voxel grid creation
- **Config key**: `data.target_size: [256, 256]`

#### 4. Event-Specific Data Augmentation
- **Paper**: SPADES (Rathinam et al., 2023), general event camera augmentation literature
- **Rationale**: Without augmentation, the model overfits to the specific event patterns in training data. Event cameras have unique augmentation opportunities not available in RGB.
- **Augmentations implemented** (train-only):
  - **Random horizontal flip** (50%): Flips voxel grid width and corrects Tx (negate) and quaternion (negate Qy, Qz)
  - **Random polarity flip** (30%): Swaps ON and OFF event channels — tests invariance to sensor polarity inversion
  - **Random additive noise** (20%): Adds small uniform noise (0-0.05) to simulate sensor noise
- **Expected impact**: 5-15% generalization improvement.
- **Files modified**:
  - `src/dataset.py` — Added augmentation logic in `__getitem__`, `augmentation` parameter in constructor and `get_dataloaders`
- **Config key**: `data.augmentation: true`

#### 5. Loss & Training Strategy Improvements
- **Rationale**: 
  - **rot_weight=3.0**: Since rotation error is the dominant bottleneck, increasing its weight forces the optimizer to allocate more gradient capacity to rotation learning.
  - **Cosine Annealing with Warm Restarts**: Instead of monotonic LR decay, periodic restarts allow the optimizer to escape local minima and explore new loss landscape regions.
  - **Differential LR**: Pretrained backbone layers learn at 10× lower LR (0.0001) than new head layers (0.001), preventing catastrophic forgetting of pretrained features.
  - **Staged unfreezing**: Backbone is frozen for the first 5 epochs so the head learns meaningful features first, then the backbone is unfrozen for joint fine-tuning.
- **Expected impact**: 5-10% from loss rebalancing, faster convergence from training strategy.
- **Files modified**:
  - `train.py` — Added differential LR param groups, staged unfreezing logic, `CosineAnnealingWarmRestarts` scheduler option
- **Config keys**: `loss.rot_weight: 3.0`, `training.scheduler: "cosine_warm_restarts"`, `model.backbone_lr_scale: 0.1`, `model.unfreeze_at_epoch: 5`

### New Files
- `config/rtx4090_sota.yaml` — Configuration enabling all 5 methods

### Verified
- Fast dev test on RTX 4090: All features operational, 857K trainable params (backbone frozen), stable loss ~9.0

---

## [v0.1.1] — 2026-02-11: RTX 4090 Configuration

### Summary
Added optimized training configuration for RTX 4090 (24GB VRAM).

### Changes
- Created `config/rtx4090.yaml` with batch_size=32, full model dimensions (feature_dim=512, hidden_dim=256), num_workers=4, AMP enabled.
- **Rationale**: The default config was designed for RTX 3070 (8GB). The 4090's 24GB allows larger batches and full-size model without OOM.

### New Files
- `config/rtx4090.yaml`

---

## [v0.1.0] — 2026-02-10: Initial Codebase

### Summary
Initial project setup with baseline model and training pipeline.

### Architecture
- **Backbone**: Custom `EventResNet` (ResNet-like 2D CNN, trained from scratch)
- **Pose head**: `PoseHead` (shared layers → translation + quaternion branches)
- **Input**: 10-channel voxel grid (5 temporal bins × 2 polarities, 1280×720)
- **Loss**: `PoseLoss` (Smooth L1 translation + Geodesic rotation)
- **Optimizer**: AdamW + Cosine annealing LR
- **Metrics**: Position error (m), Rotation error (°)

### Dataset
- SPARK2026 / SPADES: 300 HDF5 sequences, ~598 samples each
- Event data: (xs, ys, ts, ps) at 100µs resolution
- Labels: 6DoF pose (Tx, Ty, Tz, Qx, Qy, Qz, Qw) at 10 Hz
- Split: 240 train / 30 val / 30 test

### Files
- `src/models/backbone.py` — `VoxelCNN`, `EventResNet`
- `src/models/pose_head.py` — `PoseHead`, `PoseHeadUncertainty`
- `src/models/pose_net.py` — `PoseNet`
- `src/dataset.py` — `SPARK2026Dataset`, `get_dataloaders`
- `src/losses.py` — `PoseLoss`, `PoseLossUncertainty`
- `src/metrics.py` — `MetricTracker`, `position_error`, `rotation_error_deg`
- `src/event_representations.py` — `events_to_voxel_grid_fast`
- `train.py` — Main training script
- `evaluate.py` — Evaluation script
- `visualize_events.py` — Dataset visualization
- `config/default.yaml`, `config/rtx3070.yaml`
