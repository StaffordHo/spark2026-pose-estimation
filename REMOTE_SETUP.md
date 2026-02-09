# SPARK2026 Pose Estimation - Remote Linux PC Setup Guide

This guide provides commands to run on your remote Linux PC with RTX 4090.

## 1. Transfer Files to Remote PC

On your Windows machine, use one of these methods:

### Option A: SCP (if SSH is set up)
```powershell
# From Windows PowerShell
scp -r C:\XSITESITRSE\spark2026 disco_lab@<REMOTE_IP>:~/
```

### Option B: Use AnyDesk file transfer
- Open AnyDesk
- Connect to remote PC
- Use the file transfer feature to copy `spark2026` folder

---

## 2. Commands to Run on Remote Linux PC

### Environment Setup (one-time)

```bash
# Navigate to the project
cd ~/spark2026

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch with CUDA (if not already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU Access

```bash
# Check GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

### Verify Data Loading

```bash
# Test data loading (quick check)
cd ~/spark2026
python -c "
from src.dataset import SPARK2026Dataset
ds = SPARK2026Dataset('.', split='train')
print(f'Train samples: {len(ds)}')
sample = ds[0]
print(f'Voxel shape: {sample[\"voxel\"].shape}')
print(f'Translation: {sample[\"translation\"]}')
print('Data loading OK!')
"
```

---

## 3. Training Commands

### Quick Test Run (verify everything works)

```bash
cd ~/spark2026
source venv/bin/activate

# Fast dev run - 2 epochs, minimal logging
python train.py --config config/default.yaml --fast_dev_run
```

### Full Training Run

```bash
cd ~/spark2026
source venv/bin/activate

# Standard training (100 epochs)
python train.py --config config/default.yaml

# With custom settings
python train.py --config config/default.yaml --epochs 50 --batch_size 32 --lr 0.001

# Resume from checkpoint
python train.py --config config/default.yaml --resume checkpoints/epoch_50.pth
```

### Monitor Training (in separate terminal)

```bash
cd ~/spark2026
source venv/bin/activate

# Launch TensorBoard
tensorboard --logdir logs --port 6006

# Access via browser: http://localhost:6006
# Or via SSH tunnel: ssh -L 6006:localhost:6006 disco_lab@<REMOTE_IP>
```

---

## 4. Evaluation Commands

```bash
cd ~/spark2026
source venv/bin/activate

# Evaluate on test set
python evaluate.py --checkpoint checkpoints/best.pth --split test

# Evaluate on validation set
python evaluate.py --checkpoint checkpoints/best.pth --split val

# Save results to JSON
python evaluate.py --checkpoint checkpoints/best.pth --split test --output results.json
```

---

## 5. Running in Background (Screen/Tmux)

For long training runs, use screen or tmux to prevent disconnection issues:

```bash
# Using screen
screen -S spark_training
cd ~/spark2026
source venv/bin/activate
python train.py --config config/default.yaml

# Detach: Ctrl+A, D
# Reattach: screen -r spark_training
```

```bash
# Using tmux
tmux new -s spark_training
cd ~/spark2026
source venv/bin/activate
python train.py --config config/default.yaml

# Detach: Ctrl+B, D
# Reattach: tmux attach -t spark_training
```

---

## 6. Expected Training Time

With RTX 4090:
- **Batch size 16**: ~15-20 minutes per epoch (240 train sequences × ~598 samples each)
- **Batch size 32**: ~10-12 minutes per epoch (with AMP enabled)
- **Full 100 epochs**: ~16-20 hours

---

## 7. Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train.py --config config/default.yaml --batch_size 8
```

### Slow data loading
```bash
# Increase workers (if CPU allows)
# Edit config/default.yaml: num_workers: 8
```

### Check GPU utilization during training
```bash
# In separate terminal
watch -n 1 nvidia-smi
```
