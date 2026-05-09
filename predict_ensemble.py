import os
import argparse
import pandas as pd
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

from src.dataset import SPARK2026Dataset, events_to_voxel_grid_fast
from src.models import PoseNet

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_voxel_for_window(data, pose_idx, num_bins, height, width, target_size=None, sequence_length=1, is_test_set=False):
    t_end = data["timestamp"][pose_idx]
    
    if is_test_set:
        t_start = t_end - (sequence_length * 1000) # Reverted back to 1000 (1ms) for real test data to match training
    else:
        t_start = t_end - (sequence_length * 1000)

    event_mask = (data["ts"] >= t_start) & (data["ts"] < t_end)
    
    xs = data["xs"][event_mask]
    ys = data["ys"][event_mask]
    ts = data["ts"][event_mask]
    ps = data["ps"][event_mask]
    
    if target_size is not None:
        target_h, target_w = target_size
        xs_scaled = (xs.astype(np.float32) * target_w / width)
        ys_scaled = (ys.astype(np.float32) * target_h / height)
        voxel = events_to_voxel_grid_fast(
            xs_scaled, ys_scaled, ts, ps,
            num_bins=num_bins, height=target_h, width=target_w
        )
    else:
        voxel = events_to_voxel_grid_fast(
            xs, ys, ts, ps,
            num_bins=num_bins, height=height, width=width
        )
    return voxel

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    model = PoseNet(
        backbone=config["model"]["backbone"],
        pretrained=False,
        in_channels=config["model"]["in_channels"],
        feature_dim=config["model"]["feature_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"],
        with_uncertainty=config["model"].get("with_uncertainty", False),
        rotation_repr=config["model"].get("rotation_repr", "quaternion")
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    if config["model"]["in_channels"] == 10:
        return model, config
    else:
        raise ValueError("Sequence length must be 1 for original ensemble")

@torch.no_grad()
def generate_ensemble_predictions(rot_expert, trans_expert, config, args, device="cuda"):
    test_dir = args.test_dir
    template_path = os.path.join(test_dir, "template.csv")
    
    template_df = pd.read_csv(template_path)
    predictions = []
    
    # Group by sequence
    template_df['seq'] = template_df['timestamp'].apply(lambda x: x.split('_')[0])
    seq_to_indices = {}
    for i, row in template_df.iterrows():
        seq = row['seq']
        idx = int(row['timestamp'].split('_')[1])
        if seq not in seq_to_indices:
            seq_to_indices[seq] = []
        seq_to_indices[seq].append((i, idx))
        
    num_bins = config["data"]["num_bins"]
    height = config["data"]["height"]
    width = config["data"]["width"]
    target_size = tuple(config["data"]["target_size"])
    batch_size = config["data"]["batch_size"]

    print(f"\nGenerating Ensemble predictions for {len(template_df)} poses...")
    
    for seq_name in sorted(seq_to_indices.keys()):
        h5_path = os.path.join(test_dir, f"{seq_name}.h5")
        if not os.path.exists(h5_path):
            print(f"File not found: {h5_path}")
            continue
            
        indices = seq_to_indices[seq_name]
        print(f"Processing {seq_name}: {len(indices)} poses...")
        
        max_idx = max(idx for _, idx in indices)
        is_test_set = seq_name.startswith('T')
        
        # Debunked: Time units are 1us for both Synthetic and Real, so both are 1000
        timestamps = np.arange(0, max_idx + 1) * 1000
        
        with h5py.File(h5_path, "r") as f:
            data = {
                "xs": f["events"]["xs"][()],
                "ys": f["events"]["ys"][()],
                "ts": f["events"]["ts"][()],
                "ps": f["events"]["ps"][()],
                "timestamp": timestamps,
            }
            
        batch_voxels = []
        batch_ids = []
        
        for row_id, pose_idx in indices:
            voxel = build_voxel_for_window(
                data, pose_idx, num_bins, height, width, target_size, 1, is_test_set
            )
            batch_voxels.append(voxel)
            batch_ids.append(row_id)
            
            if len(batch_voxels) == batch_size or row_id == indices[-1][0]:
                voxel_tensor = torch.stack(batch_voxels).to(device)
                
                # Forward Translation Expert
                trans_out = trans_expert(voxel_tensor)
                trans_pred = trans_out["translation"]
                
                # Forward Rotation Expert
                rot_out = rot_expert(voxel_tensor)
                rot_pred = rot_out["quaternion"]
                
                # Denormalize translation natively here
                # Our models use normalize_translation=true which means we need the proper STD and MEAN
                trans_mean = SPARK2026Dataset.TRANS_MEAN.to(device)
                trans_std = SPARK2026Dataset.TRANS_STD.to(device)
                trans_pred = trans_pred * trans_std + trans_mean
                
                # Normalize quaternion natively here
                rot_pred = nn.functional.normalize(rot_pred, p=2, dim=1)
                
                trans_np = trans_pred.cpu().numpy()
                rot_np = rot_pred.cpu().numpy()
                
                for b_idx, df_idx in enumerate(batch_ids):
                    predictions.append({
                        "id": df_idx,
                        "Tx": trans_np[b_idx, 0],
                        "Ty": trans_np[b_idx, 1],
                        "Tz": trans_np[b_idx, 2],
                        "Qx": rot_np[b_idx, 0],
                        "Qy": rot_np[b_idx, 1],
                        "Qz": rot_np[b_idx, 2],
                        "Qw": rot_np[b_idx, 3]
                    })
                    
                batch_voxels = []
                batch_ids = []

    # Write output
    print(f"Writing to {args.output}...")
    predictions.sort(key=lambda x: x["id"])
    for p in predictions:
        idx = p["id"]
        template_df.loc[idx, "Tx": "Qw"] = [p["Tx"], p["Ty"], p["Tz"], p["Qx"], p["Qy"], p["Qz"], p["Qw"]]
        
    template_df = template_df.drop(columns=['seq'])
    template_df.to_csv(args.output, index=False)
    
    zip_output = args.output.replace('.csv', '.zip')
    print(f"Zipping to {zip_output}...")
    import zipfile
    with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output, arcname="submission.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rot_checkpoint", type=str, required=True, help="V15 rotation expert model")
    parser.add_argument("--trans_checkpoint", type=str, required=True, help="V18 translation expert model")
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--output", type=str, default="submission_ensemble.csv")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading models...")
    rot_expert, config = load_model(args.rot_checkpoint, device)
    trans_expert, _ = load_model(args.trans_checkpoint, device)
    
    generate_ensemble_predictions(rot_expert, trans_expert, config, args, device)
