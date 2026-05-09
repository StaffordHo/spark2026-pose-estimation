"""Temporal smoothing of submission predictions.

Applies SLERP-based quaternion smoothing and translation moving average
within each RT sequence to reduce TTA non-determinism noise.
"""
import csv
import argparse
import numpy as np
import zipfile
import shutil
from collections import defaultdict


def normalize_quat(q):
    """Normalize quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    
    dot = np.dot(q1, q2)
    
    # Ensure shortest path
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    dot = np.clip(dot, -1.0, 1.0)
    
    if dot > 0.9995:
        # Linear interpolation for very close quaternions
        result = q1 + t * (q2 - q1)
        return normalize_quat(result)
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    
    return normalize_quat(s1 * q1 + s2 * q2)


def smooth_quaternions_slerp(quats, window_size=5):
    """Smooth a sequence of quaternions using iterative SLERP averaging.
    
    For each quaternion, compute a SLERP-weighted average with its neighbors.
    """
    n = len(quats)
    if n <= 1:
        return quats.copy()
    
    smoothed = quats.copy()
    half_w = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        
        # Use iterative SLERP averaging
        # Start with the first quaternion in the window
        window_quats = quats[start:end]
        
        # Ensure all quaternions are in the same hemisphere as the center
        center_q = quats[i]
        aligned = []
        for q in window_quats:
            if np.dot(q, center_q) < 0:
                aligned.append(-q)
            else:
                aligned.append(q.copy())
        
        # Simple average of aligned quaternions (works well for small windows)
        avg = np.mean(aligned, axis=0)
        smoothed[i] = normalize_quat(avg)
    
    return smoothed


def smooth_translations(translations, window_size=5):
    """Smooth translations using a simple moving average."""
    n = len(translations)
    if n <= 1:
        return translations.copy()
    
    smoothed = translations.copy()
    half_w = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        smoothed[i] = np.mean(translations[start:end], axis=0)
    
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Smooth submission predictions temporally")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, default="submission_smoothed.csv", help="Output CSV file")
    parser.add_argument("--quat_window", type=int, default=5, help="Quaternion smoothing window size (odd)")
    parser.add_argument("--trans_window", type=int, default=3, help="Translation smoothing window size (odd)")
    parser.add_argument("--no_trans_smooth", action="store_true", help="Skip translation smoothing")
    parser.add_argument("--no_quat_smooth", action="store_true", help="Skip quaternion smoothing")
    args = parser.parse_args()
    
    # Read input CSV
    rows = []
    with open(args.input, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)
    
    print(f"Read {len(rows)} predictions from {args.input}")
    print(f"Quaternion window: {args.quat_window}, Translation window: {args.trans_window}")
    
    # Group by sequence
    sequences = defaultdict(list)
    for row in rows:
        timestamp = row[0]
        seq_name = timestamp.rsplit("_", 1)[0]  # e.g., RT901 from RT901_001
        pose_idx = int(timestamp.rsplit("_", 1)[1])
        tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
        qx, qy, qz, qw = float(row[4]), float(row[5]), float(row[6]), float(row[7])
        sequences[seq_name].append({
            "timestamp": timestamp,
            "pose_idx": pose_idx,
            "translation": np.array([tx, ty, tz]),
            "quaternion": np.array([qx, qy, qz, qw]),
        })
    
    print(f"Found {len(sequences)} sequences")
    
    # Sort each sequence by pose index
    for seq_name in sequences:
        sequences[seq_name].sort(key=lambda x: x["pose_idx"])
    
    # Smooth each sequence
    smoothed_rows = {}
    total_quat_change = 0.0
    total_trans_change = 0.0
    count = 0
    
    for seq_name in sorted(sequences.keys()):
        seq = sequences[seq_name]
        n = len(seq)
        
        # Extract arrays
        translations = np.array([s["translation"] for s in seq])
        quaternions = np.array([s["quaternion"] for s in seq])
        
        # Smooth quaternions
        if not args.no_quat_smooth:
            smoothed_quats = smooth_quaternions_slerp(quaternions, args.quat_window)
        else:
            smoothed_quats = quaternions.copy()
        
        # Smooth translations
        if not args.no_trans_smooth:
            smoothed_trans = smooth_translations(translations, args.trans_window)
        else:
            smoothed_trans = translations.copy()
        
        # Track changes
        for i in range(n):
            quat_diff = np.linalg.norm(smoothed_quats[i] - quaternions[i])
            trans_diff = np.linalg.norm(smoothed_trans[i] - translations[i])
            total_quat_change += quat_diff
            total_trans_change += trans_diff
            count += 1
            
            ts = seq[i]["timestamp"]
            smoothed_rows[ts] = {
                "translation": smoothed_trans[i],
                "quaternion": smoothed_quats[i],
            }
        
        print(f"  {seq_name}: {n} poses smoothed")
    
    print(f"\nAvg quaternion change: {total_quat_change/count:.6f}")
    print(f"Avg translation change: {total_trans_change/count:.6f}m")
    
    # Write output CSV in original order
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            ts = row[0]
            s = smoothed_rows[ts]
            t = s["translation"]
            q = s["quaternion"]
            writer.writerow([ts, f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}",
                           f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}"])
    
    print(f"\nSmoothed predictions written to {args.output}")
    
    # Create zip
    zip_path = args.output.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output, arcname="submission.csv")
    print(f"Zipped to {zip_path}")
    
    # Copy to Desktop
    desktop_path = r"C:\Users\kk\Desktop" + "\\" + zip_path.split("\\")[-1].split("/")[-1]
    try:
        shutil.copy(zip_path, desktop_path)
        print(f"Copied to {desktop_path}")
    except Exception as e:
        print(f"Could not copy to Desktop: {e}")
    
    print("Done!")


if __name__ == "__main__":
    main()
