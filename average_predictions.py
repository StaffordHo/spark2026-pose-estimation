"""Average predictions from multiple checkpoints into a consensus ensemble.

Takes multiple submission CSVs and averages translations (simple mean)
and quaternions (hemisphere-aligned averaging + normalization).
"""
import csv
import argparse
import numpy as np
import zipfile
import shutil


def main():
    parser = argparse.ArgumentParser(description="Average multiple submission CSVs")
    parser.add_argument("--inputs", type=str, required=True, 
                       help="Comma-separated list of input CSV files")
    parser.add_argument("--output", type=str, default="submission_ensemble_avg.csv",
                       help="Output CSV file")
    args = parser.parse_args()
    
    input_files = [f.strip() for f in args.inputs.split(",")]
    print(f"Averaging {len(input_files)} prediction files:")
    for f in input_files:
        print(f"  - {f}")
    
    # Read all CSVs
    all_predictions = []
    header = None
    timestamps = None
    
    for filepath in input_files:
        preds = {}
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            h = next(reader)
            if header is None:
                header = h
            for row in reader:
                ts = row[0]
                tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
                qx, qy, qz, qw = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                preds[ts] = {
                    "translation": np.array([tx, ty, tz]),
                    "quaternion": np.array([qx, qy, qz, qw]),
                }
        all_predictions.append(preds)
        
        if timestamps is None:
            timestamps = list(preds.keys())
        print(f"  Loaded {len(preds)} predictions from {filepath}")
    
    # Verify all files have the same timestamps
    for i, preds in enumerate(all_predictions):
        assert set(preds.keys()) == set(timestamps), \
            f"File {input_files[i]} has different timestamps!"
    
    # Average predictions
    print(f"\nAveraging {len(timestamps)} poses across {len(all_predictions)} models...")
    
    averaged = {}
    total_quat_spread = 0.0
    
    for ts in timestamps:
        # Collect all predictions for this timestamp
        translations = [preds[ts]["translation"] for preds in all_predictions]
        quaternions = [preds[ts]["quaternion"] for preds in all_predictions]
        
        # Average translations (simple mean)
        avg_trans = np.mean(translations, axis=0)
        
        # Average quaternions with hemisphere alignment
        # Use first quaternion as reference
        q_ref = quaternions[0]
        aligned_quats = [q_ref.copy()]
        for q in quaternions[1:]:
            # Align to same hemisphere as reference
            if np.dot(q, q_ref) < 0:
                aligned_quats.append(-q)
            else:
                aligned_quats.append(q.copy())
        
        avg_quat = np.mean(aligned_quats, axis=0)
        # Normalize
        avg_quat = avg_quat / (np.linalg.norm(avg_quat) + 1e-10)
        
        # Track spread (how much the models disagree)
        spreads = [min(np.linalg.norm(q - avg_quat), np.linalg.norm(q + avg_quat)) 
                   for q in quaternions]
        total_quat_spread += np.mean(spreads)
        
        averaged[ts] = {"translation": avg_trans, "quaternion": avg_quat}
    
    avg_spread = total_quat_spread / len(timestamps)
    print(f"Average quaternion spread across models: {avg_spread:.4f}")
    print(f"  (lower = more agreement between checkpoints)")
    
    # Write output
    # Preserve original row order from first file
    with open(input_files[0], "r") as f:
        original_rows = list(csv.reader(f))[1:]
    
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in original_rows:
            ts = row[0]
            t = averaged[ts]["translation"]
            q = averaged[ts]["quaternion"]
            writer.writerow([ts, f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}",
                           f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}"])
    
    print(f"\nEnsemble written to {args.output}")
    
    # Create zip
    zip_path = args.output.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output, arcname="submission.csv")
    print(f"Zipped to {zip_path}")
    
    # Copy to Desktop
    desktop_name = zip_path.split("\\")[-1].split("/")[-1]
    desktop_path = f"C:\\Users\\kk\\Desktop\\{desktop_name}"
    try:
        shutil.copy(zip_path, desktop_path)
        print(f"Copied to {desktop_path}")
    except Exception as e:
        print(f"Could not copy to Desktop: {e}")
    
    print("Done!")


if __name__ == "__main__":
    main()
