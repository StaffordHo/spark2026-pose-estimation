import zipfile
import csv
import os
import shutil

# Step 1: Extract CSVs from the zips
def extract_csv(zip_path, out_csv_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Codabench submission zips usually contain exactly one file 'submission.csv'
        filename = zf.namelist()[0]
        with zf.open(filename) as f_in, open(out_csv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extracted {out_csv_path} from {zip_path}")

extract_csv('submission_v7_ep7.zip', 'v7_ep7.csv')
extract_csv('submission_v15_latest.zip', 'v15_latest.csv')

# Step 2: Merge V7 Translation (0.264m) with V15 Rotation (1.48 rad)
def merge_trans_rot(trans_csv, rot_csv, out_csv):
    with open(trans_csv, 'r') as ft, open(rot_csv, 'r') as fr, open(out_csv, 'w', newline='') as fo:
        reader_t = csv.reader(ft)
        reader_r = csv.reader(fr)
        writer = csv.writer(fo)
        
        for rt, rr in zip(reader_t, reader_r):
            if rt[0] == 'timestamp' or rt[0] == 'Tx': # header
                writer.writerow(rt)
                continue
            # Output: V7_T[0:4] (ts, Tx, Ty, Tz) + V15_R[4:8] (Qx, Qy, Qz, Qw)
            out_row = rt[0:4] + rr[4:8]
            writer.writerow(out_row)
    print(f"Created {out_csv} via hybrid merge")

merge_trans_rot('v7_ep7.csv', 'v15_latest.csv', 'hybrid_v7T_v15R.csv')

# Step 3: Zip it up
def create_zip(csv_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(csv_path, arcname='submission.csv')
    print(f"Created {zip_path}")

create_zip('hybrid_v7T_v15R.csv', 'submission_hybrid_v7T_v15R.zip')

# Step 4: Apply smoothing to V15 rotations, then combine with V7 translations
print("Now running smoothing...")
import subprocess
# We will use the existing smooth_predictions.py
subprocess.run(['python', 'smooth_predictions.py', '--input', 'v15_latest.csv', '--output', 'v15_latest_smoothed.csv', '--quat_window', '5'])

# Merge V7 Trans + Smoothed V15 Rot
merge_trans_rot('v7_ep7.csv', 'v15_latest_smoothed.csv', 'hybrid_v7T_smoothedV15R.csv')
create_zip('hybrid_v7T_smoothedV15R.csv', 'submission_hybrid_v7T_smoothedV15R.zip')
