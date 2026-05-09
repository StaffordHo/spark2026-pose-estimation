"""
Simple CSV merge script for V15+V18 ensemble predictions.

Takes V15 predictions (for rotation) and V18 predictions (for translation),
and merges them into a single submission file.

Usage:
    python merge_ensemble.py --v15 submission_v15.csv --v18 submission_v18.csv --output submission_ensemble.csv
"""

import argparse
import csv
import os
import zipfile


def main():
    parser = argparse.ArgumentParser(description="Merge V15 (rotation) + V18 (translation) predictions")
    parser.add_argument("--v15", type=str, required=True, help="V15 predictions CSV (rotation expert)")
    parser.add_argument("--v18", type=str, required=True, help="V18 predictions CSV (translation expert)")
    parser.add_argument("--output", type=str, default="submission_ensemble.csv", help="Output CSV path")
    args = parser.parse_args()

    # Read V15 predictions (we use its rotation: Qx, Qy, Qz, Qw)
    v15_data = {}
    with open(args.v15, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            timestamp = row[0]
            # row format: timestamp, Tx, Ty, Tz, Qx, Qy, Qz, Qw
            v15_data[timestamp] = {
                "Tx": row[1], "Ty": row[2], "Tz": row[3],
                "Qx": row[4], "Qy": row[5], "Qz": row[6], "Qw": row[7]
            }

    # Read V18 predictions (we use its translation: Tx, Ty, Tz)
    v18_data = {}
    with open(args.v18, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            timestamp = row[0]
            v18_data[timestamp] = {
                "Tx": row[1], "Ty": row[2], "Tz": row[3],
                "Qx": row[4], "Qy": row[5], "Qz": row[6], "Qw": row[7]
            }

    # Merge: V18 translation + V15 rotation
    print(f"V15 predictions: {len(v15_data)}")
    print(f"V18 predictions: {len(v18_data)}")

    merged_count = 0
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for timestamp in v15_data:
            if timestamp in v18_data:
                # Take translation from V18, rotation from V15
                writer.writerow([
                    timestamp,
                    v18_data[timestamp]["Tx"],  # V18 translation
                    v18_data[timestamp]["Ty"],
                    v18_data[timestamp]["Tz"],
                    v15_data[timestamp]["Qx"],  # V15 rotation
                    v15_data[timestamp]["Qy"],
                    v15_data[timestamp]["Qz"],
                    v15_data[timestamp]["Qw"],
                ])
                merged_count += 1
            else:
                # Fallback to V15 for everything
                writer.writerow([
                    timestamp,
                    v15_data[timestamp]["Tx"],
                    v15_data[timestamp]["Ty"],
                    v15_data[timestamp]["Tz"],
                    v15_data[timestamp]["Qx"],
                    v15_data[timestamp]["Qy"],
                    v15_data[timestamp]["Qz"],
                    v15_data[timestamp]["Qw"],
                ])

    print(f"Merged {merged_count} predictions (V18 trans + V15 rot)")

    # Zip for submission
    zip_output = args.output.replace('.csv', '.zip')
    print(f"Zipping to {zip_output}...")
    with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output, arcname="submission.csv")

    print(f"Done! Submit {zip_output} to CodaBench.")


if __name__ == "__main__":
    main()
