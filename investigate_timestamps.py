import h5py
import numpy as np
import os

def analyze_h5(filepath, is_test=False):
    print(f"\n--- Analyzing {filepath} ---")
    with h5py.File(filepath, "r") as f:
        ts = f["events"]["ts"][()]
        print(f"Total events: {len(ts)}")
        print(f"Min ts: {ts.min()}")
        print(f"Max ts: {ts.max()}")
        
        # Look at the first 10,000 events to see time density
        if len(ts) > 10000:
            time_for_10k = ts[10000] - ts[0]
            print(f"Time units covered by first 10k events: {time_for_10k}")
            
        if not is_test:
            # Look at labels if available
            labels_ts = f["labels"]["data"]["timestamp"][()]
            print(f"\nLabels min ts: {labels_ts.min()}")
            print(f"Labels max ts: {labels_ts.max()}")
            print(f"Number of poses: {len(labels_ts)}")
            print(f"First 5 label timestamps: {labels_ts[:5]}")
            
            # How many events in first 1000 units (100ms synthetic)?
            t1 = labels_ts[1]
            t0 = t1 - 1000
            mask1 = (ts >= t0) & (ts < t1)
            print(f"\nEvents between {t0} and {t1} (100ms window): {mask1.sum()}")
            
        else:
            # For test set, simulate predict.py logic
            # predict.py uses 100,000 units for 100ms
            t1 = 100000
            t0 = 0
            mask1 = (ts >= t0) & (ts < t1)
            print(f"\nEvents between 0 and 100,000 (100ms real data window): {mask1.sum()}")
            
            # Predict.py USED TO DO THIS for test set:
            t1_old = 100000
            t0_old = 100000 - 1000
            mask2 = (ts >= t0_old) & (ts < t1_old)
            print(f"Events between 99000 and 100,000 (Old 1ms window): {mask2.sum()}")

if __name__ == "__main__":
    analyze_h5("h5/RT000.h5", is_test=False)
    analyze_h5("test/RT901.h5", is_test=True)
