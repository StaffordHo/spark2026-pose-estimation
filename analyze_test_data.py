"""
Analyze test sequences vs training data distribution.
Checks event density, timestamp ranges, and pose statistics to detect domain shift.
"""

import os
import sys
import glob
import h5py
import numpy as np

def analyze_sequence(h5_path: str) -> dict:
    """Analyze a single H5 sequence."""
    with h5py.File(h5_path, "r") as f:
        xs = f["events"]["xs"][()]
        ys = f["events"]["ys"][()]
        ts = f["events"]["ts"][()]
        ps = f["events"]["ps"][()]
        
        has_labels = "labels" in f
        labels = {}
        if has_labels:
            labels = {
                "Tx": f["labels"]["data"]["Tx"][()],
                "Ty": f["labels"]["data"]["Ty"][()],
                "Tz": f["labels"]["data"]["Tz"][()],
                "timestamp": f["labels"]["data"]["timestamp"][()],
            }
    
    stats = {
        "name": os.path.basename(h5_path).replace(".h5", ""),
        "num_events": len(xs),
        "x_range": (int(xs.min()), int(xs.max())),
        "y_range": (int(ys.min()), int(ys.max())),
        "t_range": (int(ts.min()), int(ts.max())),
        "t_duration_s": (ts.max() - ts.min()) / 1e4,  # 100µs units → seconds
        "polarity_ratio": ps.sum() / len(ps),  # Fraction positive
        "events_per_sec": len(xs) / max((ts.max() - ts.min()) / 1e4, 1e-6),
    }
    
    if has_labels:
        stats["num_poses"] = len(labels["Tx"])
        stats["Tx_range"] = (float(labels["Tx"].min()), float(labels["Tx"].max()))
        stats["Ty_range"] = (float(labels["Ty"].min()), float(labels["Ty"].max()))
        stats["Tz_range"] = (float(labels["Tz"].min()), float(labels["Tz"].max()))
        stats["Tz_mean"] = float(labels["Tz"].mean())
    
    return stats


def print_stats_table(all_stats, title):
    """Print statistics in a table format."""
    print(f"\n{'='*90}")
    print(f" {title}")
    print(f"{'='*90}")
    print(f"{'Seq':<8} {'Events':>12} {'Ev/sec':>12} {'Duration':>8} {'Pol+%':>6} {'X range':>12} {'Y range':>12}")
    print(f"{'-'*90}")
    
    for s in all_stats:
        print(f"{s['name']:<8} {s['num_events']:>12,} {s['events_per_sec']:>12,.0f} "
              f"{s['t_duration_s']:>7.1f}s {s['polarity_ratio']:>5.1%} "
              f"{str(s['x_range']):>12} {str(s['y_range']):>12}")
    
    # Summary
    print(f"{'-'*90}")
    total_events = sum(s["num_events"] for s in all_stats)
    avg_ev_per_sec = np.mean([s["events_per_sec"] for s in all_stats])
    avg_pol = np.mean([s["polarity_ratio"] for s in all_stats])
    print(f"{'TOTAL':<8} {total_events:>12,} {avg_ev_per_sec:>12,.0f} "
          f"{'':>8} {avg_pol:>5.1%}")
    
    # Pose stats if available
    if "Tz_range" in all_stats[0]:
        print(f"\n{'Seq':<8} {'Poses':>6} {'Tz range':>20} {'Tz mean':>10}")
        print(f"{'-'*50}")
        for s in all_stats:
            print(f"{s['name']:<8} {s['num_poses']:>6} "
                  f"[{s['Tz_range'][0]:>7.3f}, {s['Tz_range'][1]:>7.3f}] "
                  f"{s['Tz_mean']:>10.3f}")


def main():
    # Analyze training data (first 10 sequences for sample)
    train_files = sorted(glob.glob(os.path.join("h5", "RT*.h5")))[:10]
    test_files = sorted(glob.glob(os.path.join("test", "RT*.h5")))
    
    if not test_files:
        test_files = sorted(glob.glob(os.path.join("test", "*.h5")))
    
    print("Analyzing training sequences (sample of 10)...")
    train_stats = [analyze_sequence(f) for f in train_files]
    print_stats_table(train_stats, "TRAINING DATA (Sample)")
    
    print("\n\nAnalyzing test sequences...")
    test_stats = [analyze_sequence(f) for f in test_files]
    print_stats_table(test_stats, "TEST DATA")
    
    # Compare distributions
    print(f"\n{'='*90}")
    print(" DISTRIBUTION COMPARISON")
    print(f"{'='*90}")
    
    train_ev = [s["events_per_sec"] for s in train_stats]
    test_ev = [s["events_per_sec"] for s in test_stats]
    
    print(f"\nEvents/sec:  Train avg={np.mean(train_ev):,.0f} ± {np.std(train_ev):,.0f}")
    print(f"             Test  avg={np.mean(test_ev):,.0f} ± {np.std(test_ev):,.0f}")
    
    if "Tz_mean" in train_stats[0] and "Tz_mean" in test_stats[0]:
        train_tz = [s["Tz_mean"] for s in train_stats]
        test_tz = [s["Tz_mean"] for s in test_stats]
        
        print(f"\nTz (depth):  Train avg={np.mean(train_tz):.3f} ± {np.std(train_tz):.3f}")
        print(f"             Test  avg={np.mean(test_tz):.3f} ± {np.std(test_tz):.3f}")
        
        # Check for domain shift
        shift = abs(np.mean(test_tz) - np.mean(train_tz))
        if shift > 1.0:
            print(f"\n⚠️  WARNING: Significant Tz domain shift detected ({shift:.2f}m)!")
            print(f"   Consider updating TRANS_MEAN/TRANS_STD for test data.")
        else:
            print(f"\n✅ Tz distributions look compatible (shift = {shift:.2f}m)")


if __name__ == "__main__":
    main()
