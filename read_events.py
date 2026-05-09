from tensorboard.backend.event_processing import event_accumulator
import os

def extract_metrics(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    
    metrics = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]
    
    print(f"Available Tags: {list(metrics.keys())}")
    
    tags_to_show = ["val_loss", "pos_error_mean", "rot_error_mean"]
    for tag in tags_to_show:
        if tag in metrics:
            print(f"\n--- {tag} ---")
            # Show last 5 epochs
            for step, val in metrics[tag][-5:]:
                print(f"Step {step}: {val:.4f}")
        else:
            # Try to find similar tags
            matching = [t for t in metrics.keys() if tag in t]
            if matching:
                print(f"\n--- {matching[0]} ---")
                for step, val in metrics[matching[0]][-5:]:
                    print(f"Step {step}: {val:.4f}")

if __name__ == "__main__":
    log_dir = r"logs\spark2026_v12_platinum_20260225_094044"
    # Actually, let me check the list_dir output again to be sure of the path.
    # Ah, I see spark2026_v12_platinum_20260225_094044 and spark2026_v12_platinum_20260225_094256
    # The running one is likely the latter.
    log_dir = r"logs\spark2026_v12_platinum_20260225_094256"
    if os.path.exists(log_dir):
        extract_metrics(log_dir)
    else:
        print(f"Log dir not found: {log_dir}")
