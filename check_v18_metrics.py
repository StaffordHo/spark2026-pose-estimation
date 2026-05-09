import os
from tensorboard.backend.event_processing import event_accumulator

def extract_metrics(logdir):
    print(f"Reading logs from: {logdir}")
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    tags = ea.Tags()['scalars']
    
    if 'Val_Error/Translation' in tags:
        trans_events = ea.Scalars('Val_Error/Translation')
        for e in trans_events:
            print(f"Epoch {e.step}: Trans Error = {e.value:.4f} m")
            
    if 'Val_Error/Rotation' in tags:
        rot_events = ea.Scalars('Val_Error/Rotation')
        for e in rot_events:
            print(f"Epoch {e.step}: Rot Error = {e.value:.4f} degrees")
            
    if 'Val_Loss/Total' in tags:
        loss_events = ea.Scalars('Val_Loss/Total')
        best_loss = float('inf')
        best_epoch = -1
        for e in loss_events:
            if e.value < best_loss:
                best_loss = e.value
                best_epoch = e.step
        print(f"\nBest Total Loss: {best_loss:.4f} at Epoch {best_epoch}")

if __name__ == '__main__':
    logdir = "logs"
    latest_run = max([os.path.join(logdir, d) for d in os.listdir(logdir)], key=os.path.getmtime)
    print(f"Latest run: {latest_run}")
    extract_metrics(latest_run)
