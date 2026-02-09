import h5py
import numpy as np
import pandas as pd
import json
import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def load_camera_intrinsics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['cameraMatrix']), data['Nu'], data['Nv']

def load_h5_to_pandas(h5_path):
    with h5py.File(h5_path, "r") as f:
        event_grp = f["events"]
        # Use pandas for ELOPE-style slicing
        ev_data = pd.DataFrame({
            "x": event_grp["xs"][()],
            "y": event_grp["ys"][()],
            "p": event_grp["ps"][()],
            "t": event_grp["ts"][()]
        })
        labels_ds = f["labels"]["data"]
        labels_df = pd.DataFrame({
            "filename": labels_ds["filename"].astype(str),
            "Tx": labels_ds["Tx"], "Ty": labels_ds["Ty"], "Tz": labels_ds["Tz"],
            "Qx": labels_ds["Qx"], "Qy": labels_ds["Qy"], "Qz": labels_ds["Qz"], "Qw": labels_ds["Qw"],
            "timestamp": labels_ds["timestamp"],
        })
    return ev_data, labels_df

def project(q, r, K):
    """ Quaternion projection to image plane """
    p_axes = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    pose_mat = np.hstack((Rotation.from_quat(q).as_matrix(), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, p_axes.T)
    p_cam = p_cam / p_cam[2]
    points_img = K.dot(p_cam)
    return points_img[0], points_img[1]

def create_interlabel_frame(ev_data, t_start, t_end, width, height):
    """ 
    Accumulates ALL events between two pose timestamps.
    Colors: White (Pos), Blue (Neg) - ELOPE Style.
    """
    # Slice the events
    ev_slice = ev_data.loc[(ev_data['t'] >= t_start) & (ev_data['t'] < t_end)]

    # Initialize frame [width, height, 3]
    ev_frame = np.zeros([width, height, 3], dtype=np.uint8)

    if ev_slice.empty:
        return np.swapaxes(ev_frame, 0, 1)

    # Separate polarity
    pos_ev = ev_slice.loc[ev_slice['p'] == 1]
    neg_ev = ev_slice.loc[ev_slice['p'] == 0]

    # Map to pixels
    px = np.clip(pos_ev['x'].values.astype(int), 0, width - 1)
    py = np.clip(pos_ev['y'].values.astype(int), 0, height - 1)
    # White: R=255, G=255, B=255
    ev_frame[px, py] = [255, 255, 255] 

    nx = np.clip(neg_ev['x'].values.astype(int), 0, width - 1)
    ny = np.clip(neg_ev['y'].values.astype(int), 0, height - 1)
    # ELOPE Blue (RGB): R=80, G=137, B=204 
    # (Setting R to 80 and B to 204 makes it Blue)
    ev_frame[nx, ny] = [80, 137, 204] 

    # --- Visibility ---
    # Dilate makes the points thicker so they are visible at 1280x720
    kernel = np.ones((3, 3), np.uint8)
    ev_frame = cv2.dilate(ev_frame, kernel)

    # Swap axes so x is horizontal and y is vertical in imshow
    return np.swapaxes(ev_frame, 0, 1)

def main():
    h5_path = "./h5/RT000.h5" 
    camera_json = "camera.json"
    
    K, width, height = load_camera_intrinsics(camera_json)
    ev_data, labels_df = load_h5_to_pandas(h5_path)

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.flatten()

    # We start from index 1 because we need the window (idx-1) to (idx)
    # Sampling 6 poses throughout the sequence
    indices = np.linspace(1, len(labels_df) - 1, rows * cols, dtype=int)

    for i, idx in enumerate(indices):
        prev_label = labels_df.iloc[idx-1]
        curr_label = labels_df.iloc[idx]
        
        # WINDOW: All events between previous pose and current pose
        t_start = prev_label['timestamp']
        t_end = curr_label['timestamp']
        
        # Generate Frame
        img = create_interlabel_frame(ev_data, t_start, t_end, width, height)
        
        # Current Pose info
        r = np.array([curr_label['Tx'], curr_label['Ty'], curr_label['Tz']])
        q = np.array([curr_label['Qx'], curr_label['Qy'], curr_label['Qz'], curr_label['Qw']])
        
        # Visualize
        axes[i].imshow(img)
        
        # Project and draw axes
        xa, ya = project(q, r, K)
        c = (xa[0], ya[0])
        colors = ['r', 'g', 'b']
        for j in range(3):
            vx, vy = (xa[j+1] - xa[0]), (ya[j+1] - ya[0])
            mag = np.sqrt(vx**2 + vy**2)
            if mag > 0:
                axes[i].arrow(c[0], c[1], vx/mag*120, vy/mag*120, 
                             head_width=20, color=colors[j], linewidth=3, zorder=10)

        axes[i].set_title(f"Pose {idx} | Window: {t_start}-{t_end}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle("Event accumulation between consecutive pose labels", fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()
