"""
Visualize predicted trajectories from submission.csv.
Plots 3D trajectories (Tx, Ty, Tz) for each sequence.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

def visualize_submission(csv_path):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Add sequence ID column
    df['seq_id'] = df['timestamp'].apply(lambda x: x.split('_')[0])
    sequences = df['seq_id'].unique()
    
    print(f"Found {len(sequences)} sequences: {sorted(sequences)}")
    
    # Setup plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.jet(np.linspace(0, 1, len(sequences)))
    
    for i, seq in enumerate(sorted(sequences)):
        seq_df = df[df['seq_id'] == seq]
        
        xs = seq_df['Tx'].values
        ys = seq_df['Ty'].values
        zs = seq_df['Tz'].values
        
        ax.plot(xs, ys, zs, label=seq, color=colors[i], linewidth=2)
        ax.scatter(xs[0], ys[0], zs[0], color=colors[i], marker='o')  # Start point
        ax.scatter(xs[-1], ys[-1], zs[-1], color=colors[i], marker='x')  # End point
    
    ax.set_xlabel('Tx (m)')
    ax.set_ylabel('Ty (m)')
    ax.set_zlabel('Tz (m)')
    ax.set_title('Predicted Trajectories (Stream-2 Test Set)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set equal aspect ratio for realistic visualization
    # Matplotlib 3D doesn't have "axis equal" so we fake it
    all_x = df['Tx'].values
    all_y = df['Ty'].values
    all_z = df['Tz'].values
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    output_path = csv_path.replace('.csv', '_plot.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    # plt.show() # Can't show GUI in this environment, saving is better

if __name__ == "__main__":
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', default='submission.csv', nargs='?')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: {args.csv_path} not found.")
        sys.exit(1)
        
    visualize_submission(args.csv_path)
