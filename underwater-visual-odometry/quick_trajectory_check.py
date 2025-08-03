#!/usr/bin/env python3
"""
Quick trajectory check to verify the complete dataset trajectory
"""

import pandas as pd
import numpy as np

def quick_trajectory_analysis():
    # Load data
    data_csv = 'data/processed/training_dataset/training_data.csv'
    df = pd.read_csv(data_csv)
    
    print("Quick Trajectory Analysis")
    print("=" * 40)
    
    # Check validation split
    val_df = df[df['split'] == 'val'] if 'split' in df.columns else df
    print(f"Total samples: {len(df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Integrate complete trajectory (validation only, as used in visualization)
    val_df_sorted = val_df.sort_values('timestamp').reset_index(drop=True)
    
    trajectory = []
    current_pose = np.array([0.0, 0.0, 0.0])
    trajectory.append(current_pose.copy())
    
    for _, row in val_df_sorted.iterrows():
        delta = np.array([row['delta_x'], row['delta_y'], row['delta_z']])
        current_pose += delta
        trajectory.append(current_pose.copy())
    
    trajectory = np.array(trajectory)
    total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    
    print(f"\nValidation Trajectory:")
    print(f"  Points: {len(trajectory)}")
    print(f"  Distance: {total_distance:.3f} m")
    print(f"  Start: ({trajectory[0, 0]:.3f}, {trajectory[0, 1]:.3f}, {trajectory[0, 2]:.3f})")
    print(f"  End: ({trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f}, {trajectory[-1, 2]:.3f})")
    
    # Check sequence length impact
    sequence_length = 2
    num_sequences = len(val_df_sorted) - sequence_length + 1
    print(f"\nSequence Analysis:")
    print(f"  Sequence length: {sequence_length}")  
    print(f"  Number of sequences: {num_sequences}")
    print(f"  This explains why trajectory visualization had limited samples!")
    
    return trajectory, total_distance

if __name__ == '__main__':
    trajectory, distance = quick_trajectory_analysis()