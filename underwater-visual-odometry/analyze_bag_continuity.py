#!/usr/bin/env python3
"""
Analyze bag continuity and create complete trajectory
Check if bags are continuous or separate recordings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
from datetime import datetime

def analyze_bag_continuity(data_csv):
    """Analyze if bags are continuous and should be connected"""
    
    print("Bag Continuity Analysis")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(data_csv)
    
    # Analyze each bag
    bag_info = []
    
    for bag_name, bag_df in df.groupby('bag_name'):
        bag_df = bag_df.sort_values('timestamp').reset_index(drop=True)
        
        bag_stats = {
            'bag_name': bag_name,
            'num_samples': len(bag_df),
            'start_timestamp': bag_df['timestamp'].iloc[0],
            'end_timestamp': bag_df['timestamp'].iloc[-1],
            'duration': bag_df['timestamp'].iloc[-1] - bag_df['timestamp'].iloc[0],
            'start_pose': [bag_df['pose_x'].iloc[0], bag_df['pose_y'].iloc[0], bag_df['pose_z'].iloc[0]],
            'end_pose': [bag_df['pose_x'].iloc[-1], bag_df['pose_y'].iloc[-1], bag_df['pose_z'].iloc[-1]]
        }
        
        bag_info.append(bag_stats)
        
        print(f"\nBag: {bag_name}")
        print(f"  Samples: {bag_stats['num_samples']}")
        print(f"  Start time: {bag_stats['start_timestamp']:.3f}")
        print(f"  End time: {bag_stats['end_timestamp']:.3f}")
        print(f"  Duration: {bag_stats['duration']:.2f} s")
        print(f"  Start pose: ({bag_stats['start_pose'][0]:.6f}, {bag_stats['start_pose'][1]:.6f}, {bag_stats['start_pose'][2]:.6f})")
        print(f"  End pose: ({bag_stats['end_pose'][0]:.6f}, {bag_stats['end_pose'][1]:.6f}, {bag_stats['end_pose'][2]:.6f})")
    
    # Sort by timestamp
    bag_info.sort(key=lambda x: x['start_timestamp'])
    
    # Check continuity
    print(f"\nContinuity Analysis:")
    print("-" * 30)
    
    continuous_bags = True
    for i in range(1, len(bag_info)):
        prev_bag = bag_info[i-1]
        curr_bag = bag_info[i]
        
        time_gap = curr_bag['start_timestamp'] - prev_bag['end_timestamp']
        
        print(f"Gap between {prev_bag['bag_name']} and {curr_bag['bag_name']}:")
        print(f"  Time gap: {time_gap:.3f} s")
        print(f"  Previous end pose: {prev_bag['end_pose']}")
        print(f"  Current start pose: {curr_bag['start_pose']}")
        
        # Check if poses match (allowing small tolerance)
        pose_diff = np.linalg.norm(np.array(curr_bag['start_pose']) - np.array(prev_bag['end_pose']))
        print(f"  Pose difference: {pose_diff:.6f} m")
        
        if pose_diff > 0.1:  # 10cm tolerance
            continuous_bags = False
            print(f"  -> DISCONTINUOUS (large pose jump)")
        elif time_gap > 1.0:  # 1 second tolerance
            continuous_bags = False
            print(f"  -> DISCONTINUOUS (large time gap)")
        else:
            print(f"  -> CONTINUOUS")
    
    return bag_info, continuous_bags

def create_complete_trajectory(data_csv, continuous=True):
    """Create complete trajectory, either continuous or separate bags"""
    
    df = pd.read_csv(data_csv)
    
    if continuous:
        print(f"\nCreating CONTINUOUS trajectory...")
        # Sort all data by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Integrate complete trajectory
        trajectory = []
        current_pose = np.array([0.0, 0.0, 0.0])
        trajectory.append(current_pose.copy())
        
        for _, row in df_sorted.iterrows():
            delta = np.array([row['delta_x'], row['delta_y'], row['delta_z']])
            current_pose += delta
            trajectory.append(current_pose.copy())
        
        complete_trajectory = np.array(trajectory)
        total_distance = np.sum(np.linalg.norm(np.diff(complete_trajectory, axis=0), axis=1))
        
        print(f"Complete continuous trajectory:")
        print(f"  Total points: {len(complete_trajectory)}")
        print(f"  Total distance: {total_distance:.3f} m")
        print(f"  Start: ({complete_trajectory[0, 0]:.3f}, {complete_trajectory[0, 1]:.3f}, {complete_trajectory[0, 2]:.3f})")
        print(f"  End: ({complete_trajectory[-1, 0]:.3f}, {complete_trajectory[-1, 1]:.3f}, {complete_trajectory[-1, 2]:.3f})")
        
        return complete_trajectory, df_sorted
        
    else:
        print(f"\nCreating SEPARATE bag trajectories...")
        bag_trajectories = {}
        
        for bag_name, bag_df in df.groupby('bag_name'):
            bag_df = bag_df.sort_values('timestamp').reset_index(drop=True)
            
            # Integrate bag trajectory
            trajectory = []
            current_pose = np.array([0.0, 0.0, 0.0])
            trajectory.append(current_pose.copy())
            
            for _, row in bag_df.iterrows():
                delta = np.array([row['delta_x'], row['delta_y'], row['delta_z']])
                current_pose += delta
                trajectory.append(current_pose.copy())
            
            bag_trajectories[bag_name] = np.array(trajectory)
            
            distance = np.sum(np.linalg.norm(np.diff(bag_trajectories[bag_name], axis=0), axis=1))
            print(f"  {bag_name}: {len(trajectory)} points, {distance:.3f} m")
        
        return bag_trajectories, df

def plot_complete_trajectory(trajectory_data, continuous=True, output_file='complete_trajectory.png'):
    """Plot the complete trajectory"""
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 15))
    
    if continuous:
        # Single continuous trajectory
        trajectory = trajectory_data
        
        # 3D plot
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'cyan', linewidth=2, alpha=0.8, label='Complete Trajectory')
        ax1.scatter([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]], 
                   c='green', s=100, marker='o', label='Start')
        ax1.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]], 
                   c='red', s=100, marker='s', label='End')
        
        ax1.set_xlabel('X (m)', color='white')
        ax1.set_ylabel('Y (m)', color='white')
        ax1.set_zlabel('Z (m)', color='white')
        ax1.set_title('Complete 3D Trajectory', color='white', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Top view
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'cyan', linewidth=2, alpha=0.8)
        ax2.scatter([trajectory[0, 0]], [trajectory[0, 1]], c='green', s=80, marker='o', label='Start')
        ax2.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], c='red', s=80, marker='s', label='End')
        ax2.set_xlabel('X (m)', color='white')
        ax2.set_ylabel('Y (m)', color='white')
        ax2.set_title('Top View (XY)', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Side view
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(trajectory[:, 0], trajectory[:, 2], 'cyan', linewidth=2, alpha=0.8)
        ax3.set_xlabel('X (m)', color='white')
        ax3.set_ylabel('Z (m)', color='white')
        ax3.set_title('Side View (XZ)', color='white')
        ax3.grid(True, alpha=0.3)
        
        # Distance over time
        distances = np.cumsum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(distances, 'orange', linewidth=2)
        ax4.set_xlabel('Sample Index', color='white')
        ax4.set_ylabel('Cumulative Distance (m)', color='white')
        ax4.set_title('Distance Traveled', color='white')
        ax4.grid(True, alpha=0.3)
        
    else:
        # Multiple separate trajectories
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'yellow']
        
        # 3D plot
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        for i, (bag_name, trajectory) in enumerate(trajectory_data.items()):
            color = colors[i % len(colors)]
            ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                    color=color, linewidth=2, alpha=0.8, label=bag_name[:20])
            ax1.scatter([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]], 
                       c=color, s=50, marker='o')
            ax1.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]], 
                       c=color, s=50, marker='s')
        
        ax1.set_xlabel('X (m)', color='white')
        ax1.set_ylabel('Y (m)', color='white')
        ax1.set_zlabel('Z (m)', color='white')
        ax1.set_title('Separate Bag Trajectories - 3D', color='white', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Top view
        ax2 = fig.add_subplot(2, 2, 2)
        for i, (bag_name, trajectory) in enumerate(trajectory_data.items()):
            color = colors[i % len(colors)]
            ax2.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2, 
                    alpha=0.8, label=bag_name[:20])
            ax2.scatter([trajectory[0, 0]], [trajectory[0, 1]], c=color, s=50, marker='o')
            ax2.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], c=color, s=50, marker='s')
        
        ax2.set_xlabel('X (m)', color='white')
        ax2.set_ylabel('Y (m)', color='white')
        ax2.set_title('Top View (XY)', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
    
    print(f"Complete trajectory plot saved: {output_file}")

if __name__ == '__main__':
    data_csv = 'data/processed/training_dataset/training_data.csv'
    
    if not Path(data_csv).exists():
        print(f"Error: {data_csv} not found!")
        exit(1)
    
    # Analyze bag continuity
    bag_info, is_continuous = analyze_bag_continuity(data_csv)
    
    print(f"\n" + "="*50)
    print(f"CONCLUSION: Bags are {'CONTINUOUS' if is_continuous else 'SEPARATE'}")
    print(f"="*50)
    
    # Create and plot complete trajectory
    trajectory_data, df_data = create_complete_trajectory(data_csv, continuous=is_continuous)
    plot_complete_trajectory(trajectory_data, continuous=is_continuous)
    
    # Save analysis
    analysis = {
        'is_continuous': is_continuous,
        'bag_info': bag_info,
        'recommendation': 'Use continuous trajectory' if is_continuous else 'Use separate bag trajectories'
    }
    
    with open('bag_continuity_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis saved: bag_continuity_analysis.json")