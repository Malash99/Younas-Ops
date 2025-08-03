#!/usr/bin/env python3
"""
Plot Ground Truth Trajectory Analysis
Visualizes the complete ground truth trajectory to verify data correctness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path

def integrate_trajectory(df):
    """Integrate delta poses to get full trajectory"""
    trajectory = []
    current_pose = np.array([0.0, 0.0, 0.0])  # Start at origin
    
    trajectory.append(current_pose.copy())
    
    for _, row in df.iterrows():
        # Add delta to current pose
        delta = np.array([row['delta_x'], row['delta_y'], row['delta_z']])
        current_pose += delta
        trajectory.append(current_pose.copy())
    
    return np.array(trajectory)

def analyze_ground_truth_trajectory(data_csv):
    """Analyze and visualize ground truth trajectory"""
    
    print("Ground Truth Trajectory Analysis")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} data points")
    
    # Basic statistics
    print(f"\nDelta Statistics:")
    print(f"Delta X: min={df['delta_x'].min():.6f}, max={df['delta_x'].max():.6f}, mean={df['delta_x'].mean():.6f}")
    print(f"Delta Y: min={df['delta_y'].min():.6f}, max={df['delta_y'].max():.6f}, mean={df['delta_y'].mean():.6f}")
    print(f"Delta Z: min={df['delta_z'].min():.6f}, max={df['delta_z'].max():.6f}, mean={df['delta_z'].mean():.6f}")
    
    # Calculate step sizes
    step_sizes = np.sqrt(df['delta_x']**2 + df['delta_y']**2 + df['delta_z']**2)
    print(f"\nStep Sizes:")
    print(f"Min step: {step_sizes.min():.6f} m")
    print(f"Max step: {step_sizes.max():.6f} m") 
    print(f"Mean step: {step_sizes.mean():.6f} m")
    print(f"Total steps: {len(step_sizes)}")
    
    # Time analysis
    if 'dt' in df.columns:
        print(f"\nTemporal Analysis:")
        print(f"Min dt: {df['dt'].min():.4f} s")
        print(f"Max dt: {df['dt'].max():.4f} s")
        print(f"Mean dt: {df['dt'].mean():.4f} s")
        print(f"Total time: {df['dt'].sum():.2f} s")
        
        # Calculate velocities
        velocities = step_sizes / df['dt']
        print(f"Mean velocity: {velocities.mean():.6f} m/s")
        print(f"Max velocity: {velocities.max():.6f} m/s")
    
    # Process each bag separately
    bag_trajectories = {}
    total_distance = 0
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 15))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'yellow']
    
    for i, (bag_name, bag_df) in enumerate(df.groupby('bag_name')):
        print(f"\nProcessing bag: {bag_name}")
        bag_df = bag_df.sort_values('timestamp').reset_index(drop=True)
        
        # Integrate trajectory
        trajectory = integrate_trajectory(bag_df)
        bag_trajectories[bag_name] = trajectory
        
        # Calculate bag statistics
        bag_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        total_distance += bag_distance
        
        print(f"  Points: {len(trajectory)}")
        print(f"  Distance: {bag_distance:.6f} m")
        print(f"  Start: ({trajectory[0, 0]:.6f}, {trajectory[0, 1]:.6f}, {trajectory[0, 2]:.6f})")
        print(f"  End: ({trajectory[-1, 0]:.6f}, {trajectory[-1, 1]:.6f}, {trajectory[-1, 2]:.6f})")
        
        # Plot trajectory
        color = colors[i % len(colors)]
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color=color, linewidth=2, label=f'{bag_name[:20]}...', alpha=0.8)
        
        # Mark start and end
        ax1.scatter([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]], 
                   c=color, s=100, marker='o', alpha=1.0)
        ax1.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]], 
                   c=color, s=100, marker='s', alpha=1.0)
    
    print(f"\nTotal trajectory distance: {total_distance:.6f} m ({total_distance*100:.2f} cm)")
    
    ax1.set_xlabel('X (m)', color='white')
    ax1.set_ylabel('Y (m)', color='white') 
    ax1.set_zlabel('Z (m)', color='white')
    ax1.set_title('Ground Truth 3D Trajectory', color='white', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2D top view
    ax2 = fig.add_subplot(2, 3, 2)
    for i, (bag_name, trajectory) in enumerate(bag_trajectories.items()):
        color = colors[i % len(colors)]
        ax2.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2, 
                label=f'{bag_name[:20]}...', alpha=0.8)
        ax2.scatter([trajectory[0, 0]], [trajectory[0, 1]], c=color, s=50, marker='o')
        ax2.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], c=color, s=50, marker='s')
    
    ax2.set_xlabel('X (m)', color='white')
    ax2.set_ylabel('Y (m)', color='white')
    ax2.set_title('Top View (XY)', color='white')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Side view
    ax3 = fig.add_subplot(2, 3, 3)
    for i, (bag_name, trajectory) in enumerate(bag_trajectories.items()):
        color = colors[i % len(colors)]
        ax3.plot(trajectory[:, 0], trajectory[:, 2], color=color, linewidth=2,
                label=f'{bag_name[:20]}...', alpha=0.8)
    
    ax3.set_xlabel('X (m)', color='white')
    ax3.set_ylabel('Z (m)', color='white')
    ax3.set_title('Side View (XZ)', color='white')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Delta distribution
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(step_sizes, bins=50, color='cyan', alpha=0.7, edgecolor='white')
    ax4.set_xlabel('Step Size (m)', color='white')
    ax4.set_ylabel('Frequency', color='white')
    ax4.set_title('Step Size Distribution', color='white')
    ax4.grid(True, alpha=0.3)
    
    # Velocity distribution (if dt available)
    if 'dt' in df.columns:
        ax5 = fig.add_subplot(2, 3, 5)
        velocities = step_sizes / df['dt']
        ax5.hist(velocities, bins=50, color='orange', alpha=0.7, edgecolor='white')
        ax5.set_xlabel('Velocity (m/s)', color='white')
        ax5.set_ylabel('Frequency', color='white')
        ax5.set_title('Velocity Distribution', color='white')
        ax5.grid(True, alpha=0.3)
    
    # Time series of step sizes
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(step_sizes, color='magenta', linewidth=1, alpha=0.8)
    ax6.set_xlabel('Sample Index', color='white')
    ax6.set_ylabel('Step Size (m)', color='white')
    ax6.set_title('Step Size Over Time', color='white')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ground_truth_analysis.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.show()
    
    # Save analysis results
    analysis = {
        'total_samples': len(df),
        'total_distance_m': float(total_distance),
        'total_distance_cm': float(total_distance * 100),
        'num_bags': len(bag_trajectories),
        'step_statistics': {
            'min_step_m': float(step_sizes.min()),
            'max_step_m': float(step_sizes.max()),
            'mean_step_m': float(step_sizes.mean()),
            'std_step_m': float(step_sizes.std())
        },
        'delta_statistics': {
            'delta_x': {'min': float(df['delta_x'].min()), 'max': float(df['delta_x'].max()), 'mean': float(df['delta_x'].mean())},
            'delta_y': {'min': float(df['delta_y'].min()), 'max': float(df['delta_y'].max()), 'mean': float(df['delta_y'].mean())},
            'delta_z': {'min': float(df['delta_z'].min()), 'max': float(df['delta_z'].max()), 'mean': float(df['delta_z'].mean())}
        }
    }
    
    if 'dt' in df.columns:
        analysis['temporal_statistics'] = {
            'total_time_s': float(df['dt'].sum()),
            'mean_dt_s': float(df['dt'].mean()),
            'mean_velocity_m_s': float(velocities.mean()),
            'max_velocity_m_s': float(velocities.max())
        }
    
    with open('ground_truth_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Visualization saved: ground_truth_analysis.png")
    print(f"Analysis data saved: ground_truth_analysis.json")
    
    return analysis, bag_trajectories

if __name__ == '__main__':
    data_csv = 'data/processed/training_dataset/training_data.csv'
    
    if not Path(data_csv).exists():
        print(f"Error: {data_csv} not found!")
        print("Available CSV files:")
        for csv_file in Path('.').rglob('*.csv'):
            print(f"  {csv_file}")
    else:
        analysis, trajectories = analyze_ground_truth_trajectory(data_csv)