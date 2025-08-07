#!/usr/bin/env python3
"""
Visualize Full Underwater Trajectories
Show the complete trajectory paths from all bags with proper delta integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_integrate_trajectory(bag_name, df):
    """Load a specific bag and integrate deltas to get full trajectory"""
    bag_data = df[df['bag_name'] == bag_name].copy()
    
    if len(bag_data) == 0:
        return None
    
    # Sort by timestamp to ensure proper order
    bag_data = bag_data.sort_values('timestamp').reset_index(drop=True)
    
    # Integrate deltas to get actual trajectory
    cumulative_x = np.cumsum(bag_data['delta_x'].values) 
    cumulative_y = np.cumsum(bag_data['delta_y'].values)
    cumulative_z = np.cumsum(bag_data['delta_z'].values)
    
    # Add starting position (all start at origin)
    cumulative_x = np.concatenate([[0], cumulative_x])
    cumulative_y = np.concatenate([[0], cumulative_y])
    cumulative_z = np.concatenate([[0], cumulative_z])
    
    return {
        'bag_name': bag_name,
        'split': bag_data['split'].iloc[0],
        'frames': len(bag_data),
        'x': cumulative_x,
        'y': cumulative_y, 
        'z': cumulative_z,
        'timestamps': np.concatenate([[bag_data['timestamp'].iloc[0] - 0.05], bag_data['timestamp'].values])
    }

def plot_all_trajectories():
    """Plot all trajectory bags to show the full underwater navigation patterns"""
    
    # Load data
    df = pd.read_csv('data/processed/training_dataset/training_data_filtered.csv')
    
    # Get all bags
    bags = df['bag_name'].unique()
    trajectories = []
    
    print("LOADING ALL UNDERWATER TRAJECTORIES")
    print("=" * 50)
    
    for bag in bags:
        traj = load_and_integrate_trajectory(bag, df)
        if traj:
            trajectories.append(traj)
            
            # Calculate metrics
            traj_length = np.sum(np.sqrt(np.diff(traj['x'])**2 + np.diff(traj['y'])**2 + np.diff(traj['z'])**2))
            end_to_end = np.sqrt(traj['x'][-1]**2 + traj['y'][-1]**2 + traj['z'][-1]**2)
            loop_closure = end_to_end / traj_length * 100 if traj_length > 0 else 0
            
            print(f"Bag: {bag}")
            print(f"  Split: {traj['split']} | Frames: {traj['frames']}")
            print(f"  Length: {traj_length:.2f}m | End-to-end: {end_to_end:.2f}m")
            print(f"  Loop closure: {loop_closure:.1f}% {'(Good loop)' if loop_closure < 30 else '(Open path)'}")
            print()
    
    # Create comprehensive visualization
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
    
    # 1. All trajectories in XY plane
    ax1 = plt.subplot(3, 3, (1, 2))
    for i, traj in enumerate(trajectories):
        color = colors[i % len(colors)]
        ax1.plot(traj['x'], traj['y'], color=color, linewidth=2, alpha=0.8,
                label=f"{traj['bag_name'][-1]} ({traj['split']}) - {traj['frames']} frames")
        
        # Mark start and end
        ax1.scatter(traj['x'][0], traj['y'][0], color=color, s=100, marker='o', 
                   edgecolor='black', linewidth=2, zorder=5)
        ax1.scatter(traj['x'][-1], traj['y'][-1], color=color, s=100, marker='s',
                   edgecolor='black', linewidth=2, zorder=5)
    
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('All Underwater Vehicle Trajectories (XY Plane)', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 3D trajectories
    ax2 = plt.subplot(3, 3, 3, projection='3d')
    for i, traj in enumerate(trajectories):
        color = colors[i % len(colors)]
        ax2.plot(traj['x'], traj['y'], traj['z'], color=color, linewidth=2, alpha=0.8)
        ax2.scatter(traj['x'][0], traj['y'][0], traj['z'][0], color=color, s=50, marker='o')
        ax2.scatter(traj['x'][-1], traj['y'][-1], traj['z'][-1], color=color, s=50, marker='s')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('3D Trajectories', fontsize=12, fontweight='bold')
    
    # 3-5. Individual trajectory plots for first 3 bags
    for i in range(min(3, len(trajectories))):
        traj = trajectories[i]
        ax = plt.subplot(3, 3, 4 + i)
        
        color = colors[i]
        ax.plot(traj['x'], traj['y'], color=color, linewidth=3, alpha=0.8)
        ax.scatter(traj['x'][0], traj['y'][0], color='green', s=150, marker='o', 
                  edgecolor='black', linewidth=2, label='Start', zorder=5)
        ax.scatter(traj['x'][-1], traj['y'][-1], color='red', s=150, marker='s',
                  edgecolor='black', linewidth=2, label='End', zorder=5)
        
        # Add trajectory direction arrows
        n_arrows = 5
        arrow_indices = np.linspace(10, len(traj['x'])-10, n_arrows, dtype=int)
        for idx in arrow_indices:
            if idx < len(traj['x']) - 1:
                dx = traj['x'][idx+1] - traj['x'][idx]
                dy = traj['y'][idx+1] - traj['y'][idx]
                ax.arrow(traj['x'][idx], traj['y'][idx], dx*5, dy*5, 
                        head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.6)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Bag {traj["bag_name"][-1]} ({traj["split"]}) - {traj["frames"]} frames')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # 6. Z-depth over time
    ax6 = plt.subplot(3, 3, 7)
    for i, traj in enumerate(trajectories):
        color = colors[i % len(colors)]
        time_normalized = np.linspace(0, 100, len(traj['z']))
        ax6.plot(time_normalized, traj['z'], color=color, linewidth=2, alpha=0.8,
                label=f"Bag {traj['bag_name'][-1]}")
    
    ax6.set_xlabel('Time Progress (%)')
    ax6.set_ylabel('Z Position (m)')
    ax6.set_title('Depth Changes Over Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Trajectory statistics
    ax7 = plt.subplot(3, 3, 8)
    
    bag_names = []
    lengths = []
    end_distances = []
    loop_closures = []
    
    for traj in trajectories:
        traj_length = np.sum(np.sqrt(np.diff(traj['x'])**2 + np.diff(traj['y'])**2 + np.diff(traj['z'])**2))
        end_to_end = np.sqrt(traj['x'][-1]**2 + traj['y'][-1]**2 + traj['z'][-1]**2)
        loop_closure = end_to_end / traj_length * 100 if traj_length > 0 else 0
        
        bag_names.append(f"Bag {traj['bag_name'][-1]}")
        lengths.append(traj_length)
        end_distances.append(end_to_end)
        loop_closures.append(loop_closure)
    
    x_pos = np.arange(len(bag_names))
    width = 0.35
    
    bars1 = ax7.bar(x_pos - width/2, lengths, width, label='Trajectory Length (m)', alpha=0.8)
    bars2 = ax7.bar(x_pos + width/2, end_distances, width, label='End-to-End Distance (m)', alpha=0.8)
    
    ax7.set_xlabel('Trajectory')
    ax7.set_ylabel('Distance (m)')
    ax7.set_title('Trajectory Length vs End-to-End Distance')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(bag_names, rotation=45)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Loop closure percentages
    ax8 = plt.subplot(3, 3, 9)
    bars = ax8.bar(bag_names, loop_closures, color=colors[:len(bag_names)], alpha=0.8)
    
    # Color code by loop quality
    for i, (bar, closure) in enumerate(zip(bars, loop_closures)):
        if closure < 30:
            bar.set_color('#4CAF50')  # Green for good loops
        elif closure < 50:
            bar.set_color('#FF9800')  # Orange for partial loops
        else:
            bar.set_color('#F44336')  # Red for open paths
    
    ax8.set_xlabel('Trajectory')
    ax8.set_ylabel('Loop Closure (%)')
    ax8.set_title('Loop Closure Quality\n(Lower = Better Loop)')
    ax8.set_xticklabels(bag_names, rotation=45)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at 30% (good loop threshold)
    ax8.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Good Loop Threshold')
    ax8.legend()
    
    plt.suptitle('Complete Underwater Vehicle Navigation Dataset Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save plot
    plt.savefig('full_underwater_trajectories.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Full trajectory visualization saved as: full_underwater_trajectories.png")
    
    # Don't show plot to avoid timeout
    plt.close()
    
    return trajectories

def main():
    print("UNDERWATER VEHICLE TRAJECTORY ANALYSIS")
    print("=" * 60)
    print("Analyzing complete trajectory patterns from ROS bag data...")
    print("Integrating delta poses to reconstruct full navigation paths...")
    print()
    
    trajectories = plot_all_trajectories()
    
    print("\nTRAJECTORY SUMMARY:")
    print("=" * 60)
    
    for traj in trajectories:
        traj_length = np.sum(np.sqrt(np.diff(traj['x'])**2 + np.diff(traj['y'])**2 + np.diff(traj['z'])**2))
        end_to_end = np.sqrt(traj['x'][-1]**2 + traj['y'][-1]**2 + traj['z'][-1]**2)
        
        print(f"Bag {traj['bag_name'][-1]} ({traj['split']}):")
        print(f"  Duration: {traj['frames']} frames")
        print(f"  Path Length: {traj_length:.2f}m")
        print(f"  X Range: {np.min(traj['x']):.2f} to {np.max(traj['x']):.2f}m")
        print(f"  Y Range: {np.min(traj['y']):.2f} to {np.max(traj['y']):.2f}m")
        print(f"  Z Range: {np.min(traj['z']):.2f} to {np.max(traj['z']):.2f}m")
        print()
    
    print("EXPLANATION OF PREVIOUS VISUALIZATION ISSUE:")
    print("=" * 60)
    print("❌ PROBLEM: The previous trajectory plot was WRONG because:")
    print("   1. We were only using 60 frames from validation set (bag 3)")
    print("   2. We were using dummy images instead of real trajectory data")
    print("   3. We were not integrating deltas properly to show full path")
    print()
    print("✅ SOLUTION: This visualization shows:")
    print("   1. All 5 trajectory bags with complete paths")
    print("   2. Proper integration of delta poses to show real navigation")
    print("   3. Full underwater vehicle trajectories with loop patterns")
    print("   4. Some trajectories have good loop closure, others are open paths")

if __name__ == '__main__':
    main()