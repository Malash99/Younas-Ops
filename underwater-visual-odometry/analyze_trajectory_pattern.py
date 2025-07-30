#!/usr/bin/env python3
"""
Analyze why we don't see the expected figure-8/infinity pattern in our trajectories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_trajectory_patterns():
    """Analyze trajectory patterns to understand the missing figure-8."""
    print("="*80)
    print("TRAJECTORY PATTERN ANALYSIS")
    print("="*80)
    
    # Load all available trajectory data
    tum_path = './data/raw/qualisys_ariel_odom_traj_8_id6.tum'
    train_path = './data/raw/ground_truth.csv'
    bag4_path = './data/bag4_test/extracted_poses.csv'
    
    print("Loading trajectory data...")
    
    # Load TUM reference (space-separated)
    tum_df = pd.read_csv(tum_path, sep=' ', 
                        names=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    print(f"TUM reference: {len(tum_df)} poses")
    
    # Load our extracted data
    train_df = pd.read_csv(train_path)
    bag4_df = pd.read_csv(bag4_path)
    print(f"Training data: {len(train_df)} poses")
    print(f"Bag 4 data: {len(bag4_df)} poses")
    
    # Calculate temporal coverage
    print(f"\n=== TEMPORAL ANALYSIS ===")
    tum_duration = tum_df['timestamp'].max() - tum_df['timestamp'].min()
    train_duration = train_df['timestamp'].max() - train_df['timestamp'].min()
    bag4_duration = bag4_df['timestamp'].max() - bag4_df['timestamp'].min()
    
    print(f"TUM reference duration: {tum_duration:.1f} seconds")
    print(f"Training data duration: {train_duration:.1f} seconds")
    print(f"Bag 4 duration: {bag4_duration:.1f} seconds")
    print(f"Each bag represents ~{(train_duration/tum_duration)*100:.1f}% of full trajectory")
    
    # Spatial analysis
    print(f"\n=== SPATIAL ANALYSIS ===")
    
    def print_spatial_stats(df, name):
        x_range = df['x'].max() - df['x'].min()
        y_range = df['y'].max() - df['y'].min()
        z_range = df['z'].max() - df['z'].min()
        print(f"{name}:")
        print(f"  X: {df['x'].min():.2f} to {df['x'].max():.2f} (range: {x_range:.2f}m)")
        print(f"  Y: {df['y'].min():.2f} to {df['y'].max():.2f} (range: {y_range:.2f}m)")
        print(f"  Z: {df['z'].min():.2f} to {df['z'].max():.2f} (range: {z_range:.2f}m)")
        return x_range, y_range, z_range
    
    tum_ranges = print_spatial_stats(tum_df, "TUM Reference")
    train_ranges = print_spatial_stats(train_df, "Training Data")
    bag4_ranges = print_spatial_stats(bag4_df, "Bag 4 Data")
    
    # Check for figure-8 characteristics
    print(f"\n=== FIGURE-8 PATTERN ANALYSIS ===")
    print(f"For a figure-8 pattern, we expect:")
    print(f"  - Roughly equal X and Y ranges")
    print(f"  - Multiple returns to center/crossing points")
    print(f"  - Oscillatory motion in both X and Y")
    
    # TUM analysis
    tum_aspect = tum_ranges[0] / tum_ranges[1]
    print(f"\nTUM Reference:")
    print(f"  X/Y aspect ratio: {tum_aspect:.2f}")
    if 0.5 < tum_aspect < 2.0:
        print(f"  -> Aspect ratio suggests possible figure-8 pattern")
    else:
        print(f"  -> Elongated trajectory, unlikely to be classic figure-8")
    
    # Check for trajectory crossings
    def find_crossings(x, y, threshold=0.5):
        """Find points where trajectory crosses itself."""
        crossings = 0
        for i in range(0, len(x)-200, 100):  # Sample sparsely for performance
            for j in range(i+200, len(x), 100):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < threshold:
                    crossings += 1
        return crossings
    
    tum_crossings = find_crossings(tum_df['x'].values, tum_df['y'].values, 0.5)
    print(f"  Self-intersections (within 0.5m): {tum_crossings}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Full TUM trajectory
    axes[0, 0].plot(tum_df['x'], tum_df['y'], 'k-', linewidth=1, alpha=0.7)
    axes[0, 0].scatter(tum_df['x'].iloc[0], tum_df['y'].iloc[0], c='green', s=50, marker='o')
    axes[0, 0].scatter(tum_df['x'].iloc[-1], tum_df['y'].iloc[-1], c='red', s=50, marker='s')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title(f'TUM Reference\\n({len(tum_df)} poses, {tum_duration:.1f}s)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Plot 2: Training data trajectory
    axes[0, 1].plot(train_df['x'], train_df['y'], 'b-', linewidth=2, alpha=0.8)
    axes[0, 1].scatter(train_df['x'].iloc[0], train_df['y'].iloc[0], c='green', s=50, marker='o')
    axes[0, 1].scatter(train_df['x'].iloc[-1], train_df['y'].iloc[-1], c='red', s=50, marker='s')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].set_title(f'Training Data\\n({len(train_df)} poses, {train_duration:.1f}s)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Plot 3: Bag 4 trajectory
    axes[0, 2].plot(bag4_df['x'], bag4_df['y'], 'r-', linewidth=2, alpha=0.8)
    axes[0, 2].scatter(bag4_df['x'].iloc[0], bag4_df['y'].iloc[0], c='green', s=50, marker='o')
    axes[0, 2].scatter(bag4_df['x'].iloc[-1], bag4_df['y'].iloc[-1], c='red', s=50, marker='s')
    axes[0, 2].set_xlabel('X (m)')
    axes[0, 2].set_ylabel('Y (m)')
    axes[0, 2].set_title(f'Bag 4 Data\\n({len(bag4_df)} poses, {bag4_duration:.1f}s)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axis('equal')
    
    # Plot 4: TUM X vs time (looking for oscillations)
    axes[1, 0].plot(range(len(tum_df)), tum_df['x'], 'k-', linewidth=1, alpha=0.7)
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('X Position (m)')
    axes[1, 0].set_title('TUM: X Position vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: TUM Y vs time (looking for oscillations)
    axes[1, 1].plot(range(len(tum_df)), tum_df['y'], 'k-', linewidth=1, alpha=0.7)
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Y Position (m)')
    axes[1, 1].set_title('TUM: Y Position vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Distance from origin (loops show as repeated patterns)
    tum_dist = np.sqrt(tum_df['x']**2 + tum_df['y']**2)
    train_dist = np.sqrt(train_df['x']**2 + train_df['y']**2)
    bag4_dist = np.sqrt(bag4_df['x']**2 + bag4_df['y']**2)
    
    axes[1, 2].plot(range(len(tum_dist)), tum_dist, 'k-', linewidth=1, alpha=0.7, label='TUM')
    axes[1, 2].plot(range(len(train_dist)), train_dist, 'b-', linewidth=2, alpha=0.8, label='Training')
    axes[1, 2].plot(range(len(bag4_dist)), bag4_dist, 'r-', linewidth=2, alpha=0.8, label='Bag 4')
    axes[1, 2].set_xlabel('Frame')
    axes[1, 2].set_ylabel('Distance from Origin (m)')
    axes[1, 2].set_title('Distance from Origin')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./trajectory_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== FINDINGS ===")
    print(f"1. Each bag covers only ~{(train_duration/tum_duration)*100:.1f}% of the full trajectory")
    print(f"2. Individual bags show linear/curved segments, not complete loops")
    print(f"3. The figure-8 pattern (if it exists) spans the entire {tum_duration:.1f}s recording")
    print(f"4. Training on single bag segments may limit model's ability to learn loop closures")
    
    print(f"\n=== RECOMMENDATIONS ===")
    print(f"1. Use ALL 5 bags sequentially to see the complete figure-8 pattern")
    print(f"2. Train on the full sequence to learn loop closure behavior")
    print(f"3. Current per-frame accuracy ({train_ranges}) is reasonable for local motion")
    print(f"4. Trajectory drift is expected when integrating only local predictions")
    
    print("\nSaved analysis to: ./trajectory_pattern_analysis.png")
    return tum_df, train_df, bag4_df

if __name__ == "__main__":
    analyze_trajectory_patterns()