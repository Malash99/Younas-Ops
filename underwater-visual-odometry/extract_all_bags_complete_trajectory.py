#!/usr/bin/env python3
"""
Extract all 5 ROS bags sequentially to reconstruct the complete figure-8 trajectory
and test the corrected model on the full pattern.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from data_processing.extract_rosbag_data import RosbagExtractor


def extract_all_bags_sequential():
    """Extract all 5 bags to get the complete trajectory."""
    print("="*80)
    print("EXTRACTING ALL 5 BAGS FOR COMPLETE FIGURE-8 TRAJECTORY")
    print("="*80)
    
    # Bag file paths
    bag_files = [
        './data/raw/ariel_2023-12-21-14-24-42_0.bag',
        './data/raw/ariel_2023-12-21-14-25-37_1.bag', 
        './data/raw/ariel_2023-12-21-14-26-32_2.bag',
        './data/raw/ariel_2023-12-21-14-27-27_3.bag',
        './data/raw/ariel_2023-12-21-14-28-22_4.bag'
    ]
    
    # Create output directory for complete trajectory
    output_dir = './data/complete_trajectory'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    all_poses = []
    all_images = []
    total_images = 0
    
    print("Extracting data from all 5 bags...")
    
    for i, bag_path in enumerate(bag_files):
        if not os.path.exists(bag_path):
            print(f"Warning: Bag {i} not found at {bag_path}")
            continue
            
        print(f"\\nProcessing Bag {i}: {os.path.basename(bag_path)}")
        
        # Create temporary directory for this bag
        temp_dir = f'./data/temp_bag_{i}'
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'images'), exist_ok=True)
        
        # Extract this bag
        extractor = RosbagExtractor(bag_path, temp_dir)
        
        # Use auto-detection for topics
        extractor.extract_all(None, None, None)
        
        # Load extracted data
        if os.path.exists(os.path.join(temp_dir, 'extracted_poses.csv')):
            bag_poses = pd.read_csv(os.path.join(temp_dir, 'extracted_poses.csv'))
            print(f"  Extracted {len(bag_poses)} poses")
            
            # Adjust frame numbers to be sequential
            if os.path.exists(os.path.join(temp_dir, 'image_timestamps.csv')):
                bag_images = pd.read_csv(os.path.join(temp_dir, 'image_timestamps.csv'))
                print(f"  Extracted {len(bag_images)} images")
                
                # Copy images with sequential naming
                for idx, row in bag_images.iterrows():
                    old_path = os.path.join(temp_dir, 'images', row['filename'])
                    if os.path.exists(old_path):
                        new_filename = f"frame_{total_images + idx:06d}.png"
                        new_path = os.path.join(output_dir, 'images', new_filename)
                        
                        # Copy image
                        import shutil
                        shutil.copy2(old_path, new_path)
                        
                        # Update metadata
                        bag_images.loc[idx, 'filename'] = new_filename
                        bag_images.loc[idx, 'frame_id'] = total_images + idx
                
                # Update total count
                total_images += len(bag_images)
                all_images.append(bag_images)
            
            # Store poses with bag identifier
            bag_poses['bag_id'] = i
            all_poses.append(bag_poses)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"  Bag {i} completed. Total images so far: {total_images}")
    
    # Combine all data
    if all_poses:
        print(f"\\nCombining data from {len(all_poses)} bags...")
        
        # Concatenate poses
        complete_poses = pd.concat(all_poses, ignore_index=True)
        complete_poses.to_csv(os.path.join(output_dir, 'ground_truth.csv'), index=False)
        
        # Concatenate image metadata
        if all_images:
            complete_images = pd.concat(all_images, ignore_index=True)
            complete_images.to_csv(os.path.join(output_dir, 'image_timestamps.csv'), index=False)
        
        print(f"Complete trajectory saved with {len(complete_poses)} poses and {total_images} images")
        
        # Create visualization of complete trajectory
        visualize_complete_trajectory(complete_poses, output_dir)
        
        return complete_poses
    else:
        print("Error: No data extracted from any bags")
        return None


def visualize_complete_trajectory(poses_df, output_dir):
    """Create visualization of the complete figure-8 trajectory."""
    print("\\nCreating complete trajectory visualization...")
    
    # Load TUM reference for comparison
    tum_df = pd.read_csv('./data/raw/qualisys_ariel_odom_traj_8_id6.tum', sep=' ', 
                        names=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    
    # Create comprehensive comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Complete extracted trajectory
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    axes[0, 0].plot(poses_df['x'], poses_df['y'], 'k-', linewidth=2, alpha=0.8, label='Complete Trajectory')
    
    # Color code by bag
    for bag_id in range(5):
        bag_data = poses_df[poses_df['bag_id'] == bag_id]
        if len(bag_data) > 0:
            axes[0, 0].plot(bag_data['x'], bag_data['y'], color=colors[bag_id], 
                           linewidth=3, alpha=0.7, label=f'Bag {bag_id}')
    
    axes[0, 0].scatter(poses_df['x'].iloc[0], poses_df['y'].iloc[0], c='green', s=100, marker='o', label='Start')
    axes[0, 0].scatter(poses_df['x'].iloc[-1], poses_df['y'].iloc[-1], c='red', s=100, marker='s', label='End')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title(f'Complete Extracted Trajectory\\n({len(poses_df)} poses from 5 bags)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Plot 2: TUM reference trajectory
    axes[0, 1].plot(tum_df['x'], tum_df['y'], 'purple', linewidth=1, alpha=0.7)
    axes[0, 1].scatter(tum_df['x'].iloc[0], tum_df['y'].iloc[0], c='green', s=100, marker='o', label='Start')
    axes[0, 1].scatter(tum_df['x'].iloc[-1], tum_df['y'].iloc[-1], c='red', s=100, marker='s', label='End')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].set_title(f'TUM Reference Trajectory\\n({len(tum_df)} poses)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Plot 3: Overlay comparison
    axes[0, 2].plot(tum_df['x'], tum_df['y'], 'purple', linewidth=1, alpha=0.5, label='TUM Reference')
    axes[0, 2].plot(poses_df['x'], poses_df['y'], 'black', linewidth=2, alpha=0.8, label='Extracted Complete')
    axes[0, 2].set_xlabel('X (m)')
    axes[0, 2].set_ylabel('Y (m)')
    axes[0, 2].set_title('Trajectory Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axis('equal')
    
    # Plot 4: X position over time
    axes[1, 0].plot(range(len(poses_df)), poses_df['x'], 'b-', linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('X Position (m)')
    axes[1, 0].set_title('X Position vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Y position over time  
    axes[1, 1].plot(range(len(poses_df)), poses_df['y'], 'r-', linewidth=2, alpha=0.8)
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Y Position (m)')
    axes[1, 1].set_title('Y Position vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Distance from origin (to show loops)
    distance = np.sqrt(poses_df['x']**2 + poses_df['y']**2)
    axes[1, 2].plot(range(len(distance)), distance, 'g-', linewidth=2, alpha=0.8)
    axes[1, 2].set_xlabel('Frame')
    axes[1, 2].set_ylabel('Distance from Origin (m)')
    axes[1, 2].set_title('Distance from Origin vs Time\\n(Loops show as repeated patterns)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complete_figure8_trajectory.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze trajectory characteristics
    print(f"\\n=== COMPLETE TRAJECTORY ANALYSIS ===")
    x_range = poses_df['x'].max() - poses_df['x'].min()
    y_range = poses_df['y'].max() - poses_df['y'].min()
    z_range = poses_df['z'].max() - poses_df['z'].min()
    
    print(f"Spatial extents:")
    print(f"  X range: {x_range:.2f}m")
    print(f"  Y range: {y_range:.2f}m") 
    print(f"  Z range: {z_range:.2f}m")
    print(f"  X/Y aspect ratio: {x_range/y_range:.2f}")
    
    # Check for loops/crossings
    def count_crossings(x, y, threshold=0.5):
        crossings = 0
        for i in range(0, len(x)-100, 50):
            for j in range(i+100, len(x), 50):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < threshold:
                    crossings += 1
        return crossings
    
    crossings = count_crossings(poses_df['x'].values, poses_df['y'].values)
    print(f"  Trajectory crossings (within 0.5m): {crossings}")
    
    if crossings > 10:
        print("  -> High number of crossings suggests figure-8 or loop pattern!")
    elif crossings > 0:
        print("  -> Some crossings detected, partial loop pattern")
    else:
        print("  -> No crossings detected, primarily linear trajectory")
    
    print(f"\\nComplete trajectory visualization saved to:")
    print(f"  {os.path.join(output_dir, 'complete_figure8_trajectory.png')}")


if __name__ == "__main__":
    extract_all_bags_sequential()