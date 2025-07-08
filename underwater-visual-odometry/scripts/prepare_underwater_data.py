#!/usr/bin/env python3
"""
Complete data preparation pipeline for underwater visual odometry.
Extracts data from ROS bag and aligns with TUM trajectory.
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def run_command(cmd, description):
    """Run a command and check for errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    return True


def verify_data_quality(data_dir):
    """Verify the quality of prepared data."""
    print(f"\n{'='*60}")
    print("Verifying data quality...")
    print(f"{'='*60}")
    
    # Check if required files exist
    required_files = [
        'ground_truth.csv',
        'image_timestamps.csv',
        'images'
    ]
    
    missing_files = []
    for file in required_files:
        path = os.path.join(data_dir, file)
        if not os.path.exists(path):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return False
    
    # Load and check data
    gt_df = pd.read_csv(os.path.join(data_dir, 'ground_truth.csv'))
    img_df = pd.read_csv(os.path.join(data_dir, 'image_timestamps.csv'))
    
    print(f"\nData summary:")
    print(f"  Ground truth poses: {len(gt_df)}")
    print(f"  Image timestamps: {len(img_df)}")
    
    # Check if numbers match
    if len(gt_df) != len(img_df):
        print(f"Warning: Number of poses ({len(gt_df)}) doesn't match number of images ({len(img_df)})")
    
    # Check image files
    images_dir = os.path.join(data_dir, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"  Image files: {len(image_files)}")
    
    # Check time synchronization
    if len(gt_df) > 0 and len(img_df) > 0:
        time_diff = abs(gt_df['timestamp'].iloc[0] - img_df['timestamp'].iloc[0])
        if time_diff > 0.1:  # More than 100ms difference
            print(f"Warning: Large time difference between first pose and image: {time_diff:.3f}s")
    
    # Compute frame rate
    if len(img_df) > 1:
        time_diffs = np.diff(img_df['timestamp'].values)
        avg_fps = 1.0 / np.mean(time_diffs)
        print(f"  Average frame rate: {avg_fps:.1f} FPS")
    
    return True


def create_data_statistics(data_dir):
    """Create statistics and visualizations of the prepared data."""
    print(f"\n{'='*60}")
    print("Creating data statistics...")
    print(f"{'='*60}")
    
    gt_df = pd.read_csv(os.path.join(data_dir, 'ground_truth.csv'))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. XY trajectory
    ax = axes[0, 0]
    ax.plot(gt_df['x'], gt_df['y'], 'b-', linewidth=2)
    ax.scatter(gt_df['x'].iloc[0], gt_df['y'].iloc[0], c='green', s=100, label='Start', zorder=5)
    ax.scatter(gt_df['x'].iloc[-1], gt_df['y'].iloc[-1], c='red', s=100, label='End', zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Trajectory')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Altitude over time
    ax = axes[0, 1]
    time_rel = gt_df['timestamp'] - gt_df['timestamp'].iloc[0]
    ax.plot(time_rel, gt_df['z'], 'b-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Altitude over Time')
    ax.grid(True, alpha=0.3)
    
    # 3. Velocity
    ax = axes[1, 0]
    if len(gt_df) > 1:
        positions = gt_df[['x', 'y', 'z']].values
        time_diffs = np.diff(gt_df['timestamp'].values)
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        velocities = distances / time_diffs
        ax.plot(time_rel[1:], velocities, 'r-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity over Time')
        ax.grid(True, alpha=0.3)
    
    # 4. Inter-frame motion statistics
    ax = axes[1, 1]
    if len(gt_df) > 1:
        # Compute relative poses
        rel_translations = []
        rel_rotations = []
        
        for i in range(len(gt_df) - 1):
            # Translation
            dt = positions[i+1] - positions[i]
            rel_translations.append(np.linalg.norm(dt))
            
            # Rotation (simplified - just using quaternion angle)
            q1 = gt_df.iloc[i][['qw', 'qx', 'qy', 'qz']].values
            q2 = gt_df.iloc[i+1][['qw', 'qx', 'qy', 'qz']].values
            # Quaternion difference angle
            dot_product = np.dot(q1, q2)
            angle = 2 * np.arccos(np.clip(abs(dot_product), -1, 1))
            rel_rotations.append(angle)
        
        ax.hist(rel_translations, bins=50, alpha=0.7, label='Translation (m)', density=True)
        ax2 = ax.twinx()
        ax2.hist(np.degrees(rel_rotations), bins=50, alpha=0.7, color='orange', label='Rotation (deg)', density=True)
        ax.set_xlabel('Inter-frame Motion')
        ax.set_ylabel('Translation Density', color='blue')
        ax2.set_ylabel('Rotation Density', color='orange')
        ax.set_title('Inter-frame Motion Distribution')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    plt.suptitle('Dataset Statistics', fontsize=16)
    plt.tight_layout()
    
    stats_path = os.path.join(data_dir, 'dataset_statistics.png')
    plt.savefig(stats_path, dpi=150)
    print(f"Saved statistics visualization to: {stats_path}")
    plt.close()


def prepare_data(args):
    """Main data preparation pipeline."""
    print(f"\n{'='*60}")
    print("UNDERWATER VISUAL ODOMETRY DATA PREPARATION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Step 1: Extract data from ROS bag
    if not args.skip_extraction:
        extract_cmd = f"python3 /app/scripts/extract_rosbag_data.py --bag {args.bag} --output {args.output}"
        if args.image_topic:
            extract_cmd += f" --image_topic {args.image_topic}"
        if args.pose_topic:
            extract_cmd += f" --pose_topic {args.pose_topic}"
        if args.max_images:
            extract_cmd += f" --max_images {args.max_images}"
        
        success = run_command(extract_cmd, "Step 1: Extracting data from ROS bag")
        if not success and not args.force:
            print("Extraction failed. Use --force to continue anyway.")
            return False
    else:
        print("\nSkipping ROS bag extraction (--skip_extraction)")
    
    # Step 2: Process TUM trajectory
    if not args.skip_tum:
        tum_cmd = f"python3 /app/scripts/process_tum_trajectory.py --tum {args.tum} --output {args.output}"
        if os.path.exists(os.path.join(args.output, 'image_timestamps.csv')):
            tum_cmd += f" --image_timestamps {os.path.join(args.output, 'image_timestamps.csv')}"
        
        success = run_command(tum_cmd, "Step 2: Processing TUM trajectory")
        if not success and not args.force:
            print("TUM processing failed. Use --force to continue anyway.")
            return False
    else:
        print("\nSkipping TUM trajectory processing (--skip_tum)")
    
    # Step 3: Verify data quality
    if verify_data_quality(args.output):
        print("\n✓ Data quality check passed!")
    else:
        if not args.force:
            print("\n✗ Data quality check failed. Use --force to continue anyway.")
            return False
        else:
            print("\n⚠ Data quality check failed, but continuing due to --force flag")
    
    # Step 4: Create statistics
    try:
        create_data_statistics(args.output)
    except Exception as e:
        print(f"Warning: Failed to create statistics: {e}")
    
    print(f"\n{'='*60}")
    print("DATA PREPARATION COMPLETE!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"\nData saved to: {args.output}")
    print("\nNext steps:")
    print("1. Review the dataset_statistics.png to understand your data")
    print("2. Check the trajectory visualizations")
    print("3. Run the training script:")
    print("   python3 /app/scripts/train_baseline.py")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare underwater visual odometry data from ROS bag and TUM trajectory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Full pipeline with auto-detection
  python3 prepare_underwater_data.py
  
  # Specify specific topics
  python3 prepare_underwater_data.py --image_topic /camera/image_raw --pose_topic /odometry
  
  # Process only first 1000 images
  python3 prepare_underwater_data.py --max_images 1000
  
  # Skip extraction if already done
  python3 prepare_underwater_data.py --skip_extraction
        """
    )
    
    # Input files
    parser.add_argument('--bag', type=str, 
                        default='/app/data/raw/ariel_2023-12-21-14-26-32_2.bag',
                        help='Path to ROS bag file')
    parser.add_argument('--tum', type=str,
                        default='/app/data/raw/qualisys_ariel_odom_traj_8_id6.tum',
                        help='Path to TUM trajectory file')
    
    # Output
    parser.add_argument('--output', type=str, default='/app/data/raw',
                        help='Output directory')
    
    # ROS bag options
    parser.add_argument('--image_topic', type=str, default=None,
                        help='Image topic name (auto-detect if not specified)')
    parser.add_argument('--pose_topic', type=str, default=None,
                        help='Pose topic name (auto-detect if not specified)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to extract')
    
    # Processing options
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip ROS bag extraction')
    parser.add_argument('--skip_tum', action='store_true',
                        help='Skip TUM trajectory processing')
    parser.add_argument('--force', action='store_true',
                        help='Continue even if some steps fail')
    
    args = parser.parse_args()
    
    # Prepare data
    success = prepare_data(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()