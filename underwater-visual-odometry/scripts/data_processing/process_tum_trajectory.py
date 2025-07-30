#!/usr/bin/env python3
"""
Process TUM format trajectory file and align with extracted images.
TUM format: timestamp tx ty tz qx qy qz qw
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import matplotlib.pyplot as plt


class TUMProcessor:
    """Process TUM format trajectory files."""
    
    def __init__(self, tum_file: str, output_dir: str):
        """
        Initialize processor.
        
        Args:
            tum_file: Path to TUM format file
            output_dir: Directory to save processed data
        """
        self.tum_file = tum_file
        self.output_dir = output_dir
        
    def load_tum_trajectory(self):
        """Load TUM format trajectory."""
        print(f"Loading TUM trajectory from: {self.tum_file}")
        
        # TUM format: timestamp tx ty tz qx qy qz qw
        data = []
        with open(self.tum_file, 'r') as f:
            for line in f:
                # Skip comments
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split()
                if len(parts) == 8:
                    data.append([float(x) for x in parts])
                else:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        
        print(f"Loaded {len(df)} poses")
        print(f"Time range: {df['timestamp'].min():.3f} to {df['timestamp'].max():.3f} seconds")
        print(f"Duration: {df['timestamp'].max() - df['timestamp'].min():.3f} seconds")
        
        # Check for quaternion normalization
        q_norms = np.sqrt(df['qx']**2 + df['qy']**2 + df['qz']**2 + df['qw']**2)
        if not np.allclose(q_norms, 1.0, atol=0.01):
            print("Warning: Quaternions are not normalized. Normalizing...")
            for i in range(len(df)):
                norm = q_norms[i]
                df.loc[i, ['qx', 'qy', 'qz', 'qw']] /= norm
        
        return df
    
    def align_with_images(self, tum_df: pd.DataFrame, image_timestamps_file: str):
        """Align TUM trajectory with image timestamps."""
        print(f"\nAligning trajectory with images...")
        
        # Load image timestamps
        if not os.path.exists(image_timestamps_file):
            print(f"Error: Image timestamps file not found: {image_timestamps_file}")
            return None
        
        image_df = pd.read_csv(image_timestamps_file)
        print(f"Loaded {len(image_df)} image timestamps")
        
        # Check time alignment
        tum_start, tum_end = tum_df['timestamp'].min(), tum_df['timestamp'].max()
        img_start, img_end = image_df['timestamp'].min(), image_df['timestamp'].max()
        
        print(f"\nTUM time range: {tum_start:.3f} to {tum_end:.3f}")
        print(f"Image time range: {img_start:.3f} to {img_end:.3f}")
        
        # Check if timestamps are in the same reference frame
        time_offset = 0
        if abs(tum_start - img_start) > 100:  # More than 100 seconds difference
            print("\nWarning: Large time offset detected between TUM and images!")
            print("This might indicate different time references (e.g., ROS time vs wall time)")
            
            # Try to find offset by matching motion patterns
            # For now, we'll just align the start times
            time_offset = img_start - tum_start
            print(f"Applying time offset: {time_offset:.3f} seconds")
            tum_df['timestamp'] = tum_df['timestamp'] + time_offset
        
        # Interpolate poses for each image timestamp
        aligned_poses = []
        
        # Create interpolation functions for each pose component
        interp_funcs = {}
        for col in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']:
            interp_funcs[col] = interpolate.interp1d(
                tum_df['timestamp'].values,
                tum_df[col].values,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
        
        # Interpolate pose for each image
        for idx, row in image_df.iterrows():
            img_timestamp = row['timestamp']
            
            # Check if timestamp is within bounds
            if img_timestamp < tum_df['timestamp'].min() or img_timestamp > tum_df['timestamp'].max():
                print(f"Warning: Image timestamp {img_timestamp:.3f} is outside TUM trajectory range")
            
            # Interpolate pose
            pose = {
                'timestamp': img_timestamp,
                'filename': row['filename'],
                'frame_id': row['frame_id']
            }
            
            for col in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']:
                pose[col] = float(interp_funcs[col](img_timestamp))
            
            # Normalize quaternion after interpolation
            q_norm = np.sqrt(pose['qx']**2 + pose['qy']**2 + pose['qz']**2 + pose['qw']**2)
            pose['qx'] /= q_norm
            pose['qy'] /= q_norm
            pose['qz'] /= q_norm
            pose['qw'] /= q_norm
            
            aligned_poses.append(pose)
        
        aligned_df = pd.DataFrame(aligned_poses)
        print(f"\nAligned {len(aligned_df)} poses with images")
        
        return aligned_df
    
    def visualize_trajectory(self, df: pd.DataFrame, title: str = "Trajectory"):
        """Visualize the trajectory."""
        fig = plt.figure(figsize=(15, 5))
        
        # 3D trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(df['x'], df['y'], df['z'], 'b-', linewidth=2)
        ax1.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], 
                   c='green', s=100, label='Start')
        ax1.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], 
                   c='red', s=100, label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'{title} - 3D View')
        ax1.legend()
        
        # XY view
        ax2 = fig.add_subplot(132)
        ax2.plot(df['x'], df['y'], 'b-', linewidth=2)
        ax2.scatter(df['x'].iloc[0], df['y'].iloc[0], c='green', s=100, label='Start')
        ax2.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='red', s=100, label='End')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'{title} - XY View')
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Time vs position
        ax3 = fig.add_subplot(133)
        time_rel = df['timestamp'] - df['timestamp'].iloc[0]
        ax3.plot(time_rel, df['x'], 'r-', label='X', linewidth=2)
        ax3.plot(time_rel, df['y'], 'g-', label='Y', linewidth=2)
        ax3.plot(time_rel, df['z'], 'b-', label='Z', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Position (m)')
        ax3.set_title('Position over Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'{title.lower().replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to: {save_path}")
        plt.close()
    
    def compute_statistics(self, df: pd.DataFrame):
        """Compute trajectory statistics."""
        print("\nTrajectory Statistics:")
        
        # Position stats
        print(f"\nPosition ranges:")
        print(f"  X: [{df['x'].min():.3f}, {df['x'].max():.3f}] m")
        print(f"  Y: [{df['y'].min():.3f}, {df['y'].max():.3f}] m")
        print(f"  Z: [{df['z'].min():.3f}, {df['z'].max():.3f}] m")
        
        # Total distance traveled
        positions = df[['x', 'y', 'z']].values
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        total_distance = np.sum(distances)
        print(f"\nTotal distance traveled: {total_distance:.3f} m")
        
        # Velocity statistics
        if len(df) > 1:
            time_diffs = np.diff(df['timestamp'].values)
            velocities = distances / time_diffs
            print(f"\nVelocity statistics:")
            print(f"  Mean: {np.mean(velocities):.3f} m/s")
            print(f"  Max: {np.max(velocities):.3f} m/s")
            print(f"  Std: {np.std(velocities):.3f} m/s")
        
        # Rotation statistics
        quaternions = df[['qw', 'qx', 'qy', 'qz']].values
        rotations = R.from_quat(quaternions[:, [1, 2, 3, 0]])  # scipy uses xyzw format
        euler_angles = rotations.as_euler('xyz', degrees=True)
        
        print(f"\nRotation ranges (degrees):")
        print(f"  Roll:  [{euler_angles[:, 0].min():.1f}, {euler_angles[:, 0].max():.1f}]")
        print(f"  Pitch: [{euler_angles[:, 1].min():.1f}, {euler_angles[:, 1].max():.1f}]")
        print(f"  Yaw:   [{euler_angles[:, 2].min():.1f}, {euler_angles[:, 2].max():.1f}]")
    
    def process(self, image_timestamps_file: str = None):
        """Process the TUM trajectory file."""
        # Load TUM trajectory
        tum_df = self.load_tum_trajectory()
        
        # Visualize original trajectory
        self.visualize_trajectory(tum_df, "Original TUM Trajectory")
        
        # Compute statistics
        self.compute_statistics(tum_df)
        
        # Save original TUM data in our format
        tum_df.to_csv(os.path.join(self.output_dir, 'tum_trajectory.csv'), index=False)
        
        # If image timestamps available, create aligned ground truth
        if image_timestamps_file and os.path.exists(image_timestamps_file):
            aligned_df = self.align_with_images(tum_df, image_timestamps_file)
            
            if aligned_df is not None:
                # Save as ground truth
                gt_df = aligned_df[['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']]
                gt_df.to_csv(os.path.join(self.output_dir, 'ground_truth.csv'), index=False)
                print(f"\nSaved aligned ground truth to: ground_truth.csv")
                
                # Visualize aligned trajectory
                self.visualize_trajectory(aligned_df, "Aligned Ground Truth")
        else:
            print("\nNo image timestamps provided. Saving TUM trajectory as ground truth.")
            tum_df.to_csv(os.path.join(self.output_dir, 'ground_truth.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description='Process TUM trajectory file')
    parser.add_argument('--tum', type=str, 
                        default='/app/data/raw/qualisys_ariel_odom_traj_8_id6.tum',
                        help='Path to TUM trajectory file')
    parser.add_argument('--output', type=str, default='/app/data/raw',
                        help='Output directory')
    parser.add_argument('--image_timestamps', type=str, 
                        default='/app/data/raw/image_timestamps.csv',
                        help='Path to image timestamps CSV')
    
    args = parser.parse_args()
    
    # Process TUM file
    processor = TUMProcessor(args.tum, args.output)
    processor.process(args.image_timestamps)


if __name__ == "__main__":
    main()