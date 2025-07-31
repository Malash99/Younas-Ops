#!/usr/bin/env python3
"""
Prepare training data from sequential frames for UW-TransVO

Converts the existing frame_sequence.csv to the format expected by our training pipeline
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R


def load_ground_truth_trajectory(tum_file: str):
    """Load ground truth trajectory from TUM format file"""
    if not os.path.exists(tum_file):
        print(f"Warning: Ground truth file not found: {tum_file}")
        return None
    
    # TUM format: timestamp tx ty tz qx qy qz qw
    gt_data = pd.read_csv(tum_file, sep=' ', header=None, 
                         names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    return gt_data


def compute_relative_pose(pose1, pose2):
    """Compute relative pose between two poses"""
    # Position difference (simple)
    delta_pos = pose2[:3] - pose1[:3]
    
    # Rotation difference using quaternions
    # Convert to rotation matrices
    q1 = pose1[3:]  # qx, qy, qz, qw format
    q2 = pose2[3:]
    
    # Scipy expects [x,y,z,w] format
    R1 = R.from_quat(q1).as_matrix()
    R2 = R.from_quat(q2).as_matrix()
    
    # Relative rotation: R_rel = R1^T * R2
    R_rel = R1.T @ R2
    
    # Convert to Euler angles (roll, pitch, yaw)
    euler_rel = R.from_matrix(R_rel).as_euler('xyz', degrees=False)
    
    return np.concatenate([delta_pos, euler_rel])


def create_training_data():
    """Create training data CSV from sequential frames"""
    
    # Paths
    base_dir = Path('data/sequential_frames')
    frame_csv = base_dir / 'data' / 'frame_sequence.csv'
    gt_file = Path('data/raw/qualisys_ariel_odom_traj_8_id6.tum')
    
    output_dir = Path('data/processed/training_dataset')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load frame data
    print("Loading frame sequence data...")
    df = pd.read_csv(frame_csv)
    
    # Load ground truth if available
    gt_data = load_ground_truth_trajectory(str(gt_file))
    
    training_samples = []
    sample_id = 0
    
    # Group by bag and create sequential samples
    for bag_name, bag_df in df.groupby('bag_name'):
        bag_df = bag_df.sort_values('timestamp').reset_index(drop=True)
        print(f"Processing bag: {bag_name} ({len(bag_df)} frames)")
        
        # Create pairs of consecutive frames
        for i in range(len(bag_df) - 1):
            curr_row = bag_df.iloc[i + 1]  # Current frame
            prev_row = bag_df.iloc[i]      # Previous frame
            
            # Calculate time difference
            dt = curr_row['timestamp'] - prev_row['timestamp']
            
            # Try to get ground truth poses
            delta_x, delta_y, delta_z = 0.0, 0.0, 0.0
            delta_roll, delta_pitch, delta_yaw = 0.0, 0.0, 0.0
            
            if gt_data is not None:
                # Find closest ground truth poses
                curr_gt_idx = np.argmin(np.abs(gt_data['timestamp'] - curr_row['timestamp']))
                prev_gt_idx = np.argmin(np.abs(gt_data['timestamp'] - prev_row['timestamp']))
                
                if (np.abs(gt_data.iloc[curr_gt_idx]['timestamp'] - curr_row['timestamp']) < 0.1 and
                    np.abs(gt_data.iloc[prev_gt_idx]['timestamp'] - prev_row['timestamp']) < 0.1):
                    
                    # Get poses
                    curr_pose = gt_data.iloc[curr_gt_idx][['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']].values
                    prev_pose = gt_data.iloc[prev_gt_idx][['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']].values
                    
                    # Compute relative pose
                    relative_pose = compute_relative_pose(prev_pose, curr_pose)
                    delta_x, delta_y, delta_z = relative_pose[:3]
                    delta_roll, delta_pitch, delta_yaw = relative_pose[3:]
            
            # Create training sample
            sample = {
                'sample_id': sample_id,
                'bag_name': bag_name,
                'timestamp': curr_row['timestamp'],
                'dt': dt,
                'delta_x': delta_x,
                'delta_y': delta_y,
                'delta_z': delta_z,
                'delta_roll': delta_roll,
                'delta_pitch': delta_pitch,
                'delta_yaw': delta_yaw,
                'pose_x': 0.0,  # Placeholder
                'pose_y': 0.0,  # Placeholder
                'pose_z': 0.0,  # Placeholder
            }
            
            # Add camera paths (update paths to be relative to training_dataset)
            for cam_id in range(5):
                cam_col = f'cam{cam_id}_path'
                if cam_col in curr_row and pd.notna(curr_row[cam_col]):
                    # Convert path to be relative to training dataset
                    original_path = curr_row[cam_col]
                    if original_path:
                        sample[cam_col] = f"../sequential_frames/{original_path}"
                else:
                    sample[cam_col] = None
            
            training_samples.append(sample)
            sample_id += 1
    
    # Create DataFrame and save
    training_df = pd.DataFrame(training_samples)
    
    # Save training data
    output_csv = output_dir / 'training_data.csv'
    training_df.to_csv(output_csv, index=False)
    
    # Create metadata
    metadata = {
        'total_samples': len(training_samples),
        'cameras': 5,
        'coordinate_frame': 'relative_motion',
        'output_variables': ['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw'],
        'input_sources': ['cam0', 'cam1', 'cam2', 'cam3', 'cam4'],
        'description': 'Training data for UW-TransVO with relative motion ground truth',
        'has_ground_truth': gt_data is not None,
        'data_source': 'sequential_frames'
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Training data created!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total samples: {len(training_samples)}")
    print(f"📄 Training CSV: {output_csv}")
    print(f"📄 Metadata: {metadata_file}")
    
    if gt_data is not None:
        print(f"🎯 Ground truth poses: Available")
    else:
        print(f"⚠️  Ground truth poses: Not available (using zeros)")
    
    # Print sample statistics
    stats = training_df[['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw']].describe()
    print(f"\n📈 Motion statistics:")
    print(stats)
    
    return output_csv, metadata_file


if __name__ == '__main__':
    create_training_data()