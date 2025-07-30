"""
CORRECTED data loader for underwater visual odometry dataset.
Ensures consistent reference frames between ground truth and model predictions.
"""

import os
import numpy as np
import pandas as pd
import cv2
from typing import Tuple, List, Dict, Optional
import tensorflow as tf
from pyquaternion import Quaternion

from utils.coordinate_transforms_fixed import compute_relative_pose_correct


class UnderwaterVODatasetFixed:
    """Fixed dataset class with consistent reference frame handling."""
    
    def __init__(self, data_dir: str, sequence_length: int = 2, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to dataset directory
            sequence_length: Number of consecutive frames to use  
            image_size: Target size for images (height, width)
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.image_size = image_size
        
        # Load ground truth poses
        self.gt_file = os.path.join(data_dir, 'ground_truth.csv')
        self.image_dir = os.path.join(data_dir, 'images')
        
        # Will be populated by load_data()
        self.poses_df = None
        self.image_files = []
        self.timestamps = []
        
    def load_data(self):
        """Load ground truth poses and image file list."""
        # Load ground truth CSV
        # Expected columns: timestamp, x, y, z, qx, qy, qz, qw
        if os.path.exists(self.gt_file):
            self.poses_df = pd.read_csv(self.gt_file)
            print(f"Loaded {len(self.poses_df)} ground truth poses")
            
            # Validate quaternion columns
            quat_cols = ['qx', 'qy', 'qz', 'qw']
            if not all(col in self.poses_df.columns for col in quat_cols):
                print("Warning: Missing quaternion columns, will use identity quaternions")
                for col in quat_cols:
                    if col not in self.poses_df.columns:
                        self.poses_df[col] = 1.0 if col == 'qw' else 0.0
                        
        else:
            print(f"Warning: Ground truth file not found at {self.gt_file}")
            # Create dummy data for testing
            self.poses_df = self._create_dummy_data()
        
        # Get list of image files
        if os.path.exists(self.image_dir):
            self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                     if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"Found {len(self.image_files)} images")
        else:
            print(f"Warning: Image directory not found at {self.image_dir}")
            self.image_files = []
            
    def _create_dummy_data(self) -> pd.DataFrame:
        """Create dummy data for testing when real data isn't available."""
        n_frames = 1000
        timestamps = np.arange(n_frames) * 0.1  # 10 Hz
        
        # Simulate simple forward motion with slight variations
        x = np.cumsum(np.random.normal(0.1, 0.02, n_frames))
        y = np.cumsum(np.random.normal(0, 0.01, n_frames))
        z = np.cumsum(np.random.normal(0, 0.01, n_frames))
        
        # Small rotations
        angles = np.cumsum(np.random.normal(0, 0.01, n_frames))
        quaternions = [Quaternion(axis=[0, 0, 1], angle=a) for a in angles]
        
        data = {
            'timestamp': timestamps,
            'x': x, 'y': y, 'z': z,
            'qx': [q.x for q in quaternions],
            'qy': [q.y for q in quaternions], 
            'qz': [q.z for q in quaternions],
            'qw': [q.w for q in quaternions]
        }
        
        return pd.DataFrame(data)
    
    def get_pose_dict(self, idx: int) -> Dict:
        """Get pose at index as dictionary."""
        pose = self.poses_df.iloc[idx]
        return {
            'position': np.array([pose['x'], pose['y'], pose['z']], dtype=np.float32),
            'quaternion': np.array([pose['qw'], pose['qx'], pose['qy'], pose['qz']], dtype=np.float32)
        }
    
    def get_relative_pose(self, idx1: int, idx2: int) -> np.ndarray:
        """
        Calculate relative pose between two frames using CORRECTED method.
        
        Args:
            idx1: Index of first frame
            idx2: Index of second frame
            
        Returns:
            6D relative pose [dx, dy, dz, d_rx, d_ry, d_rz] in frame1's local coordinate system
        """
        pose1 = self.get_pose_dict(idx1)
        pose2 = self.get_pose_dict(idx2)
        
        return compute_relative_pose_correct(pose1, pose2)
    
    def load_image(self, idx: int) -> np.ndarray:
        """Load and preprocess image."""
        if idx < len(self.image_files):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Create dummy image for testing
                img = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        else:
            # Create dummy image for testing
            img = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
            
        # Resize to target size
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def create_tf_dataset(self, indices: List[int], batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset for training with CORRECTED ground truth."""
        def generator():
            for i in indices[:-1]:  # Skip last index since we need pairs
                try:
                    # Load consecutive frames
                    img1 = self.load_image(i)
                    img2 = self.load_image(i + 1)
                    
                    # Stack images along channel dimension
                    img_pair = np.concatenate([img1, img2], axis=-1)
                    
                    # Get CORRECTED relative pose
                    target = self.get_relative_pose(i, i + 1)
                    
                    # Ensure target is float32
                    target = target.astype(np.float32)
                    
                    yield img_pair, target
                    
                except Exception as e:
                    print(f"Error loading data at index {i}: {e}")
                    continue
        
        # Define output signature
        output_signature = (
            tf.TensorSpec(shape=(*self.image_size, 6), dtype=tf.float32),
            tf.TensorSpec(shape=(6,), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
            
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
        """Split indices into training and validation sets."""
        n_samples = len(self.poses_df) - 1  # -1 because we need pairs
        n_val = int(n_samples * val_ratio)
        
        # Use fixed seed for reproducibility
        np.random.seed(seed)
        indices = list(range(n_samples))
        np.random.shuffle(indices)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        print(f"Split dataset: {len(train_indices)} training, {len(val_indices)} validation samples")
        
        return train_indices, val_indices
    
    def validate_ground_truth_consistency(self, n_samples: int = 10):
        """
        Validate that ground truth relative poses are computed correctly.
        """
        print("="*60)
        print("VALIDATING GROUND TRUTH CONSISTENCY")
        print("="*60)
        
        for i in range(min(n_samples, len(self.poses_df) - 1)):
            # Get poses
            pose1 = self.get_pose_dict(i)
            pose2 = self.get_pose_dict(i + 1)
            
            # Get relative pose
            rel_pose = self.get_relative_pose(i, i + 1)
            
            print(f"\\nFrame pair {i} -> {i+1}:")
            print(f"  Position 1: {pose1['position']}")
            print(f"  Position 2: {pose2['position']}")
            print(f"  Relative pose: {rel_pose}")
            print(f"  Translation magnitude: {np.linalg.norm(rel_pose[:3]):.4f}")
            print(f"  Rotation magnitude: {np.linalg.norm(rel_pose[3:]):.4f}")
            
            # Validate by integrating back
            from utils.coordinate_transforms_fixed import integrate_trajectory_correct
            trajectory = integrate_trajectory_correct(pose1, [rel_pose])
            reconstructed_pose2 = trajectory[-1]
            
            pos_error = np.linalg.norm(reconstructed_pose2['position'] - pose2['position'])
            print(f"  Reconstruction error: {pos_error:.6f}")
            
            if pos_error > 1e-5:
                print(f"  WARNING: Large reconstruction error!")
        
        print("="*60)


def test_fixed_data_loader():
    """Test the fixed data loader."""
    # Create test data directory
    test_dir = "/tmp/test_underwater_vo"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test dataset
    dataset = UnderwaterVODatasetFixed(test_dir)
    dataset.load_data()  # Will create dummy data
    
    # Validate ground truth
    dataset.validate_ground_truth_consistency(5)
    
    # Test TensorFlow dataset creation
    train_indices, val_indices = dataset.train_val_split()
    tf_dataset = dataset.create_tf_dataset(train_indices[:100], batch_size=4)
    
    print("\\nTesting TensorFlow dataset:")
    for batch_imgs, batch_targets in tf_dataset.take(1):
        print(f"  Batch image shape: {batch_imgs.shape}")
        print(f"  Batch target shape: {batch_targets.shape}")
        print(f"  Sample target: {batch_targets[0].numpy()}")
        
    print("\\nFixed data loader test completed successfully!")


if __name__ == "__main__":
    test_fixed_data_loader()