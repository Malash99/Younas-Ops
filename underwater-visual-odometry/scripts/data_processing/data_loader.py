"""
Data loader for underwater visual odometry dataset.
Handles loading of image sequences and ground truth poses.
"""

import os
import numpy as np
import pandas as pd
import cv2
from typing import Tuple, List, Dict, Optional
import tensorflow as tf
from pyquaternion import Quaternion


class UnderwaterVODataset:
    """Dataset class for underwater visual odometry."""
    
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
        else:
            print(f"Warning: Ground truth file not found at {self.gt_file}")
            # Create dummy data for testing
            self.poses_df = self._create_dummy_data()
        
        # Get list of image files
        if os.path.exists(self.image_dir):
            self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
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
    
    def get_relative_pose(self, idx1: int, idx2: int) -> Dict[str, np.ndarray]:
        """
        Calculate relative pose between two frames.
        
        Returns:
            Dictionary with 'translation' and 'rotation' arrays
        """
        pose1 = self.poses_df.iloc[idx1]
        pose2 = self.poses_df.iloc[idx2]
        
        # Get positions
        p1 = np.array([pose1['x'], pose1['y'], pose1['z']])
        p2 = np.array([pose2['x'], pose2['y'], pose2['z']])
        
        # Get quaternions
        q1 = Quaternion(pose1['qw'], pose1['qx'], pose1['qy'], pose1['qz'])
        q2 = Quaternion(pose2['qw'], pose2['qx'], pose2['qy'], pose2['qz'])
        
        # Calculate relative transformation
        q_rel = q1.inverse * q2
        t_rel = q1.inverse.rotate(p2 - p1)
        
        # Convert to Euler angles for easier learning
        euler = q_rel.yaw_pitch_roll  # Returns (yaw, pitch, roll)
        
        return {
            'translation': t_rel.astype(np.float32),
            'rotation': np.array(euler, dtype=np.float32),
            'quaternion': np.array([q_rel.w, q_rel.x, q_rel.y, q_rel.z], dtype=np.float32)
        }
    
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
        """Create TensorFlow dataset for training."""
        def generator():
            for i in indices[:-1]:  # Skip last index since we need pairs
                # Load consecutive frames
                img1 = self.load_image(i)
                img2 = self.load_image(i + 1)
                
                # Stack images along channel dimension
                img_pair = np.concatenate([img1, img2], axis=-1)
                
                # Get relative pose
                pose = self.get_relative_pose(i, i + 1)
                
                # Combine translation and rotation
                target = np.concatenate([
                    pose['translation'],
                    pose['rotation']
                ])
                
                yield img_pair, target
        
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
    
    def train_val_split(self, val_ratio: float = 0.2) -> Tuple[List[int], List[int]]:
        """Split indices into training and validation sets."""
        n_samples = len(self.poses_df) - 1  # -1 because we need pairs
        n_val = int(n_samples * val_ratio)
        
        indices = list(range(n_samples))
        np.random.shuffle(indices)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        return train_indices, val_indices


# Underwater-specific data augmentation
class UnderwaterAugmentation:
    """Augmentation specifically designed for underwater images."""
    
    @staticmethod
    def add_particles(image: np.ndarray, density: float = 0.001) -> np.ndarray:
        """Add floating particles to simulate underwater environment."""
        h, w = image.shape[:2]
        n_particles = int(h * w * density)
        
        img_aug = image.copy()
        for _ in range(n_particles):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            size = np.random.randint(1, 3)
            brightness = np.random.uniform(0.8, 1.0)
            
            cv2.circle(img_aug, (x, y), size, (brightness, brightness, brightness), -1)
            
        return img_aug
    
    @staticmethod
    def add_turbidity(image: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Add turbidity effect using gaussian blur."""
        ksize = int(strength * 10) * 2 + 1  # Ensure odd number
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    @staticmethod
    def color_attenuation(image: np.ndarray, depth: float = 10.0) -> np.ndarray:
        """Simulate color attenuation based on water depth."""
        # Red attenuates fastest, then green, blue slowest
        attenuation = np.array([0.8 ** (depth/10), 0.9 ** (depth/10), 0.95 ** (depth/10)])
        return image * attenuation
    
    @staticmethod
    def augment(image: np.ndarray) -> np.ndarray:
        """Apply random underwater augmentations."""
        if np.random.random() > 0.5:
            image = UnderwaterAugmentation.add_particles(image, density=np.random.uniform(0.0005, 0.002))
        
        if np.random.random() > 0.5:
            image = UnderwaterAugmentation.add_turbidity(image, strength=np.random.uniform(0.05, 0.2))
            
        if np.random.random() > 0.5:
            image = UnderwaterAugmentation.color_attenuation(image, depth=np.random.uniform(5, 20))
            
        return np.clip(image, 0, 1)


if __name__ == "__main__":
    # Test the data loader
    dataset = UnderwaterVODataset("/app/data/raw", sequence_length=2)
    dataset.load_data()
    
    # Create train/val split
    train_indices, val_indices = dataset.train_val_split()
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    # Create TF dataset
    train_dataset = dataset.create_tf_dataset(train_indices, batch_size=32)
    
    # Test loading one batch
    for img_batch, pose_batch in train_dataset.take(1):
        print(f"Image batch shape: {img_batch.shape}")
        print(f"Pose batch shape: {pose_batch.shape}")
        print(f"Sample pose: {pose_batch[0].numpy()}")