"""
Underwater Visual Odometry Dataset

Windowed dataset loader with bag-based train/test splits for TSformer-VO training.
Supports sequence windowing, data augmentation, and proper train/test separation.

Author: Underwater Visual Odometry Research Team  
Date: January 2025
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import random


class UnderwaterVODataset(Dataset):
    """
    Underwater Visual Odometry Dataset for TSformer training.
    
    Creates windowed sequences from continuous trajectories with proper
    bag-based splitting to prevent data leakage.
    """
    
    def __init__(self,
                 csv_path,
                 data_root,
                 sequence_length=8,
                 overlap_frames=4,
                 image_size=224,
                 mode='train',
                 test_bags=None,
                 augment=True,
                 camera='cam0'):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file with pose data
            data_root: Root directory containing images
            sequence_length: Number of frames per window
            overlap_frames: Frame overlap between windows
            image_size: Target image size (will be resized)
            mode: 'train', 'val', or 'test'
            test_bags: List of bag names to use for testing
            augment: Apply data augmentation
            camera: Camera to use ('cam0', 'cam1', etc.)
        """
        self.csv_path = csv_path
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.overlap_frames = overlap_frames
        self.image_size = image_size
        self.mode = mode
        self.camera = camera
        
        # Load data
        print(f"Loading dataset from {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"Total frames in dataset: {len(self.df)}")
        
        # Create bag-based splits
        self.train_data, self.val_data, self.test_data = self._create_bag_splits(test_bags)
        
        # Select data based on mode
        if mode == 'train':
            self.data = self.train_data
            print(f"Training frames: {len(self.data)}")
        elif mode == 'val':
            self.data = self.val_data
            print(f"Validation frames: {len(self.data)}")
        elif mode == 'test':
            self.data = self.test_data
            print(f"Test frames: {len(self.data)}")
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Create sequence windows
        self.windows = self._create_windows()
        print(f"Total windows: {len(self.windows)}")
        
        # Data transforms
        self.transforms = self._create_transforms(augment and mode == 'train')
        
    def _create_bag_splits(self, test_bags=None):
        """Create train/val/test splits based on bags to prevent data leakage."""
        
        # Get unique bags
        unique_bags = self.df['bag_name'].unique()
        print(f"Available bags: {list(unique_bags)}")
        
        if test_bags is None:
            # Automatically select one bag for testing (largest bag)
            bag_sizes = self.df.groupby('bag_name').size()
            test_bags = [bag_sizes.idxmax()]
            print(f"Auto-selected test bag: {test_bags}")
        else:
            print(f"Using specified test bags: {test_bags}")
        
        # Split data
        test_data = self.df[self.df['bag_name'].isin(test_bags)].copy()
        train_val_data = self.df[~self.df['bag_name'].isin(test_bags)].copy()
        
        # Further split train_val into train/val (80/20 split within each bag)
        train_data_list = []
        val_data_list = []
        
        for bag in train_val_data['bag_name'].unique():
            bag_data = train_val_data[train_val_data['bag_name'] == bag]
            
            # Use first 80% for training, last 20% for validation
            # This maintains temporal order which is important for VO
            split_idx = int(len(bag_data) * 0.8)
            bag_train = bag_data.iloc[:split_idx]
            bag_val = bag_data.iloc[split_idx:]
            
            train_data_list.append(bag_train)
            val_data_list.append(bag_val)
        
        train_data = pd.concat(train_data_list, ignore_index=True) if train_data_list else pd.DataFrame()
        val_data = pd.concat(val_data_list, ignore_index=True) if val_data_list else pd.DataFrame()
        
        print(f"Data split:")
        print(f"  Train: {len(train_data)} frames from bags {list(train_data['bag_name'].unique())}")
        print(f"  Val: {len(val_data)} frames from bags {list(val_data['bag_name'].unique())}")
        print(f"  Test: {len(test_data)} frames from bags {list(test_data['bag_name'].unique())}")
        
        return train_data, val_data, test_data
    
    def _create_windows(self):
        """Create sliding windows from continuous sequences."""
        windows = []
        
        # Process each bag separately to maintain temporal consistency
        for bag_name in self.data['bag_name'].unique():
            bag_data = self.data[self.data['bag_name'] == bag_name].copy()
            bag_data = bag_data.sort_values('frame_index').reset_index(drop=True)
            
            # Create sliding windows
            stride = max(1, self.sequence_length - self.overlap_frames)  # Ensure stride >= 1
            
            for i in range(0, len(bag_data) - self.sequence_length + 1, stride):
                window_data = bag_data.iloc[i:i + self.sequence_length]
                
                # Skip if any frames are missing required data
                if self._is_valid_window(window_data):
                    windows.append({
                        'bag_name': bag_name,
                        'start_idx': i,
                        'window_data': window_data,
                        'target_pose': self._extract_pose(window_data.iloc[-1]),  # Single frame delta
                        'relative_pose': self._extract_relative_pose(window_data)  # NEW: Full sequence relative pose
                    })
        
        # Shuffle windows for training (but keep temporal order within each window)
        if self.mode == 'train':
            random.shuffle(windows)
        
        return windows
    
    def _is_valid_window(self, window_data):
        """Check if window contains valid data."""
        # Check for ground truth
        if not all(window_data['has_ground_truth']):
            return False
        
        # Check for valid image paths
        camera_col = f'{self.camera}_path'
        if camera_col not in window_data.columns:
            return False
        
        if any(pd.isna(window_data[camera_col])):
            return False
        
        # Check if image files exist
        for _, row in window_data.iterrows():
            img_path = self.data_root / row[camera_col]
            if not img_path.exists():
                return False
        
        return True
    
    def _extract_pose(self, frame_row):
        """Extract 6-DOF pose from dataframe row."""
        pose = np.array([
            frame_row['delta_x'],
            frame_row['delta_y'], 
            frame_row['delta_z'],
            frame_row['delta_roll'],
            frame_row['delta_pitch'],
            frame_row['delta_yaw']
        ], dtype=np.float32)
        
        return pose
    
    def _extract_relative_pose(self, window_data):
        """Extract relative pose from first to last frame of sequence."""
        # Accumulate deltas using SE(3) composition
        accumulated_T = np.eye(4)
        
        for _, frame in window_data.iterrows():
            # Extract delta
            delta = np.array([
                frame['delta_x'], frame['delta_y'], frame['delta_z'],
                frame['delta_roll'], frame['delta_pitch'], frame['delta_yaw']
            ])
            
            # Convert to SE(3) matrix
            delta_T = self._pose_to_se3_matrix(delta)
            
            # Compose: T_new = T_current @ T_delta
            accumulated_T = accumulated_T @ delta_T
        
        # Convert back to 6DOF representation
        relative_pose = self._se3_matrix_to_pose(accumulated_T)
        
        return relative_pose.astype(np.float32)
    
    def _pose_to_se3_matrix(self, pose):
        """Convert 6DOF pose to SE(3) transformation matrix."""
        translation = pose[:3]
        rotation = pose[3:]  # Euler angles
        
        # Convert Euler to rotation matrix
        from scipy.spatial.transform import Rotation as R
        rot_matrix = R.from_euler('xyz', rotation).as_matrix()
        
        # Create SE(3) matrix
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = translation
        
        return T
    
    def _se3_matrix_to_pose(self, T):
        """Convert SE(3) transformation matrix to 6DOF pose."""
        from scipy.spatial.transform import Rotation as R
        
        # Extract translation
        translation = T[:3, 3]
        
        # Extract rotation and convert to Euler angles
        rotation_matrix = T[:3, :3]
        rotation = R.from_matrix(rotation_matrix).as_euler('xyz')
        
        # Combine into 6DOF pose
        pose = np.concatenate([translation, rotation])
        
        return pose
    
    def _create_transforms(self, augment=True):
        """Create image preprocessing transforms."""
        transforms_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ]
        
        if augment:
            # Add minimal augmentations for training (removed harmful ones)
            augment_transforms = [
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                # Removed RandomHorizontalFlip (changes motion direction)
                # Removed GaussianBlur (removes important edge features)
            ]
            transforms_list = augment_transforms + transforms_list
        
        return transforms.Compose(transforms_list)
    
    def _load_image(self, image_path):
        """Load and preprocess image."""
        try:
            # Load image
            img_full_path = self.data_root / image_path
            image = Image.open(img_full_path).convert('RGB')
            
            # Apply transforms
            image = self.transforms(image)
            
            return image
        
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            return torch.zeros(3, self.image_size, self.image_size)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        window_data = window['window_data']
        target_pose = window['target_pose']
        relative_pose = window['relative_pose']  # NEW: Full sequence relative pose
        
        # Load image sequence
        images = []
        camera_col = f'{self.camera}_path'
        
        for _, row in window_data.iterrows():
            image = self._load_image(row[camera_col])
            images.append(image)
        
        # Stack images into sequence tensor
        image_sequence = torch.stack(images, dim=0)  # (seq_len, 3, H, W)
        
        # Convert poses to tensors
        pose_tensor = torch.from_numpy(target_pose).float()
        relative_pose_tensor = torch.from_numpy(relative_pose).float()
        
        return {
            'images': image_sequence,
            'poses': pose_tensor,               # Single frame delta (for local loss)
            'relative_poses': relative_pose_tensor,  # Full sequence relative pose (for multi-step loss)
            'bag_name': window['bag_name'],
            'frame_indices': window_data['frame_index'].tolist(),
            'timestamps': window_data['timestamp'].tolist()
        }
    
    def get_statistics(self):
        """Get dataset statistics."""
        if len(self.windows) == 0:
            return {}
        
        poses = np.array([window['target_pose'] for window in self.windows])
        
        stats = {
            'num_windows': len(self.windows),
            'num_bags': len(set(w['bag_name'] for w in self.windows)),
            'sequence_length': self.sequence_length,
            'overlap_frames': self.overlap_frames,
            'pose_stats': {
                'translation': {
                    'mean': poses[:, :3].mean(axis=0),
                    'std': poses[:, :3].std(axis=0),
                    'min': poses[:, :3].min(axis=0),
                    'max': poses[:, :3].max(axis=0)
                },
                'rotation': {
                    'mean': poses[:, 3:].mean(axis=0),
                    'std': poses[:, 3:].std(axis=0),
                    'min': poses[:, 3:].min(axis=0),
                    'max': poses[:, 3:].max(axis=0)
                }
            }
        }
        
        return stats


def create_data_loaders(csv_path, 
                       data_root,
                       sequence_length=8,
                       overlap_frames=4,
                       image_size=224,
                       batch_size=8,
                       test_bags=None,
                       num_workers=4,
                       camera='cam0'):
    """
    Create train/validation/test data loaders.
    
    Args:
        csv_path: Path to dataset CSV
        data_root: Root directory containing images
        sequence_length: Frames per sequence
        overlap_frames: Frame overlap
        image_size: Target image resolution
        batch_size: Batch size for training
        test_bags: Bags to reserve for testing
        num_workers: DataLoader worker processes
        camera: Which camera to use
    
    Returns:
        dict with train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = UnderwaterVODataset(
        csv_path=csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        overlap_frames=overlap_frames,
        image_size=image_size,
        mode='train',
        test_bags=test_bags,
        augment=True,
        camera=camera
    )
    
    val_dataset = UnderwaterVODataset(
        csv_path=csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        overlap_frames=overlap_frames,
        image_size=image_size,
        mode='val',
        test_bags=test_bags,
        augment=False,
        camera=camera
    )
    
    test_dataset = UnderwaterVODataset(
        csv_path=csv_path,
        data_root=data_root,
        sequence_length=sequence_length,
        overlap_frames=overlap_frames,
        image_size=image_size,
        mode='test',
        test_bags=test_bags,
        augment=False,
        camera=camera
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        stats = dataset.get_statistics()
        if stats:
            print(f"\n{name} Dataset:")
            print(f"  Windows: {stats['num_windows']}")
            print(f"  Bags: {stats['num_bags']}")
            print(f"  Translation range: {stats['pose_stats']['translation']['min']} to {stats['pose_stats']['translation']['max']}")
            print(f"  Rotation range: {stats['pose_stats']['rotation']['min']} to {stats['pose_stats']['rotation']['max']}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader, 
        'test_loader': test_loader,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }


if __name__ == "__main__":
    # Test the dataset
    csv_path = "data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv"
    data_root = "data/processed/visual_odometry_dataset"
    
    # Create data loaders
    data_loaders = create_data_loaders(
        csv_path=csv_path,
        data_root=data_root,
        sequence_length=8,
        batch_size=4,
        test_bags=["ariel_2023-12-21-14-28-22_4"]  # Reserve bag 4 for testing
    )
    
    # Test train loader
    print("\nTesting train loader...")
    train_loader = data_loaders['train_loader']
    
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Poses shape: {batch['poses'].shape}")
        print(f"  Bag names: {batch['bag_name']}")
        print(f"  Sample pose: {batch['poses'][0]}")
        
        if i >= 2:  # Test first few batches
            break
    
    print("Dataset test completed successfully!")