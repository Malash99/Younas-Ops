"""
Sub-Trajectory Dataset for Trajectory-Aware Training
Creates sequences of N frames for ATE-aware visual odometry training
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .preprocessing import UnderwaterImageProcessor
from .augmentation import UnderwaterAugmentation


class SubTrajectoryDataset(Dataset):
    """
    Sub-trajectory dataset for trajectory-aware training
    
    Creates overlapping sub-trajectories of length N frames
    Each sample contains N images and N-1 relative poses
    Enables ATE supervision over short trajectory segments
    """
    
    def __init__(
        self,
        data_csv: str,
        data_root: str,
        sub_trajectory_length: int = 8,
        overlap: int = 4,
        camera_ids: List[int] = [0, 1, 2, 3],
        img_size: int = 224,
        use_imu: bool = False,
        use_pressure: bool = False,
        augmentation: bool = True,
        split: str = 'train',
        max_samples: Optional[int] = None
    ):
        """
        Args:
            sub_trajectory_length: Number of frames in each sub-trajectory (e.g., 8)
            overlap: Overlap between consecutive sub-trajectories (e.g., 4)
            Other args same as original dataset
        """
        self.data_root = Path(data_root)
        self.sub_trajectory_length = sub_trajectory_length
        self.overlap = overlap
        self.camera_ids = camera_ids
        self.img_size = img_size
        self.use_imu = use_imu
        self.use_pressure = use_pressure
        self.split = split
        
        # Load dataset
        self.df = pd.read_csv(data_csv)
        
        # Filter by split
        if 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        # Create sub-trajectories
        self.sub_trajectories = self._create_sub_trajectories()
        
        # Limit samples for quick testing
        if max_samples:
            self.sub_trajectories = self.sub_trajectories[:max_samples]
        
        # Initialize processors
        self.image_processor = UnderwaterImageProcessor(
            img_size=img_size,
            normalize=True
        )
        
        # Initialize augmentation
        if augmentation and split == 'train':
            self.augmentation = UnderwaterAugmentation()
        else:
            self.augmentation = None
            
        print(f"SubTrajectoryDataset loaded:")
        print(f"  Split: {split}")
        print(f"  Sub-trajectory length: {sub_trajectory_length}")
        print(f"  Overlap: {overlap}")
        print(f"  Total sub-trajectories: {len(self.sub_trajectories)}")
        print(f"  Cameras: {camera_ids}")
        
    def _create_sub_trajectories(self) -> List[Dict]:
        """Create overlapping sub-trajectories from the data"""
        sub_trajectories = []
        step_size = self.sub_trajectory_length - self.overlap
        
        if self.split == 'train':
            # For training: create overlapping sub-trajectories from each bag, then shuffle
            for bag_name, bag_df in self.df.groupby('bag_name'):
                bag_df = bag_df.sort_values('timestamp').reset_index(drop=True)
                
                # Create overlapping windows
                for start_idx in range(0, len(bag_df) - self.sub_trajectory_length + 1, step_size):
                    end_idx = start_idx + self.sub_trajectory_length
                    
                    sub_traj = {
                        'bag_name': bag_name,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'indices': list(range(start_idx, end_idx)),
                        'bag_df': bag_df.iloc[start_idx:end_idx].reset_index(drop=True)
                    }
                    sub_trajectories.append(sub_traj)
            
            # Shuffle for training
            import random
            random.shuffle(sub_trajectories)
            print(f"Created {len(sub_trajectories)} shuffled sub-trajectories for training")
            
        else:
            # For val/test: sequential sub-trajectories (no shuffling)
            for bag_name, bag_df in self.df.groupby('bag_name'):
                bag_df = bag_df.sort_values('timestamp').reset_index(drop=True)
                
                for start_idx in range(0, len(bag_df) - self.sub_trajectory_length + 1, step_size):
                    end_idx = start_idx + self.sub_trajectory_length
                    
                    sub_traj = {
                        'bag_name': bag_name,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'indices': list(range(start_idx, end_idx)),
                        'bag_df': bag_df.iloc[start_idx:end_idx].reset_index(drop=True)
                    }
                    sub_trajectories.append(sub_traj)
            
            print(f"Created {len(sub_trajectories)} sequential sub-trajectories for {self.split}")
        
        return sub_trajectories
    
    def __len__(self) -> int:
        return len(self.sub_trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sub-trajectory sample
        
        Returns:
            Dictionary containing:
            - images: [sub_traj_len, num_cameras, 3, H, W]
            - camera_ids: [num_cameras]
            - camera_mask: [num_cameras]
            - pose_targets: [sub_traj_len-1, 6] (N-1 relative poses)
            - accumulated_poses: [sub_traj_len-1, 6] (cumulative poses for ATE)
            - metadata: Additional info
        """
        sub_traj = self.sub_trajectories[idx]
        sub_traj_df = sub_traj['bag_df']
        
        # Load images for all cameras and time steps
        all_images = []  # [sub_traj_len, num_cameras, 3, H, W]
        camera_mask = []
        
        for camera_id in self.camera_ids:
            camera_images = []
            camera_available = True
            
            for _, row in sub_traj_df.iterrows():
                img_path_col = f'cam{camera_id}_path'
                
                if img_path_col in row and pd.notna(row[img_path_col]):
                    img_path = self.data_root / row[img_path_col]
                    
                    if img_path.exists():
                        image = cv2.imread(str(img_path))
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = self.image_processor(image)
                            camera_images.append(image)
                        else:
                            camera_available = False
                            break
                    else:
                        camera_available = False
                        break
                else:
                    camera_available = False
                    break
            
            if camera_available and len(camera_images) == self.sub_trajectory_length:
                camera_mask.append(False)  # Camera available
            else:
                # Create dummy images for missing camera
                camera_images = [torch.zeros(3, self.img_size, self.img_size) 
                               for _ in range(self.sub_trajectory_length)]
                camera_mask.append(True)  # Camera missing
            
            # Add camera images to all_images
            for t, img in enumerate(camera_images):
                if t >= len(all_images):
                    all_images.append([])
                all_images[t].append(img)
        
        # Convert to tensor: [sub_traj_len, num_cameras, 3, H, W]
        images = torch.stack([torch.stack(time_step) for time_step in all_images])
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            # Augmentation expects [seq_len, num_cameras, 3, H, W]
            # Our images are [sub_traj_len, num_cameras, 3, H, W] - perfect!
            images = self.augmentation(images)
        
        # Get relative pose targets (N-1 poses)
        pose_targets = []
        for i in range(len(sub_traj_df) - 1):
            row = sub_traj_df.iloc[i + 1]  # Next frame's delta
            pose = torch.tensor([
                row['delta_x'], row['delta_y'], row['delta_z'],
                row['delta_roll'], row['delta_pitch'], row['delta_yaw']
            ], dtype=torch.float32)
            pose_targets.append(pose)
        
        pose_targets = torch.stack(pose_targets)  # [N-1, 6]
        
        # Calculate accumulated poses for ATE supervision
        accumulated_poses = torch.zeros_like(pose_targets)
        current_pose = torch.zeros(6)
        
        for i in range(len(pose_targets)):
            current_pose += pose_targets[i]
            accumulated_poses[i] = current_pose.clone()
        
        # Prepare output dictionary
        sample = {
            'images': images,  # [sub_traj_len, num_cameras, 3, H, W]
            'camera_ids': torch.tensor(self.camera_ids, dtype=torch.long),
            'camera_mask': torch.tensor(camera_mask, dtype=torch.bool),
            'pose_targets': pose_targets,  # [N-1, 6] relative poses
            'accumulated_poses': accumulated_poses,  # [N-1, 6] for ATE loss
            'metadata': {
                'bag_name': sub_traj['bag_name'],
                'start_idx': sub_traj['start_idx'],
                'end_idx': sub_traj['end_idx'],
                'timestamps': sub_traj_df['timestamp'].tolist(),
                'sub_traj_length': self.sub_trajectory_length
            }
        }
        
        # Add IMU data if requested
        if self.use_imu:
            imu_data = []
            for _, row in sub_traj_df.iterrows():
                if all(col in row for col in ['imu_accel_x', 'imu_accel_y', 'imu_accel_z',
                                            'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z']):
                    imu_sample = [
                        row['imu_accel_x'], row['imu_accel_y'], row['imu_accel_z'],
                        row['imu_gyro_x'], row['imu_gyro_y'], row['imu_gyro_z']
                    ]
                    imu_data.append(imu_sample)
                else:
                    imu_data.append([0.0] * 6)
            
            sample['imu_data'] = torch.tensor(imu_data, dtype=torch.float32)
        
        # Add pressure data if requested
        if self.use_pressure:
            pressure_data = []
            for _, row in sub_traj_df.iterrows():
                if 'pressure' in row and pd.notna(row['pressure']):
                    pressure_data.append([row['pressure']])
                else:
                    pressure_data.append([0.0])
            
            sample['pressure_data'] = torch.tensor(pressure_data, dtype=torch.float32)
        
        return sample


def create_sub_trajectory_dataloaders(
    train_csv: str,
    val_csv: str,
    data_root: str = '.',
    sub_trajectory_length: int = 8,
    overlap: int = 4,
    camera_ids: List[int] = [0, 1, 2, 3],
    batch_size: int = 4,
    num_workers: int = 2,
    max_samples_train: Optional[int] = None,
    max_samples_val: Optional[int] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders for sub-trajectory training"""
    
    # Create datasets
    train_dataset = SubTrajectoryDataset(
        data_csv=train_csv,
        data_root=data_root,
        sub_trajectory_length=sub_trajectory_length,
        overlap=overlap,
        camera_ids=camera_ids,
        augmentation=True,
        split='train',
        max_samples=max_samples_train
    )
    
    val_dataset = SubTrajectoryDataset(
        data_csv=val_csv,
        data_root=data_root,
        sub_trajectory_length=sub_trajectory_length,
        overlap=overlap,
        camera_ids=camera_ids,
        augmentation=False,
        split='val',
        max_samples=max_samples_val
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader