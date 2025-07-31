"""
Dataset classes for scalable multi-camera underwater visual odometry

Supports:
- Variable number of cameras (1-5)
- Multi-modal data (vision + IMU + pressure)
- Different sequence lengths (2, 4 frames)
- Robust handling of missing data
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path

from .preprocessing import UnderwaterImageProcessor, SensorDataProcessor
from .augmentation import UnderwaterAugmentation


class UnderwaterVODataset(Dataset):
    """
    Multi-camera underwater visual odometry dataset
    
    Supports scalable camera configurations and multi-modal sensor fusion
    """
    
    def __init__(
        self,
        data_csv: str,
        data_root: str,
        camera_ids: List[int] = [0, 1, 2, 3],
        sequence_length: int = 2,
        img_size: int = 224,
        use_imu: bool = True,
        use_pressure: bool = True,
        augmentation: bool = True,
        split: str = 'train',
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_csv: Path to CSV file with sample data
            data_root: Root directory containing images and data
            camera_ids: List of camera IDs to use (e.g., [0, 1, 2, 3])
            sequence_length: Number of frames in sequence (2 or 4)
            img_size: Target image size for preprocessing
            use_imu: Whether to include IMU data
            use_pressure: Whether to include pressure data
            augmentation: Whether to apply data augmentation
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples (for debugging)
        """
        self.data_root = Path(data_root)
        self.camera_ids = camera_ids
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.use_imu = use_imu
        self.use_pressure = use_pressure
        self.split = split
        self.max_cameras = 5  # Maximum cameras in dataset
        
        # Load dataset
        self.df = pd.read_csv(data_csv)
        
        # Filter by split if split information is available
        if 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        # Limit samples for debugging
        if max_samples:
            self.df = self.df.head(max_samples)
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        # Initialize processors
        self.image_processor = UnderwaterImageProcessor(
            img_size=img_size,
            normalize=True
        )
        
        self.sensor_processor = SensorDataProcessor()
        
        # Initialize augmentation
        if augmentation and split == 'train':
            self.augmentation = UnderwaterAugmentation()
        else:
            self.augmentation = None
            
        print(f"Loaded {len(self.sequences)} sequences for {split} split")
        print(f"Using cameras: {camera_ids}")
        print(f"Sequence length: {sequence_length}")
        
    def _create_sequences(self) -> List[Dict]:
        """
        Create sequence samples from the dataframe
        
        Uses shuffled frame pairs instead of sequential pairs for better generalization.
        For training: random pairs across dataset
        For val/test: sequential pairs for consistent evaluation
        """
        sequences = []
        
        if self.split == 'train':
            # SHUFFLED TRAINING: Random frame pairs across entire dataset
            all_valid_frames = []
            
            # Collect all valid frames that can form sequences
            for bag_name, bag_df in self.df.groupby('bag_name'):
                bag_df = bag_df.sort_values('timestamp').reset_index(drop=True)
                
                # Find frames with sufficient data for sequences
                for i in range(len(bag_df) - self.sequence_length + 1):
                    frame_data = {
                        'bag_name': bag_name,
                        'bag_df': bag_df,
                        'start_idx': i
                    }
                    all_valid_frames.append(frame_data)
            
            # Shuffle for random frame pair selection
            import random
            random.shuffle(all_valid_frames)
            
            # Create sequences from shuffled frames
            for frame_data in all_valid_frames:
                sequence_data = {
                    'indices': list(range(frame_data['start_idx'], 
                                        frame_data['start_idx'] + self.sequence_length)),
                    'bag_name': frame_data['bag_name'],
                    'bag_df': frame_data['bag_df']
                }
                sequences.append(sequence_data)
                
            print(f"Created {len(sequences)} SHUFFLED training sequences")
            
        else:
            # SEQUENTIAL for validation/test: maintain temporal order for evaluation
            for bag_name, bag_df in self.df.groupby('bag_name'):
                bag_df = bag_df.sort_values('timestamp').reset_index(drop=True)
                
                # Create sequences of specified length
                for i in range(len(bag_df) - self.sequence_length + 1):
                    sequence_data = {
                        'indices': list(range(i, i + self.sequence_length)),
                        'bag_name': bag_name,
                        'bag_df': bag_df
                    }
                    sequences.append(sequence_data)
            
            print(f"Created {len(sequences)} SEQUENTIAL {self.split} sequences")
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Returns:
            Dictionary containing:
            - images: [seq_len, num_cameras, 3, H, W]
            - camera_ids: [num_cameras]
            - camera_mask: [num_cameras] (True for missing cameras)
            - pose_target: [6] (delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw)
            - imu_data: [seq_len, 6] (if use_imu)
            - pressure_data: [seq_len, 1] (if use_pressure)
            - metadata: Dictionary with additional info
        """
        sequence = self.sequences[idx]
        bag_df = sequence['bag_df']
        indices = sequence['indices']
        
        # Get sequence rows
        sequence_rows = bag_df.iloc[indices]
        
        # Load images for all cameras and time steps
        images = []
        camera_mask = []
        
        for camera_id in self.camera_ids:
            camera_images = []
            camera_available = True
            
            for _, row in sequence_rows.iterrows():
                img_path_col = f'cam{camera_id}_path'
                
                if img_path_col in row and pd.notna(row[img_path_col]):
                    img_path = self.data_root / row[img_path_col]
                    
                    if img_path.exists():
                        # Load and process image
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
            
            if camera_available and len(camera_images) == self.sequence_length:
                images.append(torch.stack(camera_images))  # [seq_len, 3, H, W]
                camera_mask.append(False)  # Camera is available
            else:
                # Create dummy images for missing camera
                dummy_image = torch.zeros(self.sequence_length, 3, self.img_size, self.img_size)
                images.append(dummy_image)
                camera_mask.append(True)  # Camera is missing
        
        # Stack images: [num_cameras, seq_len, 3, H, W] -> [seq_len, num_cameras, 3, H, W]
        images = torch.stack(images).transpose(0, 1)
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            images = self.augmentation(images)
        
        # Get pose target (relative motion from first to last frame)
        last_row = sequence_rows.iloc[-1]
        pose_target = torch.tensor([
            last_row['delta_x'],
            last_row['delta_y'], 
            last_row['delta_z'],
            last_row['delta_roll'],
            last_row['delta_pitch'],
            last_row['delta_yaw']
        ], dtype=torch.float32)
        
        # Prepare output dictionary
        sample = {
            'images': images,
            'camera_ids': torch.tensor(self.camera_ids, dtype=torch.long),
            'camera_mask': torch.tensor(camera_mask, dtype=torch.bool),
            'pose_target': pose_target,
            'metadata': {
                'bag_name': sequence['bag_name'],
                'timestamps': sequence_rows['timestamp'].tolist(),
                'sample_ids': sequence_rows['sample_id'].tolist()
            }
        }
        
        # Add IMU data if requested
        if self.use_imu:
            imu_data = []
            for _, row in sequence_rows.iterrows():
                if all(col in row for col in ['imu_accel_x', 'imu_accel_y', 'imu_accel_z',
                                            'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z']):
                    imu_sample = [
                        row['imu_accel_x'], row['imu_accel_y'], row['imu_accel_z'],
                        row['imu_gyro_x'], row['imu_gyro_y'], row['imu_gyro_z']
                    ]
                    imu_data.append(imu_sample)
                else:
                    # Use zeros for missing IMU data
                    imu_data.append([0.0] * 6)
            
            sample['imu_data'] = torch.tensor(imu_data, dtype=torch.float32)
        
        # Add pressure data if requested
        if self.use_pressure:
            pressure_data = []
            for _, row in sequence_rows.iterrows():
                if 'pressure' in row and pd.notna(row['pressure']):
                    pressure_data.append([row['pressure']])
                else:
                    # Use zero for missing pressure data
                    pressure_data.append([0.0])
            
            sample['pressure_data'] = torch.tensor(pressure_data, dtype=torch.float32)
        
        return sample


class MultiModalDataset(UnderwaterVODataset):
    """
    Extended dataset class with advanced multi-modal capabilities
    
    Adds support for:
    - Dynamic camera selection based on image quality
    - Temporal consistency checking
    - Advanced sensor synchronization
    """
    
    def __init__(
        self,
        *args,
        quality_threshold: float = 0.5,
        sync_tolerance: float = 0.1,
        **kwargs
    ):
        """
        Args:
            quality_threshold: Minimum image quality score (0-1)
            sync_tolerance: Maximum time difference for sensor sync (seconds)
        """
        super().__init__(*args, **kwargs)
        self.quality_threshold = quality_threshold
        self.sync_tolerance = sync_tolerance
        
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """
        Assess image quality using various metrics
        
        Args:
            image: Input image [H, W, 3]
        Returns:
            Quality score (0-1, higher is better)
        """
        # Convert to grayscale for quality assessment
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute metrics
        # 1. Laplacian variance (sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # 2. Mean brightness (avoid too dark/bright images)
        mean_brightness = gray.mean() / 255.0
        brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
        
        # 3. Contrast (standard deviation)
        contrast_score = min(gray.std() / 128.0, 1.0)
        
        # Combined quality score
        quality = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
        
        return quality
    
    def _select_best_cameras(
        self,
        sequence_rows: pd.DataFrame,
        available_cameras: List[int]
    ) -> List[int]:
        """
        Select best cameras based on image quality and availability
        
        Args:
            sequence_rows: DataFrame rows for this sequence
            available_cameras: List of camera IDs to consider
            
        Returns:
            List of selected camera IDs
        """
        camera_scores = {}
        
        for camera_id in available_cameras:
            img_path_col = f'cam{camera_id}_path'
            total_quality = 0.0
            valid_frames = 0
            
            for _, row in sequence_rows.iterrows():
                if img_path_col in row and pd.notna(row[img_path_col]):
                    img_path = self.data_root / row[img_path_col]
                    
                    if img_path.exists():
                        image = cv2.imread(str(img_path))
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            quality = self._assess_image_quality(image)
                            total_quality += quality
                            valid_frames += 1
            
            if valid_frames == len(sequence_rows):
                camera_scores[camera_id] = total_quality / valid_frames
            else:
                camera_scores[camera_id] = 0.0  # Penalize missing frames
        
        # Select cameras above threshold, sorted by quality
        good_cameras = [(cam_id, score) for cam_id, score in camera_scores.items() 
                       if score >= self.quality_threshold]
        good_cameras.sort(key=lambda x: x[1], reverse=True)
        
        # Return at least one camera, preferably the requested number
        if good_cameras:
            selected = [cam_id for cam_id, _ in good_cameras[:len(self.camera_ids)]]
            # Pad with original camera_ids if needed
            while len(selected) < len(self.camera_ids) and len(selected) < len(available_cameras):
                for cam_id in self.camera_ids:
                    if cam_id not in selected and cam_id in available_cameras:
                        selected.append(cam_id)
                        break
        else:
            selected = self.camera_ids[:min(len(self.camera_ids), len(available_cameras))]
        
        return selected


def create_dataset(config: Dict) -> UnderwaterVODataset:
    """Factory function to create dataset from configuration"""
    dataset_class = MultiModalDataset if config.get('advanced_multimodal', False) else UnderwaterVODataset
    
    return dataset_class(
        data_csv=config['data_csv'],
        data_root=config['data_root'],
        camera_ids=config.get('camera_ids', [0, 1, 2, 3]),
        sequence_length=config.get('sequence_length', 2),
        img_size=config.get('img_size', 224),
        use_imu=config.get('use_imu', True),
        use_pressure=config.get('use_pressure', True),
        augmentation=config.get('augmentation', True),
        split=config.get('split', 'train'),
        max_samples=config.get('max_samples', None)
    )


def create_dataloaders(
    train_config: Dict,
    val_config: Dict,
    test_config: Dict,
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    # Create datasets
    train_dataset = create_dataset(train_config)
    val_dataset = create_dataset(val_config)
    test_dataset = create_dataset(test_config)
    
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
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader