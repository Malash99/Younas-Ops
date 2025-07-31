"""
Preprocessing modules for underwater images and sensor data

Handles:
- Underwater-specific image enhancement
- Sensor data normalization
- Data standardization
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional
from torchvision import transforms
import torch.nn.functional as F


class UnderwaterImageProcessor:
    """
    Underwater image preprocessing pipeline
    
    Includes:
    - Resizing and normalization
    - Underwater-specific enhancements
    - Data format conversion
    """
    
    def __init__(
        self,
        img_size: int = 224,
        normalize: bool = True,
        enhance_underwater: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Args:
            img_size: Target image size
            normalize: Whether to normalize with ImageNet stats
            enhance_underwater: Whether to apply underwater enhancement
            mean: Normalization mean values
            std: Normalization std values
        """
        self.img_size = img_size
        self.normalize = normalize
        self.enhance_underwater = enhance_underwater
        self.mean = mean
        self.std = std
        
        # Build transform pipeline
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize((img_size, img_size)))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if requested
        if normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        self.transform = transforms.Compose(transform_list)
        
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Process a single image
        
        Args:
            image: Input image [H, W, 3] in RGB format
        Returns:
            Processed image tensor [3, H, W]
        """
        # Apply underwater enhancement if enabled
        if self.enhance_underwater:
            image = self._enhance_underwater_image(image)
        
        # Apply transforms
        if isinstance(image, np.ndarray):
            # Convert numpy to PIL for torchvision transforms
            from PIL import Image
            image = Image.fromarray(image.astype(np.uint8))
        
        processed = self.transform(image)
        
        return processed
    
    def _enhance_underwater_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply underwater-specific image enhancements
        
        Args:
            image: Input image [H, W, 3] in RGB format
        Returns:
            Enhanced image [H, W, 3]
        """
        # Convert to LAB color space for better color correction
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 1. Histogram equalization on L channel
        l_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        
        # 2. White balance correction
        # Estimate illuminant and correct color cast
        a_corrected = self._white_balance_correction(a)
        b_corrected = self._white_balance_correction(b)
        
        # Merge back to LAB
        lab_enhanced = cv2.merge([l_eq, a_corrected, b_corrected])
        
        # Convert back to RGB
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # 3. Contrast enhancement
        rgb_enhanced = self._enhance_contrast(rgb_enhanced)
        
        # 4. Reduce noise
        rgb_enhanced = cv2.bilateralFilter(rgb_enhanced, 5, 75, 75)
        
        return rgb_enhanced
    
    def _white_balance_correction(self, channel: np.ndarray) -> np.ndarray:
        """Simple white balance correction for a single channel"""
        # Estimate the illuminant using gray-world assumption
        mean_val = np.mean(channel)
        target_val = 128  # Target gray value
        
        # Apply correction
        correction_factor = target_val / (mean_val + 1e-6)
        corrected = np.clip(channel * correction_factor, 0, 255).astype(np.uint8)
        
        return corrected
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using adaptive histogram equalization"""
        # Convert to YUV for luminance adjustment
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(yuv)
        
        # Apply CLAHE to Y channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)
        
        # Merge back
        yuv_enhanced = cv2.merge([y_eq, u, v])
        rgb_enhanced = cv2.cvtColor(yuv_enhanced, cv2.COLOR_YUV2RGB)
        
        return rgb_enhanced


class SensorDataProcessor:
    """
    Sensor data preprocessing for IMU and pressure sensors
    
    Handles:
    - Data normalization
    - Outlier removal
    - Missing data interpolation
    """
    
    def __init__(
        self,
        imu_accel_range: Tuple[float, float] = (-20.0, 20.0),  # m/s^2
        imu_gyro_range: Tuple[float, float] = (-10.0, 10.0),   # rad/s
        pressure_range: Tuple[float, float] = (95000, 110000), # Pa (typical range)
        outlier_threshold: float = 3.0
    ):
        """
        Args:
            imu_accel_range: Expected accelerometer range (min, max)
            imu_gyro_range: Expected gyroscope range (min, max)
            pressure_range: Expected pressure range (min, max)
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.imu_accel_range = imu_accel_range
        self.imu_gyro_range = imu_gyro_range
        self.pressure_range = pressure_range
        self.outlier_threshold = outlier_threshold
        
        # Statistics for normalization (will be computed from data or set manually)
        self.imu_stats = None
        self.pressure_stats = None
        
    def set_normalization_stats(
        self,
        imu_mean: Optional[np.ndarray] = None,
        imu_std: Optional[np.ndarray] = None,
        pressure_mean: Optional[float] = None,
        pressure_std: Optional[float] = None
    ):
        """Set normalization statistics manually"""
        if imu_mean is not None and imu_std is not None:
            self.imu_stats = {'mean': imu_mean, 'std': imu_std}
        
        if pressure_mean is not None and pressure_std is not None:
            self.pressure_stats = {'mean': pressure_mean, 'std': pressure_std}
    
    def compute_normalization_stats(
        self,
        imu_data: np.ndarray,
        pressure_data: Optional[np.ndarray] = None
    ):
        """
        Compute normalization statistics from data
        
        Args:
            imu_data: IMU data array [N, 6] (ax, ay, az, gx, gy, gz)
            pressure_data: Pressure data array [N, 1] (optional)
        """
        # Remove outliers before computing stats
        imu_clean = self._remove_outliers(imu_data)
        
        self.imu_stats = {
            'mean': np.mean(imu_clean, axis=0),
            'std': np.std(imu_clean, axis=0) + 1e-6  # Add small epsilon
        }
        
        if pressure_data is not None:
            pressure_clean = self._remove_outliers(pressure_data.reshape(-1, 1))
            self.pressure_stats = {
                'mean': np.mean(pressure_clean),
                'std': np.std(pressure_clean) + 1e-6
            }
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using Z-score method"""
        z_scores = np.abs((data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6))
        
        # Keep only samples where all dimensions are within threshold
        valid_mask = np.all(z_scores < self.outlier_threshold, axis=1)
        
        return data[valid_mask]
    
    def process_imu_data(self, imu_data: np.ndarray) -> torch.Tensor:
        """
        Process IMU data
        
        Args:
            imu_data: Raw IMU data [seq_len, 6] or [6]
        Returns:
            Processed IMU tensor
        """
        # Convert to numpy if needed
        if isinstance(imu_data, torch.Tensor):
            imu_data = imu_data.numpy()
        
        # Handle single sample vs sequence
        original_shape = imu_data.shape
        if imu_data.ndim == 1:
            imu_data = imu_data.reshape(1, -1)
        
        # Clamp to expected ranges
        imu_data[:, :3] = np.clip(imu_data[:, :3], *self.imu_accel_range)  # Accelerometer
        imu_data[:, 3:] = np.clip(imu_data[:, 3:], *self.imu_gyro_range)   # Gyroscope
        
        # Normalize if statistics available
        if self.imu_stats is not None:
            imu_data = (imu_data - self.imu_stats['mean']) / self.imu_stats['std']
        
        # Convert to tensor
        processed = torch.tensor(imu_data, dtype=torch.float32)
        
        # Restore original shape
        if len(original_shape) == 1:
            processed = processed.squeeze(0)
        
        return processed
    
    def process_pressure_data(self, pressure_data: np.ndarray) -> torch.Tensor:
        """
        Process pressure data
        
        Args:
            pressure_data: Raw pressure data [seq_len, 1] or [1]
        Returns:
            Processed pressure tensor
        """
        # Convert to numpy if needed
        if isinstance(pressure_data, torch.Tensor):
            pressure_data = pressure_data.numpy()
        
        # Handle single sample vs sequence
        original_shape = pressure_data.shape
        if pressure_data.ndim == 1:
            pressure_data = pressure_data.reshape(-1, 1)
        
        # Clamp to expected range
        pressure_data = np.clip(pressure_data, *self.pressure_range)
        
        # Normalize if statistics available
        if self.pressure_stats is not None:
            pressure_data = (pressure_data - self.pressure_stats['mean']) / self.pressure_stats['std']
        
        # Convert to tensor
        processed = torch.tensor(pressure_data, dtype=torch.float32)
        
        # Restore original shape
        if len(original_shape) == 1:
            processed = processed.squeeze(0)
        
        return processed
    
    def process_batch(
        self,
        imu_batch: Optional[torch.Tensor] = None,
        pressure_batch: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Process batched sensor data
        
        Args:
            imu_batch: Batched IMU data [batch_size, seq_len, 6]
            pressure_batch: Batched pressure data [batch_size, seq_len, 1]
            
        Returns:
            Tuple of processed (imu_batch, pressure_batch)
        """
        processed_imu = None
        processed_pressure = None
        
        if imu_batch is not None:
            batch_size, seq_len = imu_batch.shape[:2]
            # Process each sample in batch
            processed_samples = []
            for i in range(batch_size):
                processed = self.process_imu_data(imu_batch[i])
                processed_samples.append(processed)
            processed_imu = torch.stack(processed_samples)
        
        if pressure_batch is not None:
            batch_size, seq_len = pressure_batch.shape[:2]
            # Process each sample in batch
            processed_samples = []
            for i in range(batch_size):
                processed = self.process_pressure_data(pressure_batch[i])
                processed_samples.append(processed)
            processed_pressure = torch.stack(processed_samples)
        
        return processed_imu, processed_pressure


class DataStandardizer:
    """
    Standardizes data across the entire dataset
    
    Computes and applies dataset-level normalization statistics
    """
    
    def __init__(self):
        self.pose_stats = None
        self.image_processor = UnderwaterImageProcessor()
        self.sensor_processor = SensorDataProcessor()
        
    def fit(self, dataset: 'UnderwaterVODataset'):
        """
        Compute normalization statistics from dataset
        
        Args:
            dataset: Dataset to compute statistics from
        """
        print("Computing dataset statistics...")
        
        # Collect pose data
        poses = []
        imu_data = []
        pressure_data = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            poses.append(sample['pose_target'].numpy())
            
            if 'imu_data' in sample:
                imu_data.append(sample['imu_data'].numpy())
            
            if 'pressure_data' in sample:
                pressure_data.append(sample['pressure_data'].numpy())
        
        # Compute pose statistics
        poses = np.array(poses)
        self.pose_stats = {
            'mean': np.mean(poses, axis=0),
            'std': np.std(poses, axis=0) + 1e-6
        }
        
        # Compute sensor statistics
        if imu_data:
            imu_data = np.concatenate(imu_data, axis=0)
            self.sensor_processor.compute_normalization_stats(imu_data)
        
        if pressure_data:
            pressure_data = np.concatenate(pressure_data, axis=0)
            self.sensor_processor.compute_normalization_stats(
                np.zeros((len(pressure_data), 6)),  # Dummy IMU data
                pressure_data
            )
        
        print(f"Pose statistics - Mean: {self.pose_stats['mean']}")
        print(f"Pose statistics - Std: {self.pose_stats['std']}")
        
    def normalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Normalize pose data"""
        if self.pose_stats is None:
            return pose
        
        mean = torch.tensor(self.pose_stats['mean'], dtype=pose.dtype, device=pose.device)
        std = torch.tensor(self.pose_stats['std'], dtype=pose.dtype, device=pose.device)
        
        return (pose - mean) / std
    
    def denormalize_pose(self, normalized_pose: torch.Tensor) -> torch.Tensor:
        """Denormalize pose data back to original scale"""
        if self.pose_stats is None:
            return normalized_pose
        
        mean = torch.tensor(self.pose_stats['mean'], dtype=normalized_pose.dtype, device=normalized_pose.device)
        std = torch.tensor(self.pose_stats['std'], dtype=normalized_pose.dtype, device=normalized_pose.device)
        
        return normalized_pose * std + mean