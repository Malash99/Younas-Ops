import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnderwaterVIODataset(Dataset):
    def __init__(self, csv_path, data_root, sequence_length=10, transform=None, use_ground_truth_only=True):
        """
        Dataset for underwater Visual-Inertial Odometry
        
        Args:
            csv_path: Path to CSV file
            data_root: Root directory for data
            sequence_length: Length of temporal sequences for LSTM
            transform: Image transformations
            use_ground_truth_only: If True, only use samples with ground truth
        """
        self.data = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.transform = transform or self.default_transform()
        
        # Filter out samples without ground truth if requested
        if use_ground_truth_only:
            self.data = self.data[self.data['has_ground_truth'] == True]
            logger.info(f"Filtered dataset to {len(self.data)} samples with ground truth")
        
        # Group by bag_name to create sequences
        self.sequences = []
        for bag_name, group in self.data.groupby('bag_name'):
            group = group.sort_values('frame_index').reset_index(drop=True)
            if len(group) >= sequence_length:
                for i in range(len(group) - sequence_length + 1):
                    self.sequences.append((bag_name, i, i + sequence_length))
        
        logger.info(f"Created {len(self.sequences)} sequences of length {sequence_length}")
        
        # Compute normalization statistics
        self.compute_normalization_stats()
    
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def compute_normalization_stats(self):
        """Compute mean and std for pose and IMU data normalization"""
        # Pose deltas (translation and rotation)
        pose_cols = ['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw']
        self.pose_mean = self.data[pose_cols].mean().values.astype(np.float32)
        self.pose_std = self.data[pose_cols].std().values.astype(np.float32)
        
        # IMU data
        imu_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        self.imu_mean = self.data[imu_cols].mean().values.astype(np.float32)
        self.imu_std = self.data[imu_cols].std().values.astype(np.float32)
        
        logger.info(f"Pose normalization - Mean: {self.pose_mean}, Std: {self.pose_std}")
        logger.info(f"IMU normalization - Mean: {self.imu_mean}, Std: {self.imu_std}")
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        full_path = self.data_root / image_path
        try:
            image = Image.open(full_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            logger.warning(f"Failed to load image {full_path}: {e}")
            # Return black image as fallback
            return torch.zeros(3, 224, 224)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        bag_name, start_idx, end_idx = self.sequences[idx]
        
        # Get sequence data
        bag_data = self.data[self.data['bag_name'] == bag_name].iloc[start_idx:end_idx]
        
        # Load images (using cam0 for simplicity, can be extended to multi-camera)
        images = []
        for _, row in bag_data.iterrows():
            image = self.load_image(row['cam0_path'])
            images.append(image)
        images = torch.stack(images)  # Shape: (seq_len, 3, H, W)
        
        # Get IMU data
        imu_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        imu_data = bag_data[imu_cols].values.astype(np.float32)
        # Normalize IMU data
        imu_data = (imu_data - self.imu_mean) / (self.imu_std + 1e-8)
        imu_tensor = torch.from_numpy(imu_data)  # Shape: (seq_len, 6)
        
        # Get pose targets (6-DOF: x, y, z, roll, pitch, yaw)
        pose_cols = ['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw']
        pose_targets = bag_data[pose_cols].values.astype(np.float32)
        # Normalize pose data
        pose_targets = (pose_targets - self.pose_mean) / (self.pose_std + 1e-8)
        pose_tensor = torch.from_numpy(pose_targets)  # Shape: (seq_len, 6)
        
        return {
            'images': images,
            'imu': imu_tensor,
            'pose': pose_tensor,
            'timestamps': torch.from_numpy(bag_data['timestamp'].values.astype(np.float32))
        }


class CNNFeatureExtractor(nn.Module):
    """CNN for extracting visual features from images"""
    def __init__(self, feature_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        # Using ResNet-inspired architecture but lighter for underwater
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 3, H, W)
        batch_size, seq_len = x.shape[:2]
        x = x.view(-1, *x.shape[2:])  # (batch_size * seq_len, 3, H, W)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc(x))
        
        # Reshape back to sequence format
        x = x.view(batch_size, seq_len, -1)
        return x


class IMUEncoder(nn.Module):
    """Simple encoder for IMU data"""
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=256):
        super(IMUEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class UnderwaterVIOLSTM(nn.Module):
    """Main VIO model combining CNN visual features and IMU data"""
    def __init__(self, 
                 visual_feature_dim=512,
                 imu_feature_dim=256, 
                 lstm_hidden_dim=512,
                 lstm_layers=2,
                 output_dim=6):
        super(UnderwaterVIOLSTM, self).__init__()
        
        self.visual_encoder = CNNFeatureExtractor(visual_feature_dim)
        self.imu_encoder = IMUEncoder(output_dim=imu_feature_dim)
        
        # Fusion layer
        fusion_dim = visual_feature_dim + imu_feature_dim
        self.fusion_fc = nn.Linear(fusion_dim, lstm_hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )
        
        # Output layers for 6-DOF pose
        self.output_fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, images, imu_data):
        batch_size, seq_len = images.shape[:2]
        
        # Extract visual features
        visual_features = self.visual_encoder(images)  # (batch, seq_len, visual_dim)
        
        # Extract IMU features
        imu_features = self.imu_encoder(imu_data)  # (batch, seq_len, imu_dim)
        
        # Fuse features
        fused_features = torch.cat([visual_features, imu_features], dim=-1)
        fused_features = F.relu(self.fusion_fc(fused_features))
        
        # LSTM processing
        lstm_out, _ = self.lstm(fused_features)
        
        # Output prediction
        pose_predictions = self.output_fc(lstm_out)
        
        return pose_predictions


def create_model():
    """Factory function to create the model"""
    return UnderwaterVIOLSTM(
        visual_feature_dim=512,
        imu_feature_dim=256,
        lstm_hidden_dim=512,
        lstm_layers=2,
        output_dim=6
    )


if __name__ == "__main__":
    # Test dataset loading
    csv_path = "data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr_clean.csv"
    data_root = "data/processed/visual_odometry_dataset"
    
    dataset = UnderwaterVIODataset(csv_path, data_root, sequence_length=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Test model
    model = create_model()
    
    for batch in dataloader:
        images = batch['images']
        imu = batch['imu']
        pose = batch['pose']
        
        print(f"Images shape: {images.shape}")
        print(f"IMU shape: {imu.shape}")
        print(f"Pose shape: {pose.shape}")
        
        with torch.no_grad():
            predictions = model(images, imu)
            print(f"Predictions shape: {predictions.shape}")
        
        break