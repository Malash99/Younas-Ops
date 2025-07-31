"""
UW-TransVO: Scalable Multi-Camera Underwater Visual Odometry with Transformers

Main architecture that combines:
1. Underwater image enhancement
2. Multi-camera feature extraction  
3. Spatial cross-camera attention
4. Temporal self-attention
5. Multi-modal sensor fusion
6. 6-DOF pose regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from .vision_transformer import VisionTransformer
from .multimodal_fusion import MultiModalFusion
from .pose_regression import PoseRegressionHead


class UnderwaterImageEnhancement(nn.Module):
    """Underwater-specific image enhancement module"""
    
    def __init__(self, channels: int = 3):
        super().__init__()
        
        # Color correction layers
        self.color_correction = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        
        # Contrast enhancement
        self.contrast_enhancement = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        # Noise reduction
        self.noise_reduction = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [batch_size, channels, height, width]
        Returns:
            Enhanced images [batch_size, channels, height, width]
        """
        # Color correction
        color_corrected = self.color_correction(x) + x
        
        # Contrast enhancement
        contrast_weights = self.contrast_enhancement(color_corrected)
        contrast_enhanced = color_corrected * contrast_weights
        
        # Noise reduction
        noise_reduced = self.noise_reduction(contrast_enhanced) + contrast_enhanced
        
        return noise_reduced


class CameraPositionalEncoding(nn.Module):
    """Positional encoding for different cameras"""
    
    def __init__(self, d_model: int, max_cameras: int = 5):
        super().__init__()
        self.d_model = d_model
        self.max_cameras = max_cameras
        
        # Create learnable positional embeddings for each camera
        self.camera_embeddings = nn.Parameter(torch.randn(max_cameras, d_model))
        
    def forward(self, camera_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_ids: Camera indices [batch_size, num_cameras]
        Returns:
            Camera positional encodings [batch_size, num_cameras, d_model]
        """
        batch_size, num_cameras = camera_ids.shape
        
        # Get embeddings for specified camera IDs
        embeddings = self.camera_embeddings[camera_ids]  # [batch_size, num_cameras, d_model]
        
        return embeddings


class TemporalPositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_seq_len: int = 10):
        super().__init__()
        self.d_model = d_model
        
        # Create sinusoidal positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: Sequence length
        Returns:
            Temporal positional encodings [seq_len, d_model]
        """
        return self.pe[:seq_len]


class SpatialCrossCameraAttention(nn.Module):
    """Cross-attention between simultaneous camera views"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor, camera_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: Camera features [batch_size, num_cameras, d_model]
            camera_mask: Mask for missing cameras [batch_size, num_cameras]
        Returns:
            Attended features [batch_size, num_cameras, d_model]
        """
        # Self-attention across cameras
        attn_output, attn_weights = self.multihead_attn(
            features, features, features, 
            key_padding_mask=camera_mask
        )
        
        # Residual connection and normalization
        output = self.norm(features + self.dropout(attn_output))
        
        return output


class TemporalSelfAttention(nn.Module):
    """Self-attention across temporal sequence"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Temporal features [batch_size, seq_len, d_model]
        Returns:
            Attended features [batch_size, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.multihead_attn(features, features, features)
        features = self.norm(features + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(features)
        output = self.norm2(features + ffn_output)
        
        return output


class UWTransVO(nn.Module):
    """
    UW-TransVO: Scalable Multi-Camera Underwater Visual Odometry
    
    Architecture:
    1. Underwater image enhancement
    2. Multi-camera feature extraction
    3. Spatial cross-camera attention  
    4. Temporal self-attention
    5. Multi-modal sensor fusion
    6. 6-DOF pose regression
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        max_cameras: int = 5,
        max_seq_len: int = 10,
        dropout: float = 0.1,
        use_imu: bool = True,
        use_pressure: bool = True,
        uncertainty_estimation: bool = True
    ):
        super().__init__()
        
        self.max_cameras = max_cameras
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.use_imu = use_imu
        self.use_pressure = use_pressure
        self.uncertainty_estimation = uncertainty_estimation
        
        # 1. Underwater image enhancement
        self.image_enhancement = UnderwaterImageEnhancement()
        
        # 2. Vision transformer for feature extraction
        self.vision_transformer = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 3. Positional encodings
        self.camera_pos_encoding = CameraPositionalEncoding(d_model, max_cameras)
        self.temporal_pos_encoding = TemporalPositionalEncoding(d_model, max_seq_len)
        
        # 4. Spatial cross-camera attention
        self.spatial_attention_layers = nn.ModuleList([
            SpatialCrossCameraAttention(d_model, num_heads, dropout)
            for _ in range(2)  # 2 layers of spatial attention
        ])
        
        # 5. Temporal self-attention
        self.temporal_attention_layers = nn.ModuleList([
            TemporalSelfAttention(d_model, num_heads, dropout)
            for _ in range(2)  # 2 layers of temporal attention
        ])
        
        # 6. Multi-modal fusion
        self.multimodal_fusion = MultiModalFusion(
            d_model=d_model,
            use_imu=use_imu,
            use_pressure=use_pressure,
            dropout=dropout
        )
        
        # 7. Pose regression head
        self.pose_head = PoseRegressionHead(
            d_model=d_model,
            uncertainty_estimation=uncertainty_estimation,
            dropout=dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
            
    def forward(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        camera_mask: Optional[torch.Tensor] = None,
        imu_data: Optional[torch.Tensor] = None,
        pressure_data: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of UW-TransVO
        
        Args:
            images: Input images [batch_size, seq_len, num_cameras, 3, H, W]
            camera_ids: Camera indices [batch_size, num_cameras] 
            camera_mask: Mask for missing cameras [batch_size, num_cameras]
            imu_data: IMU data [batch_size, seq_len, 6] (accel + gyro)
            pressure_data: Pressure data [batch_size, seq_len, 1]
            
        Returns:
            Dictionary containing:
            - pose: 6-DOF pose [batch_size, 6] (tx, ty, tz, rx, ry, rz)
            - uncertainty: Pose uncertainty [batch_size, 6] (if enabled)
        """
        batch_size, seq_len, num_cameras = images.shape[:3]
        
        # Reshape for processing: [batch_size * seq_len * num_cameras, 3, H, W]
        images_flat = images.view(-1, *images.shape[-3:])
        
        # 1. Underwater image enhancement
        enhanced_images = self.image_enhancement(images_flat)
        
        # 2. Extract visual features using vision transformer
        visual_features = self.vision_transformer(enhanced_images)  # [batch*seq*cam, d_model]
        
        # Reshape back to sequence structure
        visual_features = visual_features.view(batch_size, seq_len, num_cameras, self.d_model)
        
        # 3. Add camera positional encoding
        camera_pos = self.camera_pos_encoding(camera_ids).unsqueeze(1)  # [batch, 1, num_cameras, d_model]
        visual_features = visual_features + camera_pos
        
        # 4. Spatial cross-camera attention (for each time step)
        spatial_features = []
        for t in range(seq_len):
            step_features = visual_features[:, t]  # [batch_size, num_cameras, d_model]
            
            # Apply spatial attention layers
            for spatial_layer in self.spatial_attention_layers:
                step_features = spatial_layer(step_features, camera_mask)
            
            # Aggregate across cameras (mean pooling)
            step_aggregated = step_features.mean(dim=1)  # [batch_size, d_model]
            spatial_features.append(step_aggregated)
        
        # Stack temporal features
        temporal_features = torch.stack(spatial_features, dim=1)  # [batch_size, seq_len, d_model]
        
        # 5. Add temporal positional encoding
        temporal_pos = self.temporal_pos_encoding(seq_len).unsqueeze(0)  # [1, seq_len, d_model]
        temporal_features = temporal_features + temporal_pos
        
        # 6. Temporal self-attention
        for temporal_layer in self.temporal_attention_layers:
            temporal_features = temporal_layer(temporal_features)
        
        # Aggregate temporal features (use the last time step)
        aggregated_features = temporal_features[:, -1]  # [batch_size, d_model]
        
        # 7. Multi-modal fusion
        fused_features = self.multimodal_fusion(
            visual_features=aggregated_features,
            imu_data=imu_data,
            pressure_data=pressure_data
        )
        
        # 8. Pose regression
        output = self.pose_head(fused_features)
        
        return output
    
    def get_attention_weights(self, layer_idx: int = -1) -> torch.Tensor:
        """Get attention weights for visualization"""
        # This would be implemented to extract attention weights
        # for interpretability analysis
        pass
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_uw_transvo_model(config: Dict) -> UWTransVO:
    """Factory function to create UW-TransVO model from config"""
    return UWTransVO(
        img_size=config.get('img_size', 224),
        patch_size=config.get('patch_size', 16),
        d_model=config.get('d_model', 768),
        num_heads=config.get('num_heads', 12),
        num_layers=config.get('num_layers', 6),
        max_cameras=config.get('max_cameras', 5),
        max_seq_len=config.get('max_seq_len', 10),
        dropout=config.get('dropout', 0.1),
        use_imu=config.get('use_imu', True),
        use_pressure=config.get('use_pressure', True),
        uncertainty_estimation=config.get('uncertainty_estimation', True)
    )