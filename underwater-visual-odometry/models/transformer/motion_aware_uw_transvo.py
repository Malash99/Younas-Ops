"""
Motion-Aware UW-TransVO: Modified architecture for frame-to-frame motion learning

This version outputs pose predictions for each frame in the sequence,
enabling supervision of relative motion between consecutive frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from .vision_transformer import VisionTransformer
from .multimodal_fusion import MultiModalFusion
from .pose_regression import PoseRegressionHead
from .uw_transvo import (
    UnderwaterImageEnhancement, 
    CameraPositionalEncoding,
    TemporalPositionalEncoding,
    SpatialCrossCameraAttention,
    TemporalSelfAttention
)


class SequentialPoseHead(nn.Module):
    """
    Pose regression head that outputs poses for each frame in sequence
    """
    
    def __init__(
        self,
        d_model: int,
        sequence_length: int = 5,
        uncertainty_estimation: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.uncertainty_estimation = uncertainty_estimation
        
        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Pose prediction heads for each frame
        self.pose_heads = nn.ModuleList([
            nn.Linear(d_model // 2, 6) for _ in range(sequence_length)
        ])
        
        if uncertainty_estimation:
            self.uncertainty_heads = nn.ModuleList([
                nn.Linear(d_model // 2, 6) for _ in range(sequence_length)
            ])
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Temporal features [batch_size, seq_len, d_model]
        Returns:
            Dictionary with poses and uncertainties for each frame
        """
        batch_size, seq_len = features.shape[:2]
        processed_features = self.feature_processor(features)
        
        # Predict pose for each frame
        poses = []
        uncertainties = [] if self.uncertainty_estimation else None
        
        for t in range(min(seq_len, self.sequence_length)):
            frame_features = processed_features[:, t]  # [batch_size, d_model//2]
            
            # Pose prediction
            pose = self.pose_heads[t](frame_features)
            poses.append(pose)
            
            # Uncertainty prediction
            if self.uncertainty_estimation:
                uncertainty = torch.exp(self.uncertainty_heads[t](frame_features))
                uncertainties.append(uncertainty)
        
        poses = torch.stack(poses, dim=1)  # [batch_size, seq_len, 6]
        
        output = {'pose': poses}
        if self.uncertainty_estimation:
            output['uncertainty'] = torch.stack(uncertainties, dim=1)
        
        return output


class MotionAwareUWTransVO(nn.Module):
    """
    Motion-Aware UW-TransVO that learns frame-to-frame motion patterns
    
    Key changes from original:
    1. Outputs poses for each frame in sequence (not just last frame)
    2. Uses temporal attention to model motion patterns
    3. Designed for motion-aware loss functions
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        max_cameras: int = 5,
        max_seq_len: int = 5,  # Reduced for motion learning
        dropout: float = 0.1,
        use_imu: bool = False,  # Disable for pure vision learning
        use_pressure: bool = False,
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
            for _ in range(2)
        ])
        
        # 5. Enhanced temporal self-attention for motion modeling
        self.temporal_attention_layers = nn.ModuleList([
            TemporalSelfAttention(d_model, num_heads, dropout)
            for _ in range(3)  # More layers for better temporal modeling
        ])
        
        # 6. Motion-specific feature enhancement
        self.motion_enhancement = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 7. Multi-modal fusion (optional)
        if use_imu or use_pressure:
            self.multimodal_fusion = MultiModalFusion(
                d_model=d_model,
                use_imu=use_imu,
                use_pressure=use_pressure,
                dropout=dropout
            )
        
        # 8. Sequential pose regression head
        self.pose_head = SequentialPoseHead(
            d_model=d_model,
            sequence_length=max_seq_len,
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
        Forward pass optimized for motion learning
        
        Args:
            images: [batch_size, seq_len, num_cameras, 3, H, W]
            camera_ids: [batch_size, num_cameras]
            camera_mask: [batch_size, num_cameras]
            imu_data: [batch_size, seq_len, 6] (optional)
            pressure_data: [batch_size, seq_len, 1] (optional)
            
        Returns:
            Dictionary containing:
            - pose: [batch_size, seq_len, 6] - pose for each frame
            - uncertainty: [batch_size, seq_len, 6] (if enabled)
        """
        batch_size, seq_len, num_cameras = images.shape[:3]
        
        # Reshape for processing
        images_flat = images.view(-1, *images.shape[-3:])
        
        # 1. Underwater image enhancement
        enhanced_images = self.image_enhancement(images_flat)
        
        # 2. Extract visual features
        visual_features = self.vision_transformer(enhanced_images)
        visual_features = visual_features.view(batch_size, seq_len, num_cameras, self.d_model)
        
        # 3. Add camera positional encoding
        camera_pos = self.camera_pos_encoding(camera_ids).unsqueeze(1)
        visual_features = visual_features + camera_pos
        
        # 4. Spatial cross-camera attention (for each time step)
        spatial_features = []
        for t in range(seq_len):
            step_features = visual_features[:, t]
            
            for spatial_layer in self.spatial_attention_layers:
                step_features = spatial_layer(step_features, camera_mask)
            
            # Aggregate across cameras (attention-weighted)
            camera_attention = F.softmax(
                torch.sum(step_features, dim=-1), dim=-1
            ).unsqueeze(-1)
            step_aggregated = torch.sum(step_features * camera_attention, dim=1)
            spatial_features.append(step_aggregated)
        
        # Stack temporal features
        temporal_features = torch.stack(spatial_features, dim=1)
        
        # 5. Add temporal positional encoding
        temporal_pos = self.temporal_pos_encoding(seq_len).unsqueeze(0)
        temporal_features = temporal_features + temporal_pos
        
        # 6. Enhanced temporal self-attention for motion modeling
        for temporal_layer in self.temporal_attention_layers:
            temporal_features = temporal_layer(temporal_features)
        
        # 7. Motion-specific feature enhancement
        motion_features = self.motion_enhancement(temporal_features)
        
        # 8. Multi-modal fusion (if enabled)
        if hasattr(self, 'multimodal_fusion'):
            # Apply fusion to each frame
            fused_features = []
            for t in range(seq_len):
                frame_imu = imu_data[:, t:t+1] if imu_data is not None else None
                frame_pressure = pressure_data[:, t:t+1] if pressure_data is not None else None
                
                fused_frame = self.multimodal_fusion(
                    visual_features=motion_features[:, t],
                    imu_data=frame_imu,
                    pressure_data=frame_pressure
                )
                fused_features.append(fused_frame)
            
            motion_features = torch.stack(fused_features, dim=1)
        
        # 9. Sequential pose regression
        output = self.pose_head(motion_features)
        
        return output
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_motion_aware_model(config: Dict) -> MotionAwareUWTransVO:
    """Factory function to create motion-aware UW-TransVO model"""
    return MotionAwareUWTransVO(
        img_size=config.get('img_size', 224),
        patch_size=config.get('patch_size', 16),
        d_model=config.get('d_model', 512),  # Reduced for faster training
        num_heads=config.get('num_heads', 8),  # Reduced for faster training
        num_layers=config.get('num_layers', 4),  # Reduced for faster training
        max_cameras=config.get('max_cameras', 1),  # Start with single camera
        max_seq_len=config.get('max_seq_len', 5),
        dropout=config.get('dropout', 0.1),
        use_imu=config.get('use_imu', False),
        use_pressure=config.get('use_pressure', False),
        uncertainty_estimation=config.get('uncertainty_estimation', True)
    )