"""
Multi-Scale UW-TransVO Architecture

Enhanced UW-TransVO that predicts poses at multiple temporal scales:
- Fine-grained: Frame-to-frame deltas
- Medium-scale: 5-frame accumulated motions  
- Long-term: 20-frame trajectory segments

This architecture is designed to prevent straight-line predictions by explicitly
modeling trajectory shape at different temporal resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from .vision_transformer import VisionTransformer
from .multimodal_fusion import MultiModalFusion
from .uw_transvo import (
    UnderwaterImageEnhancement, 
    CameraPositionalEncoding,
    TemporalPositionalEncoding,
    SpatialCrossCameraAttention,
    TemporalSelfAttention
)


class MultiScalePoseHead(nn.Module):
    """
    Multi-scale pose regression head that outputs predictions at different temporal scales
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 20,
        uncertainty_estimation: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.uncertainty_estimation = uncertainty_estimation
        
        # Shared feature processing
        self.shared_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        
        # Scale-specific feature processors
        self.delta_processor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.short_processor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.long_processor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(), 
            nn.Dropout(dropout)
        )
        
        # Scale-specific pose prediction heads
        self.delta_head = nn.Linear(d_model // 2, 6)  # Frame-to-frame deltas
        self.short_head = nn.Linear(d_model // 2, 6)  # 5-frame accumulated
        self.long_head = nn.Linear(d_model // 2, 6)   # 20-frame accumulated
        
        # NEW: Scale prediction heads for fixing magnitude issues
        self.delta_scale_head = nn.Linear(d_model // 2, 1)  # Scale factor for delta poses
        self.short_scale_head = nn.Linear(d_model // 2, 1)  # Scale factor for short poses
        self.long_scale_head = nn.Linear(d_model // 2, 1)   # Scale factor for long poses
        
        # Uncertainty heads (optional)
        if uncertainty_estimation:
            self.delta_uncertainty = nn.Linear(d_model // 2, 6)
            self.short_uncertainty = nn.Linear(d_model // 2, 6)
            self.long_uncertainty = nn.Linear(d_model // 2, 6)
        
        # Cross-scale fusion for final prediction
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model // 2 * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2)
        )
        
        self.final_pose_head = nn.Linear(d_model // 2, 6)
        if uncertainty_estimation:
            self.final_uncertainty_head = nn.Linear(d_model // 2, 6)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Temporal features [batch_size, seq_len, d_model]
        Returns:
            Dictionary with multi-scale pose predictions
        """
        batch_size, seq_len = features.shape[:2]
        
        # Shared processing
        processed_features = self.shared_processor(features)
        
        # Multi-scale predictions
        predictions = {}
        
        # Scale 1: Frame-to-frame deltas (for each frame)
        delta_features = self.delta_processor(processed_features)
        delta_poses = self.delta_head(delta_features)  # [batch, seq_len, 6]
        
        # NEW: Predict scale factors and apply to translation components
        delta_scales = torch.exp(self.delta_scale_head(delta_features)).squeeze(-1)  # [batch, seq_len]
        scaled_delta_poses = delta_poses.clone()
        scaled_delta_poses[..., :3] *= delta_scales.unsqueeze(-1)  # Apply scale to XYZ
        
        predictions['delta_poses'] = scaled_delta_poses
        predictions['delta_scales'] = delta_scales
        predictions['delta_poses_unscaled'] = delta_poses  # Keep original for debugging
        
        if self.uncertainty_estimation:
            delta_uncertainty = torch.exp(self.delta_uncertainty(delta_features))
            predictions['delta_uncertainty'] = delta_uncertainty
        
        # Scale 2: Short-term accumulated (5-frame windows) - use middle frames
        if seq_len >= 5:
            # Use features from middle of each 5-frame window
            short_indices = torch.arange(2, seq_len-2, device=features.device)  # Middle frames
            if len(short_indices) > 0:
                short_features = processed_features[:, short_indices]
                short_processed = self.short_processor(short_features)
                short_poses = self.short_head(short_processed)
                
                # NEW: Apply scale factors to short-term poses
                short_scales = torch.exp(self.short_scale_head(short_processed)).squeeze(-1)
                scaled_short_poses = short_poses.clone()
                scaled_short_poses[..., :3] *= short_scales.unsqueeze(-1)
                
                predictions['short_poses'] = scaled_short_poses
                predictions['short_scales'] = short_scales
                predictions['short_poses_unscaled'] = short_poses
                
                if self.uncertainty_estimation:
                    short_uncertainty = torch.exp(self.short_uncertainty(short_processed))
                    predictions['short_uncertainty'] = short_uncertainty
        
        # Scale 3: Long-term accumulated (use global average of all features)
        if seq_len >= 10:  # Need reasonable sequence length
            # Global temporal pooling for long-term prediction
            long_features = processed_features.mean(dim=1)  # [batch, d_model]
            long_processed = self.long_processor(long_features)
            long_poses = self.long_head(long_processed)  # [batch, 6]
            
            # NEW: Apply scale factors to long-term poses
            long_scales = torch.exp(self.long_scale_head(long_processed)).squeeze(-1)  # [batch]
            scaled_long_poses = long_poses.clone()
            scaled_long_poses[..., :3] *= long_scales.unsqueeze(-1)
            
            predictions['long_poses'] = scaled_long_poses
            predictions['long_scales'] = long_scales
            predictions['long_poses_unscaled'] = long_poses
            
            if self.uncertainty_estimation:
                long_uncertainty = torch.exp(self.long_uncertainty(long_processed))
                predictions['long_uncertainty'] = long_uncertainty
        
        # Cross-scale fusion for primary prediction
        # Use last frame features for final prediction
        final_features = processed_features[:, -1]  # [batch, d_model]
        
        # Collect all scale features for fusion
        fusion_input_list = [self.delta_processor(final_features)]
        
        if seq_len >= 5:
            fusion_input_list.append(self.short_processor(final_features))
        else:
            fusion_input_list.append(torch.zeros_like(fusion_input_list[0]))
            
        if seq_len >= 10:
            fusion_input_list.append(self.long_processor(final_features))
        else:
            fusion_input_list.append(torch.zeros_like(fusion_input_list[0]))
        
        fusion_input = torch.cat(fusion_input_list, dim=-1)  # [batch, d_model//2 * 3]
        fused_features = self.fusion_layer(fusion_input)
        
        # Final pose prediction (primary output)
        final_pose = self.final_pose_head(fused_features)
        predictions['pose'] = final_pose
        
        if self.uncertainty_estimation:
            final_uncertainty = torch.exp(self.final_uncertainty_head(fused_features))
            predictions['uncertainty'] = final_uncertainty
        
        return predictions


class MultiScaleUWTransVO(nn.Module):
    """
    Multi-Scale UW-TransVO for trajectory-aware underwater visual odometry
    
    Key features:
    - Multi-scale temporal modeling (frame, short-term, long-term)
    - Enhanced temporal attention for trajectory understanding
    - Cross-scale feature fusion
    - Designed to prevent straight-line predictions
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_cameras: int = 1,
        max_seq_len: int = 20,
        dropout: float = 0.1,
        use_imu: bool = False,
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
        
        # 5. Enhanced temporal self-attention for multi-scale modeling
        self.temporal_attention_layers = nn.ModuleList([
            TemporalSelfAttention(d_model, num_heads, dropout)
            for _ in range(4)  # More layers for better temporal modeling
        ])
        
        # 6. Multi-scale trajectory modeling layers
        self.trajectory_modeling = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='relu',
                batch_first=True
            ) for _ in range(2)
        ])
        
        # 7. Multi-modal fusion (optional)
        if use_imu or use_pressure:
            self.multimodal_fusion = MultiModalFusion(
                d_model=d_model,
                use_imu=use_imu,
                use_pressure=use_pressure,
                dropout=dropout
            )
        
        # 8. Multi-scale pose regression head
        self.pose_head = MultiScalePoseHead(
            d_model=d_model,
            max_seq_len=max_seq_len,
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
        Forward pass with multi-scale trajectory modeling
        
        Args:
            images: [batch_size, seq_len, num_cameras, 3, H, W]
            camera_ids: [batch_size, num_cameras]
            camera_mask: [batch_size, num_cameras]
            imu_data: [batch_size, seq_len, 6] (optional)
            pressure_data: [batch_size, seq_len, 1] (optional)
            
        Returns:
            Dictionary containing multi-scale predictions:
            - pose: [batch_size, 6] - primary pose prediction
            - delta_poses: [batch_size, seq_len, 6] - frame-to-frame deltas
            - short_poses: [batch_size, num_short_windows, 6] - 5-frame accumulated
            - long_poses: [batch_size, 6] - full sequence accumulated
            - uncertainty: uncertainty estimates (if enabled)
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
            
            # Weighted aggregation across cameras
            if num_cameras > 1:
                camera_attention = F.softmax(
                    torch.sum(step_features, dim=-1), dim=-1
                ).unsqueeze(-1)
                step_aggregated = torch.sum(step_features * camera_attention, dim=1)
            else:
                step_aggregated = step_features.squeeze(1)
                
            spatial_features.append(step_aggregated)
        
        # Stack temporal features
        temporal_features = torch.stack(spatial_features, dim=1)  # [batch, seq_len, d_model]
        
        # 5. Add temporal positional encoding
        temporal_pos = self.temporal_pos_encoding(seq_len).unsqueeze(0)
        temporal_features = temporal_features + temporal_pos
        
        # 6. Enhanced temporal self-attention for multi-scale modeling
        for temporal_layer in self.temporal_attention_layers:
            temporal_features = temporal_layer(temporal_features)
        
        # 7. Multi-scale trajectory modeling
        trajectory_features = temporal_features
        for traj_layer in self.trajectory_modeling:
            trajectory_features = traj_layer(trajectory_features)
        
        # 8. Multi-modal fusion (if enabled)
        if hasattr(self, 'multimodal_fusion'):
            # Apply fusion to each frame
            fused_features = []
            for t in range(seq_len):
                frame_imu = imu_data[:, t:t+1] if imu_data is not None else None
                frame_pressure = pressure_data[:, t:t+1] if pressure_data is not None else None
                
                fused_frame = self.multimodal_fusion(
                    visual_features=trajectory_features[:, t],
                    imu_data=frame_imu,
                    pressure_data=frame_pressure
                )
                fused_features.append(fused_frame)
            
            trajectory_features = torch.stack(fused_features, dim=1)
        
        # 9. Multi-scale pose regression
        outputs = self.pose_head(trajectory_features)
        
        return outputs
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_multiscale_model(config: Dict) -> MultiScaleUWTransVO:
    """Factory function to create multi-scale UW-TransVO model"""
    return MultiScaleUWTransVO(
        img_size=config.get('img_size', 224),
        patch_size=config.get('patch_size', 16),
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 6),
        max_cameras=config.get('max_cameras', 1),
        max_seq_len=config.get('max_seq_len', 20),
        dropout=config.get('dropout', 0.1),
        use_imu=config.get('use_imu', False),
        use_pressure=config.get('use_pressure', False),
        uncertainty_estimation=config.get('uncertainty_estimation', True)
    )