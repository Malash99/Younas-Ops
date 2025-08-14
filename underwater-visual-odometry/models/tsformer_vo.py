"""
TSformer Visual Odometry Model

Implementation of transformer-based visual odometry using Vision Transformer (ViT)
backbone with transfer learning for underwater ROV navigation.

Author: Underwater Visual Odometry Research Team
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TSformerVO(nn.Module):
    """
    Transformer-based Visual Odometry Model
    
    Architecture:
    1. ViT backbone (pretrained on ImageNet) extracts spatial features
    2. Temporal transformer processes sequence of features
    3. Pose regression head predicts 6-DOF pose deltas
    """
    
    def __init__(self, 
                 sequence_length=8,
                 pretrained_model="google/vit-base-patch16-224",
                 hidden_dim=768,
                 num_transformer_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 freeze_backbone=False,
                 image_size=224):
        super(TSformerVO, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # 1. Vision Transformer Backbone (Transfer Learning)
        print(f"Loading pretrained ViT: {pretrained_model}")
        self.vit = ViTModel.from_pretrained(pretrained_model)
        
        # Handle different image sizes by enabling interpolation
        if image_size != 224:
            print(f"Warning: Using image size {image_size} with ViT trained on 224. Position embeddings will be interpolated.")
            self.interpolate_pos_encoding = True
        else:
            self.interpolate_pos_encoding = False
        
        # Optionally freeze backbone
        if freeze_backbone:
            print("Freezing ViT backbone parameters")
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Get ViT output dimension
        vit_hidden_size = self.vit.config.hidden_size
        
        # 2. Feature projection (if needed)
        if vit_hidden_size != hidden_dim:
            self.feature_projection = nn.Linear(vit_hidden_size, hidden_dim)
        else:
            self.feature_projection = nn.Identity()
        
        # 3. Temporal positional encoding (sequence + 1 for CLS token)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=sequence_length + 2)
        
        # 4. Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # 5. Pose Regression Head
        self.pose_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6)  # [dx, dy, dz, droll, dpitch, dyaw]
        )
        
        # 6. Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        print(f"TSformer-VO initialized:")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Hidden dimension: {hidden_dim}")
        print(f"  - Transformer layers: {num_transformer_layers}")
        print(f"  - Attention heads: {num_heads}")
        print(f"  - Backbone frozen: {freeze_backbone}")
    
    def forward(self, image_sequence):
        """
        Forward pass of TSformer-VO
        
        Args:
            image_sequence: (batch_size, sequence_length, 3, H, W)
            
        Returns:
            pose_deltas: (batch_size, 6) - [dx, dy, dz, droll, dpitch, dyaw]
        """
        batch_size, seq_len, channels, height, width = image_sequence.shape
        
        # 1. Extract spatial features using ViT
        # Reshape to process all frames at once
        images = image_sequence.view(-1, channels, height, width)  # (B*T, C, H, W)
        
        # Extract features from ViT
        if self.interpolate_pos_encoding:
            vit_outputs = self.vit(pixel_values=images, interpolate_pos_encoding=True)
        else:
            vit_outputs = self.vit(pixel_values=images)
        spatial_features = vit_outputs.last_hidden_state[:, 0, :]  # Use CLS token (B*T, hidden_size)
        
        # Project features if needed
        spatial_features = self.feature_projection(spatial_features)  # (B*T, hidden_dim)
        
        # 2. Reshape back to sequence format
        spatial_features = spatial_features.view(batch_size, seq_len, self.hidden_dim)  # (B, T, hidden_dim)
        
        # 3. Add CLS token for sequence-level representation
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, hidden_dim)
        sequence_features = torch.cat([cls_tokens, spatial_features], dim=1)  # (B, T+1, hidden_dim)
        
        # 4. Add positional encoding
        sequence_features = sequence_features.transpose(0, 1)  # (T+1, B, hidden_dim)
        sequence_features = self.pos_encoding(sequence_features)
        sequence_features = sequence_features.transpose(0, 1)  # (B, T+1, hidden_dim)
        
        # 5. Temporal transformer processing
        temporal_features = self.temporal_transformer(sequence_features)  # (B, T+1, hidden_dim)
        
        # 6. Use CLS token for final prediction
        sequence_representation = temporal_features[:, 0, :]  # (B, hidden_dim)
        
        # 7. Pose regression
        pose_deltas = self.pose_head(sequence_representation)  # (B, 6)
        
        return pose_deltas
    
    def get_num_parameters(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SE3GeometricLoss(nn.Module):
    """
    SE(3) Geometric Loss for Visual Odometry
    
    Uses proper SE(3) manifold geometry for pose regression.
    Prevents straight-line predictions through geometric constraints.
    """
    
    def __init__(self, geodesic_weight=1.0, consistency_weight=0.5, magnitude_weight=0.3):
        super().__init__()
        self.geodesic_weight = geodesic_weight
        self.consistency_weight = consistency_weight 
        self.magnitude_weight = magnitude_weight
        
        # Small epsilon for numerical stability
        self.eps = 1e-8
        
    def forward(self, pred_poses, gt_poses):
        """
        Compute SE(3) geometric loss.
        
        Args:
            pred_poses: (batch_size, 6) - [dx,dy,dz,droll,dpitch,dyaw] 
            gt_poses: (batch_size, 6) - [dx,dy,dz,droll,dpitch,dyaw]
            
        Returns:
            loss: SE(3) geometric loss
            loss_dict: Dictionary with loss components
        """
        batch_size = pred_poses.shape[0]
        device = pred_poses.device
        
        # 1. SE(3) Geodesic Distance Loss
        geodesic_loss = self.se3_geodesic_loss(pred_poses, gt_poses)
        
        # 2. SE(3) Chain Consistency Loss (for sequences)
        consistency_loss = torch.tensor(0.0, device=device)
        if batch_size >= 2:
            consistency_loss = self.se3_chain_consistency_loss(pred_poses, gt_poses)
        
        # 3. Motion Magnitude Loss (prevents zero motion)
        magnitude_loss = self.motion_magnitude_loss(pred_poses, gt_poses)
        
        # Combined loss
        total_loss = (self.geodesic_weight * geodesic_loss + 
                     self.consistency_weight * consistency_loss +
                     self.magnitude_weight * magnitude_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'geodesic_loss': geodesic_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'magnitude_loss': magnitude_loss.item()
        }
        
        return total_loss, loss_dict
    
    def se3_geodesic_loss(self, pred_poses, gt_poses):
        """
        Compute geodesic distance on SE(3) manifold.
        
        Uses the Frobenius norm of the logarithm of the relative transformation.
        """
        # Convert 6DoF to SE(3) matrices
        pred_T = self.pose_to_se3(pred_poses)  # (B, 4, 4)
        gt_T = self.pose_to_se3(gt_poses)      # (B, 4, 4)
        
        # Compute relative transformation: T_rel = T_pred^{-1} * T_gt
        pred_T_inv = self.se3_inverse(pred_T)
        T_rel = torch.bmm(pred_T_inv, gt_T)  # (B, 4, 4)
        
        # Compute geodesic distance using matrix logarithm
        geodesic_dist = self.se3_log_frobenius_norm(T_rel)
        
        return geodesic_dist.mean()
    
    def se3_chain_consistency_loss(self, pred_poses, gt_poses):
        """
        Enforce SE(3) chain consistency: T_01 * T_12 = T_02
        """
        if pred_poses.shape[0] < 2:
            return torch.tensor(0.0, device=pred_poses.device)
        
        consistency_losses = []
        
        # Check consistency for all possible pairs
        for i in range(pred_poses.shape[0] - 1):
            # Single step transformation
            T_step_pred = self.pose_to_se3(pred_poses[i:i+1])  # T_i
            T_step_gt = self.pose_to_se3(gt_poses[i:i+1])
            
            # Next step
            T_next_pred = self.pose_to_se3(pred_poses[i+1:i+2])  # T_{i+1} 
            T_next_gt = self.pose_to_se3(gt_poses[i+1:i+2])
            
            # Chain composition vs direct prediction
            T_composed_pred = torch.bmm(T_step_pred, T_next_pred)  # T_i * T_{i+1}
            T_composed_gt = torch.bmm(T_step_gt, T_next_gt)
            
            # Compute relative error
            T_composed_inv = self.se3_inverse(T_composed_pred)
            T_rel = torch.bmm(T_composed_inv, T_composed_gt)
            
            consistency_dist = self.se3_log_frobenius_norm(T_rel)
            consistency_losses.append(consistency_dist)
        
        if consistency_losses:
            return torch.stack(consistency_losses).mean()
        else:
            return torch.tensor(0.0, device=pred_poses.device)
    
    def motion_magnitude_loss(self, pred_poses, gt_poses):
        """
        Prevent zero motion predictions while respecting SE(3) geometry.
        """
        # Translation magnitude
        pred_trans_mag = torch.norm(pred_poses[:, :3], dim=1)
        gt_trans_mag = torch.norm(gt_poses[:, :3], dim=1)
        trans_mag_loss = F.mse_loss(pred_trans_mag, gt_trans_mag)
        
        # Rotation magnitude (angle of rotation)
        pred_rot_mag = torch.norm(pred_poses[:, 3:], dim=1)
        gt_rot_mag = torch.norm(gt_poses[:, 3:], dim=1)
        rot_mag_loss = F.mse_loss(pred_rot_mag, gt_rot_mag)
        
        # Penalty for very small motions (prevents degeneracy)
        small_motion_penalty = torch.exp(-pred_trans_mag - pred_rot_mag).mean()
        
        return trans_mag_loss + rot_mag_loss + 0.1 * small_motion_penalty
    
    def pose_to_se3(self, poses):
        """
        Convert 6DoF pose to SE(3) transformation matrix.
        
        Args:
            poses: (B, 6) - [dx, dy, dz, droll, dpitch, dyaw]
            
        Returns:
            T: (B, 4, 4) - SE(3) transformation matrices
        """
        batch_size = poses.shape[0]
        device = poses.device
        
        # Extract translation and rotation
        translation = poses[:, :3]  # (B, 3)
        rotation = poses[:, 3:]     # (B, 3) - Euler angles
        
        # Convert Euler angles to rotation matrices
        R = self.euler_to_rotation_matrix(rotation)  # (B, 3, 3)
        
        # Create SE(3) matrices
        T = torch.zeros(batch_size, 4, 4, device=device)
        T[:, :3, :3] = R
        T[:, :3, 3] = translation
        T[:, 3, 3] = 1.0
        
        return T
    
    def euler_to_rotation_matrix(self, euler_angles):
        """
        Convert Euler angles (roll, pitch, yaw) to rotation matrices.
        
        Args:
            euler_angles: (B, 3) - [roll, pitch, yaw] in radians
            
        Returns:
            R: (B, 3, 3) - Rotation matrices
        """
        batch_size = euler_angles.shape[0]
        device = euler_angles.device
        
        roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
        
        # Compute trigonometric values
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch) 
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        
        # Construct rotation matrix (ZYX convention)
        R = torch.zeros(batch_size, 3, 3, device=device)
        
        R[:, 0, 0] = cos_y * cos_p
        R[:, 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
        R[:, 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
        
        R[:, 1, 0] = sin_y * cos_p
        R[:, 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
        R[:, 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
        
        R[:, 2, 0] = -sin_p
        R[:, 2, 1] = cos_p * sin_r
        R[:, 2, 2] = cos_p * cos_r
        
        return R
    
    def se3_inverse(self, T):
        """
        Compute inverse of SE(3) transformation matrices.
        
        Args:
            T: (B, 4, 4) - SE(3) matrices
            
        Returns:
            T_inv: (B, 4, 4) - Inverse SE(3) matrices
        """
        batch_size = T.shape[0]
        device = T.device
        
        # Extract rotation and translation
        R = T[:, :3, :3]  # (B, 3, 3)
        t = T[:, :3, 3]   # (B, 3)
        
        # Inverse: R^T and -R^T * t
        R_inv = R.transpose(-2, -1)  # (B, 3, 3)
        t_inv = -torch.bmm(R_inv, t.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        
        # Construct inverse matrix
        T_inv = torch.zeros(batch_size, 4, 4, device=device)
        T_inv[:, :3, :3] = R_inv
        T_inv[:, :3, 3] = t_inv
        T_inv[:, 3, 3] = 1.0
        
        return T_inv
    
    def se3_log_frobenius_norm(self, T):
        """
        Compute Frobenius norm of SE(3) matrix logarithm.
        
        This gives the geodesic distance on the SE(3) manifold.
        
        Args:
            T: (B, 4, 4) - SE(3) matrices
            
        Returns:
            dist: (B,) - Geodesic distances
        """
        batch_size = T.shape[0]
        device = T.device
        
        # Extract rotation and translation
        R = T[:, :3, :3]  # (B, 3, 3)
        t = T[:, :3, 3]   # (B, 3)
        
        # Compute rotation angle using trace
        trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)  # (B,)
        cos_angle = (trace_R - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1 + self.eps, 1 - self.eps)
        angle = torch.arccos(cos_angle)  # (B,)
        
        # Handle small angles (linearization near identity)
        small_angle_mask = angle < self.eps
        
        # For small angles, use linearized version
        geodesic_dist = torch.zeros(batch_size, device=device)
        
        # Small angle case: ||log(T)||_F ≈ ||t||_2 + ||skew(R-I)||_F
        if small_angle_mask.any():
            t_small = t[small_angle_mask]
            R_small = R[small_angle_mask]
            
            # Skew-symmetric part of (R - I)
            R_minus_I = R_small - torch.eye(3, device=device).unsqueeze(0)
            skew_norm = torch.norm(R_minus_I, dim=(-2, -1))  # Frobenius norm
            
            dist_small = torch.norm(t_small, dim=-1) + skew_norm
            geodesic_dist[small_angle_mask] = dist_small
        
        # Large angle case: full SE(3) logarithm
        large_angle_mask = ~small_angle_mask
        if large_angle_mask.any():
            t_large = t[large_angle_mask]
            angle_large = angle[large_angle_mask]
            
            # Translation part scaled by sinc function
            sinc_val = torch.sin(angle_large) / (angle_large + self.eps)
            scale_factor = angle_large / (2 * sinc_val + self.eps)
            
            t_scaled = t_large * scale_factor.unsqueeze(-1)
            
            # Rotation part contribution
            rot_contrib = angle_large
            
            # Combined distance
            dist_large = torch.sqrt(torch.norm(t_scaled, dim=-1)**2 + rot_contrib**2)
            geodesic_dist[large_angle_mask] = dist_large
        
        return geodesic_dist


def create_tsformer_vo(sequence_length=8, pretrained=True, freeze_backbone=False, image_size=224, use_multi_scale_loss=True):
    """
    Factory function to create TSformer-VO model with advanced loss function.
    
    Args:
        sequence_length: Number of frames in input sequence
        pretrained: Use pretrained ViT backbone  
        freeze_backbone: Freeze ViT parameters
        image_size: Input image resolution
        use_multi_scale_loss: Use multi-scale SE(3) loss (recommended for better scale matching)
        
    Returns:
        model: TSformerVO instance
        loss_fn: Loss function instance
    """
    if pretrained:
        model_name = "google/vit-base-patch16-224"
    else:
        # Create from scratch (not recommended)
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        model_name = config
    
    model = TSformerVO(
        sequence_length=sequence_length,
        pretrained_model=model_name,
        freeze_backbone=freeze_backbone,
        image_size=image_size
    )
    
    # Choose loss function
    if use_multi_scale_loss:
        # Use multi-scale SE(3) loss for better trajectory scale matching
        from .multi_scale_se3_loss import create_multi_scale_loss
        loss_fn = create_multi_scale_loss(
            single_step_weight=1.0,      # λ₁: Local accuracy
            multi_step_weight=2.0,       # λ₂: Trajectory scale (higher!)
            chain_consistency_weight=0.5, # λ₃: Geometric consistency
            sequence_length=sequence_length
        )
        print("Using MultiScaleSE3Loss for improved trajectory scale matching!")
    else:
        # Use original SE(3) geometric loss
        loss_fn = SE3GeometricLoss(
            geodesic_weight=1.0,
            consistency_weight=0.5, 
            magnitude_weight=0.3
        )
        print("Using original SE3GeometricLoss")
    
    return model, loss_fn


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing TSformer-VO on {device}")
    
    # Create model
    model, loss_fn = create_tsformer_vo(sequence_length=8)
    model = model.to(device)
    
    # Test forward pass
    batch_size = 2
    seq_len = 8
    dummy_images = torch.randn(batch_size, seq_len, 3, 224, 224).to(device)
    dummy_poses = torch.randn(batch_size, 6).to(device)
    
    print(f"Input shape: {dummy_images.shape}")
    
    # Forward pass
    with torch.no_grad():
        pred_poses = model(dummy_images)
        print(f"Output shape: {pred_poses.shape}")
        
        # Test SE(3) geometric loss
        loss, loss_dict = loss_fn(pred_poses, dummy_poses)
        print(f"SE(3) Geometric Loss: {loss.item():.6f}")
        print("Loss components:", loss_dict)
    
    # Model info
    print(f"\nModel parameters:")
    print(f"  Total: {model.get_num_parameters():,}")
    print(f"  Trainable: {model.get_num_trainable_parameters():,}")