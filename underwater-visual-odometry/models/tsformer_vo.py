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


class TSformerVOLoss(nn.Module):
    """
    Combined loss function for TSformer-VO training.
    
    Combines translation and rotation losses with appropriate weighting.
    """
    
    def __init__(self, trans_weight=1.0, rot_weight=100.0):
        super().__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        
    def forward(self, pred_poses, gt_poses):
        """
        Compute combined pose loss.
        
        Args:
            pred_poses: (batch_size, 6) - predicted [dx,dy,dz,droll,dpitch,dyaw]
            gt_poses: (batch_size, 6) - ground truth [dx,dy,dz,droll,dpitch,dyaw]
            
        Returns:
            loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        # Split translation and rotation
        pred_trans = pred_poses[:, :3]  # [dx, dy, dz]
        pred_rot = pred_poses[:, 3:]    # [droll, dpitch, dyaw]
        
        gt_trans = gt_poses[:, :3]      # [dx, dy, dz]
        gt_rot = gt_poses[:, 3:]        # [droll, dpitch, dyaw]
        
        # Translation loss (L1 + L2)
        trans_l1 = F.l1_loss(pred_trans, gt_trans)
        trans_l2 = F.mse_loss(pred_trans, gt_trans)
        trans_loss = trans_l1 + trans_l2
        
        # Rotation loss (L1 + L2)
        rot_l1 = F.l1_loss(pred_rot, gt_rot)
        rot_l2 = F.mse_loss(pred_rot, gt_rot)
        rot_loss = rot_l1 + rot_l2
        
        # Combined loss
        total_loss = self.trans_weight * trans_loss + self.rot_weight * rot_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'trans_loss': trans_loss.item(),
            'rot_loss': rot_loss.item(),
            'trans_l1': trans_l1.item(),
            'trans_l2': trans_l2.item(),
            'rot_l1': rot_l1.item(),
            'rot_l2': rot_l2.item()
        }
        
        return total_loss, loss_dict


def create_tsformer_vo(sequence_length=8, pretrained=True, freeze_backbone=False, image_size=224):
    """
    Factory function to create TSformer-VO model.
    
    Args:
        sequence_length: Number of frames in input sequence
        pretrained: Use pretrained ViT backbone
        freeze_backbone: Freeze ViT parameters
        image_size: Input image resolution
        
    Returns:
        model: TSformerVO instance
        loss_fn: TSformerVOLoss instance
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
    
    loss_fn = TSformerVOLoss(trans_weight=1.0, rot_weight=100.0)
    
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
        
        # Test loss
        loss, loss_dict = loss_fn(pred_poses, dummy_poses)
        print(f"Loss: {loss.item():.6f}")
        print("Loss components:", loss_dict)
    
    # Model info
    print(f"\nModel parameters:")
    print(f"  Total: {model.get_num_parameters():,}")
    print(f"  Trainable: {model.get_num_trainable_parameters():,}")