"""
Motion-Aware Loss Functions for Underwater Visual Odometry

Implements losses that supervise frame-to-frame motion rather than just absolute poses.
This helps the model learn to extract meaningful visual motion cues from consecutive frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class MotionAwareLoss(nn.Module):
    """
    Motion-aware loss that supervises relative motion between consecutive frames
    
    This loss encourages the model to learn visual motion patterns rather than 
    just predicting constant velocity or straight-line trajectories.
    """
    
    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 10.0,
        motion_weight: float = 5.0,
        consistency_weight: float = 2.0,
        sequence_length: int = 5
    ):
        """
        Args:
            translation_weight: Weight for absolute translation loss
            rotation_weight: Weight for absolute rotation loss
            motion_weight: Weight for relative motion loss
            consistency_weight: Weight for motion consistency loss
            sequence_length: Length of input sequences
        """
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.motion_weight = motion_weight
        self.consistency_weight = consistency_weight
        self.sequence_length = sequence_length
        
    def compute_relative_pose(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Compute relative poses between consecutive frames
        
        Args:
            poses: Absolute poses [batch_size, seq_len, 6]
        Returns:
            Relative poses [batch_size, seq_len-1, 6]
        """
        if poses.dim() == 2:
            # Single pose, return zero motion
            return torch.zeros_like(poses[:, :0])
        
        # Translation differences
        trans_diff = poses[:, 1:, :3] - poses[:, :-1, :3]
        
        # Rotation differences (simple angle differences for small angles)
        rot_diff = poses[:, 1:, 3:] - poses[:, :-1, 3:]
        
        # Wrap angle differences to [-pi, pi]
        rot_diff = torch.atan2(torch.sin(rot_diff), torch.cos(rot_diff))
        
        relative_poses = torch.cat([trans_diff, rot_diff], dim=-1)
        return relative_poses
    
    def motion_consistency_loss(self, predicted_motions: torch.Tensor) -> torch.Tensor:
        """
        Encourage smooth motion (penalize sudden changes in velocity)
        
        Args:
            predicted_motions: Relative motions [batch_size, seq_len-1, 6]
        Returns:
            Consistency loss scalar
        """
        if predicted_motions.size(1) < 2:
            return torch.tensor(0.0, device=predicted_motions.device)
        
        # Second-order differences (acceleration)
        motion_diff = predicted_motions[:, 1:] - predicted_motions[:, :-1]
        
        # L2 penalty on acceleration
        consistency_loss = torch.norm(motion_diff, dim=-1).mean()
        
        return consistency_loss
    
    def forward(
        self, 
        pred_poses: torch.Tensor, 
        target_poses: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - main interface for the loss function
        """
        return self.visual_motion_loss(pred_poses, target_poses, visual_features)
    
    def visual_motion_loss(
        self, 
        pred_poses: torch.Tensor, 
        target_poses: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss that emphasizes learning from visual motion cues
        
        Args:
            pred_poses: Predicted poses [batch_size, seq_len, 6] or [batch_size, 6]
            target_poses: Ground truth poses [batch_size, seq_len, 6] or [batch_size, 6]
            visual_features: Optional visual features for additional supervision
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Handle single frame case
        if pred_poses.dim() == 2:
            pred_poses = pred_poses.unsqueeze(1)
        if target_poses.dim() == 2:
            target_poses = target_poses.unsqueeze(1)
        
        batch_size, seq_len = pred_poses.shape[:2]
        
        # 1. Absolute pose loss (standard)
        pred_trans = pred_poses[:, :, :3]
        pred_rot = pred_poses[:, :, 3:]
        target_trans = target_poses[:, :, :3]
        target_rot = target_poses[:, :, 3:]
        
        abs_trans_loss = F.mse_loss(pred_trans, target_trans)
        abs_rot_loss = F.mse_loss(pred_rot, target_rot)
        
        losses['absolute_translation_loss'] = abs_trans_loss
        losses['absolute_rotation_loss'] = abs_rot_loss
        
        # 2. Relative motion loss (key innovation)
        if seq_len > 1:
            pred_motions = self.compute_relative_pose(pred_poses)
            target_motions = self.compute_relative_pose(target_poses)
            
            motion_trans_loss = F.mse_loss(pred_motions[:, :, :3], target_motions[:, :, :3])
            motion_rot_loss = F.mse_loss(pred_motions[:, :, 3:], target_motions[:, :, 3:])
            
            losses['motion_translation_loss'] = motion_trans_loss
            losses['motion_rotation_loss'] = motion_rot_loss
            
            # 3. Motion consistency loss
            consistency_loss = self.motion_consistency_loss(pred_motions)
            losses['motion_consistency_loss'] = consistency_loss
        else:
            losses['motion_translation_loss'] = torch.tensor(0.0, device=pred_poses.device)
            losses['motion_rotation_loss'] = torch.tensor(0.0, device=pred_poses.device)
            losses['motion_consistency_loss'] = torch.tensor(0.0, device=pred_poses.device)
        
        # 4. Total weighted loss
        total_loss = (
            self.translation_weight * abs_trans_loss +
            self.rotation_weight * abs_rot_loss +
            self.motion_weight * (losses['motion_translation_loss'] + losses['motion_rotation_loss']) +
            self.consistency_weight * losses['motion_consistency_loss']
        )
        
        losses['total_loss'] = total_loss
        losses['translation_loss'] = abs_trans_loss  # For compatibility
        losses['rotation_loss'] = abs_rot_loss      # For compatibility
        
        return losses


class SequentialMotionLoss(nn.Module):
    """
    Loss function specifically designed for sequential motion prediction
    
    This loss supervises the model to predict pose changes for each frame
    in a sequence, encouraging it to learn visual motion patterns.
    """
    
    def __init__(
        self,
        motion_weight: float = 10.0,
        direction_weight: float = 5.0,
        speed_weight: float = 3.0,
        smoothness_weight: float = 1.0
    ):
        super().__init__()
        self.motion_weight = motion_weight
        self.direction_weight = direction_weight
        self.speed_weight = speed_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(
        self,
        pred_poses: torch.Tensor,
        target_poses: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_poses: [batch_size, seq_len, 6]
            target_poses: [batch_size, seq_len, 6]
        """
        batch_size, seq_len = pred_poses.shape[:2]
        
        if seq_len < 2:
            # Fallback to simple MSE for single frames
            return {
                'total_loss': F.mse_loss(pred_poses, target_poses),
                'translation_loss': F.mse_loss(pred_poses[:, :, :3], target_poses[:, :, :3]),
                'rotation_loss': F.mse_loss(pred_poses[:, :, 3:], target_poses[:, :, 3:])
            }
        
        # Compute motion vectors
        pred_motions = pred_poses[:, 1:] - pred_poses[:, :-1]
        target_motions = target_poses[:, 1:] - target_poses[:, :-1]
        
        # 1. Motion magnitude loss
        pred_trans_motion = pred_motions[:, :, :3]
        target_trans_motion = target_motions[:, :, :3]
        
        motion_loss = F.mse_loss(pred_trans_motion, target_trans_motion)
        
        # 2. Motion direction loss (normalized vectors)
        pred_speed = torch.norm(pred_trans_motion, dim=-1, keepdim=True) + 1e-8
        target_speed = torch.norm(target_trans_motion, dim=-1, keepdim=True) + 1e-8
        
        pred_direction = pred_trans_motion / pred_speed
        target_direction = target_trans_motion / target_speed
        
        direction_loss = F.mse_loss(pred_direction, target_direction)
        
        # 3. Speed loss
        speed_loss = F.mse_loss(pred_speed.squeeze(-1), target_speed.squeeze(-1))
        
        # 4. Smoothness loss (second-order differences)
        if seq_len > 2:
            pred_accel = pred_motions[:, 1:] - pred_motions[:, :-1]
            target_accel = target_motions[:, 1:] - target_motions[:, :-1]
            smoothness_loss = F.mse_loss(pred_accel, target_accel)
        else:
            smoothness_loss = torch.tensor(0.0, device=pred_poses.device)
        
        # Rotation losses
        pred_rot_motion = pred_motions[:, :, 3:]
        target_rot_motion = target_motions[:, :, 3:]
        rot_motion_loss = F.mse_loss(pred_rot_motion, target_rot_motion)
        
        # Total loss
        total_loss = (
            self.motion_weight * motion_loss +
            self.direction_weight * direction_loss +
            self.speed_weight * speed_loss +
            self.smoothness_weight * smoothness_loss +
            self.motion_weight * rot_motion_loss
        )
        
        return {
            'total_loss': total_loss,
            'translation_loss': F.mse_loss(pred_poses[:, :, :3], target_poses[:, :, :3]),
            'rotation_loss': F.mse_loss(pred_poses[:, :, 3:], target_poses[:, :, 3:]),
            'motion_loss': motion_loss,
            'direction_loss': direction_loss,
            'speed_loss': speed_loss,
            'smoothness_loss': smoothness_loss,
            'rotation_motion_loss': rot_motion_loss
        }


class MotionContrastiveLoss(nn.Module):
    """
    Contrastive loss that encourages the model to distinguish between
    different types of motion patterns
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        visual_features: torch.Tensor,
        motion_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_features: Features from vision transformer [batch, seq_len, d_model]
            motion_labels: Motion type labels [batch, seq_len]
        """
        # Implement contrastive learning to distinguish motion patterns
        # This would encourage the visual features to be discriminative
        # for different types of underwater motion (turn left, right, up, down, etc.)
        
        batch_size, seq_len, d_model = visual_features.shape
        
        # For now, return zero - this would be implemented based on motion clustering
        return torch.tensor(0.0, device=visual_features.device)


def create_motion_aware_loss(config: Dict) -> nn.Module:
    """Factory function to create motion-aware loss function"""
    loss_type = config.get('loss_type', 'motion_aware')
    
    if loss_type == 'motion_aware':
        return MotionAwareLoss(
            translation_weight=config.get('translation_weight', 1.0),
            rotation_weight=config.get('rotation_weight', 10.0),
            motion_weight=config.get('motion_weight', 5.0),
            consistency_weight=config.get('consistency_weight', 2.0),
            sequence_length=config.get('sequence_length', 5)
        )
    elif loss_type == 'sequential_motion':
        return SequentialMotionLoss(
            motion_weight=config.get('motion_weight', 10.0),
            direction_weight=config.get('direction_weight', 5.0),
            speed_weight=config.get('speed_weight', 3.0),
            smoothness_weight=config.get('smoothness_weight', 1.0)
        )
    else:
        raise ValueError(f"Unknown motion loss type: {loss_type}")