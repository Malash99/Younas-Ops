"""
Multi-Scale Loss Functions for Visual Odometry

Implements loss functions that supervise motion at multiple temporal scales:
1. Frame-to-frame deltas (fine-grained)
2. Short-term accumulated poses (5-frame windows)
3. Long-term trajectory segments (20-frame windows)

This prevents straight-line predictions by explicitly supervising trajectory shape
at multiple temporal resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss that supervises motion at three temporal scales:
    - Scale 1: Frame-to-frame deltas (local consistency)
    - Scale 2: 5-frame accumulated poses (short-term shape)
    - Scale 3: 20-frame accumulated poses (long-term trajectory)
    """
    
    def __init__(
        self,
        delta_weight: float = 1.0,      # Frame-to-frame
        short_weight: float = 3.0,      # 5-frame accumulated  
        long_weight: float = 5.0,       # 20-frame accumulated
        magnitude_weight: float = 2.0,  # Motion magnitude consistency
        smoothness_weight: float = 1.0, # Trajectory smoothness
        translation_weight: float = 1.0,
        rotation_weight: float = 5.0
    ):
        super().__init__()
        self.delta_weight = delta_weight
        self.short_weight = short_weight
        self.long_weight = long_weight
        self.magnitude_weight = magnitude_weight
        self.smoothness_weight = smoothness_weight
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
    
    def accumulate_poses(self, deltas: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Accumulate pose deltas over sliding windows
        
        Args:
            deltas: [batch, seq_len, 6] - frame-to-frame pose deltas
            window_size: Size of accumulation window
            
        Returns:
            accumulated: [batch, seq_len - window_size + 1, 6] - accumulated poses
        """
        batch_size, seq_len, pose_dim = deltas.shape
        
        if seq_len < window_size:
            return deltas.sum(dim=1, keepdim=True)  # Sum all if sequence too short
        
        accumulated = []
        for i in range(seq_len - window_size + 1):
            window_sum = deltas[:, i:i+window_size].sum(dim=1)  # [batch, 6]
            accumulated.append(window_sum)
        
        return torch.stack(accumulated, dim=1)  # [batch, num_windows, 6]
    
    def motion_magnitude_loss(
        self, 
        pred_poses: torch.Tensor, 
        target_poses: torch.Tensor
    ) -> torch.Tensor:
        """
        Loss that ensures predicted motions have correct magnitude/energy
        Prevents model from learning to predict near-zero motions
        """
        pred_magnitudes = torch.norm(pred_poses[..., :3], dim=-1)  # Translation magnitude
        target_magnitudes = torch.norm(target_poses[..., :3], dim=-1)
        
        magnitude_loss = F.mse_loss(pred_magnitudes, target_magnitudes)
        return magnitude_loss
    
    def trajectory_smoothness_loss(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Penalize sudden changes in motion (second-order derivatives)
        """
        if poses.size(1) < 3:
            return torch.tensor(0.0, device=poses.device)
        
        # First derivative (velocity)
        velocity = poses[:, 1:] - poses[:, :-1]
        
        # Second derivative (acceleration)  
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        
        # Penalize large accelerations
        smoothness_loss = torch.mean(torch.norm(acceleration, dim=-1))
        return smoothness_loss
    
    def forward(
        self,
        pred_deltas: torch.Tensor,
        target_deltas: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale loss
        
        Args:
            pred_deltas: [batch, seq_len, 6] - predicted frame-to-frame deltas
            target_deltas: [batch, seq_len, 6] - ground truth frame-to-frame deltas
            
        Returns:
            Dictionary of loss components
        """
        batch_size, seq_len, pose_dim = pred_deltas.shape
        losses = {}
        
        # Scale 1: Frame-to-frame delta loss (fine-grained)
        pred_trans_delta = pred_deltas[..., :3]
        pred_rot_delta = pred_deltas[..., 3:]
        target_trans_delta = target_deltas[..., :3]
        target_rot_delta = target_deltas[..., 3:]
        
        delta_trans_loss = F.mse_loss(pred_trans_delta, target_trans_delta)
        delta_rot_loss = F.mse_loss(pred_rot_delta, target_rot_delta)
        delta_loss = (self.translation_weight * delta_trans_loss + 
                     self.rotation_weight * delta_rot_loss)
        
        losses['delta_loss'] = delta_loss
        losses['delta_trans_loss'] = delta_trans_loss
        losses['delta_rot_loss'] = delta_rot_loss
        
        # Scale 2: Short-term accumulated poses (5-frame windows)
        if seq_len >= 5:
            pred_short = self.accumulate_poses(pred_deltas, window_size=5)
            target_short = self.accumulate_poses(target_deltas, window_size=5)
            
            short_trans_loss = F.mse_loss(pred_short[..., :3], target_short[..., :3])
            short_rot_loss = F.mse_loss(pred_short[..., 3:], target_short[..., 3:])
            short_loss = (self.translation_weight * short_trans_loss + 
                         self.rotation_weight * short_rot_loss)
            
            losses['short_loss'] = short_loss
            losses['short_trans_loss'] = short_trans_loss
            losses['short_rot_loss'] = short_rot_loss
            
            # Motion magnitude loss for short-term
            short_mag_loss = self.motion_magnitude_loss(pred_short, target_short)
            losses['short_magnitude_loss'] = short_mag_loss
        else:
            losses['short_loss'] = torch.tensor(0.0, device=pred_deltas.device)
            losses['short_trans_loss'] = torch.tensor(0.0, device=pred_deltas.device)
            losses['short_rot_loss'] = torch.tensor(0.0, device=pred_deltas.device)
            losses['short_magnitude_loss'] = torch.tensor(0.0, device=pred_deltas.device)
        
        # Scale 3: Long-term accumulated poses (20-frame windows)
        if seq_len >= 20:
            pred_long = self.accumulate_poses(pred_deltas, window_size=20)
            target_long = self.accumulate_poses(target_deltas, window_size=20)
            
            long_trans_loss = F.mse_loss(pred_long[..., :3], target_long[..., :3])
            long_rot_loss = F.mse_loss(pred_long[..., 3:], target_long[..., 3:])
            long_loss = (self.translation_weight * long_trans_loss + 
                        self.rotation_weight * long_rot_loss)
            
            losses['long_loss'] = long_loss
            losses['long_trans_loss'] = long_trans_loss
            losses['long_rot_loss'] = long_rot_loss
            
            # Motion magnitude loss for long-term (most important)
            long_mag_loss = self.motion_magnitude_loss(pred_long, target_long)
            losses['long_magnitude_loss'] = long_mag_loss
        else:
            losses['long_loss'] = torch.tensor(0.0, device=pred_deltas.device)
            losses['long_trans_loss'] = torch.tensor(0.0, device=pred_deltas.device)
            losses['long_rot_loss'] = torch.tensor(0.0, device=pred_deltas.device)
            losses['long_magnitude_loss'] = torch.tensor(0.0, device=pred_deltas.device)
        
        # Additional losses
        
        # Motion magnitude loss (overall)
        overall_mag_loss = self.motion_magnitude_loss(pred_deltas, target_deltas)
        losses['magnitude_loss'] = overall_mag_loss
        
        # Trajectory smoothness loss
        smoothness_loss = self.trajectory_smoothness_loss(pred_deltas)
        losses['smoothness_loss'] = smoothness_loss
        
        # Total weighted loss
        total_loss = (
            self.delta_weight * losses['delta_loss'] +
            self.short_weight * losses['short_loss'] +
            self.long_weight * losses['long_loss'] +
            self.magnitude_weight * (losses['short_magnitude_loss'] + losses['long_magnitude_loss']) +
            self.smoothness_weight * losses['smoothness_loss']
        )
        
        losses['total_loss'] = total_loss
        
        # For compatibility
        losses['translation_loss'] = delta_trans_loss
        losses['rotation_loss'] = delta_rot_loss
        
        return losses


class AdaptiveMultiScaleLoss(MultiScaleLoss):
    """
    Adaptive version that adjusts scale weights during training
    
    Curriculum learning: Start with long-term supervision, gradually add fine details
    """
    
    def __init__(self, *args, **kwargs):
        # Extract curriculum-specific parameters
        self.curriculum_steps = kwargs.pop('curriculum_steps', 5000)
        self.initial_long_weight = kwargs.pop('initial_long_weight', 10.0)
        
        # Initialize parent class
        super().__init__(*args, **kwargs)
        self.training_step = 0
        
    def update_weights(self, step: int):
        """Update loss weights based on training progress"""
        self.training_step = step
        
        # Curriculum learning: Start with high long-term weight, gradually balance
        progress = min(step / self.curriculum_steps, 1.0)
        
        # Gradually reduce long-term weight and increase short-term/delta weights
        self.long_weight = self.initial_long_weight * (1.0 - 0.5 * progress)
        self.short_weight = 1.0 + 2.0 * progress  
        self.delta_weight = 0.5 + 0.5 * progress
        
    def forward(self, pred_deltas, target_deltas):
        losses = super().forward(pred_deltas, target_deltas)
        
        # Add curriculum info to losses
        losses['curriculum_progress'] = min(self.training_step / self.curriculum_steps, 1.0)
        losses['current_long_weight'] = self.long_weight
        losses['current_short_weight'] = self.short_weight
        losses['current_delta_weight'] = self.delta_weight
        
        return losses


def create_multiscale_loss(config: Dict) -> nn.Module:
    """Factory function to create multi-scale loss"""
    loss_type = config.get('loss_type', 'multiscale')
    
    if loss_type == 'multiscale':
        return MultiScaleLoss(
            delta_weight=config.get('delta_weight', 1.0),
            short_weight=config.get('short_weight', 3.0),
            long_weight=config.get('long_weight', 5.0),
            magnitude_weight=config.get('magnitude_weight', 2.0),
            smoothness_weight=config.get('smoothness_weight', 1.0),
            translation_weight=config.get('translation_weight', 1.0),
            rotation_weight=config.get('rotation_weight', 5.0)
        )
    elif loss_type == 'adaptive_multiscale':
        return AdaptiveMultiScaleLoss(
            delta_weight=config.get('delta_weight', 0.5),
            short_weight=config.get('short_weight', 1.0),
            long_weight=config.get('long_weight', 5.0),
            magnitude_weight=config.get('magnitude_weight', 2.0),
            smoothness_weight=config.get('smoothness_weight', 1.0),
            translation_weight=config.get('translation_weight', 1.0),
            rotation_weight=config.get('rotation_weight', 5.0),
            curriculum_steps=config.get('curriculum_steps', 5000),
            initial_long_weight=config.get('initial_long_weight', 10.0)
        )
    else:
        raise ValueError(f"Unknown multiscale loss type: {loss_type}")


# Utility functions for analyzing multi-scale predictions
def analyze_multiscale_predictions(pred_deltas: torch.Tensor, target_deltas: torch.Tensor) -> Dict:
    """Analyze prediction quality at different temporal scales"""
    
    with torch.no_grad():
        batch_size, seq_len = pred_deltas.shape[:2]
        analysis = {}
        
        # Frame-to-frame analysis
        delta_errors = torch.norm(pred_deltas - target_deltas, dim=-1)
        analysis['delta_mean_error'] = delta_errors.mean().item()
        analysis['delta_max_error'] = delta_errors.max().item()
        
        # Short-term analysis (5-frame windows)
        if seq_len >= 5:
            pred_short = []
            target_short = []
            for i in range(seq_len - 4):
                pred_short.append(pred_deltas[:, i:i+5].sum(dim=1))
                target_short.append(target_deltas[:, i:i+5].sum(dim=1))
            
            pred_short = torch.stack(pred_short, dim=1)
            target_short = torch.stack(target_short, dim=1)
            short_errors = torch.norm(pred_short - target_short, dim=-1)
            
            analysis['short_mean_error'] = short_errors.mean().item()
            analysis['short_max_error'] = short_errors.max().item()
        
        # Long-term analysis (20-frame windows)
        if seq_len >= 20:
            pred_long = pred_deltas.sum(dim=1)  # Full sequence
            target_long = target_deltas.sum(dim=1)
            long_errors = torch.norm(pred_long - target_long, dim=-1)
            
            analysis['long_mean_error'] = long_errors.mean().item()
            analysis['long_max_error'] = long_errors.max().item()
        
        return analysis


class ScaleDirectionLoss(nn.Module):
    """
    Improved loss function that fixes scale and direction issues
    Based on research from SC-SfMLearner++, DeepVO++, and CamPoseNet approaches
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        diversity_weight: float = 0.1,
        frame_diff_weight: float = 0.5,
        magnitude_weight: float = 3.0,      # Strong scale supervision
        direction_weight: float = 2.0,      # Strong direction consistency
        scale_reg_weight: float = 0.1       # Scale regularization
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.diversity_weight = diversity_weight
        self.frame_diff_weight = frame_diff_weight
        self.magnitude_weight = magnitude_weight
        self.direction_weight = direction_weight
        self.scale_reg_weight = scale_reg_weight
        
    def magnitude_supervision_loss(self, pred_poses: torch.Tensor, target_poses: torch.Tensor, 
                                 pred_scales: torch.Tensor = None) -> torch.Tensor:
        """
        Scale supervision: MSE between ||scaled_pred|| and ||gt||
        Fixes the 5.3x magnitude problem
        """
        if pred_scales is not None:
            # Apply predicted scale factors
            scaled_pred = pred_poses.clone()
            scaled_pred[..., :3] *= pred_scales.unsqueeze(-1)
            pred_magnitudes = torch.norm(scaled_pred[..., :3], dim=-1)
        else:
            pred_magnitudes = torch.norm(pred_poses[..., :3], dim=-1)
            
        target_magnitudes = torch.norm(target_poses[..., :3], dim=-1)
        
        return F.mse_loss(pred_magnitudes, target_magnitudes)
    
    def direction_consistency_loss(self, pred_poses: torch.Tensor, target_poses: torch.Tensor) -> torch.Tensor:
        """
        Direction consistency using cosine similarity
        Fixes the backward vs forward motion problem
        """
        # Extract translation components
        pred_trans = pred_poses[..., :3]
        target_trans = target_poses[..., :3]
        
        # Normalize to unit vectors (handle zero vectors)
        pred_dir = F.normalize(pred_trans, dim=-1, eps=1e-8)
        target_dir = F.normalize(target_trans, dim=-1, eps=1e-8)
        
        # Cosine similarity (1.0 = same direction, -1.0 = opposite)
        cosine_sim = F.cosine_similarity(pred_dir, target_dir, dim=-1)
        
        # Loss penalizes opposite directions (want cosine_sim → 1.0)
        direction_loss = 1.0 - cosine_sim.mean()
        
        return direction_loss
    
    def scale_regularization_loss(self, pred_scales: torch.Tensor) -> torch.Tensor:
        """
        Prevent extreme scale factors - keep them near 1.0
        """
        if pred_scales is None:
            return torch.tensor(0.0)
        
        # Penalize scales far from 1.0
        scale_reg = torch.mean((pred_scales - 1.0) ** 2)
        return scale_reg
    
    def forward(self, pred_deltas: torch.Tensor, target_deltas: torch.Tensor, 
                pred_scales: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_deltas: [batch, seq_len, 6] predicted pose deltas
            target_deltas: [batch, seq_len, 6] ground truth pose deltas  
            pred_scales: [batch, seq_len] optional predicted scale factors
        """
        losses = {}
        
        # 1. Base MSE loss
        mse_loss = F.mse_loss(pred_deltas, target_deltas)
        losses['mse_loss'] = mse_loss
        
        # 2. Diversity loss - prevent collapse to constant predictions
        pred_var = torch.var(pred_deltas, dim=1)  # [batch, 6]
        diversity_loss = torch.mean(torch.exp(-pred_var))
        losses['diversity_loss'] = diversity_loss
        
        # 3. Frame difference loss - ensure frame-to-frame variation
        if pred_deltas.size(1) > 1:
            pred_diffs = torch.diff(pred_deltas, dim=1)
            target_diffs = torch.diff(target_deltas, dim=1)
            frame_diff_loss = F.mse_loss(pred_diffs, target_diffs)
            losses['frame_diff_loss'] = frame_diff_loss
        else:
            losses['frame_diff_loss'] = torch.tensor(0.0, device=pred_deltas.device)
        
        # 4. NEW: Magnitude supervision - fix 5.3x scale problem
        magnitude_loss = self.magnitude_supervision_loss(pred_deltas, target_deltas, pred_scales)
        losses['magnitude_loss'] = magnitude_loss
        
        # 5. NEW: Direction consistency - fix backward/forward problem
        direction_loss = self.direction_consistency_loss(pred_deltas, target_deltas)
        losses['direction_loss'] = direction_loss
        
        # 6. NEW: Scale regularization - prevent extreme scale factors
        scale_reg_loss = self.scale_regularization_loss(pred_scales)
        losses['scale_reg_loss'] = scale_reg_loss
        
        # Total weighted loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.diversity_weight * diversity_loss +
            self.frame_diff_weight * losses['frame_diff_loss'] +
            self.magnitude_weight * magnitude_loss +
            self.direction_weight * direction_loss +
            self.scale_reg_weight * scale_reg_loss
        )
        losses['total_loss'] = total_loss
        
        # Add debugging info
        with torch.no_grad():
            pred_magnitudes = torch.norm(pred_deltas[..., :3], dim=-1)
            target_magnitudes = torch.norm(target_deltas[..., :3], dim=-1)
            losses['pred_magnitude_mean'] = pred_magnitudes.mean()
            losses['target_magnitude_mean'] = target_magnitudes.mean()
            losses['magnitude_ratio'] = (pred_magnitudes.mean() / (target_magnitudes.mean() + 1e-8))
            
            if pred_scales is not None:
                losses['scale_mean'] = pred_scales.mean()
                losses['scale_std'] = pred_scales.std()
        
        return losses