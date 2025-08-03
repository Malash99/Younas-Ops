"""
Trajectory-Aware Loss Functions for Sub-Trajectory Training
Includes ATE (Absolute Trajectory Error) supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class TrajectoryAwareLoss(nn.Module):
    """
    Combined loss for trajectory-aware training
    
    Combines:
    1. Individual pose losses (frame-to-frame)
    2. Absolute Trajectory Error (ATE) over sub-trajectory
    3. Trajectory consistency penalties
    """
    
    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 10.0,
        ate_weight: float = 5.0,
        consistency_weight: float = 1.0,
        smoothness_weight: float = 0.5
    ):
        """
        Args:
            translation_weight: Weight for individual translation losses
            rotation_weight: Weight for individual rotation losses  
            ate_weight: Weight for absolute trajectory error
            consistency_weight: Weight for trajectory consistency
            smoothness_weight: Weight for trajectory smoothness
        """
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.ate_weight = ate_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(
        self, 
        predictions: torch.Tensor,  # [batch, N-1, 6]
        pose_targets: torch.Tensor,  # [batch, N-1, 6] 
        accumulated_targets: torch.Tensor  # [batch, N-1, 6]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute trajectory-aware loss
        
        Args:
            predictions: Predicted relative poses [batch, N-1, 6]
            pose_targets: Ground truth relative poses [batch, N-1, 6]
            accumulated_targets: Ground truth accumulated poses [batch, N-1, 6]
        """
        batch_size, seq_len, pose_dim = predictions.shape
        
        # 1. Individual Pose Losses (frame-to-frame)
        trans_pred = predictions[:, :, :3]  # [batch, N-1, 3]
        rot_pred = predictions[:, :, 3:]    # [batch, N-1, 3]
        trans_target = pose_targets[:, :, :3]
        rot_target = pose_targets[:, :, 3:]
        
        # Translation loss (L2)
        translation_loss = F.mse_loss(trans_pred, trans_target)
        
        # Rotation loss (L2 on rotation vectors)
        rotation_loss = F.mse_loss(rot_pred, rot_target)
        
        # 2. Absolute Trajectory Error (ATE)
        # Accumulate predicted poses
        predicted_accumulated = torch.zeros_like(predictions)
        current_pose = torch.zeros(batch_size, pose_dim, device=predictions.device)
        
        for t in range(seq_len):
            current_pose = current_pose + predictions[:, t, :]
            predicted_accumulated[:, t, :] = current_pose
        
        # ATE loss (L2 between accumulated trajectories)
        ate_trans_loss = F.mse_loss(
            predicted_accumulated[:, :, :3], 
            accumulated_targets[:, :, :3]
        )
        ate_rot_loss = F.mse_loss(
            predicted_accumulated[:, :, 3:], 
            accumulated_targets[:, :, 3:]
        )
        ate_loss = ate_trans_loss + ate_rot_loss
        
        # 3. Trajectory Consistency Loss
        # Penalize sudden changes in predicted poses
        if seq_len > 1:
            pose_diffs = predictions[:, 1:, :] - predictions[:, :-1, :]
            consistency_loss = torch.mean(torch.norm(pose_diffs, dim=-1))
        else:
            consistency_loss = torch.tensor(0.0, device=predictions.device)
        
        # 4. Trajectory Smoothness Loss
        # Encourage smooth trajectories in accumulated space
        if seq_len > 1:
            acc_diffs = predicted_accumulated[:, 1:, :] - predicted_accumulated[:, :-1, :]
            target_diffs = accumulated_targets[:, 1:, :] - accumulated_targets[:, :-1, :]
            smoothness_loss = F.mse_loss(acc_diffs, target_diffs)
        else:
            smoothness_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combine losses
        total_loss = (
            self.translation_weight * translation_loss +
            self.rotation_weight * rotation_loss +
            self.ate_weight * ate_loss +
            self.consistency_weight * consistency_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        # Calculate final position error (drift metric)
        final_pred_pos = predicted_accumulated[:, -1, :3]  # [batch, 3]
        final_target_pos = accumulated_targets[:, -1, :3]  # [batch, 3]
        final_position_error = torch.mean(torch.norm(final_pred_pos - final_target_pos, dim=-1))
        
        return {
            'total_loss': total_loss,
            'translation_loss': translation_loss,
            'rotation_loss': rotation_loss,
            'ate_loss': ate_loss,
            'ate_trans_loss': ate_trans_loss,
            'ate_rot_loss': ate_rot_loss,
            'consistency_loss': consistency_loss,
            'smoothness_loss': smoothness_loss,
            'final_position_error': final_position_error,
            'predicted_accumulated': predicted_accumulated,
            'target_accumulated': accumulated_targets
        }


class AdaptiveTrajectoryLoss(TrajectoryAwareLoss):
    """
    Adaptive version that adjusts weights based on training progress
    Increases ATE weight as training progresses to focus on drift reduction
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_ate_weight = kwargs.get('ate_weight', 5.0)
        self.max_ate_weight = kwargs.get('max_ate_weight', 20.0)
        self.training_step = 0
        self.adaptation_steps = kwargs.get('adaptation_steps', 10000)
        
    def update_weights(self, step: int):
        """Update loss weights based on training progress"""
        self.training_step = step
        
        # Gradually increase ATE weight
        progress = min(step / self.adaptation_steps, 1.0)
        self.ate_weight = self.initial_ate_weight + progress * (self.max_ate_weight - self.initial_ate_weight)


class DriftPenaltyLoss(nn.Module):
    """
    Specialized loss that heavily penalizes drift accumulation
    """
    
    def __init__(
        self,
        base_weight: float = 1.0,
        drift_penalty: float = 10.0,
        exponential_penalty: bool = True
    ):
        super().__init__()
        self.base_weight = base_weight
        self.drift_penalty = drift_penalty
        self.exponential_penalty = exponential_penalty
        
    def forward(
        self,
        predictions: torch.Tensor,
        accumulated_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply exponentially increasing penalty for drift
        """
        # Accumulate predictions
        batch_size, seq_len, pose_dim = predictions.shape
        predicted_accumulated = torch.zeros_like(predictions)
        current_pose = torch.zeros(batch_size, pose_dim, device=predictions.device)
        
        for t in range(seq_len):
            current_pose = current_pose + predictions[:, t, :]
            predicted_accumulated[:, t, :] = current_pose
        
        # Calculate positional errors at each time step
        position_errors = torch.norm(
            predicted_accumulated[:, :, :3] - accumulated_targets[:, :, :3], 
            dim=-1
        )  # [batch, seq_len]
        
        if self.exponential_penalty:
            # Apply exponentially increasing weights over time
            time_weights = torch.exp(torch.arange(seq_len, dtype=torch.float32, device=predictions.device))
            time_weights = time_weights / time_weights[0]  # Normalize
            weighted_errors = position_errors * time_weights.unsqueeze(0)
        else:
            # Linear increasing weights
            time_weights = torch.arange(1, seq_len + 1, dtype=torch.float32, device=predictions.device)
            weighted_errors = position_errors * time_weights.unsqueeze(0)
        
        drift_loss = torch.mean(weighted_errors)
        
        return self.base_weight * drift_loss


def create_trajectory_loss(loss_type: str = 'standard', **kwargs) -> nn.Module:
    """Factory function to create trajectory loss"""
    
    if loss_type == 'standard':
        return TrajectoryAwareLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveTrajectoryLoss(**kwargs)
    elif loss_type == 'drift_penalty':
        return DriftPenaltyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Utility functions for trajectory analysis
def calculate_trajectory_metrics(
    predicted_poses: torch.Tensor,
    target_poses: torch.Tensor
) -> Dict[str, float]:
    """Calculate comprehensive trajectory metrics"""
    
    # Convert to numpy for analysis
    pred_np = predicted_poses.detach().cpu().numpy()
    target_np = target_poses.detach().cpu().numpy()
    
    metrics = {}
    
    # ATE (Absolute Trajectory Error)
    ate_errors = np.linalg.norm(pred_np[:, :3] - target_np[:, :3], axis=1)
    metrics['ate_mean'] = float(np.mean(ate_errors))
    metrics['ate_rmse'] = float(np.sqrt(np.mean(ate_errors**2)))
    metrics['ate_max'] = float(np.max(ate_errors))
    
    # Final drift
    final_error = np.linalg.norm(pred_np[-1, :3] - target_np[-1, :3])
    trajectory_length = np.sum(np.linalg.norm(np.diff(target_np[:, :3], axis=0), axis=1))
    
    metrics['final_drift_m'] = float(final_error)
    metrics['trajectory_length_m'] = float(trajectory_length)
    metrics['relative_drift_percent'] = float((final_error / trajectory_length) * 100) if trajectory_length > 0 else 0.0
    
    # Rotation errors
    rot_errors = np.linalg.norm(pred_np[:, 3:] - target_np[:, 3:], axis=1)
    metrics['rotation_mean_rad'] = float(np.mean(rot_errors))
    metrics['rotation_mean_deg'] = float(np.mean(rot_errors) * 180 / np.pi)
    
    return metrics