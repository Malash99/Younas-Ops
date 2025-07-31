"""
Loss functions for underwater visual odometry

Implements various loss functions optimized for pose regression:
- Weighted translation/rotation losses
- Uncertainty-aware losses
- Robust losses for outliers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class PoseLoss(nn.Module):
    """
    Standard pose loss with weighted translation and rotation components
    """
    
    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 10.0,
        loss_type: str = 'mse',  # 'mse', 'l1', 'huber'
        huber_delta: float = 1.0
    ):
        """
        Args:
            translation_weight: Weight for translation loss
            rotation_weight: Weight for rotation loss
            loss_type: Type of loss function ('mse', 'l1', 'huber')
            huber_delta: Delta for Huber loss
        """
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
    def forward(
        self,
        pred_pose: torch.Tensor,
        target_pose: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pose loss
        
        Args:
            pred_pose: Predicted pose [batch_size, 6] (tx, ty, tz, rx, ry, rz)
            target_pose: Target pose [batch_size, 6]
            
        Returns:
            Dictionary of losses
        """
        # Split translation and rotation
        pred_trans = pred_pose[:, :3]
        pred_rot = pred_pose[:, 3:]
        target_trans = target_pose[:, :3]
        target_rot = target_pose[:, 3:]
        
        # Compute individual losses
        if self.loss_type == 'mse':
            trans_loss = F.mse_loss(pred_trans, target_trans)
            rot_loss = F.mse_loss(pred_rot, target_rot)
        elif self.loss_type == 'l1':
            trans_loss = F.l1_loss(pred_trans, target_trans)
            rot_loss = F.l1_loss(pred_rot, target_rot)
        elif self.loss_type == 'huber':
            trans_loss = F.smooth_l1_loss(pred_trans, target_trans, beta=self.huber_delta)
            rot_loss = F.smooth_l1_loss(pred_rot, target_rot, beta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Weighted combination
        total_loss = (self.translation_weight * trans_loss + 
                     self.rotation_weight * rot_loss)
        
        return {
            'total_loss': total_loss,
            'translation_loss': trans_loss,
            'rotation_loss': rot_loss,
            'weighted_translation_loss': self.translation_weight * trans_loss,
            'weighted_rotation_loss': self.rotation_weight * rot_loss
        }


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted loss for pose regression
    
    Implements heteroscedastic uncertainty as described in:
    "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
    """
    
    def __init__(
        self,
        initial_log_var_trans: float = 0.0,
        initial_log_var_rot: float = 0.0,
        learn_uncertainty: bool = True
    ):
        """
        Args:
            initial_log_var_trans: Initial log variance for translation
            initial_log_var_rot: Initial log variance for rotation
            learn_uncertainty: Whether to learn uncertainty weights
        """
        super().__init__()
        self.learn_uncertainty = learn_uncertainty
        
        if learn_uncertainty:
            # Learnable uncertainty parameters
            self.log_var_trans = nn.Parameter(torch.tensor(initial_log_var_trans))
            self.log_var_rot = nn.Parameter(torch.tensor(initial_log_var_rot))
        else:
            # Fixed uncertainty weights
            self.register_buffer('log_var_trans', torch.tensor(initial_log_var_trans))
            self.register_buffer('log_var_rot', torch.tensor(initial_log_var_rot))
    
    def forward(
        self,
        pred_pose: torch.Tensor,
        target_pose: torch.Tensor,
        pred_uncertainty: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty-weighted loss
        
        Args:
            pred_pose: Predicted pose [batch_size, 6]
            target_pose: Target pose [batch_size, 6]
            pred_uncertainty: Predicted uncertainty [batch_size, 6] (optional)
            
        Returns:
            Dictionary of losses
        """
        # Split translation and rotation
        pred_trans = pred_pose[:, :3]
        pred_rot = pred_pose[:, 3:]
        target_trans = target_pose[:, :3]
        target_rot = target_pose[:, 3:]
        
        # Compute squared errors
        trans_error = (pred_trans - target_trans) ** 2
        rot_error = (pred_rot - target_rot) ** 2
        
        if pred_uncertainty is not None:
            # Use predicted uncertainties
            pred_trans_uncertainty = pred_uncertainty[:, :3]
            pred_rot_uncertainty = pred_uncertainty[:, 3:]
            
            # Uncertainty-weighted loss: L = 0.5 * exp(-log_var) * error^2 + 0.5 * log_var
            trans_precision = torch.exp(-torch.log(pred_trans_uncertainty + 1e-8))
            rot_precision = torch.exp(-torch.log(pred_rot_uncertainty + 1e-8))
            
            trans_loss = (0.5 * trans_precision * trans_error + 
                         0.5 * torch.log(pred_trans_uncertainty + 1e-8)).mean()
            rot_loss = (0.5 * rot_precision * rot_error + 
                       0.5 * torch.log(pred_rot_uncertainty + 1e-8)).mean()
        else:
            # Use learnable task uncertainties
            trans_precision = torch.exp(-self.log_var_trans)
            rot_precision = torch.exp(-self.log_var_rot)
            
            trans_loss = 0.5 * trans_precision * trans_error.mean() + 0.5 * self.log_var_trans
            rot_loss = 0.5 * rot_precision * rot_error.mean() + 0.5 * self.log_var_rot
        
        total_loss = trans_loss + rot_loss
        
        return {
            'total_loss': total_loss,
            'translation_loss': trans_error.mean(), 
            'rotation_loss': rot_error.mean(),
            'uncertainty_weighted_trans_loss': trans_loss,
            'uncertainty_weighted_rot_loss': rot_loss,
            'trans_precision': trans_precision.item() if isinstance(trans_precision, torch.Tensor) and trans_precision.numel() == 1 else trans_precision.mean().item(),
            'rot_precision': rot_precision.item() if isinstance(rot_precision, torch.Tensor) and rot_precision.numel() == 1 else rot_precision.mean().item()
        }


class RobustPoseLoss(nn.Module):
    """
    Robust pose loss that handles outliers
    
    Combines multiple robust loss functions
    """
    
    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 10.0,
        robust_type: str = 'huber',  # 'huber', 'cauchy', 'geman_mcclure'
        huber_delta: float = 1.0,
        cauchy_c: float = 1.0
    ):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.robust_type = robust_type
        self.huber_delta = huber_delta
        self.cauchy_c = cauchy_c
        
    def _robust_loss(self, error: torch.Tensor) -> torch.Tensor:
        """Apply robust loss function to error"""
        if self.robust_type == 'huber':
            return F.smooth_l1_loss(error, torch.zeros_like(error), beta=self.huber_delta, reduction='none')
        elif self.robust_type == 'cauchy':
            # Cauchy (Lorentzian) loss: log(1 + (error/c)^2)
            return torch.log(1 + (error / self.cauchy_c) ** 2)
        elif self.robust_type == 'geman_mcclure':
            # Geman-McClure loss: error^2 / (1 + error^2)
            error_sq = error ** 2
            return error_sq / (1 + error_sq)
        else:
            raise ValueError(f"Unknown robust loss type: {self.robust_type}")
    
    def forward(
        self,
        pred_pose: torch.Tensor,
        target_pose: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute robust pose loss"""
        # Split translation and rotation
        pred_trans = pred_pose[:, :3]
        pred_rot = pred_pose[:, 3:]
        target_trans = target_pose[:, :3]
        target_rot = target_pose[:, 3:]
        
        # Compute errors
        trans_error = pred_trans - target_trans
        rot_error = pred_rot - target_rot
        
        # Apply robust loss
        trans_robust_loss = self._robust_loss(trans_error).mean()
        rot_robust_loss = self._robust_loss(rot_error).mean()
        
        # Standard losses for comparison
        trans_mse = F.mse_loss(pred_trans, target_trans)
        rot_mse = F.mse_loss(pred_rot, target_rot)
        
        # Weighted combination
        total_loss = (self.translation_weight * trans_robust_loss + 
                     self.rotation_weight * rot_robust_loss)
        
        return {
            'total_loss': total_loss,
            'translation_loss': trans_mse,
            'rotation_loss': rot_mse,
            'robust_translation_loss': trans_robust_loss,
            'robust_rotation_loss': rot_robust_loss
        }


class GeometricPoseLoss(nn.Module):
    """
    Geometrically-aware pose loss
    
    Considers the manifold structure of rotation space
    """
    
    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 10.0,
        rotation_repr: str = 'euler'  # 'euler', 'quaternion', 'rotation_matrix'
    ):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.rotation_repr = rotation_repr
        
    def _geodesic_rotation_loss(
        self,
        pred_rot: torch.Tensor,
        target_rot: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute geodesic distance on rotation manifold
        
        For Euler angles, we approximate the geodesic distance
        """
        if self.rotation_repr == 'euler':
            # For small angles, Euler angle difference approximates geodesic distance
            angle_diff = pred_rot - target_rot
            
            # Wrap angles to [-pi, pi]
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            
            # Compute geodesic distance (approximation)
            geodesic_dist = torch.norm(angle_diff, dim=1)
            return geodesic_dist.mean()
        else:
            # Fallback to MSE for other representations
            return F.mse_loss(pred_rot, target_rot)
    
    def forward(
        self,
        pred_pose: torch.Tensor,
        target_pose: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute geometrically-aware pose loss"""
        # Split translation and rotation
        pred_trans = pred_pose[:, :3]
        pred_rot = pred_pose[:, 3:]
        target_trans = target_pose[:, :3]
        target_rot = target_pose[:, 3:]
        
        # Translation loss (Euclidean)
        trans_loss = F.mse_loss(pred_trans, target_trans)
        
        # Rotation loss (geodesic)
        rot_loss = self._geodesic_rotation_loss(pred_rot, target_rot)
        
        # Weighted combination
        total_loss = (self.translation_weight * trans_loss + 
                     self.rotation_weight * rot_loss)
        
        return {
            'total_loss': total_loss,
            'translation_loss': trans_loss,
            'rotation_loss': rot_loss,
            'geodesic_rotation_loss': rot_loss
        }


def create_loss_function(config: Dict) -> nn.Module:
    """Factory function to create loss function from config"""
    loss_type = config.get('loss_type', 'pose')
    
    if loss_type == 'pose':
        return PoseLoss(
            translation_weight=config.get('translation_weight', 1.0),
            rotation_weight=config.get('rotation_weight', 10.0),
            loss_type=config.get('base_loss', 'mse'),
            huber_delta=config.get('huber_delta', 1.0)
        )
    elif loss_type == 'uncertainty':
        return UncertaintyWeightedLoss(
            initial_log_var_trans=config.get('initial_log_var_trans', 0.0),
            initial_log_var_rot=config.get('initial_log_var_rot', 0.0),
            learn_uncertainty=config.get('learn_uncertainty', True)
        )
    elif loss_type == 'robust':
        return RobustPoseLoss(
            translation_weight=config.get('translation_weight', 1.0),
            rotation_weight=config.get('rotation_weight', 10.0),
            robust_type=config.get('robust_type', 'huber'),
            huber_delta=config.get('huber_delta', 1.0),
            cauchy_c=config.get('cauchy_c', 1.0)
        )
    elif loss_type == 'geometric':
        return GeometricPoseLoss(
            translation_weight=config.get('translation_weight', 1.0),
            rotation_weight=config.get('rotation_weight', 10.0),
            rotation_repr=config.get('rotation_repr', 'euler')
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")