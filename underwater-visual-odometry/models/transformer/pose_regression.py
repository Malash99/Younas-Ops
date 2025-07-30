"""
Pose regression head for UW-TransVO

Outputs 6-DOF pose (translation + rotation) with optional uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PoseRegressionHead(nn.Module):
    """
    6-DOF pose regression head with uncertainty estimation
    
    Outputs:
    - Translation: (delta_x, delta_y, delta_z) in meters
    - Rotation: (delta_roll, delta_pitch, delta_yaw) in radians
    - Uncertainty: Optional uncertainty estimates for each DOF
    """
    
    def __init__(
        self,
        d_model: int = 768,
        uncertainty_estimation: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.uncertainty_estimation = uncertainty_estimation
        
        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate heads for translation and rotation
        hidden_dim = d_model // 4
        
        # Translation head (delta_x, delta_y, delta_z)
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3 DOF for translation
        )
        
        # Rotation head (delta_roll, delta_pitch, delta_yaw)
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3 DOF for rotation
        )
        
        # Uncertainty estimation heads
        if uncertainty_estimation:
            # Translation uncertainty
            self.translation_uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3),
                nn.Softplus()  # Ensure positive uncertainty
            )
            
            # Rotation uncertainty
            self.rotation_uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3),
                nn.Softplus()  # Ensure positive uncertainty
            )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # Initialize pose outputs to small values
                if m in [self.translation_head[-1], self.rotation_head[-1]]:
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of pose regression head
        
        Args:
            features: Input features [batch_size, d_model]
            
        Returns:
            Dictionary containing:
            - pose: 6-DOF pose [batch_size, 6] (tx, ty, tz, rx, ry, rz)
            - translation: Translation [batch_size, 3]
            - rotation: Rotation [batch_size, 3]
            - uncertainty: Uncertainty estimates [batch_size, 6] (if enabled)
            - translation_uncertainty: Translation uncertainty [batch_size, 3] (if enabled)
            - rotation_uncertainty: Rotation uncertainty [batch_size, 3] (if enabled)
        """
        # Process features
        processed_features = self.feature_processor(features)
        
        # Predict translation and rotation
        translation = self.translation_head(processed_features)  # [batch_size, 3]
        rotation = self.rotation_head(processed_features)        # [batch_size, 3]
        
        # Combine into 6-DOF pose
        pose = torch.cat([translation, rotation], dim=1)  # [batch_size, 6]
        
        output = {
            'pose': pose,
            'translation': translation,
            'rotation': rotation
        }
        
        # Predict uncertainties if enabled
        if self.uncertainty_estimation:
            translation_uncertainty = self.translation_uncertainty_head(processed_features)
            rotation_uncertainty = self.rotation_uncertainty_head(processed_features)
            
            uncertainty = torch.cat([translation_uncertainty, rotation_uncertainty], dim=1)
            
            output.update({
                'uncertainty': uncertainty,
                'translation_uncertainty': translation_uncertainty,
                'rotation_uncertainty': rotation_uncertainty
            })
        
        return output
    
    def compute_weighted_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        translation_weight: float = 1.0,
        rotation_weight: float = 10.0,
        uncertainty_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss with optional uncertainty
        
        Args:
            predictions: Model predictions
            targets: Ground truth pose [batch_size, 6]
            translation_weight: Weight for translation loss
            rotation_weight: Weight for rotation loss
            uncertainty_weight: Weight for uncertainty regularization
            
        Returns:
            Dictionary of losses
        """
        device = predictions['pose'].device
        
        # Split targets
        target_translation = targets[:, :3]  # [batch_size, 3]
        target_rotation = targets[:, 3:]     # [batch_size, 3]
        
        # Compute basic losses
        translation_loss = F.mse_loss(predictions['translation'], target_translation)
        rotation_loss = F.mse_loss(predictions['rotation'], target_rotation)
        
        # Weighted combination
        total_loss = (translation_weight * translation_loss + 
                     rotation_weight * rotation_loss)
        
        loss_dict = {
            'total_loss': total_loss,
            'translation_loss': translation_loss,
            'rotation_loss': rotation_loss
        }
        
        # Add uncertainty-aware loss if available
        if self.uncertainty_estimation and 'uncertainty' in predictions:
            translation_uncertainty = predictions['translation_uncertainty']
            rotation_uncertainty = predictions['rotation_uncertainty']
            
            # Uncertainty-weighted loss (heteroscedastic uncertainty)
            # L = 0.5 * exp(-log_var) * ||pred - target||^2 + 0.5 * log_var
            
            # Translation uncertainty loss
            trans_precision = torch.exp(-torch.log(translation_uncertainty + 1e-8))
            trans_uncertainty_loss = (
                0.5 * trans_precision * (predictions['translation'] - target_translation) ** 2 +
                0.5 * torch.log(translation_uncertainty + 1e-8)
            ).mean()
            
            # Rotation uncertainty loss
            rot_precision = torch.exp(-torch.log(rotation_uncertainty + 1e-8))
            rot_uncertainty_loss = (
                0.5 * rot_precision * (predictions['rotation'] - target_rotation) ** 2 +
                0.5 * torch.log(rotation_uncertainty + 1e-8)
            ).mean()
            
            # Total uncertainty loss
            uncertainty_loss = trans_uncertainty_loss + rot_uncertainty_loss
            
            # Add to total loss
            total_loss = total_loss + uncertainty_weight * uncertainty_loss
            
            loss_dict.update({
                'total_loss': total_loss,
                'uncertainty_loss': uncertainty_loss,
                'translation_uncertainty_loss': trans_uncertainty_loss,
                'rotation_uncertainty_loss': rot_uncertainty_loss
            })
        
        return loss_dict


class MultiTaskPoseLoss(nn.Module):
    """
    Multi-task loss for pose regression with automatic weight balancing
    
    Implements uncertainty weighting as described in:
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """
    
    def __init__(
        self,
        initial_translation_weight: float = 1.0,
        initial_rotation_weight: float = 10.0,
        learnable_weights: bool = True
    ):
        super().__init__()
        
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            # Learnable task uncertainty parameters
            self.log_var_translation = nn.Parameter(torch.tensor(0.0))
            self.log_var_rotation = nn.Parameter(torch.tensor(0.0))
        else:
            self.translation_weight = initial_translation_weight
            self.rotation_weight = initial_rotation_weight
    
    def forward(
        self,
        translation_pred: torch.Tensor,
        rotation_pred: torch.Tensor,
        translation_target: torch.Tensor,
        rotation_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            translation_pred: Predicted translation [batch_size, 3]
            rotation_pred: Predicted rotation [batch_size, 3]
            translation_target: Target translation [batch_size, 3]
            rotation_target: Target rotation [batch_size, 3]
            
        Returns:
            Dictionary of losses
        """
        # Compute individual losses
        translation_loss = F.mse_loss(translation_pred, translation_target)
        rotation_loss = F.mse_loss(rotation_pred, rotation_target)
        
        if self.learnable_weights:
            # Automatic weight balancing using uncertainty
            precision_translation = torch.exp(-self.log_var_translation)
            precision_rotation = torch.exp(-self.log_var_rotation)
            
            # Weighted losses
            weighted_translation_loss = (
                precision_translation * translation_loss + self.log_var_translation
            )
            weighted_rotation_loss = (
                precision_rotation * rotation_loss + self.log_var_rotation
            )
            
            total_loss = weighted_translation_loss + weighted_rotation_loss
            
            return {
                'total_loss': total_loss,
                'translation_loss': translation_loss,
                'rotation_loss': rotation_loss,
                'weighted_translation_loss': weighted_translation_loss,
                'weighted_rotation_loss': weighted_rotation_loss,
                'translation_weight': precision_translation.item(),
                'rotation_weight': precision_rotation.item()
            }
        else:
            # Fixed weights
            total_loss = (
                self.translation_weight * translation_loss +
                self.rotation_weight * rotation_loss
            )
            
            return {
                'total_loss': total_loss,
                'translation_loss': translation_loss,
                'rotation_loss': rotation_loss,
                'translation_weight': self.translation_weight,
                'rotation_weight': self.rotation_weight
            }


def create_pose_regression_head(config: Dict) -> PoseRegressionHead:
    """Factory function to create pose regression head from config"""
    return PoseRegressionHead(
        d_model=config.get('d_model', 768),
        uncertainty_estimation=config.get('uncertainty_estimation', True),
        dropout=config.get('dropout', 0.1)
    )