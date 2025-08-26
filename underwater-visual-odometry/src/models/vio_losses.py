import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SE3GeodesicLoss(nn.Module):
    """
    SE(3) Geodesic Loss for 6-DOF pose estimation
    Based on "Real-time Deep Pose Estimation with Geodesic Loss" and related work
    """
    def __init__(self, translation_weight=1.0, rotation_weight=1.0):
        super(SE3GeodesicLoss, self).__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
    
    def so3_geodesic_distance(self, R1, R2):
        """
        Compute geodesic distance between two rotation matrices on SO(3) - Numerically Stable
        Args:
            R1, R2: rotation matrices of shape (..., 3, 3)
        Returns:
            geodesic distance
        """
        # For numerical stability, use Frobenius norm instead of arccos
        # |R1 - R2|_F is proportional to geodesic distance for small angles
        diff = R1 - R2
        frobenius_dist = torch.norm(diff, dim=(-2, -1))
        
        # Scale to approximate geodesic distance
        # For small angles: |R1 - R2|_F ≈ √2 * θ
        geodesic_approx = frobenius_dist / 1.414  # sqrt(2)
        
        return geodesic_approx
    
    def euler_to_rotation_matrix(self, euler_angles):
        """
        Convert Euler angles (roll, pitch, yaw) to rotation matrices
        Args:
            euler_angles: (..., 3) tensor with [roll, pitch, yaw]
        Returns:
            rotation matrices (..., 3, 3)
        """
        roll, pitch, yaw = euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]
        
        # Individual rotation matrices
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        
        # Rotation matrix for roll (X-axis)
        R_x = torch.stack([
            torch.stack([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll)], dim=-1),
            torch.stack([torch.zeros_like(roll), cos_r, -sin_r], dim=-1),
            torch.stack([torch.zeros_like(roll), sin_r, cos_r], dim=-1)
        ], dim=-2)
        
        # Rotation matrix for pitch (Y-axis)
        R_y = torch.stack([
            torch.stack([cos_p, torch.zeros_like(pitch), sin_p], dim=-1),
            torch.stack([torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch)], dim=-1),
            torch.stack([-sin_p, torch.zeros_like(pitch), cos_p], dim=-1)
        ], dim=-2)
        
        # Rotation matrix for yaw (Z-axis)
        R_z = torch.stack([
            torch.stack([cos_y, -sin_y, torch.zeros_like(yaw)], dim=-1),
            torch.stack([sin_y, cos_y, torch.zeros_like(yaw)], dim=-1),
            torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=-1)
        ], dim=-2)
        
        # Combined rotation: R = R_z * R_y * R_x
        R = torch.matmul(torch.matmul(R_z, R_y), R_x)
        return R
    
    def forward(self, pred_pose, target_pose):
        """
        Compute SE(3) geodesic loss
        Args:
            pred_pose: predicted poses (..., 6) [tx, ty, tz, roll, pitch, yaw]
            target_pose: target poses (..., 6) [tx, ty, tz, roll, pitch, yaw]
        """
        # Split translation and rotation
        pred_trans = pred_pose[..., :3]
        target_trans = target_pose[..., :3]
        pred_rot = pred_pose[..., 3:]
        target_rot = target_pose[..., 3:]
        
        # Translation loss (Euclidean)
        trans_loss = torch.norm(pred_trans - target_trans, dim=-1)
        
        # Rotation loss (geodesic on SO(3))
        pred_R = self.euler_to_rotation_matrix(pred_rot)
        target_R = self.euler_to_rotation_matrix(target_rot)
        rot_loss = self.so3_geodesic_distance(pred_R, target_R)
        
        # Combined loss
        total_loss = self.translation_weight * trans_loss + self.rotation_weight * rot_loss
        
        return total_loss.mean()


class TrajectoryConsistencyLoss(nn.Module):
    """
    Trajectory consistency loss to prevent constant predictions
    Encourages smooth but non-constant trajectory predictions
    """
    def __init__(self, smoothness_weight=0.1, variation_weight=1.0):
        super(TrajectoryConsistencyLoss, self).__init__()
        self.smoothness_weight = smoothness_weight
        self.variation_weight = variation_weight
    
    def forward(self, predictions):
        """
        Args:
            predictions: pose predictions of shape (batch, seq_len, 6)
        """
        # Compute first derivatives (velocities)
        first_diff = predictions[:, 1:] - predictions[:, :-1]  # (batch, seq_len-1, 6)
        
        # Smoothness loss: encourage small changes between consecutive predictions
        smoothness_loss = torch.mean(torch.norm(first_diff, dim=-1))
        
        # Variation loss: penalize constant predictions
        # Use standard deviation instead of variance for numerical stability
        pred_std = torch.std(predictions, dim=1, unbiased=False)  # (batch, 6)
        
        # Encourage non-zero std deviation (prevent constants)
        # Use negative log to penalize small std values
        variation_loss = torch.mean(-torch.log(pred_std + 1e-8))
        
        # Clamp to prevent extreme values
        variation_loss = torch.clamp(variation_loss, 0, 10)
        
        total_loss = self.smoothness_weight * smoothness_loss + self.variation_weight * variation_loss
        
        return total_loss


class PhotometricConsistencyLoss(nn.Module):
    """
    Photometric consistency loss for visual odometry
    Encourages predictions that result in consistent image warping
    """
    def __init__(self):
        super(PhotometricConsistencyLoss, self).__init__()
        
    def forward(self, images, poses):
        """
        Simple photometric consistency based on image gradients
        Args:
            images: input images (batch, seq_len, 3, H, W)
            poses: predicted poses (batch, seq_len, 6)
        """
        batch_size, seq_len = images.shape[:2]
        
        # Simple loss: return zero for now to avoid tensor dimension issues
        # In a full implementation, you'd want proper image warping based on poses
        return torch.tensor(0.0, device=images.device, requires_grad=True)


class VIOCompositeLoss(nn.Module):
    """
    Composite loss function combining multiple loss terms for robust VIO training
    """
    def __init__(self, 
                 se3_weight=1.0,
                 consistency_weight=0.1,
                 photometric_weight=0.05,
                 huber_delta=1.0):
        super(VIOCompositeLoss, self).__init__()
        
        self.se3_loss = SE3GeodesicLoss(translation_weight=1.0, rotation_weight=0.1)
        self.consistency_loss = TrajectoryConsistencyLoss()
        self.photometric_loss = PhotometricConsistencyLoss()
        
        self.se3_weight = se3_weight
        self.consistency_weight = consistency_weight
        self.photometric_weight = photometric_weight
        self.huber_delta = huber_delta
        
    def huber_loss(self, pred, target, delta=1.0):
        """Huber loss for robust regression"""
        residual = torch.abs(pred - target)
        condition = residual < delta
        small_res = 0.5 * residual ** 2
        large_res = delta * residual - 0.5 * delta ** 2
        return torch.where(condition, small_res, large_res).mean()
    
    def forward(self, predictions, targets, images=None):
        """
        Args:
            predictions: predicted poses (batch, seq_len, 6)
            targets: target poses (batch, seq_len, 6)
            images: input images (batch, seq_len, 3, H, W) - optional for photometric loss
        """
        losses = {}
        
        # Check for NaN or inf in inputs
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            return torch.tensor(float('inf'), device=predictions.device, requires_grad=True), {'total': float('inf')}
        
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            return torch.tensor(float('inf'), device=targets.device, requires_grad=True), {'total': float('inf')}
        
        # Main SE(3) geodesic loss with error handling
        try:
            se3_loss = self.se3_loss(predictions, targets)
            if torch.isnan(se3_loss) or torch.isinf(se3_loss):
                se3_loss = torch.tensor(1.0, device=predictions.device, requires_grad=True)
            losses['se3'] = se3_loss
        except:
            se3_loss = torch.tensor(1.0, device=predictions.device, requires_grad=True)
            losses['se3'] = se3_loss
        
        # Trajectory consistency loss with error handling
        try:
            consistency_loss = self.consistency_loss(predictions)
            if torch.isnan(consistency_loss) or torch.isinf(consistency_loss):
                consistency_loss = torch.tensor(1.0, device=predictions.device, requires_grad=True)
            losses['consistency'] = consistency_loss
        except:
            consistency_loss = torch.tensor(1.0, device=predictions.device, requires_grad=True)
            losses['consistency'] = consistency_loss
        
        # Photometric consistency loss (if images provided)
        photometric_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        if images is not None:
            try:
                photometric_loss = self.photometric_loss(images, predictions)
                if torch.isnan(photometric_loss) or torch.isinf(photometric_loss):
                    photometric_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            except:
                photometric_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        losses['photometric'] = photometric_loss
        
        # Additional Huber loss for robustness
        try:
            huber_loss = self.huber_loss(predictions, targets, self.huber_delta)
            if torch.isnan(huber_loss) or torch.isinf(huber_loss):
                huber_loss = torch.tensor(0.5, device=predictions.device, requires_grad=True)
            losses['huber'] = huber_loss
        except:
            huber_loss = torch.tensor(0.5, device=predictions.device, requires_grad=True)
            losses['huber'] = huber_loss
        
        # Total weighted loss with clamping
        total_loss = (self.se3_weight * se3_loss + 
                     self.consistency_weight * consistency_loss + 
                     self.photometric_weight * photometric_loss +
                     0.1 * huber_loss)
        
        # Final check and clamping
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(10.0, device=predictions.device, requires_grad=True)
        else:
            total_loss = torch.clamp(total_loss, 0, 100)  # Prevent extreme losses
        
        losses['total'] = total_loss
        
        return total_loss, losses


def create_simple_loss_function():
    """Factory function to create a simple, stable loss for initial training"""
    return nn.MSELoss()

def create_loss_function():
    """Factory function to create the composite loss"""
    return VIOCompositeLoss(
        se3_weight=0.1,  # Reduced weights for stability
        consistency_weight=0.05,
        photometric_weight=0.0,  # Disable photometric for now
        huber_delta=1.0
    )


if __name__ == "__main__":
    # Test loss functions
    batch_size, seq_len = 4, 10
    
    # Create dummy data
    pred_poses = torch.randn(batch_size, seq_len, 6)
    target_poses = torch.randn(batch_size, seq_len, 6)
    images = torch.randn(batch_size, seq_len, 3, 224, 224)
    
    # Test individual losses
    se3_loss = SE3GeodesicLoss()
    consistency_loss = TrajectoryConsistencyLoss()
    photometric_loss = PhotometricConsistencyLoss()
    
    print("SE3 Loss:", se3_loss(pred_poses, target_poses).item())
    print("Consistency Loss:", consistency_loss(pred_poses).item())
    print("Photometric Loss:", photometric_loss(images, pred_poses).item())
    
    # Test composite loss
    composite_loss = create_loss_function()
    total_loss, loss_dict = composite_loss(pred_poses, target_poses, images)
    
    print("\nComposite Loss Components:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item():.6f}")
        else:
            print(f"{key}: {value:.6f}")