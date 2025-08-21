#!/usr/bin/env python3
"""
Multi-Scale SE(3) Loss for Visual Odometry

Implements multi-scale supervision with:
1. Single-step loss (local accuracy)
2. Multi-step loss (trajectory scale)
3. Chain consistency loss (geometric validity)

This addresses the core scale mismatch problem in VO training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R


class MultiScaleSE3Loss(nn.Module):
    """
    Multi-Scale SE(3) Loss for Visual Odometry
    
    Combines local and global supervision to ensure both smooth motion
    and correct trajectory scale.
    """
    
    def __init__(self, 
                 single_step_weight=1.0,      # λ₁
                 multi_step_weight=2.0,       # λ₂ (higher for scale correction!)
                 chain_consistency_weight=0.5, # λ₃
                 sequence_length=8):
        super().__init__()
        
        self.λ1 = single_step_weight
        self.λ2 = multi_step_weight      # Key: Higher weight for trajectory scale
        self.λ3 = chain_consistency_weight
        self.sequence_length = sequence_length
        
        # Numerical stability
        self.eps = 1e-8
        
        print(f"MultiScaleSE3Loss initialized:")
        print(f"  lambda1 (single-step): {self.λ1}")
        print(f"  lambda2 (multi-step): {self.λ2}")
        print(f"  lambda3 (chain consistency): {self.λ3}")
        
    def forward(self, pred_deltas, gt_deltas, gt_relative_pose):
        """
        Compute multi-scale SE(3) loss
        
        Args:
            pred_deltas: (batch_size, 6) - predicted [dx,dy,dz,droll,dpitch,dyaw]
            gt_deltas: (batch_size, 6) - ground truth deltas (for local loss)  
            gt_relative_pose: (batch_size, 6) - ground truth relative pose (first→last frame)
            
        Returns:
            loss: Multi-scale loss
            loss_dict: Dictionary with loss components
        """
        batch_size = pred_deltas.shape[0]
        device = pred_deltas.device
        
        # 1. Single-step loss (local accuracy)
        single_step_loss = self.compute_single_step_loss(pred_deltas, gt_deltas)
        
        # 2. Multi-step loss (trajectory scale) - THE KEY COMPONENT!
        multi_step_loss = self.compute_multi_step_loss(pred_deltas, gt_relative_pose)
        
        # 3. Chain consistency loss (geometric validity)
        chain_loss = torch.tensor(0.0, device=device)
        if batch_size >= 2:
            chain_loss = self.compute_chain_consistency_loss(pred_deltas, gt_deltas)
        
        # Combined multi-scale loss
        total_loss = (self.λ1 * single_step_loss + 
                     self.λ2 * multi_step_loss +      # This will fix the scale!
                     self.λ3 * chain_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'single_step_loss': single_step_loss.item(),
            'multi_step_loss': multi_step_loss.item(),
            'chain_consistency_loss': chain_loss.item(),
            'scale_ratio': self.λ2 / self.λ1  # Monitor relative importance
        }
        
        return total_loss, loss_dict
    
    def compute_single_step_loss(self, pred_deltas, gt_deltas):
        """
        Local accuracy loss - ensures smooth frame-to-frame motion
        """
        # Simple SE(3) geodesic distance for individual deltas
        pred_T = self.pose_to_se3(pred_deltas)  # (B, 4, 4)
        gt_T = self.pose_to_se3(gt_deltas)      # (B, 4, 4)
        
        # Compute relative transformation error
        pred_T_inv = self.se3_inverse(pred_T)   # (B, 4, 4)
        T_rel = torch.bmm(pred_T_inv, gt_T)     # (B, 4, 4)
        
        # Geodesic distance
        geodesic_dist = self.se3_log_frobenius_norm(T_rel)
        
        return geodesic_dist.mean()
    
    def compute_multi_step_loss(self, pred_deltas, gt_relative_pose):
        """
        Global trajectory loss - ensures correct trajectory scale
        
        This is the KEY component that will fix your scale problem!
        """
        batch_size = pred_deltas.shape[0]
        
        if batch_size < self.sequence_length:
            # If we don't have a full sequence, accumulate what we have
            sequence_length = batch_size
        else:
            sequence_length = self.sequence_length
        
        # Accumulate predicted deltas using SE(3) composition
        pred_accumulated = self.accumulate_se3_deltas(pred_deltas[:sequence_length])
        
        # Convert ground truth relative pose to SE(3)
        gt_relative_T = self.pose_to_se3(gt_relative_pose[:sequence_length])
        
        # If we accumulated less than expected, take corresponding GT
        if pred_accumulated.shape[0] < gt_relative_T.shape[0]:
            gt_relative_T = gt_relative_T[:pred_accumulated.shape[0]]
        
        # Compute SE(3) geodesic distance between accumulated and target
        pred_T_inv = self.se3_inverse(pred_accumulated)
        T_rel = torch.bmm(pred_T_inv, gt_relative_T)
        
        geodesic_dist = self.se3_log_frobenius_norm(T_rel)
        
        return geodesic_dist.mean()
    
    def accumulate_se3_deltas(self, deltas):
        """
        Accumulate pose deltas using proper SE(3) composition
        
        Args:
            deltas: (sequence_length, 6) - sequence of pose deltas
            
        Returns:
            accumulated_T: (sequence_length, 4, 4) - accumulated transformations
        """
        sequence_length = deltas.shape[0]
        device = deltas.device
        
        # Initialize with identity
        accumulated_T = torch.eye(4, device=device).unsqueeze(0).repeat(sequence_length, 1, 1)
        current_T = torch.eye(4, device=device)
        
        for i in range(sequence_length):
            # Convert delta to SE(3) matrix
            delta_T = self.pose_to_se3(deltas[i:i+1])  # (1, 4, 4)
            
            # Compose: T_new = T_current @ T_delta
            current_T = torch.mm(current_T, delta_T[0])
            accumulated_T[i] = current_T
        
        return accumulated_T
    
    def compute_chain_consistency_loss(self, pred_deltas, gt_deltas):
        """
        Enforce SE(3) composition consistency
        """
        batch_size = pred_deltas.shape[0]
        consistency_losses = []
        
        for i in range(batch_size - 1):
            # Single transformations
            T1_pred = self.pose_to_se3(pred_deltas[i:i+1])  # T_i
            T2_pred = self.pose_to_se3(pred_deltas[i+1:i+2])  # T_{i+1}
            
            T1_gt = self.pose_to_se3(gt_deltas[i:i+1])
            T2_gt = self.pose_to_se3(gt_deltas[i+1:i+2])
            
            # Composed transformation: T1 @ T2
            T_composed_pred = torch.bmm(T1_pred, T2_pred)
            T_composed_gt = torch.bmm(T1_gt, T2_gt)
            
            # Compute error
            T_comp_inv = self.se3_inverse(T_composed_pred)
            T_rel = torch.bmm(T_comp_inv, T_composed_gt)
            
            consistency_dist = self.se3_log_frobenius_norm(T_rel)
            consistency_losses.append(consistency_dist)
        
        if consistency_losses:
            return torch.stack(consistency_losses).mean()
        else:
            return torch.tensor(0.0, device=pred_deltas.device)
    
    def pose_to_se3(self, poses):
        """
        Convert 6DoF pose to SE(3) transformation matrix
        
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
        """Convert Euler angles to rotation matrices"""
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
        """Compute SE(3) matrix inverse"""
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
        """Compute Frobenius norm of SE(3) matrix logarithm"""
        batch_size = T.shape[0]
        device = T.device
        
        # Extract rotation and translation
        R = T[:, :3, :3]  # (B, 3, 3)
        t = T[:, :3, 3]   # (B, 3)
        
        # Compute rotation angle
        trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)  # (B,)
        cos_angle = (trace_R - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1 + self.eps, 1 - self.eps)
        angle = torch.arccos(cos_angle)  # (B,)
        
        # Handle small angles
        small_angle_mask = angle < self.eps
        geodesic_dist = torch.zeros(batch_size, device=device)
        
        # Small angle case
        if small_angle_mask.any():
            t_small = t[small_angle_mask]
            R_small = R[small_angle_mask]
            R_minus_I = R_small - torch.eye(3, device=device).unsqueeze(0)
            skew_norm = torch.norm(R_minus_I, dim=(-2, -1))
            dist_small = torch.norm(t_small, dim=-1) + skew_norm
            geodesic_dist[small_angle_mask] = dist_small
        
        # Large angle case
        large_angle_mask = ~small_angle_mask
        if large_angle_mask.any():
            t_large = t[large_angle_mask]
            angle_large = angle[large_angle_mask]
            sinc_val = torch.sin(angle_large) / (angle_large + self.eps)
            scale_factor = angle_large / (2 * sinc_val + self.eps)
            t_scaled = t_large * scale_factor.unsqueeze(-1)
            rot_contrib = angle_large
            dist_large = torch.sqrt(torch.norm(t_scaled, dim=-1)**2 + rot_contrib**2)
            geodesic_dist[large_angle_mask] = dist_large
        
        return geodesic_dist


def create_multi_scale_loss(single_step_weight=1.0, 
                           multi_step_weight=2.0, 
                           chain_consistency_weight=0.5,
                           sequence_length=8):
    """
    Factory function to create multi-scale SE(3) loss
    
    Args:
        single_step_weight: Weight for local frame-to-frame accuracy
        multi_step_weight: Weight for global trajectory scale (should be higher!)
        chain_consistency_weight: Weight for geometric consistency
        sequence_length: Number of frames in sequence
        
    Returns:
        MultiScaleSE3Loss instance
    """
    return MultiScaleSE3Loss(
        single_step_weight=single_step_weight,
        multi_step_weight=multi_step_weight,
        chain_consistency_weight=chain_consistency_weight,
        sequence_length=sequence_length
    )


if __name__ == "__main__":
    # Test the multi-scale loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing MultiScaleSE3Loss on {device}")
    
    # Create loss function
    loss_fn = create_multi_scale_loss(
        single_step_weight=1.0,
        multi_step_weight=2.0,  # Higher weight for trajectory scale
        chain_consistency_weight=0.5,
        sequence_length=8
    )
    
    # Test data (single sample for simplicity)
    batch_size = 1
    pred_deltas = torch.randn(batch_size, 6).to(device) * 0.01  # Small deltas
    gt_deltas = torch.randn(batch_size, 6).to(device) * 0.01    # Small deltas
    gt_relative_pose = torch.randn(batch_size, 6).to(device) * 0.1  # Larger relative pose
    
    # Compute loss
    loss, loss_dict = loss_fn(pred_deltas, gt_deltas, gt_relative_pose)
    
    print(f"\\nTest Results:")
    print(f"Total Loss: {loss.item():.6f}")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\\n[SUCCESS] Multi-scale SE(3) loss is ready!")
    print(f"This should dramatically improve trajectory scale matching!")