#!/usr/bin/env python3
"""
Global Trajectory Visualization for UW-TransVO
Reconstructs and visualizes the complete robot trajectory using trained model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json
import sys
from pathlib import Path
import time
from datetime import datetime
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.datasets import UnderwaterVODataset
from training.losses import PoseLoss

class TrajectoryVisualizer:
    """Comprehensive trajectory visualization and analysis"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        
        # Load model configuration from checkpoint
        self.model_config = self._infer_model_config()
        self.model = self._load_model()
        
        print(f"Trajectory Visualizer initialized")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _infer_model_config(self):
        """Infer model configuration from checkpoint"""
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Infer d_model from embedding dimensions
        d_model = state_dict['vision_transformer.cls_token'].shape[-1]
        
        # Infer num_layers
        num_layers = 0
        for key in state_dict.keys():
            if 'vision_transformer.blocks.' in key:
                layer_num = int(key.split('.')[2])
                num_layers = max(num_layers, layer_num + 1)
        
        # Infer num_heads
        for key in state_dict.keys():
            if 'attn.qkv.weight' in key:
                qkv_dim = state_dict[key].shape[0]
                num_heads = qkv_dim // (d_model * 3)
                break
        
        return {
            'img_size': 224, 'patch_size': 16, 'd_model': d_model,
            'num_heads': num_heads, 'num_layers': num_layers,
            'max_cameras': 4, 'max_seq_len': 2, 'dropout': 0.1,
            'use_imu': False, 'use_pressure': False, 'uncertainty_estimation': True
        }
    
    def _load_model(self):
        """Load trained model"""
        model = UWTransVO(**self.model_config).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def reconstruct_trajectory(self, dataset, batch_size=1):
        """Reconstruct complete trajectory from sequential predictions"""
        print(f"\\nReconstructing trajectory from {len(dataset)} samples...")
        
        # Use sequential dataloader (no shuffling)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Storage for trajectory reconstruction
        predicted_poses = []
        ground_truth_poses = []
        relative_predictions = []
        relative_ground_truth = []
        timestamps = []
        
        # Initialize global pose (starting at origin)
        current_pose_pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, z, roll, pitch, yaw]
        current_pose_gt = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        predicted_poses.append(current_pose_pred.copy())
        ground_truth_poses.append(current_pose_gt.copy())
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                images = batch['images'].to(self.device)
                camera_ids = batch['camera_ids'].to(self.device)
                camera_mask = batch['camera_mask'].to(self.device)
                pose_target = batch['pose_target'].to(self.device)
                
                # Forward pass
                output = self.model(
                    images=images,
                    camera_ids=camera_ids,
                    camera_mask=camera_mask
                )
                
                # Get relative pose predictions
                pred_relative = output['pose'].cpu().numpy().squeeze()
                true_relative = pose_target.cpu().numpy().squeeze()
                
                relative_predictions.append(pred_relative)
                relative_ground_truth.append(true_relative)
                
                # Integrate relative motion to get global pose
                current_pose_pred = self._integrate_pose(current_pose_pred, pred_relative)
                current_pose_gt = self._integrate_pose(current_pose_gt, true_relative)
                
                predicted_poses.append(current_pose_pred.copy())
                ground_truth_poses.append(current_pose_gt.copy())
                
                timestamps.append(batch_idx * 0.1)  # Assume 10Hz
                
                if batch_idx % 20 == 0:
                    print(f"  Processed {batch_idx+1}/{len(dataloader)} sequences")
        
        return {
            'predicted_trajectory': np.array(predicted_poses),
            'ground_truth_trajectory': np.array(ground_truth_poses),
            'relative_predictions': np.array(relative_predictions),
            'relative_ground_truth': np.array(relative_ground_truth),
            'timestamps': np.array(timestamps)
        }
    
    def _integrate_pose(self, current_pose, relative_pose):
        """Integrate relative pose to get new global pose"""
        # Simple integration for demonstration
        # In practice, proper SE(3) integration should be used
        new_pose = current_pose.copy()
        
        # Translation (assuming relative motion in current frame)
        new_pose[:3] += relative_pose[:3]
        
        # Rotation (simple addition for small angles)
        new_pose[3:] += relative_pose[3:]
        
        return new_pose
    
    def calculate_drift_metrics(self, trajectory_data):
        """Calculate trajectory drift and accuracy metrics"""
        pred_traj = trajectory_data['predicted_trajectory']
        gt_traj = trajectory_data['ground_truth_trajectory']
        
        # Calculate absolute trajectory error (ATE)
        position_errors = np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1)
        rotation_errors = np.linalg.norm(pred_traj[:, 3:] - gt_traj[:, 3:], axis=1)
        
        # Calculate drift (final position error)
        final_position_error = np.linalg.norm(pred_traj[-1, :3] - gt_traj[-1, :3])
        final_rotation_error = np.linalg.norm(pred_traj[-1, 3:] - gt_traj[-1, 3:])
        
        # Calculate trajectory length
        trajectory_length = np.sum(np.linalg.norm(np.diff(gt_traj[:, :3], axis=0), axis=1))
        
        # Relative drift (drift as percentage of trajectory length)
        relative_drift = (final_position_error / trajectory_length) * 100 if trajectory_length > 0 else 0
        
        metrics = {
            'trajectory_length_m': float(trajectory_length),
            'final_position_error_m': float(final_position_error),
            'final_rotation_error_rad': float(final_rotation_error),
            'final_rotation_error_deg': float(final_rotation_error * 180 / np.pi),
            'relative_drift_percent': float(relative_drift),
            'mean_position_error_m': float(np.mean(position_errors)),
            'max_position_error_m': float(np.max(position_errors)),
            'mean_rotation_error_rad': float(np.mean(rotation_errors)),
            'mean_rotation_error_deg': float(np.mean(rotation_errors) * 180 / np.pi),
            'rmse_position_m': float(np.sqrt(np.mean(position_errors**2))),
            'rmse_rotation_rad': float(np.sqrt(np.mean(rotation_errors**2)))
        }
        
        return metrics, position_errors, rotation_errors
    
    def create_trajectory_visualizations(self, trajectory_data, metrics, output_dir='trajectory_results'):
        """Create comprehensive trajectory visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        pred_traj = trajectory_data['predicted_trajectory']
        gt_traj = trajectory_data['ground_truth_trajectory']
        timestamps = trajectory_data['timestamps']
        
        plt.style.use('dark_background')
        
        # 1. 3D Trajectory Plot
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 
               'g-', linewidth=3, label='Ground Truth', alpha=0.8)
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 
               'r--', linewidth=2, label='Predicted', alpha=0.8)
        
        # Mark start and end points
        ax.scatter([gt_traj[0, 0]], [gt_traj[0, 1]], [gt_traj[0, 2]], 
                  c='lime', s=100, marker='o', label='Start')
        ax.scatter([gt_traj[-1, 0]], [gt_traj[-1, 1]], [gt_traj[-1, 2]], 
                  c='red', s=100, marker='s', label='End (GT)')
        ax.scatter([pred_traj[-1, 0]], [pred_traj[-1, 1]], [pred_traj[-1, 2]], 
                  c='orange', s=100, marker='^', label='End (Pred)')
        
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_zlabel('Z (m)', color='white')
        ax.set_title('UW-TransVO: 3D Trajectory Reconstruction\\n' + 
                    f'Final Drift: {metrics["final_position_error_m"]:.3f}m ' +
                    f'({metrics["relative_drift_percent"]:.2f}% of trajectory)', 
                    color='white', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(output_dir / '3d_trajectory.png', dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        # 2. 2D Top-down View
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('UW-TransVO: Trajectory Analysis', fontsize=16, color='white')
        
        # XY trajectory (top-down)
        axes[0,0].plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', linewidth=3, label='Ground Truth', alpha=0.8)
        axes[0,0].plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        axes[0,0].scatter([gt_traj[0, 0]], [gt_traj[0, 1]], c='lime', s=80, marker='o', label='Start')
        axes[0,0].scatter([gt_traj[-1, 0]], [gt_traj[-1, 1]], c='red', s=80, marker='s', label='End (GT)')
        axes[0,0].scatter([pred_traj[-1, 0]], [pred_traj[-1, 1]], c='orange', s=80, marker='^', label='End (Pred)')
        axes[0,0].set_xlabel('X (m)', color='white')
        axes[0,0].set_ylabel('Y (m)', color='white')
        axes[0,0].set_title('Top-down View (XY)', color='white')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axis('equal')
        
        # Depth trajectory (XZ)
        axes[0,1].plot(gt_traj[:, 0], gt_traj[:, 2], 'g-', linewidth=3, label='Ground Truth', alpha=0.8)
        axes[0,1].plot(pred_traj[:, 0], pred_traj[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        axes[0,1].set_xlabel('X (m)', color='white')
        axes[0,1].set_ylabel('Z (m)', color='white')
        axes[0,1].set_title('Side View (XZ)', color='white')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Position error over time
        position_errors = np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1)
        # Ensure same length by using minimum length
        min_len = min(len(timestamps), len(position_errors))
        axes[1,0].plot(timestamps[:min_len], position_errors[:min_len], 'cyan', linewidth=2)
        axes[1,0].set_xlabel('Time (s)', color='white')
        axes[1,0].set_ylabel('Position Error (m)', color='white')
        axes[1,0].set_title('Position Error Over Time', color='white')
        axes[1,0].grid(True, alpha=0.3)
        
        # Rotation error over time
        rotation_errors = np.linalg.norm(pred_traj[:, 3:] - gt_traj[:, 3:], axis=1) * 180 / np.pi
        min_len = min(len(timestamps), len(rotation_errors))
        axes[1,1].plot(timestamps[:min_len], rotation_errors[:min_len], 'orange', linewidth=2)
        axes[1,1].set_xlabel('Time (s)', color='white')
        axes[1,1].set_ylabel('Rotation Error (°)', color='white')
        axes[1,1].set_title('Rotation Error Over Time', color='white')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trajectory_analysis.png', dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        # 3. Error Distribution Analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Error Analysis and Distribution', fontsize=16, color='white')
        
        # Position error histogram
        axes[0,0].hist(position_errors, bins=30, color='cyan', alpha=0.7, edgecolor='white')
        axes[0,0].set_xlabel('Position Error (m)', color='white')
        axes[0,0].set_ylabel('Frequency', color='white')
        axes[0,0].set_title('Position Error Distribution', color='white')
        axes[0,0].grid(True, alpha=0.3)
        
        # Rotation error histogram
        axes[0,1].hist(rotation_errors, bins=30, color='orange', alpha=0.7, edgecolor='white')
        axes[0,1].set_xlabel('Rotation Error (°)', color='white')
        axes[0,1].set_ylabel('Frequency', color='white')
        axes[0,1].set_title('Rotation Error Distribution', color='white')
        axes[0,1].grid(True, alpha=0.3)
        
        # Cumulative error
        cumulative_error = np.cumsum(position_errors)
        min_len = min(len(timestamps), len(cumulative_error))
        axes[1,0].plot(timestamps[:min_len], cumulative_error[:min_len], 'red', linewidth=2)
        axes[1,0].set_xlabel('Time (s)', color='white')
        axes[1,0].set_ylabel('Cumulative Error (m)', color='white')
        axes[1,0].set_title('Cumulative Position Error', color='white')
        axes[1,0].grid(True, alpha=0.3)
        
        # Error growth rate
        error_growth = np.gradient(position_errors)
        min_len = min(len(timestamps), len(error_growth))
        axes[1,1].plot(timestamps[:min_len], error_growth[:min_len], 'magenta', linewidth=2)
        axes[1,1].set_xlabel('Time (s)', color='white')
        axes[1,1].set_ylabel('Error Growth Rate (m/s)', color='white')
        axes[1,1].set_title('Error Growth Rate', color='white')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"Trajectory visualizations saved to {output_dir}")
        
        return output_dir

def main():
    print("UW-TransVO Trajectory Visualization and Analysis")
    print("=" * 60)
    
    # Configuration
    checkpoint_path = 'checkpoints/test_4cam_seq2_vision/best_model.pth'
    data_csv = 'data/processed/training_dataset/training_data.csv'
    output_dir = 'trajectory_results'
    
    # Initialize visualizer
    visualizer = TrajectoryVisualizer(checkpoint_path)
    
    # Create sequential dataset for trajectory reconstruction
    print("\\nLoading sequential dataset...")
    dataset = UnderwaterVODataset(
        data_csv=data_csv,
        data_root='.',
        camera_ids=[0, 1, 2, 3],
        sequence_length=2,
        img_size=224,
        use_imu=False,
        use_pressure=False,
        augmentation=False,
        split='val',  # Use validation data
        max_samples=50  # Limit for visualization
    )
    
    print(f"Dataset loaded: {len(dataset)} sequential samples")
    
    # Reconstruct trajectory
    trajectory_data = visualizer.reconstruct_trajectory(dataset)
    
    # Calculate drift metrics
    print("\\nCalculating drift and accuracy metrics...")
    metrics, position_errors, rotation_errors = visualizer.calculate_drift_metrics(trajectory_data)
    
    # Print results
    print("\\n" + "="*60)
    print("TRAJECTORY ANALYSIS RESULTS")
    print("="*60)
    print(f"Trajectory Length: {metrics['trajectory_length_m']:.2f} m")
    print(f"Final Position Drift: {metrics['final_position_error_m']:.4f} m")
    print(f"Final Rotation Drift: {metrics['final_rotation_error_deg']:.2f}°")
    print(f"Relative Drift: {metrics['relative_drift_percent']:.3f}% of trajectory")
    print(f"Mean Position Error: {metrics['mean_position_error_m']:.4f} m")
    print(f"RMSE Position: {metrics['rmse_position_m']:.4f} m")
    print(f"Mean Rotation Error: {metrics['mean_rotation_error_deg']:.2f}°")
    print("="*60)
    
    # Create visualizations
    print("\\nGenerating trajectory visualizations...")
    vis_dir = visualizer.create_trajectory_visualizations(trajectory_data, metrics, output_dir)
    
    # Save detailed results
    results = {
        'analysis_info': {
            'timestamp': datetime.now().isoformat(),
            'model_checkpoint': str(checkpoint_path),
            'dataset_samples': len(dataset),
            'model_parameters': sum(p.numel() for p in visualizer.model.parameters())
        },
        'trajectory_metrics': metrics,
        'model_config': visualizer.model_config
    }
    
    with open(Path(output_dir) / 'trajectory_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save trajectory data
    np.savez(Path(output_dir) / 'trajectory_data.npz',
             predicted_trajectory=trajectory_data['predicted_trajectory'],
             ground_truth_trajectory=trajectory_data['ground_truth_trajectory'],
             timestamps=trajectory_data['timestamps'])
    
    print(f"\\nComplete trajectory analysis saved to: {output_dir}")
    print("Generated files:")
    print("  - 3d_trajectory.png: 3D trajectory visualization")
    print("  - trajectory_analysis.png: 2D trajectory and error analysis")
    print("  - error_analysis.png: Detailed error distribution")
    print("  - trajectory_analysis.json: Complete metrics")
    print("  - trajectory_data.npz: Raw trajectory data")
    
    return results

if __name__ == '__main__':
    results = main()