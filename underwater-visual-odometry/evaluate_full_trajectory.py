#!/usr/bin/env python3
"""
Enhanced TSformer Visual Odometry Trajectory Evaluation

Complete trajectory reconstruction and ATE evaluation with comprehensive
3D visualizations showing all projection planes (XY, XZ, YZ).

Author: Underwater Visual Odometry Research Team
Date: January 2025
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders


def compute_ate(pred_trajectory, gt_trajectory):
    """
    Compute Absolute Trajectory Error (ATE).
    
    Args:
        pred_trajectory: (N, 6) predicted poses [x,y,z,roll,pitch,yaw]
        gt_trajectory: (N, 6) ground truth poses [x,y,z,roll,pitch,yaw]
    
    Returns:
        dict with ATE metrics
    """
    # Extract translation components
    pred_trans = pred_trajectory[:, :3]
    gt_trans = gt_trajectory[:, :3]
    
    # Compute translation errors
    trans_errors = np.linalg.norm(pred_trans - gt_trans, axis=1)
    
    # Extract rotation components and compute rotation errors
    pred_rot = pred_trajectory[:, 3:]
    gt_rot = gt_trajectory[:, 3:]
    rot_errors = np.linalg.norm(pred_rot - gt_rot, axis=1)
    
    ate_metrics = {
        'translation': {
            'rmse': np.sqrt(np.mean(trans_errors**2)),
            'mean': np.mean(trans_errors),
            'median': np.median(trans_errors),
            'std': np.std(trans_errors),
            'min': np.min(trans_errors),
            'max': np.max(trans_errors)
        },
        'rotation': {
            'rmse': np.sqrt(np.mean(rot_errors**2)),
            'mean': np.mean(rot_errors),
            'median': np.median(rot_errors),
            'std': np.std(rot_errors),
            'min': np.min(rot_errors),
            'max': np.max(rot_errors)
        },
        'errors': {
            'translation_errors': trans_errors.tolist(),
            'rotation_errors': rot_errors.tolist()
        }
    }
    
    return ate_metrics


class FullTrajectoryEvaluator:
    """Enhanced evaluator for complete trajectory reconstruction and ATE analysis"""
    
    def __init__(self, checkpoint_path, config):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model, self.loss_fn = create_tsformer_vo(
            sequence_length=config['sequence_length'],
            pretrained=config['pretrained'],
            freeze_backbone=config.get('freeze_backbone', False),
            image_size=config['image_size']
        )
        
        # Load checkpoint
        self.load_checkpoint()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load test data
        self.load_test_data()
        
    def load_checkpoint(self):
        """Load model weights from checkpoint."""
        print(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")
        
    def load_test_data(self):
        """Load test data and organize by sequential frames."""
        # Read the full dataset
        df = pd.read_csv(self.config['csv_path'])
        
        # Filter for test bag
        test_bag = self.config['test_bags'][0]
        self.test_data = df[df['bag_name'] == test_bag].copy()
        self.test_data = self.test_data.sort_values('frame_index').reset_index(drop=True)
        
        print(f"Test bag: {test_bag}")
        print(f"Total sequential frames: {len(self.test_data)}")
        
    def predict_full_trajectory(self):
        """Predict poses for the complete trajectory using sliding windows."""
        print("Predicting full trajectory...")
        
        sequence_length = self.config['sequence_length']
        overlap_frames = self.config['overlap_frames']
        stride = max(1, sequence_length - overlap_frames)
        
        # Initialize predictions array
        num_frames = len(self.test_data)
        all_predictions = np.zeros((num_frames, 6))
        prediction_counts = np.zeros(num_frames)
        
        # Create data loader for sequential prediction
        data_loaders = create_data_loaders(
            csv_path=self.config['csv_path'],
            data_root=self.config['data_root'],
            sequence_length=sequence_length,
            overlap_frames=overlap_frames,
            image_size=self.config['image_size'],
            batch_size=1,
            test_bags=self.config['test_bags'],
            num_workers=0,  # Use 0 to maintain order
            camera=self.config['camera']
        )
        
        test_loader = data_loaders['test_loader']
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Get frame indices for this window
                frame_indices = batch['frame_indices'][0]  # Remove batch dimension
                if hasattr(frame_indices, 'tolist'):
                    frame_indices = frame_indices.tolist()
                
                # Move data to device
                images = batch['images'].to(self.device)
                
                # Forward pass
                pred_poses = self.model(images).cpu().numpy()[0]  # Remove batch dimension
                
                # The prediction is for the last frame in the sequence
                last_frame_idx = frame_indices[-1]
                
                # Find the position in test_data
                data_idx = self.test_data[self.test_data['frame_index'] == last_frame_idx].index
                if len(data_idx) > 0:
                    data_idx = data_idx[0]
                    all_predictions[data_idx] += pred_poses
                    prediction_counts[data_idx] += 1
                
                if batch_idx % 100 == 0:
                    print(f"  Processed {batch_idx}/{len(test_loader)} windows")
        
        # Average predictions for overlapping windows
        mask = prediction_counts > 0
        all_predictions[mask] /= prediction_counts[mask, np.newaxis]
        
        # For frames without predictions, use ground truth deltas
        missing_frames = ~mask
        if missing_frames.any():
            print(f"Warning: {missing_frames.sum()} frames without predictions, using ground truth")
            for i in range(num_frames):
                if missing_frames[i]:
                    all_predictions[i] = np.array([
                        self.test_data.iloc[i]['delta_x'],
                        self.test_data.iloc[i]['delta_y'],
                        self.test_data.iloc[i]['delta_z'],
                        self.test_data.iloc[i]['delta_roll'],
                        self.test_data.iloc[i]['delta_pitch'],
                        self.test_data.iloc[i]['delta_yaw']
                    ])
        
        print(f"Prediction complete: {len(all_predictions)} poses")
        return all_predictions
    
    def integrate_trajectory(self, pose_deltas):
        """Integrate pose deltas to get absolute trajectory."""
        trajectory = np.zeros((len(pose_deltas), 6))
        
        # Start from origin
        current_pose = np.zeros(6)
        
        for i, delta in enumerate(pose_deltas):
            # Simple integration (good for small deltas)
            current_pose[:3] += delta[:3]  # Translation
            current_pose[3:] += delta[3:]  # Rotation
            trajectory[i] = current_pose.copy()
            
        return trajectory
    
    def get_ground_truth_trajectory(self):
        """Get ground truth trajectory from pose deltas."""
        gt_deltas = np.array([
            [row['delta_x'], row['delta_y'], row['delta_z'],
             row['delta_roll'], row['delta_pitch'], row['delta_yaw']]
            for _, row in self.test_data.iterrows()
        ])
        
        return self.integrate_trajectory(gt_deltas)
    
    def plot_comprehensive_trajectory(self, pred_traj, gt_traj, output_dir):
        """Create comprehensive trajectory visualization with all projection planes."""
        print("Creating comprehensive trajectory visualization...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 3D trajectory plot (main plot, larger)
        ax1 = fig.add_subplot(2, 3, (1, 4), projection='3d')
        ax1.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 
                'b-', linewidth=3, label='Ground Truth', alpha=0.8)
        ax1.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 
                'r--', linewidth=2, label='Predicted', alpha=0.8)
        
        # Mark start and end points
        ax1.scatter(gt_traj[0, 0], gt_traj[0, 1], gt_traj[0, 2], 
                   c='green', s=100, marker='o', label='Start')
        ax1.scatter(gt_traj[-1, 0], gt_traj[-1, 1], gt_traj[-1, 2], 
                   c='red', s=100, marker='s', label='End')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory', fontsize=16, fontweight='bold')
        ax1.legend()
        ax1.grid(True)
        
        # XY plane projection
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', linewidth=2, label='Ground Truth')
        ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, label='Predicted')
        ax2.scatter(gt_traj[0, 0], gt_traj[0, 1], c='green', s=50, marker='o', zorder=5)
        ax2.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c='red', s=50, marker='s', zorder=5)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Plane Projection')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # XZ plane projection
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(gt_traj[:, 0], gt_traj[:, 2], 'b-', linewidth=2, label='Ground Truth')
        ax3.plot(pred_traj[:, 0], pred_traj[:, 2], 'r--', linewidth=2, label='Predicted')
        ax3.scatter(gt_traj[0, 0], gt_traj[0, 2], c='green', s=50, marker='o', zorder=5)
        ax3.scatter(gt_traj[-1, 0], gt_traj[-1, 2], c='red', s=50, marker='s', zorder=5)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('XZ Plane Projection')
        ax3.legend()
        ax3.grid(True)
        ax3.axis('equal')
        
        # YZ plane projection
        ax4 = fig.add_subplot(2, 3, 5)
        ax4.plot(gt_traj[:, 1], gt_traj[:, 2], 'b-', linewidth=2, label='Ground Truth')
        ax4.plot(pred_traj[:, 1], pred_traj[:, 2], 'r--', linewidth=2, label='Predicted')
        ax4.scatter(gt_traj[0, 1], gt_traj[0, 2], c='green', s=50, marker='o', zorder=5)
        ax4.scatter(gt_traj[-1, 1], gt_traj[-1, 2], c='red', s=50, marker='s', zorder=5)
        ax4.set_xlabel('Y (m)')
        ax4.set_ylabel('Z (m)')
        ax4.set_title('YZ Plane Projection')
        ax4.legend()
        ax4.grid(True)
        ax4.axis('equal')
        
        # Translation error over time
        ax5 = fig.add_subplot(2, 3, 6)
        trans_errors = np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1)
        time_steps = np.arange(len(trans_errors))
        
        ax5.plot(time_steps, trans_errors, 'r-', linewidth=2, alpha=0.7)
        ax5.fill_between(time_steps, trans_errors, alpha=0.3, color='red')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Translation Error (m)')
        ax5.set_title(f'ATE Translation Error\nMean: {np.mean(trans_errors):.4f}m, RMSE: {np.sqrt(np.mean(trans_errors**2)):.4f}m')
        ax5.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / 'full_trajectory_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive trajectory plot saved: {output_path}")
        
    def plot_ate_analysis(self, ate_metrics, output_dir):
        """Create detailed ATE analysis plots."""
        print("Creating ATE analysis plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Translation error histogram
        trans_errors = ate_metrics['errors']['translation_errors']
        axes[0, 0].hist(trans_errors, bins=50, alpha=0.7, edgecolor='black', color='blue')
        axes[0, 0].axvline(ate_metrics['translation']['mean'], color='red', linestyle='--', 
                          label=f"Mean: {ate_metrics['translation']['mean']:.4f}m")
        axes[0, 0].axvline(ate_metrics['translation']['median'], color='green', linestyle='--',
                          label=f"Median: {ate_metrics['translation']['median']:.4f}m")
        axes[0, 0].set_xlabel('Translation Error (m)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Translation Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rotation error histogram
        rot_errors = ate_metrics['errors']['rotation_errors']
        axes[0, 1].hist(rot_errors, bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].axvline(ate_metrics['rotation']['mean'], color='red', linestyle='--',
                          label=f"Mean: {ate_metrics['rotation']['mean']:.4f}rad")
        axes[0, 1].axvline(ate_metrics['rotation']['median'], color='blue', linestyle='--',
                          label=f"Median: {ate_metrics['rotation']['median']:.4f}rad")
        axes[0, 1].set_xlabel('Rotation Error (rad)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Rotation Error Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Translation error over time
        time_steps = np.arange(len(trans_errors))
        axes[0, 2].plot(time_steps, trans_errors, 'b-', linewidth=1, alpha=0.7)
        axes[0, 2].fill_between(time_steps, trans_errors, alpha=0.3)
        axes[0, 2].set_xlabel('Time Step')
        axes[0, 2].set_ylabel('Translation Error (m)')
        axes[0, 2].set_title('Translation Error Over Time')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Rotation error over time
        axes[1, 0].plot(time_steps, rot_errors, 'g-', linewidth=1, alpha=0.7)
        axes[1, 0].fill_between(time_steps, rot_errors, alpha=0.3, color='green')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Rotation Error (rad)')
        axes[1, 0].set_title('Rotation Error Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative error distribution
        sorted_trans_errors = np.sort(trans_errors)
        cumulative_pct = np.arange(1, len(sorted_trans_errors) + 1) / len(sorted_trans_errors) * 100
        axes[1, 1].plot(sorted_trans_errors, cumulative_pct, 'b-', linewidth=2)
        axes[1, 1].axvline(ate_metrics['translation']['median'], color='red', linestyle='--', 
                          label=f"50th percentile: {ate_metrics['translation']['median']:.4f}m")
        axes[1, 1].axvline(np.percentile(trans_errors, 95), color='orange', linestyle='--',
                          label=f"95th percentile: {np.percentile(trans_errors, 95):.4f}m")
        axes[1, 1].set_xlabel('Translation Error (m)')
        axes[1, 1].set_ylabel('Cumulative Percentage')
        axes[1, 1].set_title('Cumulative Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # ATE metrics summary table
        axes[1, 2].axis('off')
        ate_summary = f"""
ATE Metrics Summary

Translation Errors:
• RMSE: {ate_metrics['translation']['rmse']:.4f} m
• Mean: {ate_metrics['translation']['mean']:.4f} m
• Median: {ate_metrics['translation']['median']:.4f} m
• Std: {ate_metrics['translation']['std']:.4f} m
• Min: {ate_metrics['translation']['min']:.4f} m
• Max: {ate_metrics['translation']['max']:.4f} m

Rotation Errors:
• RMSE: {ate_metrics['rotation']['rmse']:.4f} rad
• Mean: {ate_metrics['rotation']['mean']:.4f} rad
• Median: {ate_metrics['rotation']['median']:.4f} rad
• Std: {ate_metrics['rotation']['std']:.4f} rad
• Min: {ate_metrics['rotation']['min']:.4f} rad
• Max: {ate_metrics['rotation']['max']:.4f} rad

Percentiles (Translation):
• 50th: {np.percentile(trans_errors, 50):.4f} m
• 90th: {np.percentile(trans_errors, 90):.4f} m
• 95th: {np.percentile(trans_errors, 95):.4f} m
• 99th: {np.percentile(trans_errors, 99):.4f} m
        """
        axes[1, 2].text(0.1, 0.9, ate_summary, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / 'ate_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ATE analysis plot saved: {output_path}")
    
    def evaluate_full_trajectory(self, output_dir):
        """Run complete trajectory evaluation with ATE analysis."""
        print("\nRunning Full Trajectory Evaluation with ATE")
        print("="*60)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Predict full trajectory
        predicted_deltas = self.predict_full_trajectory()
        
        # Integrate to get absolute trajectories
        print("Integrating trajectories...")
        predicted_trajectory = self.integrate_trajectory(predicted_deltas)
        ground_truth_trajectory = self.get_ground_truth_trajectory()
        
        print(f"Trajectory length: {len(predicted_trajectory)} poses")
        
        # Compute ATE metrics
        print("Computing ATE metrics...")
        ate_metrics = compute_ate(predicted_trajectory, ground_truth_trajectory)
        
        # Print ATE results
        print("\n" + "="*60)
        print("ABSOLUTE TRAJECTORY ERROR (ATE) RESULTS")
        print("="*60)
        print(f"Translation ATE:")
        print(f"  RMSE: {ate_metrics['translation']['rmse']:.6f} m")
        print(f"  Mean: {ate_metrics['translation']['mean']:.6f} m") 
        print(f"  Median: {ate_metrics['translation']['median']:.6f} m")
        print(f"  Std: {ate_metrics['translation']['std']:.6f} m")
        print(f"  Range: [{ate_metrics['translation']['min']:.6f}, {ate_metrics['translation']['max']:.6f}] m")
        
        print(f"\nRotation ATE:")
        print(f"  RMSE: {ate_metrics['rotation']['rmse']:.6f} rad ({np.degrees(ate_metrics['rotation']['rmse']):.3f}°)")
        print(f"  Mean: {ate_metrics['rotation']['mean']:.6f} rad ({np.degrees(ate_metrics['rotation']['mean']):.3f}°)")
        print(f"  Median: {ate_metrics['rotation']['median']:.6f} rad ({np.degrees(ate_metrics['rotation']['median']):.3f}°)")
        print(f"  Std: {ate_metrics['rotation']['std']:.6f} rad ({np.degrees(ate_metrics['rotation']['std']):.3f}°)")
        print(f"  Range: [{ate_metrics['rotation']['min']:.6f}, {ate_metrics['rotation']['max']:.6f}] rad")
        
        # Save ATE metrics
        ate_path = output_dir / 'ate_metrics.json'
        with open(ate_path, 'w') as f:
            json.dump(ate_metrics, f, indent=2)
        print(f"\nATE metrics saved: {ate_path}")
        
        # Save trajectories
        trajectories_data = {
            'predicted_trajectory': predicted_trajectory.tolist(),
            'ground_truth_trajectory': ground_truth_trajectory.tolist(),
            'test_bag': self.config['test_bags'][0],
            'frame_count': len(predicted_trajectory)
        }
        
        traj_path = output_dir / 'full_trajectories.json'
        with open(traj_path, 'w') as f:
            json.dump(trajectories_data, f, indent=2)
        print(f"Trajectories saved: {traj_path}")
        
        # Generate comprehensive visualizations
        self.plot_comprehensive_trajectory(predicted_trajectory, ground_truth_trajectory, output_dir)
        self.plot_ate_analysis(ate_metrics, output_dir)
        
        print(f"\nFull trajectory evaluation complete! Results saved in: {output_dir}")
        return ate_metrics, predicted_trajectory, ground_truth_trajectory


def main():
    parser = argparse.ArgumentParser(description="Full Trajectory Evaluation with ATE Analysis")
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='full_trajectory_evaluation',
                       help='Output directory for results')
    
    # Data
    parser.add_argument('--csv_path', type=str,
                       default='data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv',
                       help='Path to dataset CSV')
    parser.add_argument('--data_root', type=str,
                       default='data/processed/visual_odometry_dataset',
                       help='Root directory containing images')
    parser.add_argument('--test_bags', nargs='+',
                       default=['ariel_2023-12-21-14-28-22_4'],
                       help='Test bag for evaluation')
    
    # Model parameters
    parser.add_argument('--sequence_length', type=int, default=4,
                       help='Number of frames per sequence')
    parser.add_argument('--overlap_frames', type=int, default=4,
                       help='Frame overlap between windows')
    parser.add_argument('--image_size', type=int, default=196,
                       help='Input image size')
    parser.add_argument('--camera', type=str, default='cam0',
                       help='Which camera to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ViT backbone')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze ViT backbone parameters')
    
    args = parser.parse_args()
    config = vars(args)
    
    print("Full Trajectory Evaluation with ATE Analysis")
    print("="*60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Create evaluator and run evaluation
    evaluator = FullTrajectoryEvaluator(args.checkpoint, config)
    ate_metrics, pred_traj, gt_traj = evaluator.evaluate_full_trajectory(args.output_dir)


if __name__ == "__main__":
    main()