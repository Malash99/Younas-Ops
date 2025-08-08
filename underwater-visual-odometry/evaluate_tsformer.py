#!/usr/bin/env python3
"""
TSformer Visual Odometry Evaluation Script

Comprehensive evaluation of trained TSformer model on unseen test data.
Includes trajectory reconstruction, error analysis, and visualizations.

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
import seaborn as sns

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders


class TSformerEvaluator:
    """TSformer Visual Odometry Evaluator"""
    
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
        
        # Create data loaders
        self.setup_data()
        
        # Results storage
        self.predictions = []
        self.ground_truths = []
        self.trajectories = {}
        
    def load_checkpoint(self):
        """Load model weights from checkpoint."""
        print(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")
        
    def setup_data(self):
        """Setup data loaders."""
        data_loaders = create_data_loaders(
            csv_path=self.config['csv_path'],
            data_root=self.config['data_root'],
            sequence_length=self.config['sequence_length'],
            overlap_frames=self.config['overlap_frames'],
            image_size=self.config['image_size'],
            batch_size=1,  # Use batch_size=1 for detailed analysis
            test_bags=self.config['test_bags'],
            num_workers=self.config['num_workers'],
            camera=self.config['camera']
        )
        
        self.test_loader = data_loaders['test_loader']
        self.test_dataset = data_loaders['datasets']['test']
        
        print(f"Test set: {len(self.test_dataset)} windows from bags {self.config['test_bags']}")
        
    def predict_all(self):
        """Run inference on entire test set."""
        print("Running inference on test set...")
        
        self.predictions = []
        self.ground_truths = []
        batch_info = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Move data to device
                images = batch['images'].to(self.device)
                poses = batch['poses'].to(self.device)
                
                # Forward pass
                pred_poses = self.model(images)
                
                # Store results
                self.predictions.append(pred_poses.cpu().numpy())
                self.ground_truths.append(poses.cpu().numpy())
                batch_info.append({
                    'bag_name': batch['bag_name'][0],
                    'frame_indices': batch['frame_indices'][0].tolist() if hasattr(batch['frame_indices'][0], 'tolist') else batch['frame_indices'][0],
                    'timestamps': batch['timestamps'][0].tolist() if hasattr(batch['timestamps'][0], 'tolist') else batch['timestamps'][0]
                })
        
        # Convert to numpy arrays
        self.predictions = np.concatenate(self.predictions, axis=0)  # (N, 6)
        self.ground_truths = np.concatenate(self.ground_truths, axis=0)  # (N, 6)
        self.batch_info = batch_info
        
        print(f"Inference complete: {len(self.predictions)} predictions")
        
    def compute_metrics(self):
        """Compute evaluation metrics."""
        # Translation errors (meters)
        trans_pred = self.predictions[:, :3]
        trans_gt = self.ground_truths[:, :3]
        trans_errors = np.abs(trans_pred - trans_gt)
        
        # Rotation errors (radians)
        rot_pred = self.predictions[:, 3:]
        rot_gt = self.ground_truths[:, 3:]
        rot_errors = np.abs(rot_pred - rot_gt)
        
        # Overall metrics
        metrics = {
            'translation': {
                'mae': np.mean(trans_errors, axis=0),  # [x, y, z]
                'rmse': np.sqrt(np.mean(trans_errors**2, axis=0)),
                'mean_mae': np.mean(trans_errors),
                'mean_rmse': np.sqrt(np.mean(trans_errors**2))
            },
            'rotation': {
                'mae': np.mean(rot_errors, axis=0),  # [roll, pitch, yaw]
                'rmse': np.sqrt(np.mean(rot_errors**2, axis=0)),
                'mean_mae': np.mean(rot_errors),
                'mean_rmse': np.sqrt(np.mean(rot_errors**2))
            }
        }
        
        return metrics
        
    def reconstruct_trajectories(self):
        """Reconstruct full trajectories from pose deltas."""
        print("Reconstructing trajectories...")
        
        self.trajectories = {}
        
        # Group predictions by bag
        bag_predictions = {}
        bag_ground_truths = {}
        bag_frame_info = {}
        
        for i, info in enumerate(self.batch_info):
            bag_name = info['bag_name']
            if bag_name not in bag_predictions:
                bag_predictions[bag_name] = []
                bag_ground_truths[bag_name] = []
                bag_frame_info[bag_name] = []
                
            bag_predictions[bag_name].append(self.predictions[i])
            bag_ground_truths[bag_name].append(self.ground_truths[i])
            bag_frame_info[bag_name].append(info)
        
        # Reconstruct trajectory for each bag
        for bag_name in bag_predictions:
            pred_deltas = np.array(bag_predictions[bag_name])
            gt_deltas = np.array(bag_ground_truths[bag_name])
            
            # Integrate deltas to get absolute trajectory
            pred_traj = self._integrate_deltas(pred_deltas)
            gt_traj = self._integrate_deltas(gt_deltas)
            
            self.trajectories[bag_name] = {
                'predicted': pred_traj,
                'ground_truth': gt_traj,
                'frame_info': bag_frame_info[bag_name]
            }
            
            print(f"  {bag_name}: {len(pred_traj)} poses")
    
    def _integrate_deltas(self, deltas):
        """Integrate pose deltas to get absolute trajectory."""
        trajectory = np.zeros((len(deltas) + 1, 6))  # Start from origin
        
        for i, delta in enumerate(deltas):
            # Simple integration (assumes small deltas)
            trajectory[i + 1, :3] = trajectory[i, :3] + delta[:3]  # Translation
            trajectory[i + 1, 3:] = trajectory[i, 3:] + delta[3:]  # Rotation
            
        return trajectory
    
    def plot_trajectories(self, output_dir):
        """Plot trajectory comparisons."""
        print("Plotting trajectories...")
        
        for bag_name, traj_data in self.trajectories.items():
            pred_traj = traj_data['predicted']
            gt_traj = traj_data['ground_truth']
            
            # 3D trajectory plot
            fig = plt.figure(figsize=(15, 10))
            
            # 3D plot
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 'b-', label='Ground Truth', linewidth=2)
            ax1.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 'r--', label='Predicted', linewidth=2)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title(f'3D Trajectory - {bag_name}')
            ax1.legend()
            
            # XY plot
            ax2 = fig.add_subplot(222)
            ax2.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='Ground Truth', linewidth=2)
            ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label='Predicted', linewidth=2)
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('XY Trajectory')
            ax2.legend()
            ax2.grid(True)
            ax2.axis('equal')
            
            # Translation errors over time
            ax3 = fig.add_subplot(223)
            trans_errors = np.abs(pred_traj[:, :3] - gt_traj[:, :3])
            ax3.plot(trans_errors[:, 0], 'r-', label='X error')
            ax3.plot(trans_errors[:, 1], 'g-', label='Y error')
            ax3.plot(trans_errors[:, 2], 'b-', label='Z error')
            ax3.set_xlabel('Time step')
            ax3.set_ylabel('Translation Error (m)')
            ax3.set_title('Translation Errors')
            ax3.legend()
            ax3.grid(True)
            
            # Rotation errors over time
            ax4 = fig.add_subplot(224)
            rot_errors = np.abs(pred_traj[:, 3:] - gt_traj[:, 3:])
            ax4.plot(rot_errors[:, 0], 'r-', label='Roll error')
            ax4.plot(rot_errors[:, 1], 'g-', label='Pitch error')
            ax4.plot(rot_errors[:, 2], 'b-', label='Yaw error')
            ax4.set_xlabel('Time step')
            ax4.set_ylabel('Rotation Error (rad)')
            ax4.set_title('Rotation Errors')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            output_path = Path(output_dir) / f'trajectory_{bag_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {output_path}")
    
    def plot_error_distributions(self, output_dir):
        """Plot error distribution histograms."""
        print("Plotting error distributions...")
        
        # Calculate errors
        trans_errors = np.abs(self.predictions[:, :3] - self.ground_truths[:, :3])
        rot_errors = np.abs(self.predictions[:, 3:] - self.ground_truths[:, 3:])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Translation error histograms
        for i, axis_name in enumerate(['X', 'Y', 'Z']):
            axes[0, i].hist(trans_errors[:, i], bins=50, alpha=0.7, edgecolor='black')
            axes[0, i].set_xlabel(f'{axis_name} Error (m)')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].set_title(f'Translation {axis_name} Error Distribution')
            axes[0, i].grid(True, alpha=0.3)
        
        # Rotation error histograms  
        for i, axis_name in enumerate(['Roll', 'Pitch', 'Yaw']):
            axes[1, i].hist(rot_errors[:, i], bins=50, alpha=0.7, edgecolor='black')
            axes[1, i].set_xlabel(f'{axis_name} Error (rad)')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_title(f'Rotation {axis_name} Error Distribution')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / 'error_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
    
    def plot_prediction_scatter(self, output_dir):
        """Plot prediction vs ground truth scatter plots."""
        print("Plotting prediction scatter plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Translation scatter plots
        axis_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
        for i in range(6):
            row = i // 3
            col = i % 3
            
            gt_vals = self.ground_truths[:, i]
            pred_vals = self.predictions[:, i]
            
            # Scatter plot
            axes[row, col].scatter(gt_vals, pred_vals, alpha=0.6, s=10)
            
            # Perfect prediction line
            min_val = min(gt_vals.min(), pred_vals.min())
            max_val = max(gt_vals.max(), pred_vals.max())
            axes[row, col].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
            
            # Formatting
            units = '(m)' if i < 3 else '(rad)'
            axes[row, col].set_xlabel(f'Ground Truth {axis_names[i]} {units}')
            axes[row, col].set_ylabel(f'Predicted {axis_names[i]} {units}')
            axes[row, col].set_title(f'{axis_names[i]} Predictions')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / 'prediction_scatter.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
    
    def evaluate(self, output_dir):
        """Run complete evaluation."""
        print("\\nRunning TSformer-VO Evaluation")
        print("="*50)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run inference
        self.predict_all()
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        # Print results
        print("\\nEvaluation Results:")
        print("-" * 30)
        print(f"Translation Errors:")
        print(f"  MAE (XYZ): {metrics['translation']['mae']}")
        print(f"  RMSE (XYZ): {metrics['translation']['rmse']}")
        print(f"  Mean MAE: {metrics['translation']['mean_mae']:.6f} m")
        print(f"  Mean RMSE: {metrics['translation']['mean_rmse']:.6f} m")
        
        print(f"\\nRotation Errors:")
        print(f"  MAE (RPY): {metrics['rotation']['mae']}")
        print(f"  RMSE (RPY): {metrics['rotation']['rmse']}")
        print(f"  Mean MAE: {metrics['rotation']['mean_mae']:.6f} rad")
        print(f"  Mean RMSE: {metrics['rotation']['mean_rmse']:.6f} rad")
        
        # Save metrics
        metrics_path = output_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda x: x.tolist())
        print(f"\\nMetrics saved: {metrics_path}")
        
        # Reconstruct trajectories
        self.reconstruct_trajectories()
        
        # Generate plots
        self.plot_trajectories(output_dir)
        self.plot_error_distributions(output_dir)
        self.plot_prediction_scatter(output_dir)
        
        # Save raw results
        results = {
            'predictions': self.predictions.tolist(),
            'ground_truth': self.ground_truths.tolist(),
            'batch_info': self.batch_info,
            'metrics': metrics
        }
        
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Raw results saved: {results_path}")
        
        print(f"\\nEvaluation complete! Results saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TSformer Visual Odometry")
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
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
                       help='Bags to use for testing')
    
    # Model parameters (should match training)
    parser.add_argument('--sequence_length', type=int, default=8,
                       help='Number of frames per sequence')
    parser.add_argument('--overlap_frames', type=int, default=4,
                       help='Frame overlap between windows')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--camera', type=str, default='cam0',
                       help='Which camera to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ViT backbone')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze ViT backbone parameters')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Convert to config dict
    config = vars(args)
    
    print("TSformer Visual Odometry Evaluation")
    print("="*50)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*50)
    
    # Create evaluator and run evaluation
    evaluator = TSformerEvaluator(args.checkpoint, config)
    evaluator.evaluate(args.output_dir)


if __name__ == "__main__":
    main()