#!/usr/bin/env python3
"""
Comprehensive TSformer Visual Odometry Evaluation Script

Generates detailed visualizations including:
- 3D trajectory plots
- 2D plane projections (XY, XZ, YZ)  
- Training and test data evaluation
- Prediction vs ground truth comparison

Author: Underwater Visual Odometry Research Team
Date: January 2025
"""

import os
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TSformerEvaluator:
    """Comprehensive TSformer Visual Odometry Evaluator"""
    
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        experiment_name = Path(self.model_path).parent.name
        self.output_dir = Path(f"evaluation_results_{experiment_name}")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Load model
        self.model, self.loss_fn = self._load_model()
        
        # Load data
        self.data_loaders = self._load_data()
        
    def _load_model(self):
        """Load trained TSformer model"""
        print("Loading trained model...")
        
        # Create model
        model, loss_fn = create_tsformer_vo(
            sequence_length=self.config['sequence_length'],
            pretrained=True,
            freeze_backbone=False,
            image_size=self.config['image_size']
        )
        
        # Load checkpoint
        if Path(self.model_path).exists():
            print(f"Loading checkpoint: {self.model_path}")
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Trying to load on CPU first...")
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Print training info
            if 'epoch' in checkpoint:
                print(f"Model trained for {checkpoint['epoch']} epochs")
            if 'best_val_loss' in checkpoint:
                print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
        else:
            print(f"Warning: Model path {self.model_path} not found. Using untrained model.")
            
        model = model.to(self.device)
        model.eval()
        
        return model, loss_fn
        
    def _load_data(self):
        """Load data loaders"""
        print("Loading data loaders...")
        
        data_loaders = create_data_loaders(
            csv_path=self.config['csv_path'],
            data_root=self.config['data_root'],
            sequence_length=self.config['sequence_length'],
            overlap_frames=self.config['overlap_frames'],
            image_size=self.config['image_size'],
            batch_size=1,  # Use batch size 1 for evaluation
            test_bags=self.config['test_bags'],
            num_workers=0,  # Avoid multiprocessing issues
            camera='cam0'  # Use camera 0 as requested
        )
        
        return data_loaders
        
    def evaluate_dataset(self, data_loader, dataset_name):
        """Evaluate model on a dataset and return predictions with true world coordinates"""
        print(f"\\nEvaluating on {dataset_name} dataset...")
        
        predictions = []
        ground_truth_deltas = []
        true_world_coords = []
        metadata = []
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i % 50 == 0:
                    print(f"  Processing batch {i}/{len(data_loader)}")
                    
                # Move to device
                images = batch['images'].to(self.device)
                poses = batch['poses'].to(self.device)
                
                # Forward pass
                pred_poses = self.model(images)
                
                # Store results
                predictions.append(pred_poses.cpu().numpy())
                ground_truth_deltas.append(poses.cpu().numpy())
                
                # Get true world coordinates for this batch
                bag_name = batch['bag_name'][0]
                frame_indices = batch['frame_indices'][0]
                
                # Load world coordinates from dataset
                world_coords = self._get_world_coordinates(bag_name, frame_indices)
                true_world_coords.append(world_coords)
                
                # Store metadata
                metadata.append({
                    'bag_name': bag_name,
                    'frame_indices': frame_indices,
                    'timestamps': batch['timestamps'][0]
                })
                
        # Concatenate results
        predictions = np.concatenate(predictions, axis=0)  # (N, 6)
        ground_truth_deltas = np.concatenate(ground_truth_deltas, axis=0)  # (N, 6)
        true_world_coords = np.concatenate(true_world_coords, axis=0)  # (N, 3)
        
        print(f"  Collected {len(predictions)} predictions")
        
        return predictions, ground_truth_deltas, true_world_coords, metadata
    
    def _get_world_coordinates(self, bag_name, frame_indices):
        """Get true world coordinates for given frames"""
        import pandas as pd
        
        # Load the dataset
        df = pd.read_csv(self.config['csv_path'])
        
        # Get the specific frames
        world_coords = []
        for frame_idx in frame_indices:
            frame_data = df[(df['bag_name'] == bag_name) & (df['frame_index'] == frame_idx)]
            if len(frame_data) > 0:
                row = frame_data.iloc[0]
                # Use the last frame's world coordinates (since we predict the last frame's pose)
                world_coords.append([row['world_x'], row['world_y'], row['world_z']])
            else:
                # Fallback to NaN if no data
                world_coords.append([np.nan, np.nan, np.nan])
        
        # Return the last frame's coordinates (what we're predicting)
        return np.array([world_coords[-1]])
        
    def compute_trajectory_from_deltas(self, deltas, initial_pose=None):
        """Compute absolute trajectory from pose deltas using proper SE(3) transformations"""
        if initial_pose is None:
            initial_pose = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
            
        trajectory = [initial_pose.copy()]
        current_pose = initial_pose.copy()
        
        for delta in deltas:
            # Convert current pose to transformation matrix
            current_T = self.pose_to_transformation_matrix(current_pose)
            
            # Convert delta to transformation matrix
            delta_T = self.pose_to_transformation_matrix(delta)
            
            # Apply transformation: T_new = T_current * T_delta
            new_T = current_T @ delta_T
            
            # Extract pose from transformation matrix
            current_pose = self.transformation_matrix_to_pose(new_T)
            trajectory.append(current_pose.copy())
            
        return np.array(trajectory)
    
    def pose_to_transformation_matrix(self, pose):
        """Convert [x,y,z,roll,pitch,yaw] to 4x4 transformation matrix"""
        x, y, z, roll, pitch, yaw = pose
        
        # Rotation matrices
        R_x = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        
        # Combined rotation: R = R_z * R_y * R_x
        R = R_z @ R_y @ R_x
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        
        return T
    
    def transformation_matrix_to_pose(self, T):
        """Convert 4x4 transformation matrix to [x,y,z,roll,pitch,yaw]"""
        # Extract translation
        x, y, z = T[:3, 3]
        
        # Extract rotation matrix
        R = T[:3, :3]
        
        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        # Using ZYX convention
        pitch = np.arcsin(-R[2, 0])
        
        if np.cos(pitch) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock case
            roll = np.arctan2(-R[1, 2], R[1, 1])
            yaw = 0
        
        return np.array([x, y, z, roll, pitch, yaw])
    
    def compute_true_trajectory(self, world_coords):
        """Convert true world coordinates to trajectory format"""
        # Remove NaN values
        valid_mask = ~np.isnan(world_coords).any(axis=1)
        valid_coords = world_coords[valid_mask]
        
        if len(valid_coords) == 0:
            print("Warning: No valid world coordinates found!")
            return np.zeros((1, 6))  # Return dummy trajectory
        
        # Create trajectory with world coordinates (x, y, z) and zero rotations
        trajectory = np.zeros((len(valid_coords), 6))
        trajectory[:, :3] = valid_coords  # Set x, y, z
        # Leave rotations as zero since we don't have world rotations
        
        return trajectory
        
    def compute_metrics(self, predictions, ground_truths):
        """Compute evaluation metrics"""
        # Translation errors
        trans_pred = predictions[:, :3]
        trans_gt = ground_truths[:, :3]
        trans_errors = np.linalg.norm(trans_pred - trans_gt, axis=1)
        
        # Rotation errors  
        rot_pred = predictions[:, 3:]
        rot_gt = ground_truths[:, 3:]
        rot_errors = np.linalg.norm(rot_pred - rot_gt, axis=1)
        
        metrics = {
            'translation': {
                'mse': np.mean(trans_errors**2),
                'rmse': np.sqrt(np.mean(trans_errors**2)),
                'mae': np.mean(trans_errors),
                'std': np.std(trans_errors),
                'max': np.max(trans_errors),
                'median': np.median(trans_errors)
            },
            'rotation': {
                'mse': np.mean(rot_errors**2),
                'rmse': np.sqrt(np.mean(rot_errors**2)),
                'mae': np.mean(rot_errors),
                'std': np.std(rot_errors),
                'max': np.max(rot_errors),
                'median': np.median(rot_errors)
            }
        }
        
        return metrics
        
    def plot_3d_trajectory(self, pred_trajectory, gt_trajectory, dataset_name):
        """Create 3D trajectory plot"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories
        ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
                'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
                'r--', linewidth=2, label='Prediction', alpha=0.8)
        
        # Mark start and end points
        ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], gt_trajectory[0, 2], 
                  c='green', s=100, label='Start', marker='o')
        ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], gt_trajectory[-1, 2], 
                  c='red', s=100, label='End', marker='s')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Global Trajectory - {dataset_name} Dataset\\nTSformer Visual Odometry (Absolute Coordinates)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.output_dir / f'3d_trajectory_{dataset_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_2d_projections(self, pred_trajectory, gt_trajectory, dataset_name):
        """Create 2D plane projection plots"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # XY plane
        axes[0].plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', linewidth=2, label='Ground Truth')
        axes[0].plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=2, label='Prediction')
        axes[0].scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], c='green', s=100, marker='o', label='Start')
        axes[0].scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], c='red', s=100, marker='s', label='End')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title('XY Plane (Top View)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].axis('equal')
        
        # XZ plane  
        axes[1].plot(gt_trajectory[:, 0], gt_trajectory[:, 2], 'b-', linewidth=2, label='Ground Truth')
        axes[1].plot(pred_trajectory[:, 0], pred_trajectory[:, 2], 'r--', linewidth=2, label='Prediction')
        axes[1].scatter(gt_trajectory[0, 0], gt_trajectory[0, 2], c='green', s=100, marker='o', label='Start')
        axes[1].scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 2], c='red', s=100, marker='s', label='End')
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Z (m)')
        axes[1].set_title('XZ Plane (Side View)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].axis('equal')
        
        # YZ plane
        axes[2].plot(gt_trajectory[:, 1], gt_trajectory[:, 2], 'b-', linewidth=2, label='Ground Truth')
        axes[2].plot(pred_trajectory[:, 1], pred_trajectory[:, 2], 'r--', linewidth=2, label='Prediction')
        axes[2].scatter(gt_trajectory[0, 1], gt_trajectory[0, 2], c='green', s=100, marker='o', label='Start')
        axes[2].scatter(gt_trajectory[-1, 1], gt_trajectory[-1, 2], c='red', s=100, marker='s', label='End')
        axes[2].set_xlabel('Y (m)')
        axes[2].set_ylabel('Z (m)')
        axes[2].set_title('YZ Plane (Front View)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].axis('equal')
        
        plt.suptitle(f'2D Global Trajectory Projections - {dataset_name} Dataset\\nTSformer Visual Odometry (Absolute Coordinates)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'2d_projections_{dataset_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_prediction_scatter(self, predictions, ground_truths, dataset_name):
        """Create scatter plots of predictions vs ground truth"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Translation components
        trans_labels = ['X (m)', 'Y (m)', 'Z (m)']
        for i in range(3):
            axes[0, i].scatter(ground_truths[:, i], predictions[:, i], alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(ground_truths[:, i].min(), predictions[:, i].min())
            max_val = max(ground_truths[:, i].max(), predictions[:, i].max())
            axes[0, i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            axes[0, i].set_xlabel(f'Ground Truth {trans_labels[i]}')
            axes[0, i].set_ylabel(f'Predicted {trans_labels[i]}')
            axes[0, i].set_title(f'Translation {trans_labels[i]} - {dataset_name}')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend()
            
            # Compute R²
            corr_coef = np.corrcoef(ground_truths[:, i], predictions[:, i])[0, 1]
            axes[0, i].text(0.05, 0.95, f'R² = {corr_coef**2:.3f}', transform=axes[0, i].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Rotation components
        rot_labels = ['Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
        for i in range(3):
            axes[1, i].scatter(ground_truths[:, i+3], predictions[:, i+3], alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(ground_truths[:, i+3].min(), predictions[:, i+3].min())
            max_val = max(ground_truths[:, i+3].max(), predictions[:, i+3].max())
            axes[1, i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            axes[1, i].set_xlabel(f'Ground Truth {rot_labels[i]}')
            axes[1, i].set_ylabel(f'Predicted {rot_labels[i]}')
            axes[1, i].set_title(f'Rotation {rot_labels[i]} - {dataset_name}')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend()
            
            # Compute R²
            corr_coef = np.corrcoef(ground_truths[:, i+3], predictions[:, i+3])[0, 1]
            axes[1, i].text(0.05, 0.95, f'R² = {corr_coef**2:.3f}', transform=axes[1, i].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'prediction_scatter_{dataset_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_error_distributions(self, predictions, ground_truths, dataset_name):
        """Plot error distributions"""
        # Compute errors
        trans_errors = np.linalg.norm(predictions[:, :3] - ground_truths[:, :3], axis=1)
        rot_errors = np.linalg.norm(predictions[:, 3:] - ground_truths[:, 3:], axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Translation error distribution
        axes[0].hist(trans_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(np.mean(trans_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(trans_errors):.4f}m')
        axes[0].axvline(np.median(trans_errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(trans_errors):.4f}m')
        axes[0].set_xlabel('Translation Error (m)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Translation Error Distribution - {dataset_name}')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Rotation error distribution
        axes[1].hist(rot_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(np.mean(rot_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rot_errors):.4f}rad')
        axes[1].axvline(np.median(rot_errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(rot_errors):.4f}rad')
        axes[1].set_xlabel('Rotation Error (rad)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Rotation Error Distribution - {dataset_name}')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'error_distributions_{dataset_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def evaluate_and_visualize(self):
        """Main evaluation and visualization function"""
        print("\\n" + "="*80)
        print("COMPREHENSIVE TSFORMER VISUAL ODOMETRY EVALUATION")
        print("="*80)
        
        results = {}
        
        # Evaluate each dataset
        for dataset_name, data_loader in [('Training', self.data_loaders['train_loader']),
                                         ('Validation', self.data_loaders['val_loader']),
                                         ('Test', self.data_loaders['test_loader'])]:
            
            if len(data_loader) == 0:
                print(f"Skipping {dataset_name} dataset (empty)")
                continue
                
            print(f"\\n{'-'*50}")
            print(f"Evaluating {dataset_name} Dataset")
            print(f"{'-'*50}")
            
            # Get predictions
            predictions, ground_truth_deltas, true_world_coords, metadata = self.evaluate_dataset(data_loader, dataset_name)
            
            # Compute metrics (still using deltas for model performance)
            metrics = self.compute_metrics(predictions, ground_truth_deltas)
            results[dataset_name.lower()] = {
                'metrics': metrics,
                'num_samples': len(predictions)
            }
            
            # Print metrics
            print(f"\\nMetrics for {dataset_name} Dataset:")
            print(f"  Translation RMSE: {metrics['translation']['rmse']:.6f} m")
            print(f"  Translation MAE:  {metrics['translation']['mae']:.6f} m")
            print(f"  Rotation RMSE:    {metrics['rotation']['rmse']:.6f} rad")
            print(f"  Rotation MAE:     {metrics['rotation']['mae']:.6f} rad")
            
            # Compute trajectories
            # For predictions: accumulate deltas from predicted poses
            pred_trajectory = self.compute_trajectory_from_deltas(predictions)
            
            # For ground truth: use TRUE world coordinates (no accumulation!)
            gt_trajectory = self.compute_true_trajectory(true_world_coords)
            
            # Create visualizations
            print(f"Creating visualizations for {dataset_name} dataset...")
            
            # 3D trajectory plot
            self.plot_3d_trajectory(pred_trajectory, gt_trajectory, dataset_name)
            
            # 2D projections
            self.plot_2d_projections(pred_trajectory, gt_trajectory, dataset_name)
            
            # Prediction scatter plots
            self.plot_prediction_scatter(predictions, ground_truths, dataset_name)
            
            # Error distributions
            self.plot_error_distributions(predictions, ground_truths, dataset_name)
            
            print(f"[COMPLETED] {dataset_name} evaluation completed!")
            
        # Save results
        results['evaluation_info'] = {
            'model_path': str(self.model_path),
            'evaluation_date': datetime.now().isoformat(),
            'config': self.config
        }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\\n{'='*80}")
        print("EVALUATION COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}")
        
        return results

def main():
    # Configuration
    config = {
        'csv_path': 'data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv',
        'data_root': 'data/processed/visual_odometry_dataset',
        'sequence_length': 3,
        'overlap_frames': 1,
        'image_size': 224,
        'test_bags': ['ariel_2023-12-21-14-28-22_4']
    }
    
    # Find the best model checkpoint
    checkpoint_dir = Path('experiments/2_TSFormer_seq3_frozen_balanced_loss_consistency')
    best_model_path = checkpoint_dir / 'checkpoint_best.pth'
    latest_model_path = checkpoint_dir / 'checkpoint_latest.pth'
    
    if best_model_path.exists():
        model_path = best_model_path
        print(f"Using best model: {model_path}")
    elif latest_model_path.exists():
        model_path = latest_model_path
        print(f"Using latest model: {model_path}")
    else:
        print("No trained model found. Please train the model first.")
        print("Expected paths:")
        print(f"  {best_model_path}")
        print(f"  {latest_model_path}")
        return
        
    # Create evaluator and run evaluation
    evaluator = TSformerEvaluator(model_path, config)
    results = evaluator.evaluate_and_visualize()
    
    print("\\nEvaluation Summary:")
    for dataset_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"{dataset_name.capitalize()} Dataset ({result['num_samples']} samples):")
            print(f"  Trans RMSE: {metrics['translation']['rmse']:.6f} m")
            print(f"  Rot RMSE: {metrics['rotation']['rmse']:.6f} rad")

if __name__ == "__main__":
    main()