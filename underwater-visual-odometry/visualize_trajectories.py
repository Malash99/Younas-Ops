#!/usr/bin/env python3
"""
Visualize TSformer-VO Predictions vs Ground Truth

Creates comprehensive trajectory visualizations in global coordinates
for train, validation, and test sets.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrajectoryVisualizer:
    """Visualize predicted vs ground truth trajectories"""
    
    def __init__(self, model_path, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Load model
        print("Loading trained model...")
        self.model, self.loss_fn = create_tsformer_vo(
            sequence_length=config['sequence_length'],
            pretrained=config['pretrained'],
            freeze_backbone=config['freeze_backbone'],
            image_size=config['image_size']
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        # Load data
        print("Loading datasets...")
        data_loaders = create_data_loaders(
            csv_path=config['csv_path'],
            data_root=config['data_root'],
            sequence_length=config['sequence_length'],
            overlap_frames=config['overlap_frames'],
            image_size=config['image_size'],
            batch_size=1,  # Use batch_size=1 for prediction
            test_bags=config['test_bags'],
            num_workers=0,  # Avoid multiprocessing issues
            camera=config['camera']
        )
        
        self.train_loader = data_loaders['train_loader']
        self.val_loader = data_loaders['val_loader'] 
        self.test_loader = data_loaders['test_loader']
        
    def predict_trajectories(self, data_loader, max_samples=None, dataset_name=""):
        """Predict trajectories for a dataset"""
        print(f"Predicting {dataset_name} trajectories...")
        
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if max_samples and i >= max_samples:
                    break
                    
                images = batch['images'].to(self.device)
                poses = batch['poses'].to(self.device)
                
                pred_poses = self.model(images)
                
                predictions.append(pred_poses.cpu().numpy())
                ground_truths.append(poses.cpu().numpy())
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{min(max_samples or len(data_loader), len(data_loader))} samples")
        
        predictions = np.vstack(predictions)
        ground_truths = np.vstack(ground_truths)
        
        print(f"  {dataset_name} predictions shape: {predictions.shape}")
        return predictions, ground_truths
    
    def poses_to_global_trajectory(self, pose_deltas):
        """Convert pose deltas to global trajectory"""
        # pose_deltas: (N, 6) - [dx, dy, dz, droll, dpitch, dyaw]
        
        global_positions = np.zeros((len(pose_deltas) + 1, 3))
        global_orientations = np.zeros((len(pose_deltas) + 1, 3))
        
        # Start at origin
        current_pos = np.array([0.0, 0.0, 0.0])
        current_rot = np.array([0.0, 0.0, 0.0])
        
        global_positions[0] = current_pos
        global_orientations[0] = current_rot
        
        for i, delta in enumerate(pose_deltas):
            # Extract deltas
            dt_pos = delta[:3]  # [dx, dy, dz]
            dt_rot = delta[3:]  # [droll, dpitch, dyaw]
            
            # Simple integration (assuming small rotations)
            # In a more sophisticated version, you'd use proper SE(3) integration
            current_pos += dt_pos
            current_rot += dt_rot
            
            global_positions[i + 1] = current_pos
            global_orientations[i + 1] = current_rot
            
        return global_positions, global_orientations
    
    def compute_trajectory_metrics(self, pred_traj, gt_traj):
        """Compute trajectory comparison metrics"""
        # Align trajectories (both start at origin)
        pred_aligned = pred_traj - pred_traj[0]
        gt_aligned = gt_traj - gt_traj[0]
        
        # Compute metrics
        mse = mean_squared_error(gt_aligned.flatten(), pred_aligned.flatten())
        mae = mean_absolute_error(gt_aligned.flatten(), pred_aligned.flatten())
        
        # Endpoint error
        endpoint_error = np.linalg.norm(pred_aligned[-1] - gt_aligned[-1])
        
        # Path length ratio
        pred_length = np.sum(np.linalg.norm(np.diff(pred_aligned, axis=0), axis=1))
        gt_length = np.sum(np.linalg.norm(np.diff(gt_aligned, axis=0), axis=1))
        length_ratio = pred_length / gt_length if gt_length > 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'endpoint_error': endpoint_error,
            'path_length_ratio': length_ratio,
            'pred_length': pred_length,
            'gt_length': gt_length
        }
    
    def plot_3d_trajectory(self, pred_traj, gt_traj, title, ax=None):
        """Plot 3D trajectory comparison"""
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 
                'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 
                'r--', linewidth=2, label='Predicted', alpha=0.8)
        
        # Mark start and end points
        ax.scatter(*gt_traj[0], color='green', s=100, label='Start', marker='o')
        ax.scatter(*gt_traj[-1], color='blue', s=100, label='GT End', marker='s')
        ax.scatter(*pred_traj[-1], color='red', s=100, label='Pred End', marker='^')
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()
        
        # Equal aspect ratio
        max_range = np.array([gt_traj.max()-gt_traj.min(), 
                             pred_traj.max()-pred_traj.min()]).max() / 2.0
        mid_x = (gt_traj[:, 0].max()+gt_traj[:, 0].min()) * 0.5
        mid_y = (gt_traj[:, 1].max()+gt_traj[:, 1].min()) * 0.5
        mid_z = (gt_traj[:, 2].max()+gt_traj[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        return ax
    
    def plot_2d_trajectory(self, pred_traj, gt_traj, title, ax=None):
        """Plot 2D trajectory comparison (XY plane)"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot trajectories
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', linewidth=3, 
                label='Ground Truth', alpha=0.8)
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=3, 
                label='Predicted', alpha=0.8)
        
        # Mark start and end points
        ax.scatter(*gt_traj[0, :2], color='green', s=150, label='Start', 
                  marker='o', edgecolor='black', linewidth=2)
        ax.scatter(*gt_traj[-1, :2], color='blue', s=150, label='GT End', 
                  marker='s', edgecolor='black', linewidth=2)
        ax.scatter(*pred_traj[-1, :2], color='red', s=150, label='Pred End', 
                  marker='^', edgecolor='black', linewidth=2)
        
        # Add direction arrows
        self.add_direction_arrows(ax, gt_traj[:, :2], 'blue', 0.7)
        self.add_direction_arrows(ax, pred_traj[:, :2], 'red', 0.5)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return ax
    
    def add_direction_arrows(self, ax, trajectory, color, alpha, arrow_interval=10):
        """Add direction arrows to trajectory"""
        for i in range(0, len(trajectory)-1, arrow_interval):
            if i + 1 < len(trajectory):
                dx = trajectory[i+1, 0] - trajectory[i, 0]
                dy = trajectory[i+1, 1] - trajectory[i, 1]
                if np.sqrt(dx**2 + dy**2) > 1e-6:  # Avoid zero-length arrows
                    ax.annotate('', xy=trajectory[i+1], xytext=trajectory[i],
                              arrowprops=dict(arrowstyle='->', color=color, 
                                            alpha=alpha, lw=1.5))
    
    def create_summary_plot(self, results, output_dir):
        """Create comprehensive summary visualization"""
        fig = plt.figure(figsize=(20, 15))
        
        # 2D trajectory plots
        ax1 = plt.subplot(2, 3, 1)
        self.plot_2d_trajectory(results['train']['pred_traj'], 
                               results['train']['gt_traj'], 
                               'Training Set Trajectory (Sample)', ax1)
        
        ax2 = plt.subplot(2, 3, 2)
        self.plot_2d_trajectory(results['val']['pred_traj'], 
                               results['val']['gt_traj'], 
                               'Validation Set Trajectory', ax2)
        
        ax3 = plt.subplot(2, 3, 3)
        self.plot_2d_trajectory(results['test']['pred_traj'], 
                               results['test']['gt_traj'], 
                               'Test Set Trajectory', ax3)
        
        # Metrics comparison
        ax4 = plt.subplot(2, 3, 4)
        datasets = ['Train', 'Val', 'Test']
        endpoint_errors = [results['train']['metrics']['endpoint_error'],
                          results['val']['metrics']['endpoint_error'],
                          results['test']['metrics']['endpoint_error']]
        
        bars = ax4.bar(datasets, endpoint_errors, color=['skyblue', 'lightgreen', 'salmon'])
        ax4.set_ylabel('Endpoint Error (m)')
        ax4.set_title('Trajectory Endpoint Error')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, endpoint_errors):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Path length ratio
        ax5 = plt.subplot(2, 3, 5)
        length_ratios = [results['train']['metrics']['path_length_ratio'],
                        results['val']['metrics']['path_length_ratio'],
                        results['test']['metrics']['path_length_ratio']]
        
        bars = ax5.bar(datasets, length_ratios, color=['skyblue', 'lightgreen', 'salmon'])
        ax5.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Ratio')
        ax5.set_ylabel('Path Length Ratio')
        ax5.set_title('Predicted/GT Path Length Ratio')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, length_ratios):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # MSE comparison
        ax6 = plt.subplot(2, 3, 6)
        mse_values = [results['train']['metrics']['mse'],
                     results['val']['metrics']['mse'],
                     results['test']['metrics']['mse']]
        
        bars = ax6.bar(datasets, mse_values, color=['skyblue', 'lightgreen', 'salmon'])
        ax6.set_ylabel('Mean Squared Error')
        ax6.set_title('Trajectory MSE')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, mse_values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{value:.2e}', ha='center', va='bottom', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        summary_path = output_dir / 'trajectory_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved: {summary_path}")
        
        return fig
    
    def generate_all_visualizations(self, output_dir="visualizations"):
        """Generate all trajectory visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Generating visualizations in: {output_dir}")
        
        # Predict trajectories
        train_pred, train_gt = self.predict_trajectories(
            self.train_loader, max_samples=100, dataset_name="Train"
        )
        val_pred, val_gt = self.predict_trajectories(
            self.val_loader, dataset_name="Validation"
        )
        test_pred, test_gt = self.predict_trajectories(
            self.test_loader, dataset_name="Test"
        )
        
        # Convert to global trajectories
        print("Converting to global coordinates...")
        train_pred_traj, _ = self.poses_to_global_trajectory(train_pred)
        train_gt_traj, _ = self.poses_to_global_trajectory(train_gt)
        
        val_pred_traj, _ = self.poses_to_global_trajectory(val_pred)
        val_gt_traj, _ = self.poses_to_global_trajectory(val_gt)
        
        test_pred_traj, _ = self.poses_to_global_trajectory(test_pred)
        test_gt_traj, _ = self.poses_to_global_trajectory(test_gt)
        
        # Compute metrics
        train_metrics = self.compute_trajectory_metrics(train_pred_traj, train_gt_traj)
        val_metrics = self.compute_trajectory_metrics(val_pred_traj, val_gt_traj)
        test_metrics = self.compute_trajectory_metrics(test_pred_traj, test_gt_traj)
        
        # Store results
        results = {
            'train': {'pred_traj': train_pred_traj, 'gt_traj': train_gt_traj, 'metrics': train_metrics},
            'val': {'pred_traj': val_pred_traj, 'gt_traj': val_gt_traj, 'metrics': val_metrics},
            'test': {'pred_traj': test_pred_traj, 'gt_traj': test_gt_traj, 'metrics': test_metrics}
        }
        
        # Create individual 3D plots
        for dataset_name, data in results.items():
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            self.plot_3d_trajectory(data['pred_traj'], data['gt_traj'], 
                                   f'{dataset_name.title()} Set - 3D Trajectory', ax)
            
            plot_path = output_dir / f'{dataset_name}_3d_trajectory.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{dataset_name.title()} 3D plot saved: {plot_path}")
        
        # Create individual 2D plots
        for dataset_name, data in results.items():
            fig, ax = plt.subplots(figsize=(12, 10))
            self.plot_2d_trajectory(data['pred_traj'], data['gt_traj'], 
                                   f'{dataset_name.title()} Set - 2D Trajectory (XY plane)', ax)
            
            plot_path = output_dir / f'{dataset_name}_2d_trajectory.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{dataset_name.title()} 2D plot saved: {plot_path}")
        
        # Create summary plot
        self.create_summary_plot(results, output_dir)
        
        # Save metrics
        metrics_summary = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        
        metrics_path = output_dir / 'trajectory_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"Metrics saved: {metrics_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("TRAJECTORY EVALUATION SUMMARY")
        print(f"{'='*60}")
        for dataset_name, metrics in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
            print(f"{dataset_name:>8} - Endpoint Error: {metrics['endpoint_error']:.4f}m, "
                  f"Path Length Ratio: {metrics['path_length_ratio']:.3f}, "
                  f"MSE: {metrics['mse']:.2e}")
        
        return results

def main():
    # Configuration (adjust paths as needed)
    config = {
        'csv_path': 'data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr_clean.csv',
        'data_root': 'data/processed/visual_odometry_dataset',
        'test_bags': ['ariel_2023-12-21-14-28-22_4'],
        'sequence_length': 8,
        'overlap_frames': 1,
        'image_size': 224,
        'camera': 'cam0',
        'pretrained': True,
        'freeze_backbone': False
    }
    
    # Model path (use best checkpoint)
    model_path = "experiments/tsformer_vo/checkpoint_best.pth"
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        print("Please make sure training is completed and checkpoint exists.")
        return
    
    # Create visualizer and generate plots
    visualizer = TrajectoryVisualizer(model_path, config)
    results = visualizer.generate_all_visualizations()
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print("Generated files:")
    print("  - trajectory_summary.png (comprehensive overview)")
    print("  - train_2d_trajectory.png, train_3d_trajectory.png")
    print("  - val_2d_trajectory.png, val_3d_trajectory.png") 
    print("  - test_2d_trajectory.png, test_3d_trajectory.png")
    print("  - trajectory_metrics.json (numerical results)")

if __name__ == "__main__":
    main()