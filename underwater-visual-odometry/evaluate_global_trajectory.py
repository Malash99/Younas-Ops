#!/usr/bin/env python3
"""
Global Frame Trajectory Evaluation for TSformer Visual Odometry

Proper trajectory reconstruction in global coordinate frame using SE(3) transformations.
Addresses coordinate frame issues and analyzes actual motion pattern learning.

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
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders


def pose_deltas_to_global_trajectory(pose_deltas):
    """
    Convert pose deltas to global trajectory using proper SE(3) transformation.
    
    Args:
        pose_deltas: (N, 6) array of [dx,dy,dz,droll,dpitch,dyaw] in local frame
    
    Returns:
        global_trajectory: (N+1, 6) array of [x,y,z,roll,pitch,yaw] in global frame
    """
    N = len(pose_deltas)
    global_trajectory = np.zeros((N + 1, 6))  # Include initial pose at origin
    
    # Current global pose [x, y, z, roll, pitch, yaw]
    current_pose = np.zeros(6)
    global_trajectory[0] = current_pose.copy()
    
    for i, delta in enumerate(pose_deltas):
        # Extract current rotation and translation
        current_translation = current_pose[:3]
        current_rotation = current_pose[3:]
        
        # Delta in local frame
        delta_translation = delta[:3]
        delta_rotation = delta[3:]
        
        # Convert current rotation to rotation matrix
        current_R = R.from_euler('xyz', current_rotation).as_matrix()
        
        # Transform delta translation from local to global frame
        global_delta_translation = current_R @ delta_translation
        
        # Update global position
        new_translation = current_translation + global_delta_translation
        
        # Update rotation (composition in global frame)
        delta_R = R.from_euler('xyz', delta_rotation)
        current_R_obj = R.from_euler('xyz', current_rotation)
        new_R = current_R_obj * delta_R  # Composition
        new_rotation = new_R.as_euler('xyz')
        
        # Update current pose
        current_pose = np.concatenate([new_translation, new_rotation])
        global_trajectory[i + 1] = current_pose.copy()
    
    return global_trajectory


def analyze_motion_patterns(predicted_deltas, ground_truth_deltas):
    """
    Analyze if the model is learning actual motion patterns or just averaging.
    """
    analysis = {}
    
    # Check if predictions are just mean values (straight line problem)
    pred_std = np.std(predicted_deltas, axis=0)
    gt_std = np.std(ground_truth_deltas, axis=0)
    
    analysis['prediction_std'] = pred_std
    analysis['ground_truth_std'] = gt_std
    analysis['std_ratio'] = pred_std / (gt_std + 1e-8)
    
    # Check for straight line patterns
    analysis['pred_range'] = np.max(predicted_deltas, axis=0) - np.min(predicted_deltas, axis=0)
    analysis['gt_range'] = np.max(ground_truth_deltas, axis=0) - np.min(ground_truth_deltas, axis=0)
    
    # Correlation analysis
    correlations = []
    for i in range(6):
        corr = np.corrcoef(predicted_deltas[:, i], ground_truth_deltas[:, i])[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0.0)
    analysis['correlations'] = np.array(correlations)
    
    # Motion diversity check
    analysis['pred_motion_diversity'] = np.mean(np.abs(np.diff(predicted_deltas, axis=0)))
    analysis['gt_motion_diversity'] = np.mean(np.abs(np.diff(ground_truth_deltas, axis=0)))
    
    return analysis


class GlobalTrajectoryEvaluator:
    """Evaluator with proper global frame trajectory reconstruction"""
    
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
        
    def predict_sequence_deltas(self):
        """Predict pose deltas for the complete sequence."""
        print("Predicting pose deltas...")
        
        # Create data loader for sequential prediction
        data_loaders = create_data_loaders(
            csv_path=self.config['csv_path'],
            data_root=self.config['data_root'],
            sequence_length=self.config['sequence_length'],
            overlap_frames=self.config['overlap_frames'],
            image_size=self.config['image_size'],
            batch_size=1,
            test_bags=self.config['test_bags'],
            num_workers=0,
            camera=self.config['camera']
        )
        
        test_loader = data_loaders['test_loader']
        
        predicted_deltas = []
        ground_truth_deltas = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move data to device
                images = batch['images'].to(self.device)
                poses = batch['poses'].to(self.device)
                
                # Forward pass
                pred_poses = self.model(images).cpu().numpy()[0]  # Remove batch dimension
                gt_poses = poses.cpu().numpy()[0]
                
                predicted_deltas.append(pred_poses)
                ground_truth_deltas.append(gt_poses)
                
                if batch_idx % 100 == 0:
                    print(f"  Processed {batch_idx}/{len(test_loader)} windows")
        
        predicted_deltas = np.array(predicted_deltas)
        ground_truth_deltas = np.array(ground_truth_deltas)
        
        print(f"Collected {len(predicted_deltas)} pose delta predictions")
        return predicted_deltas, ground_truth_deltas
    
    def plot_motion_analysis(self, pred_deltas, gt_deltas, motion_analysis, output_dir):
        """Plot motion pattern analysis."""
        print("Creating motion pattern analysis...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        axis_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
        units = ['(m)', '(m)', '(m)', '(rad)', '(rad)', '(rad)']
        
        # Plot pose delta comparisons
        for i in range(6):
            row = i // 3
            col = i % 3
            
            # Time series comparison
            time_steps = np.arange(len(pred_deltas))
            axes[row, col].plot(time_steps, gt_deltas[:, i], 'b-', linewidth=2, 
                               label='Ground Truth', alpha=0.8)
            axes[row, col].plot(time_steps, pred_deltas[:, i], 'r--', linewidth=2, 
                               label='Predicted', alpha=0.8)
            
            axes[row, col].set_xlabel('Time Step')
            axes[row, col].set_ylabel(f'{axis_names[i]} Delta {units[i]}')
            axes[row, col].set_title(f'{axis_names[i]} Motion Deltas\n'
                                   f'Corr: {motion_analysis["correlations"][i]:.3f}, '
                                   f'Std Ratio: {motion_analysis["std_ratio"][i]:.3f}')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Motion diversity comparison
        axes[2, 0].axis('off')
        diversity_text = f"""
Motion Pattern Analysis:

Prediction Statistics:
• Std Dev: {motion_analysis['prediction_std']}
• Range: {motion_analysis['pred_range']}
• Motion Diversity: {motion_analysis['pred_motion_diversity']:.6f}

Ground Truth Statistics:
• Std Dev: {motion_analysis['ground_truth_std']}  
• Range: {motion_analysis['gt_range']}
• Motion Diversity: {motion_analysis['gt_motion_diversity']:.6f}

Correlation per Axis:
• X: {motion_analysis['correlations'][0]:.3f}
• Y: {motion_analysis['correlations'][1]:.3f}
• Z: {motion_analysis['correlations'][2]:.3f}
• Roll: {motion_analysis['correlations'][3]:.3f}
• Pitch: {motion_analysis['correlations'][4]:.3f}
• Yaw: {motion_analysis['correlations'][5]:.3f}

Std Deviation Ratios:
{motion_analysis['std_ratio']}

Issues Detected:
{"• Low std ratio indicates averaging behavior" if np.any(motion_analysis['std_ratio'] < 0.1) else "• Std ratios look reasonable"}
{"• Low correlations indicate poor motion learning" if np.any(motion_analysis['correlations'] < 0.1) else "• Correlations look reasonable"}
        """
        axes[2, 0].text(0.05, 0.95, diversity_text, transform=axes[2, 0].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Distribution comparison
        axes[2, 1].hist(pred_deltas.flatten(), bins=50, alpha=0.5, label='Predicted', density=True)
        axes[2, 1].hist(gt_deltas.flatten(), bins=50, alpha=0.5, label='Ground Truth', density=True)
        axes[2, 1].set_xlabel('Delta Value')
        axes[2, 1].set_ylabel('Density')
        axes[2, 1].set_title('Overall Delta Distribution')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[2, 2].scatter(gt_deltas.flatten(), pred_deltas.flatten(), alpha=0.3, s=1)
        min_val = min(gt_deltas.min(), pred_deltas.min())
        max_val = max(gt_deltas.max(), pred_deltas.max())
        axes[2, 2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[2, 2].set_xlabel('Ground Truth Delta')
        axes[2, 2].set_ylabel('Predicted Delta')
        axes[2, 2].set_title('Prediction vs Ground Truth')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / 'motion_pattern_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Motion analysis saved: {output_path}")
    
    def plot_global_trajectories(self, pred_traj, gt_traj, output_dir):
        """Plot trajectories in global coordinate frame."""
        print("Creating global trajectory visualization...")
        
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
        
        ax1.set_xlabel('X (m) - Global')
        ax1.set_ylabel('Y (m) - Global')
        ax1.set_zlabel('Z (m) - Global')
        ax1.set_title('3D Trajectory (Global Frame)', fontsize=16, fontweight='bold')
        ax1.legend()
        ax1.grid(True)
        
        # XY plane projection (Global)
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', linewidth=2, label='Ground Truth')
        ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, label='Predicted')
        ax2.scatter(gt_traj[0, 0], gt_traj[0, 1], c='green', s=50, marker='o', zorder=5)
        ax2.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c='red', s=50, marker='s', zorder=5)
        ax2.set_xlabel('X (m) - Global')
        ax2.set_ylabel('Y (m) - Global')
        ax2.set_title('XY Plane (Global Frame)')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # XZ plane projection (Global)
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(gt_traj[:, 0], gt_traj[:, 2], 'b-', linewidth=2, label='Ground Truth')
        ax3.plot(pred_traj[:, 0], pred_traj[:, 2], 'r--', linewidth=2, label='Predicted')
        ax3.scatter(gt_traj[0, 0], gt_traj[0, 2], c='green', s=50, marker='o', zorder=5)
        ax3.scatter(gt_traj[-1, 0], gt_traj[-1, 2], c='red', s=50, marker='s', zorder=5)
        ax3.set_xlabel('X (m) - Global')
        ax3.set_ylabel('Z (m) - Global')
        ax3.set_title('XZ Plane (Global Frame)')
        ax3.legend()
        ax3.grid(True)
        ax3.axis('equal')
        
        # YZ plane projection (Global)
        ax4 = fig.add_subplot(2, 3, 5)
        ax4.plot(gt_traj[:, 1], gt_traj[:, 2], 'b-', linewidth=2, label='Ground Truth')
        ax4.plot(pred_traj[:, 1], pred_traj[:, 2], 'r--', linewidth=2, label='Predicted')
        ax4.scatter(gt_traj[0, 1], gt_traj[0, 2], c='green', s=50, marker='o', zorder=5)
        ax4.scatter(gt_traj[-1, 1], gt_traj[-1, 2], c='red', s=50, marker='s', zorder=5)
        ax4.set_xlabel('Y (m) - Global')
        ax4.set_ylabel('Z (m) - Global')
        ax4.set_title('YZ Plane (Global Frame)')
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
        ax5.set_title(f'Translation Error (Global Frame)\nMean: {np.mean(trans_errors):.4f}m, RMSE: {np.sqrt(np.mean(trans_errors**2)):.4f}m')
        ax5.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / 'global_trajectory_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Global trajectory plot saved: {output_path}")
    
    def evaluate_global_trajectory(self, output_dir):
        """Run complete evaluation with global frame reconstruction."""
        print("\nRunning Global Frame Trajectory Evaluation")
        print("="*60)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get pose delta predictions
        predicted_deltas, ground_truth_deltas = self.predict_sequence_deltas()
        
        # Analyze motion patterns
        print("Analyzing motion patterns...")
        motion_analysis = analyze_motion_patterns(predicted_deltas, ground_truth_deltas)
        
        # Report straight line problem
        print("\n" + "="*60)
        print("MOTION PATTERN ANALYSIS")
        print("="*60)
        print(f"Prediction Standard Deviations: {motion_analysis['prediction_std']}")
        print(f"Ground Truth Standard Deviations: {motion_analysis['ground_truth_std']}")
        print(f"Std Deviation Ratios: {motion_analysis['std_ratio']}")
        print(f"Correlations: {motion_analysis['correlations']}")
        print(f"Motion Diversity - Predicted: {motion_analysis['pred_motion_diversity']:.6f}")
        print(f"Motion Diversity - Ground Truth: {motion_analysis['gt_motion_diversity']:.6f}")
        
        # Check for straight line problem
        low_std_axes = motion_analysis['std_ratio'] < 0.1
        if np.any(low_std_axes):
            axes_with_issue = [['X','Y','Z','Roll','Pitch','Yaw'][i] for i in range(6) if low_std_axes[i]]
            print(f"\nWARNING: STRAIGHT LINE PROBLEM DETECTED in axes: {axes_with_issue}")
            print("   Model is averaging predictions instead of learning motion patterns!")
        else:
            print("\nMotion patterns look reasonable")
        
        # Convert to global trajectories  
        print("Converting to global coordinate frame...")
        predicted_global_traj = pose_deltas_to_global_trajectory(predicted_deltas)
        ground_truth_global_traj = pose_deltas_to_global_trajectory(ground_truth_deltas)
        
        # Compute global ATE
        global_ate = np.linalg.norm(predicted_global_traj[:, :3] - ground_truth_global_traj[:, :3], axis=1)
        print(f"\nGlobal Frame ATE:")
        print(f"  RMSE: {np.sqrt(np.mean(global_ate**2)):.6f} m")
        print(f"  Mean: {np.mean(global_ate):.6f} m")
        print(f"  Max: {np.max(global_ate):.6f} m")
        
        # Save analysis results
        analysis_results = {
            'motion_analysis': {k: (v.tolist() if isinstance(v, np.ndarray) else 
                                   (float(v) if isinstance(v, (np.float32, np.float64)) else v))
                               for k, v in motion_analysis.items()},
            'global_ate': {
                'rmse': float(np.sqrt(np.mean(global_ate**2))),
                'mean': float(np.mean(global_ate)),
                'max': float(np.max(global_ate)),
                'errors': [float(x) for x in global_ate.tolist()]
            },
            'straight_line_problem_detected': bool(np.any(low_std_axes)),
            'problematic_axes': [['X','Y','Z','Roll','Pitch','Yaw'][i] for i in range(6) if low_std_axes[i]]
        }
        
        results_path = output_dir / 'global_analysis_results.json'
        with open(results_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"Analysis results saved: {results_path}")
        
        # Generate visualizations
        self.plot_motion_analysis(predicted_deltas, ground_truth_deltas, motion_analysis, output_dir)
        self.plot_global_trajectories(predicted_global_traj, ground_truth_global_traj, output_dir)
        
        print(f"\nGlobal trajectory evaluation complete! Results saved in: {output_dir}")
        
        return analysis_results, predicted_global_traj, ground_truth_global_traj


def main():
    parser = argparse.ArgumentParser(description="Global Frame Trajectory Evaluation")
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='global_trajectory_evaluation',
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
    
    print("Global Frame Trajectory Evaluation")
    print("="*60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Create evaluator and run evaluation
    evaluator = GlobalTrajectoryEvaluator(args.checkpoint, config)
    analysis_results, pred_traj, gt_traj = evaluator.evaluate_global_trajectory(args.output_dir)


if __name__ == "__main__":
    main()