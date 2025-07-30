#!/usr/bin/env python3
"""
Comprehensive evaluation script for the corrected two-frame consecutive model.
Tests the model with fixed coordinate transformations and creates visualizations.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from data_processing.data_loader_fixed import UnderwaterVODatasetFixed
from models.baseline.baseline_cnn import create_model
from utils.coordinate_transforms_fixed import integrate_trajectory_correct, compute_trajectory_errors_correct


def load_trained_model(model_path, input_shape=(224, 224, 6)):
    """Load the trained corrected model."""
    print(f"Loading model from: {model_path}")
    
    # Create model architecture
    model = create_model('baseline', input_shape=input_shape)
    
    # Build model
    model.build((None, *input_shape))
    
    # Load weights
    model.load_weights(model_path)
    
    print("Model loaded successfully!")
    return model


def evaluate_on_test_data(model, dataset, test_indices, output_dir):
    """Evaluate model on test data and create visualizations."""
    print(f"\nEvaluating on {len(test_indices)} test samples...")
    
    # Create test dataset
    test_dataset = dataset.create_tf_dataset(test_indices, batch_size=1, shuffle=False)
    
    # Get predictions
    predictions = []
    ground_truth = []
    
    print("Getting model predictions...")
    for i, (img_batch, pose_batch) in enumerate(test_dataset):
        if i >= len(test_indices):
            break
            
        pred_batch = model.predict(img_batch, verbose=0)
        predictions.append(pred_batch[0])
        ground_truth.append(pose_batch.numpy()[0])
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(test_indices)} samples")
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    print(f"Collected {len(predictions)} predictions")
    
    # Compute per-frame errors
    trans_errors = np.linalg.norm(predictions[:, :3] - ground_truth[:, :3], axis=1)
    rot_errors = np.linalg.norm(predictions[:, 3:] - ground_truth[:, 3:], axis=1)
    
    print(f"\nPer-frame Performance:")
    print(f"Translation Error - Mean: {np.mean(trans_errors):.4f} ± {np.std(trans_errors):.4f} m")
    print(f"Rotation Error - Mean: {np.mean(rot_errors):.4f} ± {np.std(rot_errors):.4f} rad")
    print(f"Translation Error - Median: {np.median(trans_errors):.4f} m")
    print(f"Rotation Error - Median: {np.median(rot_errors):.4f} rad")
    
    # Create trajectory comparison
    print("\nIntegrating trajectories with CORRECTED transformations...")
    
    # Get initial pose
    initial_pose = {
        'position': np.array([0.0, 0.0, 0.0]),
        'quaternion': np.array([1.0, 0.0, 0.0, 0.0])
    }
    
    # Integrate trajectories using CORRECTED method
    pred_trajectory = integrate_trajectory_correct(initial_pose, predictions)
    gt_trajectory = integrate_trajectory_correct(initial_pose, ground_truth)
    
    # Compute trajectory errors using CORRECTED method
    traj_errors = compute_trajectory_errors_correct(pred_trajectory, gt_trajectory)
    
    print(f"\nTrajectory Performance (CORRECTED transformations):")
    print(f"ATE RMSE: {traj_errors['ate_rmse']:.4f} m")
    print(f"ATE Mean: {traj_errors['ate_mean']:.4f} ± {traj_errors['ate_std']:.4f} m")
    print(f"RPE Translation: {traj_errors['rpe_trans_mean']:.4f} ± {traj_errors['rpe_trans_std']:.4f} m")
    print(f"RPE Rotation: {traj_errors['rpe_rot_mean']:.4f} ± {traj_errors['rpe_rot_std']:.4f} rad")
    
    # Create comprehensive visualizations
    create_trajectory_visualizations(pred_trajectory, gt_trajectory, traj_errors, output_dir)
    create_error_analysis_plots(trans_errors, rot_errors, traj_errors, output_dir)
    
    # Save results
    results = {
        'per_frame': {
            'translation_error_mean': float(np.mean(trans_errors)),
            'translation_error_std': float(np.std(trans_errors)),
            'translation_error_median': float(np.median(trans_errors)),
            'rotation_error_mean': float(np.mean(rot_errors)),
            'rotation_error_std': float(np.std(rot_errors)),
            'rotation_error_median': float(np.median(rot_errors))
        },
        'trajectory': {
            'ate_rmse': float(traj_errors['ate_rmse']),
            'ate_mean': float(traj_errors['ate_mean']),
            'ate_std': float(traj_errors['ate_std']),
            'rpe_trans_mean': float(traj_errors['rpe_trans_mean']),
            'rpe_trans_std': float(traj_errors['rpe_trans_std']),
            'rpe_rot_mean': float(traj_errors['rpe_rot_mean']),
            'rpe_rot_std': float(traj_errors['rpe_rot_std'])
        },
        'num_test_samples': len(test_indices)
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def create_trajectory_visualizations(pred_traj, gt_traj, errors, output_dir):
    """Create comprehensive trajectory visualization plots."""
    print("Creating trajectory visualizations...")
    
    # Extract positions
    pred_positions = np.array([pose['position'] for pose in pred_traj])
    gt_positions = np.array([pose['position'] for pose in gt_traj])
    
    # Create multi-panel trajectory plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # XY trajectory plot
    axes[0, 0].plot(gt_positions[:, 0], gt_positions[:, 1], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    axes[0, 0].plot(pred_positions[:, 0], pred_positions[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.8)
    axes[0, 0].scatter(gt_positions[0, 0], gt_positions[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    axes[0, 0].scatter(gt_positions[-1, 0], gt_positions[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('XY Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # XZ trajectory plot
    axes[0, 1].plot(gt_positions[:, 0], gt_positions[:, 2], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    axes[0, 1].plot(pred_positions[:, 0], pred_positions[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.8)
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Z (m)')
    axes[0, 1].set_title('XZ Trajectory Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Position errors over time
    position_errors = errors['position_errors']
    axes[1, 0].plot(position_errors, 'g-', linewidth=2)
    axes[1, 0].axhline(y=np.mean(position_errors), color='r', linestyle='--', label=f'Mean: {np.mean(position_errors):.3f}m')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Position Error (m)')
    axes[1, 0].set_title('Position Error Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error histogram
    axes[1, 1].hist(position_errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(x=np.mean(position_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(position_errors):.3f}m')
    axes[1, 1].axvline(x=np.median(position_errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(position_errors):.3f}m')
    axes[1, 1].set_xlabel('Position Error (m)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Position Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'corrected_model_trajectory_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create 3D trajectory plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'b-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 'r--', linewidth=3, label='Predicted', alpha=0.8)
    ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], c='green', s=100, label='Start')
    ax.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], c='red', s=100, label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory Comparison - Corrected Model')
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, 'corrected_model_3d_trajectory.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_error_analysis_plots(trans_errors, rot_errors, traj_errors, output_dir):
    """Create detailed error analysis plots."""
    print("Creating error analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Translation error over time
    axes[0, 0].plot(trans_errors, 'b-', linewidth=2, alpha=0.7)
    axes[0, 0].axhline(y=np.mean(trans_errors), color='r', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(trans_errors):.4f}m')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Translation Error (m)')
    axes[0, 0].set_title('Per-Frame Translation Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rotation error over time
    axes[0, 1].plot(rot_errors, 'g-', linewidth=2, alpha=0.7)
    axes[0, 1].axhline(y=np.mean(rot_errors), color='r', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(rot_errors):.4f}rad')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Rotation Error (rad)')
    axes[0, 1].set_title('Per-Frame Rotation Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Translation error histogram
    axes[1, 0].hist(trans_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(x=np.mean(trans_errors), color='r', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(trans_errors):.4f}m')
    axes[1, 0].axvline(x=np.median(trans_errors), color='orange', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(trans_errors):.4f}m')
    axes[1, 0].set_xlabel('Translation Error (m)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Translation Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rotation error histogram
    axes[1, 1].hist(rot_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].axvline(x=np.mean(rot_errors), color='r', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(rot_errors):.4f}rad')
    axes[1, 1].axvline(x=np.median(rot_errors), color='orange', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(rot_errors):.4f}rad')
    axes[1, 1].set_xlabel('Rotation Error (rad)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Rotation Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'corrected_model_error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function."""
    print("="*80)
    print("CORRECTED MODEL EVALUATION")
    print("="*80)
    
    # Configuration
    model_path = './output/models/corrected_model_20250729_220921/best_model.h5'
    data_dir = './data/raw'
    output_dir = './output/models/corrected_model_20250729_220921/evaluation'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load dataset with CORRECTED transformations
    print("Loading dataset with CORRECTED coordinate transformations...")
    dataset = UnderwaterVODatasetFixed(data_dir, sequence_length=2, image_size=(224, 224))
    dataset.load_data()
    
    # Use validation split as test data (unseen during training)
    train_indices, test_indices = dataset.train_val_split(val_ratio=0.2, seed=42)
    print(f"Using {len(test_indices)} validation samples for testing")
    
    # Load trained model
    model = load_trained_model(model_path)
    
    # Evaluate model
    results = evaluate_on_test_data(model, dataset, test_indices, output_dir)
    
    print("="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Key Performance Metrics:")
    print(f"  - ATE RMSE: {results['trajectory']['ate_rmse']:.4f} m")
    print(f"  - Translation Error: {results['per_frame']['translation_error_mean']:.4f} ± {results['per_frame']['translation_error_std']:.4f} m")
    print(f"  - Rotation Error: {results['per_frame']['rotation_error_mean']:.4f} ± {results['per_frame']['rotation_error_std']:.4f} rad")
    print("="*80)


if __name__ == "__main__":
    main()