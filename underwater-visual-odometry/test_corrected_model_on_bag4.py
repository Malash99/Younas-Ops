#!/usr/bin/env python3
"""
Test the corrected two-frame consecutive model on unseen ROS bag 4 data.
This provides a true test of generalization performance on completely unseen underwater footage.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from data_processing.data_loader_fixed import UnderwaterVODatasetFixed
from models.baseline.baseline_cnn import create_model
from utils.coordinate_transforms_fixed import integrate_trajectory_correct, compute_trajectory_errors_correct


def test_on_unseen_bag4():
    """Test corrected model on completely unseen bag 4 data."""
    print("="*80)
    print("TESTING CORRECTED MODEL ON UNSEEN BAG 4 DATA")
    print("="*80)
    
    # Paths
    model_path = './output/models/corrected_model_20250729_220921/best_model.h5'
    bag4_data_dir = './data/bag4_test'
    output_dir = './output/models/corrected_model_20250729_220921/bag4_unseen_test'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Testing model: {model_path}")
    print(f"On unseen data: {bag4_data_dir}")
    print(f"Results will be saved to: {output_dir}")
    
    # Load bag 4 dataset with CORRECTED transformations
    print("\nLoading BAG 4 dataset with CORRECTED coordinate transformations...")
    dataset = UnderwaterVODatasetFixed(bag4_data_dir, sequence_length=2, image_size=(224, 224))
    dataset.load_data()
    
    # Use ALL bag 4 data for testing (it's completely unseen)
    all_indices = list(range(len(dataset.poses_df) - 1))  # -1 for pairs
    print(f"Testing on ALL {len(all_indices)} samples from bag 4 (completely unseen data)")
    
    # Load trained model
    print("\\nLoading trained corrected model...")
    model = create_model('baseline', input_shape=(224, 224, 6))
    model.build((None, 224, 224, 6))
    model.load_weights(model_path)
    print("Model loaded successfully!")
    
    # Create test dataset
    print("\\nCreating test dataset...")
    test_dataset = dataset.create_tf_dataset(all_indices, batch_size=1, shuffle=False)
    
    # Get predictions on ALL bag 4 data
    print("Getting predictions on unseen bag 4 data...")
    predictions = []
    ground_truth = []
    
    sample_count = 0
    for img_batch, pose_batch in test_dataset:
        if sample_count >= len(all_indices):
            break
            
        pred_batch = model.predict(img_batch, verbose=0)
        predictions.append(pred_batch[0])
        ground_truth.append(pose_batch.numpy()[0])
        
        sample_count += 1
        if sample_count % 100 == 0:
            print(f"Processed {sample_count}/{len(all_indices)} samples")
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    print(f"\\nCollected {len(predictions)} predictions on unseen data")
    
    # Compute per-frame errors
    trans_errors = np.linalg.norm(predictions[:, :3] - ground_truth[:, :3], axis=1)
    rot_errors = np.linalg.norm(predictions[:, 3:] - ground_truth[:, 3:], axis=1)
    
    print(f"\\nPer-frame Performance on UNSEEN BAG 4:")
    print(f"Translation Error - Mean: {np.mean(trans_errors):.4f} ± {np.std(trans_errors):.4f} m")
    print(f"Translation Error - Median: {np.median(trans_errors):.4f} m")
    print(f"Rotation Error - Mean: {np.mean(rot_errors):.4f} ± {np.std(rot_errors):.4f} rad")
    print(f"Rotation Error - Median: {np.median(rot_errors):.4f} rad")
    
    # Create trajectory comparison with CORRECTED transformations
    print("\\nIntegrating trajectories with CORRECTED transformations...")
    
    # Get initial pose from bag 4 data
    initial_pose = dataset.get_pose_dict(0)
    print(f"Initial pose from bag 4: {initial_pose['position']}")
    
    # Integrate trajectories using CORRECTED method
    pred_trajectory = integrate_trajectory_correct(initial_pose, predictions)
    gt_trajectory = integrate_trajectory_correct(initial_pose, ground_truth)
    
    # Compute trajectory errors using CORRECTED method
    traj_errors = compute_trajectory_errors_correct(pred_trajectory, gt_trajectory)
    
    print(f"\\nTrajectory Performance on UNSEEN BAG 4 (CORRECTED transformations):")
    print(f"ATE RMSE: {traj_errors['ate_rmse']:.4f} m")
    print(f"ATE Mean: {traj_errors['ate_mean']:.4f} ± {traj_errors['ate_std']:.4f} m")
    print(f"RPE Translation: {traj_errors['rpe_trans_mean']:.4f} ± {traj_errors['rpe_trans_std']:.4f} m")
    print(f"RPE Rotation: {traj_errors['rpe_rot_mean']:.4f} ± {traj_errors['rpe_rot_std']:.4f} rad")
    
    # Create comprehensive visualizations
    create_unseen_data_visualizations(pred_trajectory, gt_trajectory, traj_errors, output_dir)
    create_unseen_error_analysis(trans_errors, rot_errors, traj_errors, output_dir)
    
    # Save results
    results = {
        'test_data': 'bag4_completely_unseen',
        'num_test_samples': len(all_indices),
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
        }
    }
    
    with open(os.path.join(output_dir, 'bag4_unseen_test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*80)
    print("UNSEEN BAG 4 TESTING COMPLETED")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"\\nKey Findings on COMPLETELY UNSEEN DATA:")
    print(f"  - ATE RMSE: {results['trajectory']['ate_rmse']:.4f} m")
    print(f"  - Translation Error: {results['per_frame']['translation_error_mean']:.4f} ± {results['per_frame']['translation_error_std']:.4f} m")
    print(f"  - Rotation Error: {results['per_frame']['rotation_error_mean']:.4f} ± {results['per_frame']['rotation_error_std']:.4f} rad")
    print("="*80)
    
    return results


def create_unseen_data_visualizations(pred_traj, gt_traj, errors, output_dir):
    """Create visualizations specifically for unseen bag 4 testing."""
    print("Creating unseen data visualizations...")
    
    # Extract positions
    pred_positions = np.array([pose['position'] for pose in pred_traj])
    gt_positions = np.array([pose['position'] for pose in gt_traj])
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # XY trajectory plot
    axes[0, 0].plot(gt_positions[:, 0], gt_positions[:, 1], 'b-', linewidth=3, label='Ground Truth (Bag 4)', alpha=0.8)
    axes[0, 0].plot(pred_positions[:, 0], pred_positions[:, 1], 'r--', linewidth=3, label='Predicted (Corrected Model)', alpha=0.8)
    axes[0, 0].scatter(gt_positions[0, 0], gt_positions[0, 1], c='green', s=150, marker='o', label='Start', zorder=5)
    axes[0, 0].scatter(gt_positions[-1, 0], gt_positions[-1, 1], c='red', s=150, marker='s', label='End', zorder=5)
    axes[0, 0].set_xlabel('X (m)', fontsize=12)
    axes[0, 0].set_ylabel('Y (m)', fontsize=12)
    axes[0, 0].set_title('XY Trajectory - Unseen Bag 4 Data', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # XZ trajectory plot  
    axes[0, 1].plot(gt_positions[:, 0], gt_positions[:, 2], 'b-', linewidth=3, label='Ground Truth', alpha=0.8)
    axes[0, 1].plot(pred_positions[:, 0], pred_positions[:, 2], 'r--', linewidth=3, label='Predicted', alpha=0.8)
    axes[0, 1].set_xlabel('X (m)', fontsize=12)
    axes[0, 1].set_ylabel('Z (m)', fontsize=12)
    axes[0, 1].set_title('XZ Trajectory - Depth Analysis', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Position errors over time
    position_errors = errors['position_errors']
    axes[1, 0].plot(position_errors, 'g-', linewidth=2, alpha=0.7)
    axes[1, 0].axhline(y=np.mean(position_errors), color='r', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(position_errors):.3f}m')
    axes[1, 0].fill_between(range(len(position_errors)), 
                           np.mean(position_errors) - np.std(position_errors),
                           np.mean(position_errors) + np.std(position_errors),
                           alpha=0.2, color='r', label=f'±1σ: {np.std(position_errors):.3f}m')
    axes[1, 0].set_xlabel('Frame Number', fontsize=12)
    axes[1, 0].set_ylabel('Position Error (m)', fontsize=12)
    axes[1, 0].set_title('Position Error Over Time - Unseen Data', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Detailed error statistics
    axes[1, 1].hist(position_errors, bins=40, alpha=0.7, color='purple', edgecolor='black', density=True)
    axes[1, 1].axvline(x=np.mean(position_errors), color='r', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(position_errors):.3f}m')
    axes[1, 1].axvline(x=np.median(position_errors), color='orange', linestyle='--', linewidth=2, 
                      label=f'Median: {np.median(position_errors):.3f}m')
    axes[1, 1].axvline(x=np.percentile(position_errors, 95), color='purple', linestyle=':', linewidth=2,
                      label=f'95th %ile: {np.percentile(position_errors, 95):.3f}m')
    axes[1, 1].set_xlabel('Position Error (m)', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Error Distribution - Unseen Bag 4', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Corrected Model Performance on Completely Unseen Underwater Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bag4_unseen_trajectory_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create 3D trajectory plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'b-', linewidth=4, 
           label='Ground Truth (Bag 4)', alpha=0.8)
    ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 'r--', linewidth=4, 
           label='Predicted (Corrected Model)', alpha=0.8)
    ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], c='green', s=200, label='Start')
    ax.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], c='red', s=200, label='End')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('3D Trajectory Comparison - Unseen Bag 4 Data\\n(Corrected Model with Fixed Coordinate Transformations)', 
                fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.savefig(os.path.join(output_dir, 'bag4_unseen_3d_trajectory.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_unseen_error_analysis(trans_errors, rot_errors, traj_errors, output_dir):
    """Create detailed error analysis for unseen data testing."""
    print("Creating unseen data error analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Translation error over time
    axes[0, 0].plot(trans_errors, 'b-', linewidth=2, alpha=0.7)
    axes[0, 0].axhline(y=np.mean(trans_errors), color='r', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(trans_errors):.4f}m')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Translation Error (m)')
    axes[0, 0].set_title('Per-Frame Translation Error\\n(Unseen Bag 4)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rotation error over time
    axes[0, 1].plot(rot_errors, 'g-', linewidth=2, alpha=0.7)
    axes[0, 1].axhline(y=np.mean(rot_errors), color='r', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(rot_errors):.4f}rad')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Rotation Error (rad)')
    axes[0, 1].set_title('Per-Frame Rotation Error\\n(Unseen Bag 4)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Combined error scatter
    axes[0, 2].scatter(trans_errors, rot_errors, alpha=0.6, c='purple', s=20)
    axes[0, 2].set_xlabel('Translation Error (m)')
    axes[0, 2].set_ylabel('Rotation Error (rad)')
    axes[0, 2].set_title('Translation vs Rotation Error\\n(Unseen Data)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Translation error histogram with statistics
    axes[1, 0].hist(trans_errors, bins=40, alpha=0.7, color='blue', edgecolor='black', density=True)
    axes[1, 0].axvline(x=np.mean(trans_errors), color='r', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(trans_errors):.4f}m')
    axes[1, 0].axvline(x=np.median(trans_errors), color='orange', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(trans_errors):.4f}m')
    axes[1, 0].axvline(x=np.percentile(trans_errors, 95), color='purple', linestyle=':', linewidth=2,
                      label=f'95th %ile: {np.percentile(trans_errors, 95):.4f}m')
    axes[1, 0].set_xlabel('Translation Error (m)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Translation Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rotation error histogram with statistics
    axes[1, 1].hist(rot_errors, bins=40, alpha=0.7, color='green', edgecolor='black', density=True)
    axes[1, 1].axvline(x=np.mean(rot_errors), color='r', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(rot_errors):.4f}rad')
    axes[1, 1].axvline(x=np.median(rot_errors), color='orange', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(rot_errors):.4f}rad')
    axes[1, 1].axvline(x=np.percentile(rot_errors, 95), color='purple', linestyle=':', linewidth=2,
                      label=f'95th %ile: {np.percentile(rot_errors, 95):.4f}rad')
    axes[1, 1].set_xlabel('Rotation Error (rad)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Rotation Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Performance summary text
    axes[1, 2].axis('off')
    summary_text = f"""
UNSEEN BAG 4 TEST RESULTS
(Corrected Model Performance)

TRANSLATION ERRORS:
• Mean: {np.mean(trans_errors):.4f} ± {np.std(trans_errors):.4f} m
• Median: {np.median(trans_errors):.4f} m
• 95th Percentile: {np.percentile(trans_errors, 95):.4f} m
• Max: {np.max(trans_errors):.4f} m

ROTATION ERRORS:
• Mean: {np.mean(rot_errors):.4f} ± {np.std(rot_errors):.4f} rad
• Median: {np.median(rot_errors):.4f} rad
• 95th Percentile: {np.percentile(rot_errors, 95):.4f} rad
• Max: {np.max(rot_errors):.4f} rad

TRAJECTORY METRICS:
• ATE RMSE: {traj_errors['ate_rmse']:.4f} m
• RPE Translation: {traj_errors['rpe_trans_mean']:.4f} m
• RPE Rotation: {traj_errors['rpe_rot_mean']:.4f} rad

TEST DATA: Completely unseen ROS bag 4
SAMPLES: {len(trans_errors)} frame pairs
MODEL: Corrected coordinate transformations
    """
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Detailed Error Analysis - Corrected Model on Unseen Bag 4 Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bag4_unseen_error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    test_on_unseen_bag4()