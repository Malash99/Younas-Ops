#!/usr/bin/env python3
"""
Evaluate model on the complete trajectory dataset.
"""

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

import sys
sys.path.append('/app/scripts')
from data_loader import UnderwaterVODataset
from models.baseline_cnn import create_model
from coordinate_transforms import integrate_trajectory, compute_trajectory_error

# Configuration
MODEL_DIR = '/app/output/models/model_20250708_023343'
DATA_DIR = '/app/data/raw'
BATCH_SIZE = 32  # Process in batches for efficiency

print("="*70)
print("FULL TRAJECTORY EVALUATION - UNDERWATER VISUAL ODOMETRY")
print("="*70)

# Load model
print("\n1. Loading trained model...")
model = create_model('baseline')
dummy_input = tf.zeros((1, 224, 224, 6))
_ = model(dummy_input, training=False)
model.load_weights(os.path.join(MODEL_DIR, 'best_model.h5'))
print("   ✓ Model loaded successfully")

# Load dataset
print("\n2. Loading complete dataset...")
dataset = UnderwaterVODataset(DATA_DIR, image_size=(224, 224))
dataset.load_data()
total_frames = len(dataset.poses_df) - 1
print(f"   ✓ Dataset loaded: {total_frames} frame pairs")

# Process entire trajectory
print(f"\n3. Processing {total_frames} frames...")
predictions = []
ground_truth = []

# Process in batches for efficiency
for start_idx in tqdm(range(0, total_frames, BATCH_SIZE), desc="Evaluating"):
    end_idx = min(start_idx + BATCH_SIZE, total_frames)
    batch_size = end_idx - start_idx
    
    # Prepare batch
    batch_images = []
    batch_gt = []
    
    for i in range(start_idx, end_idx):
        # Load images
        img1 = dataset.load_image(i)
        img2 = dataset.load_image(i + 1)
        img_pair = np.concatenate([img1, img2], axis=-1)
        batch_images.append(img_pair)
        
        # Ground truth
        gt = dataset.get_relative_pose(i, i + 1)
        gt_pose = np.concatenate([gt['translation'], gt['rotation']])
        batch_gt.append(gt_pose)
    
    # Predict batch
    batch_images = np.array(batch_images)
    batch_pred = model.predict(batch_images, verbose=0)
    
    predictions.extend(batch_pred)
    ground_truth.extend(batch_gt)

predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

print(f"   ✓ Processed {len(predictions)} predictions")

# Integrate full trajectories
print("\n4. Integrating complete trajectory...")
initial_pose = {
    'position': np.array([0, 0, 0]),
    'quaternion': np.array([1, 0, 0, 0])
}

pred_trajectory = integrate_trajectory(initial_pose, predictions)
gt_trajectory = integrate_trajectory(initial_pose, ground_truth)

# Compute comprehensive metrics
print("\n5. Computing trajectory metrics...")
errors = compute_trajectory_error(pred_trajectory, gt_trajectory)

# Frame-by-frame errors
trans_errors = np.linalg.norm(predictions[:, :3] - ground_truth[:, :3], axis=1)
rot_errors = np.degrees(np.linalg.norm(predictions[:, 3:] - ground_truth[:, 3:], axis=1))

# Cumulative errors
position_errors = np.linalg.norm(pred_trajectory[:, :3] - gt_trajectory[:, :3], axis=1)

# Distance traveled
distances = np.sqrt(np.sum(np.diff(gt_trajectory[:, :3], axis=0)**2, axis=1))
cumulative_distance = np.concatenate([[0], np.cumsum(distances)])

# Create output directory
output_dir = os.path.join(MODEL_DIR, 'full_trajectory_evaluation')
os.makedirs(output_dir, exist_ok=True)

# Create comprehensive visualizations
print("\n6. Creating visualizations...")

# 1. Full 3D Trajectory - matching your ground truth visualization
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories
ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
        'b-', linewidth=2, label='Ground Truth')
ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
        'r--', linewidth=2, label='Predicted', alpha=0.8)

# Mark start and end
ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], gt_trajectory[0, 2], 
          color='green', s=200, marker='o', label='Start', zorder=5)
ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], gt_trajectory[-1, 2], 
          color='red', s=200, marker='s', label='End', zorder=5)

ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.set_title('Full Trajectory Comparison - 3D View', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Set view angle similar to your ground truth plot
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'full_trajectory_3d.png'), dpi=200)
plt.close()

# 2. XY, XZ, YZ views in one figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# XY view
ax = axes[0]
ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', linewidth=2, label='Ground Truth')
ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=2, label='Predicted')
ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], color='green', s=100, marker='o', zorder=5)
ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], color='red', s=100, marker='s', zorder=5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Top-Down View (XY)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

# XZ view
ax = axes[1]
ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], 'b-', linewidth=2, label='Ground Truth')
ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], 'r--', linewidth=2, label='Predicted')
ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 2], color='green', s=100, marker='o', zorder=5)
ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 2], color='red', s=100, marker='s', zorder=5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Side View (XZ)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

# YZ view
ax = axes[2]
ax.plot(gt_trajectory[:, 1], gt_trajectory[:, 2], 'b-', linewidth=2, label='Ground Truth')
ax.plot(pred_trajectory[:, 1], pred_trajectory[:, 2], 'r--', linewidth=2, label='Predicted')
ax.scatter(gt_trajectory[0, 1], gt_trajectory[0, 2], color='green', s=100, marker='o', zorder=5)
ax.scatter(gt_trajectory[-1, 1], gt_trajectory[-1, 2], color='red', s=100, marker='s', zorder=5)
ax.set_xlabel('Y (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Front View (YZ)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.suptitle('Full Trajectory - Multiple Views', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'full_trajectory_views.png'), dpi=200)
plt.close()

# 3. Error Analysis over Full Trajectory
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Position error over time
ax = axes[0]
time_stamps = np.arange(len(position_errors)) * 0.1  # Assuming 10Hz
ax.plot(time_stamps, position_errors, 'g-', linewidth=2)
ax.fill_between(time_stamps, 0, position_errors, alpha=0.3, color='green')
ax.set_ylabel('Position Error (m)', fontsize=12)
ax.set_title(f'Position Error Over Time - Final Error: {position_errors[-1]:.3f} m', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, time_stamps[-1])

# Translation and rotation errors per frame
ax = axes[1]
frame_time = time_stamps[:-1]
ax.plot(frame_time, trans_errors * 1000, 'b-', linewidth=1, label='Translation (mm)', alpha=0.7)
ax.set_ylabel('Translation Error (mm)', fontsize=12)
ax.set_title(f'Per-Frame Translation Error - Mean: {np.mean(trans_errors)*1000:.1f} mm', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, frame_time[-1])

ax2 = ax.twinx()
ax2.plot(frame_time, rot_errors, 'r-', linewidth=1, label='Rotation (deg)', alpha=0.7)
ax2.set_ylabel('Rotation Error (deg)', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Error vs distance traveled
ax = axes[2]
ax.plot(cumulative_distance, position_errors, 'purple', linewidth=2)
ax.set_xlabel('Distance Traveled (m)', fontsize=12)
ax.set_ylabel('Position Error (m)', fontsize=12)
ax.set_title(f'Error vs Distance - Drift Rate: {position_errors[-1]/cumulative_distance[-1]*100:.2f}%', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, cumulative_distance[-1])

plt.suptitle('Full Trajectory Error Analysis', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'full_trajectory_errors.png'), dpi=200)
plt.close()

# 4. Comprehensive Performance Summary
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Error distributions
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(trans_errors * 1000, bins=50, edgecolor='black', alpha=0.7, color='blue')
ax1.set_xlabel('Translation Error (mm)')
ax1.set_ylabel('Frequency')
ax1.set_title(f'Translation Error Distribution\nMean: {np.mean(trans_errors)*1000:.1f} mm')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(rot_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax2.set_xlabel('Rotation Error (deg)')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Rotation Error Distribution\nMean: {np.mean(rot_errors):.2f}°')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(position_errors, bins=50, edgecolor='black', alpha=0.7, color='green')
ax3.set_xlabel('Position Error (m)')
ax3.set_ylabel('Frequency')
ax3.set_title(f'Cumulative Position Error\nFinal: {position_errors[-1]:.3f} m')
ax3.grid(True, alpha=0.3)

# Trajectory comparison (zoomed sections)
# Beginning
ax4 = fig.add_subplot(gs[1, 0])
idx_start = min(200, len(gt_trajectory))
ax4.plot(gt_trajectory[:idx_start, 0], gt_trajectory[:idx_start, 1], 'b-', linewidth=2, label='GT')
ax4.plot(pred_trajectory[:idx_start, 0], pred_trajectory[:idx_start, 1], 'r--', linewidth=2, label='Pred')
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.set_title('Trajectory Start (First 200 frames)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axis('equal')

# Middle
ax5 = fig.add_subplot(gs[1, 1])
mid_start = len(gt_trajectory) // 2 - 100
mid_end = len(gt_trajectory) // 2 + 100
ax5.plot(gt_trajectory[mid_start:mid_end, 0], gt_trajectory[mid_start:mid_end, 1], 'b-', linewidth=2, label='GT')
ax5.plot(pred_trajectory[mid_start:mid_end, 0], pred_trajectory[mid_start:mid_end, 1], 'r--', linewidth=2, label='Pred')
ax5.set_xlabel('X (m)')
ax5.set_ylabel('Y (m)')
ax5.set_title('Trajectory Middle')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.axis('equal')

# End
ax6 = fig.add_subplot(gs[1, 2])
idx_end = max(0, len(gt_trajectory) - 200)
ax6.plot(gt_trajectory[idx_end:, 0], gt_trajectory[idx_end:, 1], 'b-', linewidth=2, label='GT')
ax6.plot(pred_trajectory[idx_end:, 0], pred_trajectory[idx_end:, 1], 'r--', linewidth=2, label='Pred')
ax6.set_xlabel('X (m)')
ax6.set_ylabel('Y (m)')
ax6.set_title('Trajectory End (Last 200 frames)')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.axis('equal')

# Summary statistics
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('off')

summary_text = f"""
FULL TRAJECTORY EVALUATION SUMMARY
{'='*80}

Dataset Statistics:
  • Total Frames: {total_frames}
  • Total Duration: {total_frames * 0.1:.1f} seconds (assuming 10 Hz)
  • Total Distance Traveled: {cumulative_distance[-1]:.2f} m
  • Average Speed: {cumulative_distance[-1] / (total_frames * 0.1):.3f} m/s

Frame-to-Frame Accuracy:
  • Translation: {np.mean(trans_errors)*1000:.1f} ± {np.std(trans_errors)*1000:.1f} mm (max: {np.max(trans_errors)*1000:.1f} mm)
  • Rotation: {np.mean(rot_errors):.2f} ± {np.std(rot_errors):.2f}° (max: {np.max(rot_errors):.2f}°)

Trajectory Accuracy:
  • Absolute Trajectory Error (ATE): {errors['ate']:.4f} ± {errors['ate_std']:.4f} m
  • Relative Pose Error (Translation): {errors['rpe_trans']:.4f} ± {errors['rpe_trans_std']:.4f} m
  • Relative Pose Error (Rotation): {np.degrees(errors['rpe_rot']):.2f} ± {np.degrees(errors['rpe_rot_std']):.2f}°
  
Overall Performance:
  • Final Position Error: {position_errors[-1]:.3f} m
  • Drift Rate: {position_errors[-1]/cumulative_distance[-1]*100:.2f}% of distance traveled
  • Error Growth Rate: {position_errors[-1]/total_frames*1000:.2f} mm per frame
"""

ax7.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', 
         verticalalignment='center', transform=ax7.transAxes)

plt.suptitle('Underwater Visual Odometry - Full Trajectory Performance Analysis', fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'full_performance_summary.png'), dpi=200)
plt.close()

# Save numerical results
results = {
    'total_frames': int(total_frames),
    'total_distance_m': float(cumulative_distance[-1]),
    'duration_s': float(total_frames * 0.1),
    'frame_to_frame': {
        'translation_mean_mm': float(np.mean(trans_errors) * 1000),
        'translation_std_mm': float(np.std(trans_errors) * 1000),
        'translation_max_mm': float(np.max(trans_errors) * 1000),
        'rotation_mean_deg': float(np.mean(rot_errors)),
        'rotation_std_deg': float(np.std(rot_errors)),
        'rotation_max_deg': float(np.max(rot_errors))
    },
    'trajectory_metrics': {
        'ate_m': float(errors['ate']),
        'ate_std_m': float(errors['ate_std']),
        'rpe_trans_m': float(errors['rpe_trans']),
        'rpe_trans_std_m': float(errors['rpe_trans_std']),
        'rpe_rot_deg': float(np.degrees(errors['rpe_rot'])),
        'rpe_rot_std_deg': float(np.degrees(errors['rpe_rot_std']))
    },
    'overall': {
        'final_position_error_m': float(position_errors[-1]),
        'drift_percentage': float(position_errors[-1]/cumulative_distance[-1]*100),
        'error_per_meter': float(position_errors[-1]/cumulative_distance[-1]),
        'error_per_frame_mm': float(position_errors[-1]/total_frames*1000)
    }
}

with open(os.path.join(output_dir, 'full_trajectory_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
print(f"\nResults saved to: {output_dir}")
print("\nGenerated files:")
print("  📊 full_trajectory_3d.png - Complete 3D trajectory comparison")
print("  📈 full_trajectory_views.png - XY, XZ, YZ projections")
print("  📉 full_trajectory_errors.png - Error analysis over time")
print("  📋 full_performance_summary.png - Comprehensive performance metrics")
print("  📄 full_trajectory_results.json - Numerical results")

print(f"\n🎯 KEY RESULTS:")
print(f"  • Processed {total_frames} frames over {cumulative_distance[-1]:.1f} meters")
print(f"  • Frame accuracy: {np.mean(trans_errors)*1000:.1f} mm, {np.mean(rot_errors):.2f}°")
print(f"  • Final drift: {position_errors[-1]:.2f} m ({position_errors[-1]/cumulative_distance[-1]*100:.1f}%)")
print(f"  • Your model maintains {100 - position_errors[-1]/cumulative_distance[-1]*100:.1f}% accuracy!")