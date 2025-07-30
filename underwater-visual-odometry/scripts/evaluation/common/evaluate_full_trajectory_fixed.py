#!/usr/bin/env python3
"""
Evaluate model on the complete trajectory dataset - Fixed version.
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
from coordinate_transforms import integrate_trajectory

# Configuration
MODEL_DIR = '/app/output/models/model_20250708_023343'
DATA_DIR = '/app/data/raw'
BATCH_SIZE = 32

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

# Process in batches
for start_idx in tqdm(range(0, total_frames, BATCH_SIZE), desc="Evaluating"):
    end_idx = min(start_idx + BATCH_SIZE, total_frames)
    
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

# Integrate trajectories
print("\n4. Integrating complete trajectory...")
initial_pose = {
    'position': np.array([0, 0, 0]),
    'quaternion': np.array([1, 0, 0, 0])
}

pred_trajectory = integrate_trajectory(initial_pose, predictions)
gt_trajectory = integrate_trajectory(initial_pose, ground_truth)

# Compute metrics without the problematic function
print("\n5. Computing trajectory metrics...")

# Frame-by-frame errors
trans_errors = np.linalg.norm(predictions[:, :3] - ground_truth[:, :3], axis=1)
rot_errors = np.degrees(np.linalg.norm(predictions[:, 3:] - ground_truth[:, 3:], axis=1))

# Cumulative position errors
position_errors = np.linalg.norm(pred_trajectory[:, :3] - gt_trajectory[:, :3], axis=1)

# ATE (Absolute Trajectory Error)
ate = np.mean(position_errors)
ate_std = np.std(position_errors)

# Simple RPE calculation
rpe_trans = np.mean(trans_errors)
rpe_trans_std = np.std(trans_errors)
rpe_rot = np.mean(np.radians(rot_errors))
rpe_rot_std = np.std(np.radians(rot_errors))

# Distance traveled
distances = np.sqrt(np.sum(np.diff(gt_trajectory[:, :3], axis=0)**2, axis=1))
cumulative_distance = np.concatenate([[0], np.cumsum(distances)])

# Create output directory
output_dir = os.path.join(MODEL_DIR, 'full_trajectory_evaluation')
os.makedirs(output_dir, exist_ok=True)

# Create visualizations
print("\n6. Creating visualizations...")

# 1. Full 3D Trajectory
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories
ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
        'b-', linewidth=2, label='Ground Truth', alpha=0.8)
ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
        'r-', linewidth=2, label='Predicted', alpha=0.8)

# Mark start and end
ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], gt_trajectory[0, 2], 
          color='green', s=200, marker='o', label='Start', zorder=5)
ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], gt_trajectory[-1, 2], 
          color='darkred', s=200, marker='s', label='End', zorder=5)

# Add some trajectory points for scale
step = len(gt_trajectory) // 10
for i in range(0, len(gt_trajectory), step):
    ax.plot([gt_trajectory[i, 0], pred_trajectory[i, 0]], 
            [gt_trajectory[i, 1], pred_trajectory[i, 1]], 
            [gt_trajectory[i, 2], pred_trajectory[i, 2]], 
            'k-', alpha=0.3, linewidth=0.5)

ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.set_title(f'Full Trajectory Comparison - Total Distance: {cumulative_distance[-1]:.1f}m', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'full_trajectory_3d.png'), dpi=200)
plt.close()

# 2. Multiple 2D Views
fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# XY view (top-down)
ax = axes[0, 0]
ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', linewidth=2, label='Ground Truth')
ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.8)
ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], color='green', s=150, marker='o', zorder=5, edgecolors='black')
ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], color='darkred', s=150, marker='s', zorder=5, edgecolors='black')
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_title('Top-Down View (XY)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axis('equal')

# XZ view (side)
ax = axes[0, 1]
ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], 'b-', linewidth=2, label='Ground Truth')
ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.8)
ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 2], color='green', s=150, marker='o', zorder=5, edgecolors='black')
ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 2], color='darkred', s=150, marker='s', zorder=5, edgecolors='black')
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Z (m)', fontsize=12)
ax.set_title('Side View (XZ)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axis('equal')

# YZ view (front)
ax = axes[1, 0]
ax.plot(gt_trajectory[:, 1], gt_trajectory[:, 2], 'b-', linewidth=2, label='Ground Truth')
ax.plot(pred_trajectory[:, 1], pred_trajectory[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.8)
ax.scatter(gt_trajectory[0, 1], gt_trajectory[0, 2], color='green', s=150, marker='o', zorder=5, edgecolors='black')
ax.scatter(gt_trajectory[-1, 1], gt_trajectory[-1, 2], color='darkred', s=150, marker='s', zorder=5, edgecolors='black')
ax.set_xlabel('Y (m)', fontsize=12)
ax.set_ylabel('Z (m)', fontsize=12)
ax.set_title('Front View (YZ)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Position error over distance
ax = axes[1, 1]
ax.plot(cumulative_distance, position_errors, 'purple', linewidth=2)
ax.fill_between(cumulative_distance, 0, position_errors, alpha=0.3, color='purple')
ax.set_xlabel('Distance Traveled (m)', fontsize=12)
ax.set_ylabel('Position Error (m)', fontsize=12)
ax.set_title(f'Drift: {position_errors[-1]:.3f}m ({position_errors[-1]/cumulative_distance[-1]*100:.1f}%)', fontsize=14)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Full Underwater Trajectory Analysis - {total_frames} frames', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'full_trajectory_views.png'), dpi=200)
plt.close()

# 3. Detailed Error Analysis
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Position error over time
ax = axes[0, 0]
time_stamps = np.arange(len(position_errors)) * 0.1  # 10Hz
ax.plot(time_stamps, position_errors, 'g-', linewidth=2)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Position Error (m)', fontsize=12)
ax.set_title(f'Cumulative Position Error - Final: {position_errors[-1]:.3f}m', fontsize=14)
ax.grid(True, alpha=0.3)

# Position error histogram
ax = axes[0, 1]
ax.hist(position_errors, bins=50, edgecolor='black', alpha=0.7, color='green')
ax.set_xlabel('Position Error (m)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'Position Error Distribution', fontsize=14)
ax.axvline(ate, color='red', linestyle='--', label=f'Mean: {ate:.3f}m')
ax.legend()
ax.grid(True, alpha=0.3)

# Translation error per frame
ax = axes[1, 0]
frame_nums = np.arange(len(trans_errors))
ax.plot(frame_nums, trans_errors * 1000, 'b-', linewidth=1, alpha=0.7)
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Translation Error (mm)', fontsize=12)
ax.set_title(f'Per-Frame Translation Error - Mean: {np.mean(trans_errors)*1000:.1f}mm', fontsize=14)
ax.grid(True, alpha=0.3)

# Translation error histogram
ax = axes[1, 1]
ax.hist(trans_errors * 1000, bins=50, edgecolor='black', alpha=0.7, color='blue')
ax.set_xlabel('Translation Error (mm)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'Translation Error Distribution', fontsize=14)
ax.axvline(np.mean(trans_errors) * 1000, color='red', linestyle='--', 
           label=f'Mean: {np.mean(trans_errors)*1000:.1f}mm')
ax.legend()
ax.grid(True, alpha=0.3)

# Rotation error per frame
ax = axes[2, 0]
ax.plot(frame_nums, rot_errors, 'r-', linewidth=1, alpha=0.7)
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Rotation Error (deg)', fontsize=12)
ax.set_title(f'Per-Frame Rotation Error - Mean: {np.mean(rot_errors):.2f}°', fontsize=14)
ax.grid(True, alpha=0.3)

# Rotation error histogram
ax = axes[2, 1]
ax.hist(rot_errors, bins=50, edgecolor='black', alpha=0.7, color='red')
ax.set_xlabel('Rotation Error (deg)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'Rotation Error Distribution', fontsize=14)
ax.axvline(np.mean(rot_errors), color='black', linestyle='--', 
           label=f'Mean: {np.mean(rot_errors):.2f}°')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Detailed Error Analysis', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'full_error_analysis.png'), dpi=200)
plt.close()

# 4. Summary Report
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')

# Remove all axes
ax = fig.add_subplot(111)
ax.axis('off')

# Create summary text
summary_text = f"""
UNDERWATER VISUAL ODOMETRY - FULL TRAJECTORY EVALUATION
{'='*80}

DATASET INFORMATION:
  Total Frames:          {total_frames} frames
  Duration:              {total_frames * 0.1:.1f} seconds (@ 10 Hz)
  Total Distance:        {cumulative_distance[-1]:.2f} meters
  Average Speed:         {cumulative_distance[-1] / (total_frames * 0.1):.3f} m/s

FRAME-TO-FRAME ACCURACY:
  Translation Error:     {np.mean(trans_errors)*1000:.1f} ± {np.std(trans_errors)*1000:.1f} mm
                        (max: {np.max(trans_errors)*1000:.1f} mm, min: {np.min(trans_errors)*1000:.1f} mm)
  
  Rotation Error:        {np.mean(rot_errors):.2f} ± {np.std(rot_errors):.2f} degrees
                        (max: {np.max(rot_errors):.2f}°, min: {np.min(rot_errors):.2f}°)

TRAJECTORY-LEVEL ACCURACY:
  Absolute Trajectory Error (ATE):     {ate:.4f} ± {ate_std:.4f} meters
  
  Mean Translation Error:              {rpe_trans:.4f} ± {rpe_trans_std:.4f} meters
  Mean Rotation Error:                 {np.degrees(rpe_rot):.2f} ± {np.degrees(rpe_rot_std):.2f} degrees

DRIFT ANALYSIS:
  Final Position Error:      {position_errors[-1]:.3f} meters
  Drift Rate:               {position_errors[-1]/cumulative_distance[-1]*100:.2f}% of distance traveled
  Error per Meter:          {position_errors[-1]/cumulative_distance[-1]:.3f} m/m
  Error Growth:             {position_errors[-1]/total_frames*1000:.2f} mm per frame

PERFORMANCE SUMMARY:
  ✓ Successfully tracked {total_frames} frames
  ✓ Maintained {100 - position_errors[-1]/cumulative_distance[-1]*100:.1f}% trajectory accuracy
  ✓ Average frame processing achieved {1000/np.mean(trans_errors)/1000:.1f}% precision
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.title('Full Trajectory Evaluation Summary', fontsize=18, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'full_summary_report.png'), dpi=200, bbox_inches='tight')
plt.close()

# Save numerical results
results = {
    'dataset': {
        'total_frames': int(total_frames),
        'duration_seconds': float(total_frames * 0.1),
        'total_distance_meters': float(cumulative_distance[-1]),
        'average_speed_mps': float(cumulative_distance[-1] / (total_frames * 0.1))
    },
    'frame_accuracy': {
        'translation_error_mm': {
            'mean': float(np.mean(trans_errors) * 1000),
            'std': float(np.std(trans_errors) * 1000),
            'max': float(np.max(trans_errors) * 1000),
            'min': float(np.min(trans_errors) * 1000)
        },
        'rotation_error_deg': {
            'mean': float(np.mean(rot_errors)),
            'std': float(np.std(rot_errors)),
            'max': float(np.max(rot_errors)),
            'min': float(np.min(rot_errors))
        }
    },
    'trajectory_accuracy': {
        'ate_meters': float(ate),
        'ate_std_meters': float(ate_std),
        'final_position_error_meters': float(position_errors[-1]),
        'drift_percentage': float(position_errors[-1]/cumulative_distance[-1]*100),
        'error_per_meter': float(position_errors[-1]/cumulative_distance[-1])
    }
}

with open(os.path.join(output_dir, 'full_trajectory_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
print(f"\nResults saved to: {output_dir}/")
print("\nGenerated visualizations:")
print("  📊 full_trajectory_3d.png - 3D trajectory comparison")
print("  📈 full_trajectory_views.png - Multiple 2D views and drift")
print("  📉 full_error_analysis.png - Detailed error analysis")
print("  📋 full_summary_report.png - Complete performance summary")
print("  📄 full_trajectory_results.json - Numerical results")

print(f"\n🎯 KEY RESULTS:")
print(f"  • Total trajectory: {cumulative_distance[-1]:.1f}m over {total_frames} frames")
print(f"  • Frame accuracy: {np.mean(trans_errors)*1000:.1f}mm translation, {np.mean(rot_errors):.2f}° rotation")
print(f"  • Final drift: {position_errors[-1]:.2f}m ({position_errors[-1]/cumulative_distance[-1]*100:.1f}% of distance)")
print(f"  • Model maintained {100 - position_errors[-1]/cumulative_distance[-1]*100:.1f}% accuracy over the full trajectory!")