#!/usr/bin/env python3
"""
Evaluate the latest corrected model and plot infinity-pattern trajectory comparison.
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

# Add paths for imports
sys.path.append('./scripts')
sys.path.append('./scripts/data_processing')
sys.path.append('./scripts/models/baseline')
sys.path.append('./scripts/utils')

from data_loader import UnderwaterVODataset
from baseline_cnn import create_model
from coordinate_transforms import integrate_trajectory

# Configuration for latest model
MODEL_DIR = './output/models/corrected_model_20250729_220921'
DATA_DIR = './data/raw'
BATCH_SIZE = 16

print("="*70)
print("INFINITY TRAJECTORY EVALUATION - LATEST CORRECTED MODEL")
print("="*70)

# Load model
print("\n1. Loading latest corrected model...")
model = create_model('baseline')
dummy_input = tf.zeros((1, 224, 224, 6))
_ = model(dummy_input, training=False)
model.load_weights(os.path.join(MODEL_DIR, 'best_model.h5'))
print("   ✓ Model loaded successfully")

# Load dataset
print("\n2. Loading complete trajectory dataset...")
dataset = UnderwaterVODataset(DATA_DIR, image_size=(224, 224))
dataset.load_data()
total_frames = len(dataset.poses_df) - 1
print(f"   ✓ Dataset loaded: {total_frames} frame pairs")

# Process trajectory in batches
print(f"\n3. Generating predictions for {total_frames} frames...")
predictions = []
ground_truth = []

for start_idx in tqdm(range(0, total_frames, BATCH_SIZE), desc="Predicting"):
    end_idx = min(start_idx + BATCH_SIZE, total_frames)
    
    batch_images = []
    batch_gt = []
    
    for i in range(start_idx, end_idx):
        # Load consecutive image pairs
        img1 = dataset.load_image(i)
        img2 = dataset.load_image(i + 1)
        img_pair = np.concatenate([img1, img2], axis=-1)
        batch_images.append(img_pair)
        
        # Get ground truth relative pose
        gt = dataset.get_relative_pose(i, i + 1)
        gt_pose = np.concatenate([gt['translation'], gt['rotation']])
        batch_gt.append(gt_pose)
    
    # Generate predictions
    batch_images = np.array(batch_images)
    batch_pred = model.predict(batch_images, verbose=0)
    
    predictions.extend(batch_pred)
    ground_truth.extend(batch_gt)

predictions = np.array(predictions)
ground_truth = np.array(ground_truth)
print(f"   ✓ Generated {len(predictions)} predictions")

# Integrate trajectories from initial pose
print("\n4. Integrating full trajectories...")
initial_pose = {
    'position': np.array([0, 0, 0]),
    'quaternion': np.array([1, 0, 0, 0])
}

pred_trajectory = integrate_trajectory(initial_pose, predictions)
gt_trajectory = integrate_trajectory(initial_pose, ground_truth)

# Compute basic metrics
print("\n5. Computing trajectory metrics...")
position_errors = np.linalg.norm(pred_trajectory[:, :3] - gt_trajectory[:, :3], axis=1)
final_error = position_errors[-1]
distances = np.sqrt(np.sum(np.diff(gt_trajectory[:, :3], axis=0)**2, axis=1))
total_distance = np.sum(distances)

print(f"   ✓ Final drift: {final_error:.3f}m")
print(f"   ✓ Total distance: {total_distance:.2f}m")
print(f"   ✓ Drift percentage: {final_error/total_distance*100:.2f}%")

# Create infinity-pattern visualization
print("\n6. Creating infinity trajectory visualization...")

# Create the main infinity plot
fig = plt.figure(figsize=(20, 14))

# Main 3D trajectory plot
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
         'b-', linewidth=3, label='Ground Truth', alpha=0.9)
ax1.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
         'r--', linewidth=3, label='Predicted', alpha=0.8)

# Mark start and end points
ax1.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], gt_trajectory[0, 2], 
           color='green', s=300, marker='o', label='Start', zorder=5, edgecolors='black', linewidth=2)
ax1.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], gt_trajectory[-1, 2], 
           color='darkred', s=300, marker='s', label='End', zorder=5, edgecolors='black', linewidth=2)

ax1.set_xlabel('X (m)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
ax1.set_zlabel('Z (m)', fontsize=14, fontweight='bold')
ax1.set_title('3D Infinity Trajectory Comparison', fontsize=16, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.view_init(elev=20, azim=45)

# Top-down view (XY) - This should show the infinity pattern clearly
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', linewidth=4, label='Ground Truth', alpha=0.9)
ax2.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=4, label='Predicted', alpha=0.8)
ax2.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], color='green', s=200, marker='o', zorder=5, edgecolors='black', linewidth=2)
ax2.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], color='darkred', s=200, marker='s', zorder=5, edgecolors='black', linewidth=2)

# Add trajectory direction arrows
step = max(1, len(gt_trajectory) // 20)
for i in range(0, len(gt_trajectory)-step, step):
    dx = gt_trajectory[i+step, 0] - gt_trajectory[i, 0]
    dy = gt_trajectory[i+step, 1] - gt_trajectory[i, 1]
    ax2.arrow(gt_trajectory[i, 0], gt_trajectory[i, 1], dx*0.3, dy*0.3,
              head_width=0.05, head_length=0.03, fc='blue', ec='blue', alpha=0.6)

ax2.set_xlabel('X (m)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
ax2.set_title('Top-Down View - Infinity Pattern', fontsize=16, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# Side view (XZ)
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], 'b-', linewidth=3, label='Ground Truth')
ax3.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], 'r--', linewidth=3, label='Predicted', alpha=0.8)
ax3.scatter(gt_trajectory[0, 0], gt_trajectory[0, 2], color='green', s=150, marker='o', zorder=5, edgecolors='black')
ax3.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 2], color='darkred', s=150, marker='s', zorder=5, edgecolors='black')
ax3.set_xlabel('X (m)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Z (m)', fontsize=14, fontweight='bold')
ax3.set_title('Side View (XZ)', fontsize=16, fontweight='bold')
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

# Position error over time
ax4 = fig.add_subplot(2, 2, 4)
time_steps = np.arange(len(position_errors))
ax4.plot(time_steps, position_errors, 'purple', linewidth=3)
ax4.fill_between(time_steps, 0, position_errors, alpha=0.3, color='purple')
ax4.set_xlabel('Frame Number', fontsize=14, fontweight='bold')
ax4.set_ylabel('Position Error (m)', fontsize=14, fontweight='bold')
ax4.set_title(f'Cumulative Error - Final: {final_error:.3f}m', fontsize=16, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.suptitle(f'Underwater Visual Odometry - Infinity Trajectory Analysis\n'
             f'Model: corrected_model_20250729_220921 | Frames: {total_frames} | Distance: {total_distance:.1f}m', 
             fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('./infinity_trajectory_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a focused infinity pattern plot
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', linewidth=5, label='Ground Truth Trajectory', alpha=0.9)
ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=4, label='Predicted Trajectory', alpha=0.8)

# Mark key points
ax.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], color='green', s=400, marker='o', 
           label='Start', zorder=5, edgecolors='black', linewidth=3)
ax.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], color='darkred', s=400, marker='s', 
           label='End', zorder=5, edgecolors='black', linewidth=3)

# Add direction arrows for infinity pattern
step = max(1, len(gt_trajectory) // 15)
for i in range(0, len(gt_trajectory)-step, step):
    dx = gt_trajectory[i+step, 0] - gt_trajectory[i, 0]
    dy = gt_trajectory[i+step, 1] - gt_trajectory[i, 1]
    if np.sqrt(dx**2 + dy**2) > 0:
        ax.arrow(gt_trajectory[i, 0], gt_trajectory[i, 1], dx*0.4, dy*0.4,
                 head_width=0.08, head_length=0.06, fc='darkblue', ec='darkblue', alpha=0.7, linewidth=2)

ax.set_xlabel('X Position (meters)', fontsize=16, fontweight='bold')
ax.set_ylabel('Y Position (meters)', fontsize=16, fontweight='bold')
ax.set_title(f'Underwater Vehicle Infinity Trajectory Pattern\n'
             f'Final Drift: {final_error:.3f}m ({final_error/total_distance*100:.2f}% of {total_distance:.1f}m total distance)', 
             fontsize=18, fontweight='bold')
ax.legend(fontsize=14, loc='upper right')
ax.grid(True, alpha=0.4)
ax.axis('equal')

# Add text box with key metrics
textstr = f'''Trajectory Metrics:
• Total Frames: {total_frames}
• Total Distance: {total_distance:.2f}m
• Final Position Error: {final_error:.3f}m
• Drift Rate: {final_error/total_distance*100:.2f}%
• Pattern: Figure-8/Infinity'''

props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props, fontfamily='monospace')

plt.tight_layout()
plt.savefig('./infinity_pattern_focused.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results
results = {
    'model_name': 'corrected_model_20250729_220921',
    'total_frames': int(total_frames),
    'total_distance_m': float(total_distance),
    'final_position_error_m': float(final_error),
    'drift_percentage': float(final_error/total_distance*100),
    'trajectory_type': 'infinity_pattern'
}

with open('./infinity_trajectory_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("INFINITY TRAJECTORY EVALUATION COMPLETE!")
print("="*70)
print(f"\n📊 Generated visualizations:")
print("  • infinity_trajectory_analysis.png - Complete 4-panel analysis")
print("  • infinity_pattern_focused.png - Focused infinity pattern view")
print("  • infinity_trajectory_results.json - Numerical results")

print(f"\n🎯 TRAJECTORY SUMMARY:")
print(f"  • Pattern: {'Infinity/Figure-8' if 'infinity' in str(results).lower() else 'Complex trajectory'}")
print(f"  • Total distance: {total_distance:.2f}m over {total_frames} frames")
print(f"  • Final drift: {final_error:.3f}m ({final_error/total_distance*100:.2f}% of distance)")
print(f"  • Model accuracy: {100 - final_error/total_distance*100:.1f}%")

if final_error/total_distance < 0.05:
    print("  ✅ Excellent trajectory tracking!")
elif final_error/total_distance < 0.10:
    print("  ✅ Good trajectory tracking!")
else:
    print("  ⚠️  Moderate drift - consider improvements")