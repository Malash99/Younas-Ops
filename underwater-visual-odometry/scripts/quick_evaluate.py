#!/usr/bin/env python3
"""
Quick evaluation without history plotting.
"""

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import sys
sys.path.append('/app/scripts')
from data_loader import UnderwaterVODataset
from models.baseline_cnn import create_model
from coordinate_transforms import integrate_trajectory

# Evaluation parameters
MODEL_DIR = '/app/output/models/model_20250708_023343'
DATA_DIR = '/app/data/raw'
SEQUENCE_LENGTH = 200
START_IDX = 100

print("Quick Underwater VO Evaluation")
print("="*50)

# Load model
print("Loading model...")
model = create_model('baseline')
dummy_input = tf.zeros((1, 224, 224, 6))
_ = model(dummy_input, training=False)
model.load_weights(os.path.join(MODEL_DIR, 'best_model.h5'))

# Load dataset
print("Loading dataset...")
dataset = UnderwaterVODataset(DATA_DIR, image_size=(224, 224))
dataset.load_data()

# Make predictions
print(f"\nEvaluating on {SEQUENCE_LENGTH} frames...")
predictions = []
ground_truth = []

for i in tqdm(range(START_IDX, START_IDX + SEQUENCE_LENGTH - 1)):
    # Load and predict
    img1 = dataset.load_image(i)
    img2 = dataset.load_image(i + 1)
    img_pair = np.concatenate([img1, img2], axis=-1)
    img_batch = np.expand_dims(img_pair, axis=0)
    
    pred = model.predict(img_batch, verbose=0)[0]
    predictions.append(pred)
    
    # Ground truth
    gt = dataset.get_relative_pose(i, i + 1)
    gt_pose = np.concatenate([gt['translation'], gt['rotation']])
    ground_truth.append(gt_pose)

predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

# Integrate trajectories
print("\nIntegrating trajectories...")
initial_pose = {'position': np.array([0, 0, 0]), 'quaternion': np.array([1, 0, 0, 0])}
pred_traj = integrate_trajectory(initial_pose, predictions)
gt_traj = integrate_trajectory(initial_pose, ground_truth)

# Create output directory
output_dir = os.path.join(MODEL_DIR, 'evaluation')
os.makedirs(output_dir, exist_ok=True)

# Plot results
print("Creating visualizations...")

# 1. 3D Trajectory
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 'g-', linewidth=2, label='Ground Truth')
ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 'r--', linewidth=2, label='Predicted')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory Comparison')
ax.legend()
plt.savefig(os.path.join(output_dir, 'trajectory_3d.png'), dpi=150)
plt.close()

# 2. Top-down view
plt.figure(figsize=(10, 10))
plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', linewidth=2, label='Ground Truth')
plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, label='Predicted')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Top-Down View (XY)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig(os.path.join(output_dir, 'trajectory_xy.png'), dpi=150)
plt.close()

# 3. Error plot
trans_errors = np.linalg.norm(predictions[:, :3] - ground_truth[:, :3], axis=1)
rot_errors = np.degrees(np.linalg.norm(predictions[:, 3:] - ground_truth[:, 3:], axis=1))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
frames = np.arange(len(trans_errors))

ax1.plot(frames, trans_errors, 'b-', linewidth=2)
ax1.set_ylabel('Translation Error (m)')
ax1.set_title(f'Mean: {np.mean(trans_errors):.4f} ± {np.std(trans_errors):.4f} m')
ax1.grid(True, alpha=0.3)

ax2.plot(frames, rot_errors, 'r-', linewidth=2)
ax2.set_xlabel('Frame')
ax2.set_ylabel('Rotation Error (deg)')
ax2.set_title(f'Mean: {np.mean(rot_errors):.2f} ± {np.std(rot_errors):.2f} deg')
ax2.grid(True, alpha=0.3)

plt.suptitle('Prediction Errors Over Time')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'errors.png'), dpi=150)
plt.close()

# 4. Summary statistics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Translation histogram
ax = axes[0, 0]
ax.hist(trans_errors, bins=30, edgecolor='black', alpha=0.7)
ax.set_xlabel('Translation Error (m)')
ax.set_ylabel('Frequency')
ax.set_title(f'Translation Error Distribution\nMean: {np.mean(trans_errors):.4f} m')
ax.grid(True, alpha=0.3)

# Rotation histogram
ax = axes[0, 1]
ax.hist(rot_errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
ax.set_xlabel('Rotation Error (deg)')
ax.set_ylabel('Frequency')
ax.set_title(f'Rotation Error Distribution\nMean: {np.mean(rot_errors):.2f} deg')
ax.grid(True, alpha=0.3)

# Trajectory error over distance
ax = axes[1, 0]
distances = np.sqrt(np.sum(np.diff(gt_traj[:, :3], axis=0)**2, axis=1))
cumulative_distance = np.cumsum(distances)
position_errors = np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1)
ax.plot(cumulative_distance, position_errors[1:], 'g-', linewidth=2)
ax.set_xlabel('Distance Traveled (m)')
ax.set_ylabel('Position Error (m)')
ax.set_title('Error vs Distance')
ax.grid(True, alpha=0.3)

# Final statistics text
ax = axes[1, 1]
ax.axis('off')
stats_text = f"""PERFORMANCE SUMMARY
{'='*30}

Sequence Length: {SEQUENCE_LENGTH} frames
Total Distance: {cumulative_distance[-1]:.2f} m

Translation Accuracy:
  Mean Error: {np.mean(trans_errors):.4f} m
  Std Dev: {np.std(trans_errors):.4f} m
  Max Error: {np.max(trans_errors):.4f} m

Rotation Accuracy:
  Mean Error: {np.mean(rot_errors):.2f}°
  Std Dev: {np.std(rot_errors):.2f}°
  Max Error: {np.max(rot_errors):.2f}°

Final Position Error: {position_errors[-1]:.3f} m
Error per meter: {position_errors[-1]/cumulative_distance[-1]:.3f}
"""
ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace', 
        verticalalignment='center', transform=ax.transAxes)

plt.suptitle('Underwater Visual Odometry - Model Performance', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=150)
plt.close()

print("\n" + "="*50)
print("EVALUATION COMPLETE!")
print("="*50)
print(f"\nResults saved to: {output_dir}")
print("\nGenerated files:")
for file in ['trajectory_3d.png', 'trajectory_xy.png', 'errors.png', 'performance_summary.png']:
    if os.path.exists(os.path.join(output_dir, file)):
        print(f"  ✓ {file}")
        
print(f"\nYour model achieved:")
print(f"  • Translation accuracy: {np.mean(trans_errors)*1000:.1f} mm per frame")
print(f"  • Rotation accuracy: {np.mean(rot_errors):.2f}° per frame")
print(f"  • Final drift: {position_errors[-1]/cumulative_distance[-1]*100:.1f}% of distance traveled")