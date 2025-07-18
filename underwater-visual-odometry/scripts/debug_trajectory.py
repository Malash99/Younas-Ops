#!/usr/bin/env python3
"""
Debug trajectory reconstruction to understand why it doesn't match ground truth.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import sys
sys.path.append('/app/scripts')
from data_loader import UnderwaterVODataset

# Load dataset
print("Loading dataset...")
dataset = UnderwaterVODataset('/app/data/raw')
dataset.load_data()

# Load the original TUM trajectory
print("\nLoading original TUM trajectory...")
tum_df = pd.read_csv('/app/data/raw/tum_trajectory.csv')
print(f"TUM trajectory shape: {tum_df.shape}")

# Check ground truth from dataset
print("\nChecking ground truth from dataset...")
gt_df = dataset.poses_df
print(f"Dataset ground truth shape: {gt_df.shape}")

# Create visualization comparing different trajectories
fig = plt.figure(figsize=(20, 15))

# 1. Original TUM trajectory (absolute poses)
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(tum_df['x'], tum_df['y'], tum_df['z'], 'b-', linewidth=2)
ax1.scatter(tum_df['x'].iloc[0], tum_df['y'].iloc[0], tum_df['z'].iloc[0], 
           color='green', s=100, marker='o')
ax1.scatter(tum_df['x'].iloc[-1], tum_df['y'].iloc[-1], tum_df['z'].iloc[-1], 
           color='red', s=100, marker='s')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('Original TUM Trajectory (Absolute)')

# 2. Dataset ground truth (should match TUM)
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.plot(gt_df['x'], gt_df['y'], gt_df['z'], 'g-', linewidth=2)
ax2.scatter(gt_df['x'].iloc[0], gt_df['y'].iloc[0], gt_df['z'].iloc[0], 
           color='green', s=100, marker='o')
ax2.scatter(gt_df['x'].iloc[-1], gt_df['y'].iloc[-1], gt_df['z'].iloc[-1], 
           color='red', s=100, marker='s')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('Z (m)')
ax2.set_title('Dataset Ground Truth')

# 3. Check relative poses
print("\nAnalyzing relative poses...")
relative_translations = []
relative_rotations = []

for i in range(len(gt_df) - 1):
    rel_pose = dataset.get_relative_pose(i, i + 1)
    relative_translations.append(rel_pose['translation'])
    relative_rotations.append(rel_pose['rotation'])

relative_translations = np.array(relative_translations)
relative_rotations = np.array(relative_rotations)

# 4. Reconstruct trajectory from relative poses
print("\nReconstructing trajectory from relative poses...")
reconstructed_positions = [np.array([0, 0, 0])]
reconstructed_quaternions = [np.array([1, 0, 0, 0])]

current_pos = np.array([0, 0, 0])
current_q = Quaternion([1, 0, 0, 0])

for i in range(len(relative_translations)):
    # Get relative transformation
    dt = relative_translations[i]
    dr = relative_rotations[i]
    
    # Convert Euler to quaternion
    q_rel = Quaternion(axis=[0, 0, 1], angle=dr[2]) * \
            Quaternion(axis=[0, 1, 0], angle=dr[1]) * \
            Quaternion(axis=[1, 0, 0], angle=dr[0])
    
    # Update pose
    current_pos = current_pos + current_q.rotate(dt)
    current_q = current_q * q_rel
    
    reconstructed_positions.append(current_pos.copy())
    reconstructed_quaternions.append(current_q.elements)

reconstructed_positions = np.array(reconstructed_positions)

# 5. Plot reconstructed trajectory
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.plot(reconstructed_positions[:, 0], reconstructed_positions[:, 1], 
         reconstructed_positions[:, 2], 'r-', linewidth=2)
ax3.scatter(reconstructed_positions[0, 0], reconstructed_positions[0, 1], 
           reconstructed_positions[0, 2], color='green', s=100, marker='o')
ax3.scatter(reconstructed_positions[-1, 0], reconstructed_positions[-1, 1], 
           reconstructed_positions[-1, 2], color='red', s=100, marker='s')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.set_zlabel('Z (m)')
ax3.set_title('Reconstructed from Relative Poses')

# 6. XY views comparison
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(tum_df['x'], tum_df['y'], 'b-', linewidth=2, label='TUM Original')
ax4.plot(reconstructed_positions[:, 0], reconstructed_positions[:, 1], 
         'r--', linewidth=2, label='Reconstructed')
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.set_title('XY View Comparison')
ax4.legend()
ax4.axis('equal')
ax4.grid(True, alpha=0.3)

# 7. Analyze differences
print("\nAnalyzing differences...")
# Compare first 100 absolute positions
n = min(100, len(gt_df), len(reconstructed_positions))
actual_positions = gt_df[['x', 'y', 'z']].values[:n]
recon_positions = reconstructed_positions[:n]

position_diffs = np.linalg.norm(actual_positions - recon_positions, axis=1)

ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(position_diffs, 'k-', linewidth=2)
ax5.set_xlabel('Frame')
ax5.set_ylabel('Position Difference (m)')
ax5.set_title('Reconstruction Error')
ax5.grid(True, alpha=0.3)

# 8. Statistics
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
stats_text = f"""
Trajectory Statistics:
====================
TUM Trajectory:
  Points: {len(tum_df)}
  X range: [{tum_df['x'].min():.2f}, {tum_df['x'].max():.2f}] m
  Y range: [{tum_df['y'].min():.2f}, {tum_df['y'].max():.2f}] m
  Z range: [{tum_df['z'].min():.2f}, {tum_df['z'].max():.2f}] m

Relative Poses:
  Count: {len(relative_translations)}
  Mean translation: {np.mean(np.linalg.norm(relative_translations, axis=1)):.4f} m
  Mean rotation: {np.mean(np.abs(relative_rotations), axis=0)} rad

Reconstruction Error:
  Mean: {np.mean(position_diffs):.4f} m
  Max: {np.max(position_diffs):.4f} m
  Final: {position_diffs[-1]:.4f} m
"""
ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
         verticalalignment='center')

plt.suptitle('Trajectory Debug Analysis', fontsize=16)
plt.tight_layout()
plt.savefig('/app/output/models/model_20250708_023343/trajectory_debug.png', dpi=150)
plt.close()

# Check if it's a coordinate system issue
print("\nChecking coordinate system...")
print(f"First 5 TUM positions:\n{tum_df[['x', 'y', 'z']].head()}")
print(f"\nFirst 5 relative translations:\n{relative_translations[:5]}")
print(f"\nFirst 5 relative rotations (rad):\n{relative_rotations[:5]}")

# Check frame transformation
print("\nChecking if we need frame transformation...")
# Calculate actual relative poses from absolute ground truth
actual_relative_trans = []
for i in range(len(gt_df) - 1):
    p1 = gt_df[['x', 'y', 'z']].iloc[i].values
    p2 = gt_df[['x', 'y', 'z']].iloc[i + 1].values
    actual_relative_trans.append(p2 - p1)
actual_relative_trans = np.array(actual_relative_trans)

print(f"\nActual relative translations (first 5):\n{actual_relative_trans[:5]}")
print(f"Dataset relative translations (first 5):\n{relative_translations[:5]}")

# Save analysis
analysis = {
    'tum_range': {
        'x': [float(tum_df['x'].min()), float(tum_df['x'].max())],
        'y': [float(tum_df['y'].min()), float(tum_df['y'].max())],
        'z': [float(tum_df['z'].min()), float(tum_df['z'].max())]
    },
    'reconstruction_error': {
        'mean': float(np.mean(position_diffs)),
        'max': float(np.max(position_diffs)),
        'final': float(position_diffs[-1])
    }
}

import json
with open('/app/output/models/model_20250708_023343/trajectory_debug.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print("\nDebug analysis saved to trajectory_debug.png")