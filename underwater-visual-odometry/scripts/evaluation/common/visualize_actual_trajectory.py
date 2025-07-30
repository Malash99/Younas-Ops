#!/usr/bin/env python3
"""
Visualize the actual trajectory from ground truth data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the full TUM trajectory
print("Loading full TUM trajectory...")
tum_df = pd.read_csv('/app/data/raw/tum_trajectory.csv')
print(f"Full TUM trajectory: {len(tum_df)} poses")

# Load the extracted ground truth (aligned with images)
print("\nLoading extracted ground truth...")
gt_df = pd.read_csv('/app/data/raw/ground_truth.csv')
print(f"Extracted ground truth: {len(gt_df)} poses")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 15))

# 1. Full TUM trajectory - 3D
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(tum_df['x'], tum_df['y'], tum_df['z'], 'b-', linewidth=1, alpha=0.6)
ax1.scatter(tum_df['x'].iloc[0], tum_df['y'].iloc[0], tum_df['z'].iloc[0], 
           color='green', s=200, marker='o', label='Start')
ax1.scatter(tum_df['x'].iloc[-1], tum_df['y'].iloc[-1], tum_df['z'].iloc[-1], 
           color='red', s=200, marker='s', label='End')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title(f'Full TUM Trajectory ({len(tum_df)} poses)')
ax1.legend()

# 2. Extracted trajectory - 3D
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.plot(gt_df['x'], gt_df['y'], gt_df['z'], 'g-', linewidth=2)
ax2.scatter(gt_df['x'].iloc[0], gt_df['y'].iloc[0], gt_df['z'].iloc[0], 
           color='green', s=200, marker='o', label='Start')
ax2.scatter(gt_df['x'].iloc[-1], gt_df['y'].iloc[-1], gt_df['z'].iloc[-1], 
           color='red', s=200, marker='s', label='End')

# Highlight every 100th point
for i in range(0, len(gt_df), 100):
    ax2.scatter(gt_df['x'].iloc[i], gt_df['y'].iloc[i], gt_df['z'].iloc[i], 
               color='orange', s=50, alpha=0.5)

ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('Z (m)')
ax2.set_title(f'Extracted Trajectory ({len(gt_df)} poses)')
ax2.legend()

# 3. Overlay comparison
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.plot(tum_df['x'], tum_df['y'], tum_df['z'], 'b-', linewidth=1, alpha=0.3, label='Full TUM')
ax3.plot(gt_df['x'], gt_df['y'], gt_df['z'], 'r-', linewidth=3, label='Extracted (with images)')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.set_zlabel('Z (m)')
ax3.set_title('Trajectory Comparison')
ax3.legend()

# 4-6. XY, XZ, YZ views of extracted trajectory
# XY view
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(gt_df['x'], gt_df['y'], 'g-', linewidth=2)
ax4.scatter(gt_df['x'].iloc[0], gt_df['y'].iloc[0], color='green', s=100, marker='o')
ax4.scatter(gt_df['x'].iloc[-1], gt_df['y'].iloc[-1], color='red', s=100, marker='s')
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.set_title('Extracted Trajectory - XY View')
ax4.axis('equal')
ax4.grid(True, alpha=0.3)

# XZ view
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(gt_df['x'], gt_df['z'], 'g-', linewidth=2)
ax5.scatter(gt_df['x'].iloc[0], gt_df['z'].iloc[0], color='green', s=100, marker='o')
ax5.scatter(gt_df['x'].iloc[-1], gt_df['z'].iloc[-1], color='red', s=100, marker='s')
ax5.set_xlabel('X (m)')
ax5.set_ylabel('Z (m)')
ax5.set_title('Extracted Trajectory - XZ View')
ax5.axis('equal')
ax5.grid(True, alpha=0.3)

# YZ view
ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(gt_df['y'], gt_df['z'], 'g-', linewidth=2)
ax6.scatter(gt_df['y'].iloc[0], gt_df['z'].iloc[0], color='green', s=100, marker='o')
ax6.scatter(gt_df['y'].iloc[-1], gt_df['z'].iloc[-1], color='red', s=100, marker='s')
ax6.set_xlabel('Y (m)')
ax6.set_ylabel('Z (m)')
ax6.set_title('Extracted Trajectory - YZ View')
ax6.axis('equal')
ax6.grid(True, alpha=0.3)

plt.suptitle('Ground Truth Trajectory Analysis', fontsize=16)
plt.tight_layout()
plt.savefig('/app/output/models/model_20250708_023343/actual_trajectory.png', dpi=200)
print("\nSaved to actual_trajectory.png")

# Analyze the extracted trajectory
print("\n" + "="*50)
print("EXTRACTED TRAJECTORY ANALYSIS")
print("="*50)

# Calculate total distance
positions = gt_df[['x', 'y', 'z']].values
distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
total_distance = np.sum(distances)

print(f"\nTrajectory Statistics:")
print(f"  Total poses: {len(gt_df)}")
print(f"  X range: [{gt_df['x'].min():.3f}, {gt_df['x'].max():.3f}] = {gt_df['x'].max() - gt_df['x'].min():.3f} m")
print(f"  Y range: [{gt_df['y'].min():.3f}, {gt_df['y'].max():.3f}] = {gt_df['y'].max() - gt_df['y'].min():.3f} m") 
print(f"  Z range: [{gt_df['z'].min():.3f}, {gt_df['z'].max():.3f}] = {gt_df['z'].max() - gt_df['z'].min():.3f} m")
print(f"  Total distance: {total_distance:.2f} m")
print(f"  Start position: ({gt_df['x'].iloc[0]:.3f}, {gt_df['y'].iloc[0]:.3f}, {gt_df['z'].iloc[0]:.3f})")
print(f"  End position: ({gt_df['x'].iloc[-1]:.3f}, {gt_df['y'].iloc[-1]:.3f}, {gt_df['z'].iloc[-1]:.3f})")
print(f"  Displacement: {np.linalg.norm(positions[-1] - positions[0]):.3f} m")

# Check sampling rate
print(f"\nSampling Analysis:")
print(f"  Full TUM: {len(tum_df)} poses")
print(f"  Extracted: {len(gt_df)} poses")
print(f"  Sampling ratio: {len(gt_df)/len(tum_df)*100:.1f}%")
print(f"  Approximate sampling: Every {len(tum_df)//len(gt_df)} frames")

# Find where in the full trajectory our extracted part is
print("\nFinding extracted portion in full trajectory...")
# Match first few positions
first_extracted = gt_df[['x', 'y', 'z']].iloc[0].values
min_dist = float('inf')
match_idx = 0

for i in range(len(tum_df)):
    pos = tum_df[['x', 'y', 'z']].iloc[i].values
    dist = np.linalg.norm(pos - first_extracted)
    if dist < min_dist:
        min_dist = dist
        match_idx = i

print(f"  Extracted trajectory starts at index {match_idx} in full TUM")
print(f"  This is {match_idx/len(tum_df)*100:.1f}% into the full trajectory")

# Create a focused plot on just the extracted portion
fig2, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot the actual extracted trajectory path
ax = axes[0]
ax.plot(gt_df['x'], gt_df['y'], 'b-', linewidth=2)
ax.scatter(gt_df['x'].iloc[0], gt_df['y'].iloc[0], color='green', s=200, marker='o', label='Start', zorder=5)
ax.scatter(gt_df['x'].iloc[-1], gt_df['y'].iloc[-1], color='red', s=200, marker='s', label='End', zorder=5)

# Add direction arrows every 100 frames
for i in range(0, len(gt_df)-1, 100):
    dx = gt_df['x'].iloc[i+1] - gt_df['x'].iloc[i]
    dy = gt_df['y'].iloc[i+1] - gt_df['y'].iloc[i]
    ax.arrow(gt_df['x'].iloc[i], gt_df['y'].iloc[i], dx*20, dy*20, 
            head_width=0.05, head_length=0.02, fc='red', ec='red', alpha=0.5)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Your Actual Ground Truth Trajectory (XY View)')
ax.legend()
ax.axis('equal')
ax.grid(True, alpha=0.3)

# Show the path characteristics
ax2 = axes[1]
ax2.plot(distances * 1000, 'g-', linewidth=2)
ax2.set_xlabel('Frame')
ax2.set_ylabel('Inter-frame Distance (mm)')
ax2.set_title('Motion Between Frames')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=np.mean(distances)*1000, color='r', linestyle='--', 
           label=f'Mean: {np.mean(distances)*1000:.1f} mm')
ax2.legend()

plt.suptitle('Your Ground Truth Trajectory for Model Evaluation', fontsize=16)
plt.tight_layout()
plt.savefig('/app/output/models/model_20250708_023343/your_actual_trajectory.png', dpi=200)
print("\nSaved focused view to your_actual_trajectory.png")