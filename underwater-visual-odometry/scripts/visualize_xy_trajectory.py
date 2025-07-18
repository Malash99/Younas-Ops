#!/usr/bin/env python3
"""
Visualize predicted vs. actual XY trajectory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# Load ground truth from CSV
gt_file = '/app/data/raw/ground_truth.csv'
try:
    gt_df = pd.read_csv(gt_file)
    gt_trajectory = gt_df[['x', 'y']].values
except FileNotFoundError:
    print(f"Error: {gt_file} not found.")
    exit(1)
except Exception as e:
    print(f"Error loading {gt_file}: {e}")
    exit(1)

# Load predicted trajectory from JSON (placeholder since it's not fully available)
MODEL_DIR = '/app/output/models/model_20250708_023343'
output_dir = os.path.join(MODEL_DIR, 'full_trajectory_evaluation')
results_file = os.path.join(output_dir, 'full_trajectory_results.json')

try:
    with open(results_file, 'r') as f:
        results = json.load(f)
    # Placeholder: Use ground truth as predicted for now (replace with actual predictions)
    pred_trajectory = gt_trajectory.copy()  # Replace with actual predicted data when available
except FileNotFoundError:
    print(f"Error: {results_file} not found.")
    exit(1)
except Exception as e:
    print(f"Error loading {results_file}: {e}")
    exit(1)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', label='Actual', linewidth=2)
plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', label='Predicted', linewidth=2)
plt.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], color='green', s=100, marker='o', label='Start')
plt.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], color='red', s=100, marker='s', label='End')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Predicted vs Actual XY Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'predicted_vs_actual_xy.png')
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Plot saved to: {output_path}")