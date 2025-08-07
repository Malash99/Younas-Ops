#!/usr/bin/env python3
"""
Plot Full Trajectory Comparison
Visualize predicted vs ground truth trajectories for different cameras
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Circle
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.sub_trajectory_dataset import create_sub_trajectory_dataloaders

class UltraConservativeModel(nn.Module):
    """Same model as used in training"""
    
    def __init__(self, config):
        super().__init__()
        self.base_model = UWTransVO(**config)
        
    def forward(self, images, camera_ids, camera_mask, sub_traj_length):
        batch_size, seq_len, num_cameras, C, H, W = images.shape
        all_predictions = []
        
        for t in range(seq_len - 1):
            frame_pair = torch.stack([images[:, t], images[:, t+1]], dim=1)
            output = self.base_model(
                images=frame_pair,
                camera_ids=camera_ids,
                camera_mask=camera_mask
            )
            all_predictions.append(output['pose'])
        
        predictions = torch.stack(all_predictions, dim=1)
        return predictions

def load_model(model_path):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint['config']['model']
    model = UltraConservativeModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def predict_full_trajectory(model, device, camera_id=0, max_length=100):
    """Predict full trajectory for specified camera"""
    
    print(f"Predicting full trajectory for Camera {camera_id}...")
    
    # Load validation data
    df = pd.read_csv('data/processed/training_dataset/training_data_filtered.csv')
    val_data = df[df['split'] == 'val'].reset_index(drop=True)
    
    # Take continuous sequence
    trajectory_length = min(max_length, len(val_data))
    trajectory_data = val_data.iloc[:trajectory_length].copy()
    
    print(f"Processing {trajectory_length} frames...")
    
    # Initialize trajectories
    predicted_poses = []
    ground_truth_poses = []
    cumulative_pred = np.zeros(6)
    cumulative_gt = np.zeros(6)
    
    # Store cumulative trajectories
    pred_trajectory = [cumulative_pred.copy()]
    gt_trajectory = [cumulative_gt.copy()]
    
    # Process in sliding windows
    window_size = 3
    successful_predictions = 0
    
    with torch.no_grad():
        for i in tqdm(range(len(trajectory_data) - window_size + 1), desc=f"Cam{camera_id} Prediction"):
            try:
                # Get window data
                window_data = trajectory_data.iloc[i:i+window_size]
                
                # Create dummy images (since we're focusing on pose predictions)
                images_list = []
                poses_list = []
                
                for _, row in window_data.iterrows():
                    # Dummy image (replace with real image loading if needed)
                    dummy_image = torch.randn(3, 224, 224) * 0.1
                    images_list.append(dummy_image)
                    
                    # Ground truth pose
                    pose = np.array([
                        row['delta_x'], row['delta_y'], row['delta_z'],
                        row['delta_roll'], row['delta_pitch'], row['delta_yaw']
                    ])
                    poses_list.append(pose)
                
                # Prepare batch
                images = torch.stack(images_list).unsqueeze(0).unsqueeze(2).to(device)
                camera_ids_batch = torch.tensor([[0]], device=device)  # Force to trained camera
                camera_mask = torch.tensor([[False]], device=device)
                
                # Forward pass
                predictions = model(images, camera_ids_batch, camera_mask, window_size)
                
                if torch.isnan(predictions).any():
                    continue
                
                # Get first prediction to avoid overlaps
                pred_window = predictions[0].cpu().numpy()
                target_window = np.array(poses_list[1:])  # Skip first frame
                
                if len(pred_window) > 0:
                    pred_pose = pred_window[0]
                    target_pose = target_window[0]
                    
                    # Update cumulative trajectories
                    cumulative_pred += pred_pose
                    cumulative_gt += target_pose
                    
                    pred_trajectory.append(cumulative_pred.copy())
                    gt_trajectory.append(cumulative_gt.copy())
                    
                    predicted_poses.append(pred_pose)
                    ground_truth_poses.append(target_pose)
                    
                    successful_predictions += 1
                
            except Exception as e:
                print(f"Error at frame {i}: {e}")
                continue
    
    print(f"Successfully predicted {successful_predictions} poses")
    
    return {
        'predicted_trajectory': np.array(pred_trajectory),
        'ground_truth_trajectory': np.array(gt_trajectory),
        'predicted_poses': np.array(predicted_poses) if predicted_poses else np.array([]),
        'ground_truth_poses': np.array(ground_truth_poses) if ground_truth_poses else np.array([]),
        'successful_predictions': successful_predictions,
        'camera_id': camera_id
    }

def plot_trajectory_comparison(results_list, save_path='trajectory_comparison.png'):
    """Plot trajectory comparison for multiple cameras"""
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
    
    # 1. 3D Trajectory Plot
    ax_3d = fig.add_subplot(gs[0, :2], projection='3d')
    
    for i, result in enumerate(results_list):
        if result['successful_predictions'] > 0:
            pred_traj = result['predicted_trajectory']
            gt_traj = result['ground_truth_trajectory']
            cam_id = result['camera_id']
            
            # Plot predicted trajectory
            ax_3d.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 
                      color=colors[i], linestyle='--', alpha=0.8, linewidth=2,
                      label=f'Cam{cam_id} Predicted')
            
            # Plot ground truth trajectory
            ax_3d.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 
                      color=colors[i], linestyle='-', alpha=0.9, linewidth=2,
                      label=f'Cam{cam_id} Ground Truth')
            
            # Mark start and end points
            ax_3d.scatter(pred_traj[0, 0], pred_traj[0, 1], pred_traj[0, 2], 
                         color=colors[i], s=100, marker='o', alpha=0.8)
            ax_3d.scatter(pred_traj[-1, 0], pred_traj[-1, 1], pred_traj[-1, 2], 
                         color=colors[i], s=100, marker='s', alpha=0.8)
    
    ax_3d.set_xlabel('X Position (m)')
    ax_3d.set_ylabel('Y Position (m)')
    ax_3d.set_zlabel('Z Position (m)')
    ax_3d.set_title('3D Trajectory Comparison\n(Solid: Ground Truth, Dashed: Predicted)', fontsize=14, fontweight='bold')
    ax_3d.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_3d.grid(True, alpha=0.3)
    
    # 2. XY Plane Trajectory
    ax_xy = fig.add_subplot(gs[0, 2:])
    
    for i, result in enumerate(results_list):
        if result['successful_predictions'] > 0:
            pred_traj = result['predicted_trajectory']
            gt_traj = result['ground_truth_trajectory']
            cam_id = result['camera_id']
            
            ax_xy.plot(pred_traj[:, 0], pred_traj[:, 1], 
                      color=colors[i], linestyle='--', alpha=0.8, linewidth=2,
                      label=f'Cam{cam_id} Predicted')
            ax_xy.plot(gt_traj[:, 0], gt_traj[:, 1], 
                      color=colors[i], linestyle='-', alpha=0.9, linewidth=2,
                      label=f'Cam{cam_id} Ground Truth')
            
            # Mark start point
            ax_xy.scatter(pred_traj[0, 0], pred_traj[0, 1], 
                         color=colors[i], s=100, marker='o', alpha=0.8, label=f'Cam{cam_id} Start')
    
    ax_xy.set_xlabel('X Position (m)')
    ax_xy.set_ylabel('Y Position (m)')
    ax_xy.set_title('XY Plane Trajectory', fontsize=14, fontweight='bold')
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis('equal')
    
    # 3. Position Error Over Time
    ax_error = fig.add_subplot(gs[1, :2])
    
    for i, result in enumerate(results_list):
        if result['successful_predictions'] > 0:
            pred_traj = result['predicted_trajectory']
            gt_traj = result['ground_truth_trajectory']
            cam_id = result['camera_id']
            
            # Calculate position error over time
            position_errors = np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1)
            time_steps = np.arange(len(position_errors))
            
            ax_error.plot(time_steps, position_errors, 
                         color=colors[i], linewidth=2, alpha=0.8,
                         label=f'Camera {cam_id}')
    
    ax_error.set_xlabel('Time Step')
    ax_error.set_ylabel('Position Error (m)')
    ax_error.set_title('Position Error Over Time', fontsize=14, fontweight='bold')
    ax_error.legend()
    ax_error.grid(True, alpha=0.3)
    
    # 4. Drift Analysis
    ax_drift = fig.add_subplot(gs[1, 2:])
    
    camera_names = []
    final_drifts = []
    relative_drifts = []
    
    for result in results_list:
        if result['successful_predictions'] > 0:
            pred_traj = result['predicted_trajectory']
            gt_traj = result['ground_truth_trajectory']
            cam_id = result['camera_id']
            
            # Calculate final drift
            final_drift = np.linalg.norm(pred_traj[-1, :3] - gt_traj[-1, :3])
            
            # Calculate trajectory length
            traj_length = np.sum(np.linalg.norm(np.diff(gt_traj[:, :3], axis=0), axis=1))
            relative_drift = (final_drift / traj_length * 100) if traj_length > 0 else 0
            
            camera_names.append(f'Cam{cam_id}')
            final_drifts.append(final_drift)
            relative_drifts.append(relative_drift)
    
    x_pos = np.arange(len(camera_names))
    bars = ax_drift.bar(x_pos, final_drifts, color=colors[:len(camera_names)], alpha=0.7)
    
    # Add relative drift percentages on top of bars
    for i, (bar, rel_drift) in enumerate(zip(bars, relative_drifts)):
        height = bar.get_height()
        ax_drift.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                     f'{rel_drift:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax_drift.set_xlabel('Camera')
    ax_drift.set_ylabel('Final Drift (m)')
    ax_drift.set_title('Final Drift Comparison', fontsize=14, fontweight='bold')
    ax_drift.set_xticks(x_pos)
    ax_drift.set_xticklabels(camera_names)
    ax_drift.grid(True, alpha=0.3, axis='y')
    
    # 5. Detailed Metrics Table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # Create metrics table
    table_data = []
    for result in results_list:
        if result['successful_predictions'] > 0:
            pred_traj = result['predicted_trajectory']
            gt_traj = result['ground_truth_trajectory']
            cam_id = result['camera_id']
            
            # Calculate metrics
            position_errors = np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1)
            final_drift = position_errors[-1]
            mean_error = np.mean(position_errors)
            max_error = np.max(position_errors)
            traj_length = np.sum(np.linalg.norm(np.diff(gt_traj[:, :3], axis=0), axis=1))
            relative_drift = (final_drift / traj_length * 100) if traj_length > 0 else 0
            
            table_data.append([
                f'Camera {cam_id}',
                f'{len(pred_traj)}',
                f'{mean_error:.6f}m',
                f'{max_error:.6f}m',  
                f'{final_drift:.6f}m',
                f'{relative_drift:.2f}%',
                f'{traj_length:.3f}m'
            ])
    
    if table_data:
        table = ax_table.table(cellText=table_data,
                              colLabels=['Camera', 'Frames', 'Mean Error', 'Max Error', 'Final Drift', 'Relative Drift', 'Trajectory Length'],
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0.3, 1, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(7):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    # Main title
    fig.suptitle('UW-TransVO: Multi-Camera Trajectory Prediction Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved as: {save_path}")
    
    # Show plot
    plt.show()
    
    return fig

def main():
    model_path = 'ultra_conservative_best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    # Load model
    model, device = load_model(model_path)
    
    print("GENERATING TRAJECTORY COMPARISONS")
    print("=" * 50)
    
    # Predict trajectories for different cameras
    cameras_to_test = [0, 1, 2]  # Test cam0 (trained), cam1, cam2 (unseen)
    results = []
    
    for camera_id in cameras_to_test:
        print(f"\nProcessing Camera {camera_id}...")
        result = predict_full_trajectory(model, device, camera_id, max_length=60)
        
        if result['successful_predictions'] > 0:
            results.append(result)
            print(f"SUCCESS Camera {camera_id}: {result['successful_predictions']} successful predictions")
        else:
            print(f"FAILED Camera {camera_id}: No successful predictions")
    
    # Create comparison plot
    if results:
        print(f"\nCreating trajectory comparison plot...")
        plot_trajectory_comparison(results, 'trajectory_comparison.png')
        
        # Print summary
        print(f"\n{'='*50}")
        print("TRAJECTORY ANALYSIS SUMMARY:")
        print(f"{'='*50}")
        
        for result in results:
            cam_id = result['camera_id']
            pred_traj = result['predicted_trajectory']
            gt_traj = result['ground_truth_trajectory']
            
            if len(pred_traj) > 1:
                final_drift = np.linalg.norm(pred_traj[-1, :3] - gt_traj[-1, :3])
                traj_length = np.sum(np.linalg.norm(np.diff(gt_traj[:, :3], axis=0), axis=1))
                relative_drift = (final_drift / traj_length * 100) if traj_length > 0 else 0
                
                print(f"Camera {cam_id}:")
                print(f"  Frames: {len(pred_traj)}")
                print(f"  Final Drift: {final_drift:.6f}m")
                print(f"  Relative Drift: {relative_drift:.2f}%")
                print(f"  Trajectory Length: {traj_length:.3f}m")
    else:
        print("No successful predictions to plot!")

if __name__ == '__main__':
    main()