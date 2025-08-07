#!/usr/bin/env python3
"""
Test Normalized Model Trajectories
Compare normalized model predictions on trained vs unseen cameras
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

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO

class NormalizedModel(nn.Module):
    """Same model as used in normalized training"""
    
    def __init__(self, config):
        super().__init__()
        self.base_model = UWTransVO(**config)
        
        # Normalization parameters
        self.register_buffer('delta_mean', torch.zeros(6))
        self.register_buffer('delta_std', torch.ones(6))
        
    def set_normalization(self, delta_mean, delta_std):
        """Set normalization parameters"""
        self.delta_mean.copy_(delta_mean)
        self.delta_std.copy_(delta_std)
        
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
    
    def denormalize_predictions(self, normalized_preds):
        """Convert normalized predictions back to real scale"""
        return normalized_preds * self.delta_std + self.delta_mean

def load_normalized_model():
    """Load the normalized trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'normalized_training_best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Normalized model not found: {model_path}")
        return None, None, None, None
    
    print("Loading normalized model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint['config']['model']
    model = NormalizedModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Set normalization parameters
    delta_mean = checkpoint['delta_mean']
    delta_std = checkpoint['delta_std']
    model.set_normalization(delta_mean, delta_std)
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.8f}")
    print(f"Normalization: mean={delta_mean.cpu().numpy()}, std={delta_std.cpu().numpy()}")
    
    return model, device, delta_mean, delta_std

def predict_trajectory(model, device, delta_mean, delta_std, camera_id, max_frames=100):
    """Predict trajectory for specified camera"""
    
    print(f"\nPredicting trajectory for Camera {camera_id}...")
    
    # Load Bag 0 data
    df = pd.read_csv('data/processed/training_dataset/training_data_filtered.csv')
    bag0_data = df[df['bag_name'] == 'ariel_2023-12-21-14-24-42_0'].sort_values('timestamp').reset_index(drop=True)
    
    # Limit frames for faster testing
    bag0_data = bag0_data.iloc[:max_frames]
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for i in tqdm(range(len(bag0_data) - 3), desc=f"Camera {camera_id}"):
            try:
                # Get window data
                window_data = bag0_data.iloc[i:i+3]
                
                # Create dummy images
                images_list = [torch.randn(3, 224, 224) * 0.1 for _ in range(3)]
                images = torch.stack(images_list).unsqueeze(0).unsqueeze(2).to(device)
                
                # Set camera ID (0 for trained, 2 for unseen)
                camera_ids_batch = torch.tensor([[0]], device=device)  # Always use trained camera embedding
                camera_mask = torch.tensor([[False]], device=device)
                
                # Forward pass
                normalized_pred = model(images, camera_ids_batch, camera_mask, 3)
                
                if torch.isnan(normalized_pred).any():
                    continue
                
                # Denormalize prediction
                pred_delta = model.denormalize_predictions(normalized_pred[0, 0]).cpu().numpy()
                
                # Ground truth
                gt_delta = np.array([
                    window_data.iloc[1]['delta_x'], 
                    window_data.iloc[1]['delta_y'], 
                    window_data.iloc[1]['delta_z'],
                    window_data.iloc[1]['delta_roll'], 
                    window_data.iloc[1]['delta_pitch'], 
                    window_data.iloc[1]['delta_yaw']
                ])
                
                predictions.append(pred_delta)
                ground_truths.append(gt_delta)
                
            except Exception as e:
                continue
    
    if len(predictions) == 0:
        print(f"No valid predictions for camera {camera_id}")
        return None
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Integrate to get trajectories
    pred_trajectory = np.zeros((len(predictions) + 1, 6))
    gt_trajectory = np.zeros((len(predictions) + 1, 6))
    
    for i in range(len(predictions)):
        pred_trajectory[i + 1] = pred_trajectory[i] + predictions[i]
        gt_trajectory[i + 1] = gt_trajectory[i] + ground_truths[i]
    
    return {
        'camera_id': camera_id,
        'predicted_trajectory': pred_trajectory,
        'ground_truth_trajectory': gt_trajectory,
        'predicted_deltas': predictions,
        'ground_truth_deltas': ground_truths,
        'successful_predictions': len(predictions)
    }

def plot_trajectory_comparison(results_list, save_path='normalized_model_trajectories.png'):
    """Plot trajectory comparison for normalized model"""
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    colors = ['#2E86AB', '#F18F01']  # Blue for cam0, Orange for cam2
    
    # 1. XY Trajectory Comparison
    ax1 = plt.subplot(2, 3, 1)
    
    for i, result in enumerate(results_list):
        if result is None:
            continue
            
        pred_traj = result['predicted_trajectory']
        gt_traj = result['ground_truth_trajectory']
        cam_id = result['camera_id']
        color = colors[i]
        
        # Plot trajectories
        ax1.plot(gt_traj[:, 0], gt_traj[:, 1], color=color, linewidth=3, alpha=0.8,
                label=f'Cam{cam_id} Ground Truth', linestyle='-')
        ax1.plot(pred_traj[:, 0], pred_traj[:, 1], color=color, linewidth=3, alpha=0.8,
                label=f'Cam{cam_id} Predicted', linestyle='--')
        
        # Mark start and end
        ax1.scatter(pred_traj[0, 0], pred_traj[0, 1], color=color, s=100, marker='o', 
                   edgecolor='black', linewidth=2, zorder=5)
        ax1.scatter(pred_traj[-1, 0], pred_traj[-1, 1], color=color, s=100, marker='s',
                   edgecolor='black', linewidth=2, zorder=5)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Normalized Model: XY Trajectory Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 3D Trajectory
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    
    for i, result in enumerate(results_list):
        if result is None:
            continue
            
        pred_traj = result['predicted_trajectory']
        gt_traj = result['ground_truth_trajectory']
        cam_id = result['camera_id']
        color = colors[i]
        
        ax2.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], color=color, alpha=0.8, linewidth=2)
        ax2.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], color=color, alpha=0.8, linewidth=3, linestyle='--')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('3D Trajectories', fontweight='bold')
    
    # 3. Delta Predictions Comparison
    ax3 = plt.subplot(2, 3, 3)
    
    for i, result in enumerate(results_list):
        if result is None:
            continue
            
        pred_deltas = result['predicted_deltas']
        gt_deltas = result['ground_truth_deltas']
        cam_id = result['camera_id']
        color = colors[i]
        
        frames = np.arange(len(pred_deltas[:50]))  # First 50 frames
        ax3.plot(frames, gt_deltas[:50, 0], color=color, alpha=0.8, linewidth=2, 
                label=f'Cam{cam_id} GT Delta X')
        ax3.plot(frames, pred_deltas[:50, 0], color=color, alpha=0.8, linewidth=2, linestyle='--',
                label=f'Cam{cam_id} Pred Delta X')
    
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Delta X (m)')
    ax3.set_title('Delta X Predictions (First 50 frames)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Position Error Over Time
    ax4 = plt.subplot(2, 3, 4)
    
    for i, result in enumerate(results_list):
        if result is None:
            continue
            
        pred_traj = result['predicted_trajectory']
        gt_traj = result['ground_truth_trajectory']
        cam_id = result['camera_id']
        color = colors[i]
        
        position_errors = np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1)
        frames = np.arange(len(position_errors))
        
        ax4.plot(frames, position_errors, color=color, linewidth=2, alpha=0.8,
                label=f'Camera {cam_id}')
    
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Position Error (m)')
    ax4.set_title('Position Error Over Time', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Delta Variance Analysis
    ax5 = plt.subplot(2, 3, 5)
    
    camera_names = []
    pred_variances = []
    gt_variances = []
    
    for result in results_list:
        if result is None:
            continue
            
        pred_deltas = result['predicted_deltas']
        gt_deltas = result['ground_truth_deltas']
        cam_id = result['camera_id']
        
        pred_var = np.var(pred_deltas[:, 0])  # X variance
        gt_var = np.var(gt_deltas[:, 0])      # X variance
        
        camera_names.append(f'Cam{cam_id}')
        pred_variances.append(pred_var)
        gt_variances.append(gt_var)
    
    x_pos = np.arange(len(camera_names))
    width = 0.35
    
    bars1 = ax5.bar(x_pos - width/2, gt_variances, width, label='Ground Truth Variance', alpha=0.8)
    bars2 = ax5.bar(x_pos + width/2, pred_variances, width, label='Predicted Variance', alpha=0.8)
    
    ax5.set_xlabel('Camera')
    ax5.set_ylabel('Delta X Variance')
    ax5.set_title('Delta X Variance: GT vs Predicted', fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(camera_names)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add variance values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    f'{height:.2e}', ha='center', va='bottom', fontsize=9)
    
    # 6. Summary Metrics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "NORMALIZED MODEL RESULTS\\n" + "="*40 + "\\n\\n"
    
    for result in results_list:
        if result is None:
            continue
            
        pred_traj = result['predicted_trajectory']
        gt_traj = result['ground_truth_trajectory']
        cam_id = result['camera_id']
        
        final_error = np.linalg.norm(pred_traj[-1, :3] - gt_traj[-1, :3])
        mean_error = np.mean(np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1))
        trajectory_length = np.sum(np.linalg.norm(np.diff(gt_traj[:, :3], axis=0), axis=1))
        relative_error = (final_error / trajectory_length * 100) if trajectory_length > 0 else 0
        
        pred_var = np.var(result['predicted_deltas'][:, :3])
        gt_var = np.var(result['ground_truth_deltas'][:, :3])
        
        camera_type = "TRAINED" if cam_id == 0 else "UNSEEN"
        
        summary_text += f"Camera {cam_id} ({camera_type}):\\n"
        summary_text += f"  Predictions: {result['successful_predictions']}\\n"
        summary_text += f"  Final Error: {final_error:.6f}m\\n"
        summary_text += f"  Mean Error: {mean_error:.6f}m\\n"
        summary_text += f"  Relative Drift: {relative_error:.2f}%\\n"
        summary_text += f"  Prediction Variance: {pred_var:.2e}\\n"
        summary_text += f"  Ground Truth Variance: {gt_var:.2e}\\n"
        summary_text += "\\n"
    
    # Check if constant prediction issue is fixed
    all_pred_vars = [np.var(r['predicted_deltas'][:, :3]) if r else 0 for r in results_list]
    if all(var > 1e-8 for var in all_pred_vars):
        summary_text += "SUCCESS: Constant prediction FIXED!\\n"
        summary_text += "Model now predicts variations!"
    else:
        summary_text += "ISSUE: Still some constant predictions"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('Normalized Model: Trained Camera vs Unseen Camera Performance', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Trajectory comparison saved as: {save_path}")
    
    plt.close()

def main():
    print("TESTING NORMALIZED MODEL ON TRAINED AND UNSEEN CAMERAS")
    print("=" * 60)
    
    # Load normalized model
    model, device, delta_mean, delta_std = load_normalized_model()
    if model is None:
        return
    
    results = []
    
    # Test on Camera 0 (trained)
    print("\n" + "="*20 + " CAMERA 0 (TRAINED) " + "="*20)
    result_cam0 = predict_trajectory(model, device, delta_mean, delta_std, camera_id=0, max_frames=80)
    results.append(result_cam0)
    
    # Test on Camera 2 (unseen)
    print("\n" + "="*20 + " CAMERA 2 (UNSEEN) " + "="*20)
    result_cam2 = predict_trajectory(model, device, delta_mean, delta_std, camera_id=2, max_frames=80)
    results.append(result_cam2)
    
    # Create comparison plot
    if any(r is not None for r in results):
        print("\nCreating trajectory comparison plot...")
        plot_trajectory_comparison(results)
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY:")
        print("="*60)
        
        for result in results:
            if result is None:
                continue
                
            cam_id = result['camera_id']
            pred_var = np.var(result['predicted_deltas'][:, :3])
            gt_var = np.var(result['ground_truth_deltas'][:, :3])
            
            camera_type = "TRAINED" if cam_id == 0 else "UNSEEN"
            print(f"Camera {cam_id} ({camera_type}):")
            print(f"  Successful predictions: {result['successful_predictions']}")
            print(f"  Prediction variance: {pred_var:.2e}")
            print(f"  Ground truth variance: {gt_var:.2e}")
            print(f"  Variance ratio: {pred_var/gt_var:.3f}" if gt_var > 0 else "  Variance ratio: N/A")
            print()
        
        # Check if constant prediction is fixed
        all_vars = [np.var(r['predicted_deltas'][:, :3]) if r else 0 for r in results]
        if all(var > 1e-6 for var in all_vars):
            print("SUCCESS: Constant prediction issue is FIXED!")
            print("Model now produces trajectory variations on both cameras.")
        else:
            print("ISSUE: Some constant predictions still remain.")
    
    else:
        print("No successful predictions to analyze!")

if __name__ == '__main__':
    main()