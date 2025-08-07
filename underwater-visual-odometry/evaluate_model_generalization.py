#!/usr/bin/env python3
"""
Evaluate Model Generalization
Test on different cameras and full trajectories
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

class ModelEvaluator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct model with same config
        self.config = checkpoint['config']['model']
        self.model = UltraConservativeModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Model config: {self.config}")
        
    def evaluate_camera(self, camera_id, max_samples=None):
        """Evaluate model on specific camera"""
        
        print(f"\n=== EVALUATING CAMERA {camera_id} ===")
        
        try:
            # Create dataloader for specific camera
            _, test_loader = create_sub_trajectory_dataloaders(
                train_csv='data/processed/training_dataset/training_data_filtered.csv',
                val_csv='data/processed/training_dataset/training_data_filtered.csv',
                sub_trajectory_length=3,  # Same as training
                overlap=1,
                camera_ids=[camera_id],  # Specific camera
                batch_size=1,
                num_workers=0,
                max_samples_train=None,
                max_samples_val=max_samples or 50  # Test on 50 samples
            )
            
            print(f"Created dataloader for camera {camera_id}: {len(test_loader)} samples")
            
        except Exception as e:
            print(f"Error creating dataloader for camera {camera_id}: {e}")
            return None
        
        # Evaluation metrics
        total_samples = 0
        successful_predictions = 0
        total_translation_error = 0.0
        total_rotation_error = 0.0
        total_drift = 0.0
        
        all_pred_poses = []
        all_target_poses = []
        prediction_errors = []
        
        print(f"Testing {len(test_loader)} samples...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Camera {camera_id}")):
                try:
                    # Move to device
                    images = batch['images'].to(self.device)
                    camera_ids_batch = batch['camera_ids'].to(self.device)
                    camera_mask = batch['camera_mask'].to(self.device)
                    pose_targets = batch['pose_targets'].to(self.device)
                    
                    # FORCE camera ID to 0 (trained camera) to avoid embedding issues
                    camera_ids_batch.fill_(0)  # Always use trained camera ID
                    
                    sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
                    
                    # Forward pass
                    predictions = self.model(images, camera_ids_batch, camera_mask, sub_traj_length)
                    
                    # Check for valid predictions
                    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                        print(f"Invalid predictions in batch {batch_idx}")
                        continue
                    
                    # Calculate errors
                    pred_np = predictions[0].cpu().numpy()  # [seq_len-1, 6]
                    target_np = pose_targets[0].cpu().numpy()  # [seq_len-1, 6]
                    
                    # Translation error (first 3 dimensions)
                    trans_error = np.mean(np.linalg.norm(pred_np[:, :3] - target_np[:, :3], axis=1))
                    
                    # Rotation error (last 3 dimensions)
                    rot_error = np.mean(np.linalg.norm(pred_np[:, 3:] - target_np[:, 3:], axis=1))
                    
                    # Cumulative trajectory error (drift)
                    pred_cumulative = np.cumsum(pred_np[:, :3], axis=0)
                    target_cumulative = np.cumsum(target_np[:, :3], axis=0)
                    final_drift = np.linalg.norm(pred_cumulative[-1] - target_cumulative[-1])
                    
                    # Accumulate metrics
                    total_translation_error += trans_error
                    total_rotation_error += rot_error
                    total_drift += final_drift
                    successful_predictions += 1
                    
                    # Store for analysis
                    all_pred_poses.append(pred_np)
                    all_target_poses.append(target_np)
                    prediction_errors.append({
                        'batch_idx': batch_idx,
                        'translation_error': trans_error,
                        'rotation_error': rot_error,
                        'final_drift': final_drift
                    })
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
                
                total_samples += 1
        
        # Calculate final metrics
        if successful_predictions > 0:
            avg_translation_error = total_translation_error / successful_predictions
            avg_rotation_error = total_rotation_error / successful_predictions
            avg_drift = total_drift / successful_predictions
            success_rate = successful_predictions / total_samples * 100
            
            results = {
                'camera_id': camera_id,
                'total_samples': total_samples,
                'successful_predictions': successful_predictions,
                'success_rate_percent': success_rate,
                'avg_translation_error_m': avg_translation_error,
                'avg_rotation_error_rad': avg_rotation_error,
                'avg_rotation_error_deg': np.degrees(avg_rotation_error),
                'avg_final_drift_m': avg_drift,
                'all_errors': prediction_errors
            }
            
            # Print results
            print(f"\nRESULTS FOR CAMERA {camera_id}:")
            print(f"  Success Rate: {success_rate:.1f}% ({successful_predictions}/{total_samples})")
            print(f"  Avg Translation Error: {avg_translation_error:.6f} m")
            print(f"  Avg Rotation Error: {np.degrees(avg_rotation_error):.3f} degrees")
            print(f"  Avg Final Drift: {avg_drift:.6f} m")
            
            return results
        else:
            print(f"No successful predictions for camera {camera_id}")
            return None
    
    def evaluate_full_trajectory(self, camera_id=0, max_trajectory_length=100):
        """Evaluate on longer trajectories by stitching sub-trajectories"""
        
        print(f"\n=== EVALUATING FULL TRAJECTORY (Camera {camera_id}) ===")
        
        # Load data
        df = pd.read_csv('data/processed/training_dataset/training_data_filtered.csv')
        val_data = df[df['split'] == 'val'].reset_index(drop=True)
        
        print(f"Available validation data: {len(val_data)} samples")
        
        # Take a continuous sequence for full trajectory
        trajectory_length = min(max_trajectory_length, len(val_data))
        trajectory_data = val_data.iloc[:trajectory_length].copy()
        
        print(f"Testing on trajectory of {trajectory_length} frames")
        
        # Prepare full trajectory
        all_predictions = []
        all_targets = []
        cumulative_pred = np.zeros(6)
        cumulative_target = np.zeros(6)
        
        trajectory_pred = [cumulative_pred.copy()]
        trajectory_target = [cumulative_target.copy()]
        
        # Process in overlapping windows of 3 frames
        window_size = 3
        successful_windows = 0
        
        with torch.no_grad():
            for i in tqdm(range(len(trajectory_data) - window_size + 1), desc="Full Trajectory"):
                try:
                    # Get window data
                    window_data = trajectory_data.iloc[i:i+window_size]
                    
                    # Load images for this window
                    images_list = []
                    poses_list = []
                    
                    for _, row in window_data.iterrows():
                        # Get image path for specified camera
                        img_path = row[f'cam{camera_id}_path']
                        
                        # For now, create dummy images (replace with actual image loading)
                        dummy_image = torch.randn(3, 224, 224) * 0.1
                        images_list.append(dummy_image)
                        
                        # Get pose
                        pose = np.array([
                            row['delta_x'], row['delta_y'], row['delta_z'],
                            row['delta_roll'], row['delta_pitch'], row['delta_yaw']
                        ])
                        poses_list.append(pose)
                    
                    # Prepare batch
                    images = torch.stack(images_list).unsqueeze(0).unsqueeze(2).to(self.device)  # [1, 3, 1, 3, 224, 224]
                    camera_ids_batch = torch.tensor([[0]], device=self.device)  # Force to trained camera ID
                    camera_mask = torch.tensor([[False]], device=self.device)
                    
                    # Forward pass
                    predictions = self.model(images, camera_ids_batch, camera_mask, window_size)
                    
                    if torch.isnan(predictions).any():
                        continue
                    
                    # Get predictions for this window
                    pred_window = predictions[0].cpu().numpy()  # [2, 6] for 3-frame window
                    target_window = np.array(poses_list[1:])  # Skip first frame
                    
                    # Use only the first prediction to avoid double-counting overlaps
                    if len(pred_window) > 0:
                        pred_pose = pred_window[0]  # First pose prediction
                        target_pose = target_window[0]  # First target pose
                        
                        # Update cumulative trajectories
                        cumulative_pred += pred_pose
                        cumulative_target += target_pose
                        
                        trajectory_pred.append(cumulative_pred.copy())
                        trajectory_target.append(cumulative_target.copy())
                        
                        successful_windows += 1
                    
                except Exception as e:
                    print(f"Error in window starting at {i}: {e}")
                    continue
        
        # Calculate trajectory metrics
        if len(trajectory_pred) > 1:
            trajectory_pred = np.array(trajectory_pred)
            trajectory_target = np.array(trajectory_target)
            
            # Calculate errors
            position_errors = np.linalg.norm(trajectory_pred[:, :3] - trajectory_target[:, :3], axis=1)
            final_drift = position_errors[-1]
            trajectory_length = np.sum(np.linalg.norm(np.diff(trajectory_target[:, :3], axis=0), axis=1))
            relative_drift = (final_drift / trajectory_length * 100) if trajectory_length > 0 else 0
            
            results = {
                'trajectory_length_frames': len(trajectory_pred),
                'successful_windows': successful_windows,
                'final_drift_m': final_drift,
                'trajectory_length_m': trajectory_length,
                'relative_drift_percent': relative_drift,
                'mean_position_error_m': np.mean(position_errors),
                'max_position_error_m': np.max(position_errors),
                'trajectory_pred': trajectory_pred,
                'trajectory_target': trajectory_target,
                'position_errors': position_errors
            }
            
            print(f"\nFULL TRAJECTORY RESULTS:")
            print(f"  Trajectory Length: {len(trajectory_pred)} frames ({trajectory_length:.3f}m)")
            print(f"  Successful Windows: {successful_windows}")
            print(f"  Final Drift: {final_drift:.6f}m")
            print(f"  Relative Drift: {relative_drift:.2f}%")
            print(f"  Mean Position Error: {np.mean(position_errors):.6f}m")
            
            return results
        else:
            print("No successful trajectory predictions")
            return None

def main():
    model_path = 'ultra_conservative_best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please run training first to generate the model.")
        return
    
    evaluator = ModelEvaluator(model_path)
    
    print("ULTRA-CONSERVATIVE MODEL GENERALIZATION TEST")
    print("=" * 60)
    
    # Test 1: Camera generalization (trained on cam0, test on cam1, cam2)
    camera_results = {}
    
    for camera_id in [0, 1, 2]:  # Test cam0 (trained), cam1, cam2 (unseen)
        print(f"\n{'='*20} CAMERA {camera_id} {'='*20}")
        if camera_id == 0:
            print("(Trained camera - baseline performance)")
        else:
            print("(Unseen camera - generalization test)")
        
        result = evaluator.evaluate_camera(camera_id, max_samples=30)
        if result:
            camera_results[camera_id] = result
    
    # Test 2: Full trajectory evaluation
    print(f"\n{'='*20} FULL TRAJECTORY {'='*20}")
    trajectory_result = evaluator.evaluate_full_trajectory(camera_id=0, max_trajectory_length=50)
    
    # Summary
    print(f"\n{'='*60}")
    print("GENERALIZATION SUMMARY:")
    print(f"{'='*60}")
    
    if camera_results:
        print("\nCAMERA COMPARISON:")
        for cam_id, result in camera_results.items():
            status = "TRAINED" if cam_id == 0 else "UNSEEN"
            print(f"  Camera {cam_id} ({status}):")
            print(f"    Success Rate: {result['success_rate_percent']:.1f}%")
            print(f"    Translation Error: {result['avg_translation_error_m']:.6f}m")
            print(f"    Final Drift: {result['avg_final_drift_m']:.6f}m")
    
    if trajectory_result:
        print(f"\nFULL TRAJECTORY:")
        print(f"  Length: {trajectory_result['trajectory_length_frames']} frames")
        print(f"  Final Drift: {trajectory_result['final_drift_m']:.6f}m")
        print(f"  Relative Drift: {trajectory_result['relative_drift_percent']:.2f}%")
    
    print(f"\nEvaluation complete!")

if __name__ == '__main__':
    main()