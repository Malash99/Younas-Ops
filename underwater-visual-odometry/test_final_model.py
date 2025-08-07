"""
Test the Final Improved Model

Test the model after continued training to see trajectory improvements.
"""

import torch
import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer.multiscale_uw_transvo import create_multiscale_model

def test_final_model():
    """Test the final trained model on real data"""
    
    print("=" * 60)
    print("TESTING FINAL IMPROVED MODEL")
    print("=" * 60)
    
    # Model configuration
    config = {
        'img_size': 192,
        'd_model': 256,
        'num_heads': 4,
        'num_layers': 3,
        'max_cameras': 1,
        'max_seq_len': 10,
        'dropout': 0.1,
        'uncertainty_estimation': False
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_multiscale_model(config).to(device)
    
    # Load the final improved model
    model_path = 'final_fixed_model.pth'
    if not os.path.exists(model_path):
        print(f"ERROR: Final model not found at {model_path}")
        return
    
    print(f"Loading final model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    final_epoch = checkpoint.get('epoch', 'unknown')
    final_val_loss = checkpoint['val_loss']
    print(f"Final model from epoch {final_epoch}, validation loss: {final_val_loss:.6f}")
    
    # Test 1: Synthetic input diversity test
    print(f"\\n1. SYNTHETIC INPUT DIVERSITY TEST")
    print("-" * 40)
    
    batch_size, seq_len = 1, 10
    images = torch.randn(batch_size, seq_len, 1, 3, config['img_size'], config['img_size']).to(device)
    # Create visually distinct frames
    for i in range(seq_len):
        images[:, i] = images[:, i] + i * 0.4
    
    camera_ids = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(images=images, camera_ids=camera_ids)
        pred_deltas = outputs['delta_poses'][0].cpu().numpy()
    
    # Calculate diversity metrics
    frame_diffs = np.diff(pred_deltas, axis=0)
    max_diff = np.abs(frame_diffs).max()
    std_per_axis = np.std(pred_deltas, axis=0)
    mean_std = np.mean(std_per_axis[:3])
    
    print(f"Frame-by-frame predictions:")
    for i in range(seq_len):
        print(f"  Frame {i}: X={pred_deltas[i, 0]:8.6f}, Y={pred_deltas[i, 1]:8.6f}, Z={pred_deltas[i, 2]:8.6f}")
    
    print(f"\\nDiversity metrics:")
    print(f"  Max frame difference: {max_diff:.8f}")
    print(f"  Mean translation std: {mean_std:.6f}")
    
    # Test 2: Real data trajectory prediction
    print(f"\\n2. REAL DATA TRAJECTORY TEST")
    print("-" * 40)
    
    csv_file = "data/processed/training_dataset/training_data.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        
        # Select a test sequence with good motion
        start_idx = 1000
        seq_len = 20
        test_sequence = df.iloc[start_idx:start_idx + seq_len]
        
        # Get ground truth trajectory
        gt_deltas = test_sequence[['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw']].values.astype(float)
        gt_trajectory = np.cumsum(gt_deltas, axis=0)
        
        # Create dummy images for prediction (since we're testing architecture)
        images = torch.randn(1, seq_len, 1, 3, config['img_size'], config['img_size']).to(device) * 0.1
        camera_ids = torch.zeros(1, 1, dtype=torch.long).to(device)
        
        # Predict in overlapping windows
        predicted_deltas = []
        window_size = 10
        
        with torch.no_grad():
            for start in range(0, seq_len, 5):  # Stride of 5
                end = min(start + window_size, seq_len)
                if end - start < window_size:
                    # Pad the last window
                    window_images = images[:, start:end]
                    padding = torch.zeros(1, window_size - (end - start), 1, 3, config['img_size'], config['img_size']).to(device)
                    window_images = torch.cat([window_images, padding], dim=1)
                else:
                    window_images = images[:, start:end]
                
                outputs = model(images=window_images, camera_ids=camera_ids)
                pred_deltas = outputs['delta_poses'][0].cpu().numpy()
                
                # Take only valid predictions
                valid_len = min(end - start, window_size)
                predicted_deltas.append(pred_deltas[:valid_len])
        
        # Combine predictions (take first 10 frames)
        if predicted_deltas:
            pred_deltas = predicted_deltas[0][:min(10, len(gt_deltas))]
            gt_deltas_trimmed = gt_deltas[:len(pred_deltas)]
            
            # Compute trajectories
            pred_trajectory = np.cumsum(pred_deltas, axis=0)
            gt_trajectory_trimmed = np.cumsum(gt_deltas_trimmed, axis=0)
            
            print(f"Ground truth vs Prediction comparison (first {len(pred_deltas)} frames):")
            print(f"{'Frame':<5} {'GT_X':<10} {'GT_Y':<10} {'Pred_X':<10} {'Pred_Y':<10} {'Error':<10}")
            print("-" * 60)
            
            errors = []
            for i in range(len(pred_deltas)):
                gt_x, gt_y = gt_deltas_trimmed[i, 0], gt_deltas_trimmed[i, 1]
                pred_x, pred_y = pred_deltas[i, 0], pred_deltas[i, 1]
                error = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                errors.append(error)
                
                print(f"{i:<5} {gt_x:<10.6f} {gt_y:<10.6f} {pred_x:<10.6f} {pred_y:<10.6f} {error:<10.6f}")
            
            mean_error = np.mean(errors)
            print(f"\\nMean frame error: {mean_error:.6f}m")
            
            # Trajectory analysis
            gt_length = np.sum(np.linalg.norm(np.diff(gt_trajectory_trimmed[:, :3], axis=0), axis=1))
            pred_length = np.sum(np.linalg.norm(np.diff(pred_trajectory[:, :3], axis=0), axis=1))
            
            gt_direct = np.linalg.norm(gt_trajectory_trimmed[-1, :3] - gt_trajectory_trimmed[0, :3])
            pred_direct = np.linalg.norm(pred_trajectory[-1, :3] - pred_trajectory[0, :3])
            
            gt_curvature = gt_length / (gt_direct + 1e-8)
            pred_curvature = pred_length / (pred_direct + 1e-8)
            
            print(f"\\nTrajectory Analysis:")
            print(f"  Ground Truth - Length: {gt_length:.6f}m, Curvature: {gt_curvature:.4f}")
            print(f"  Prediction   - Length: {pred_length:.6f}m, Curvature: {pred_curvature:.4f}")
            
            curvature_error = abs(gt_curvature - pred_curvature)
            print(f"  Curvature error: {curvature_error:.4f}")
            
            # Classification
            if pred_curvature > 1.1:
                print(f"  SUCCESS: Model predicts CURVED trajectory!")
            elif pred_curvature > 1.02:
                print(f"  MODERATE: Model predicts slightly curved trajectory")
            else:
                print(f"  STRAIGHT: Model predicts mostly straight trajectory")
                
            # Visual plot
            try:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(gt_trajectory_trimmed[:, 0], gt_trajectory_trimmed[:, 1], 'b-o', 
                        label='Ground Truth', linewidth=2, markersize=4)
                plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r-s', 
                        label='Prediction', linewidth=2, markersize=3)
                plt.plot(gt_trajectory_trimmed[0, 0], gt_trajectory_trimmed[0, 1], 'go', 
                        label='Start', markersize=8)
                plt.xlabel('X Position (m)')
                plt.ylabel('Y Position (m)')
                plt.title('XY Trajectory Comparison')
                plt.legend()
                plt.grid(True)
                plt.axis('equal')
                
                plt.subplot(1, 2, 2)
                time_steps = np.arange(len(pred_deltas))
                plt.plot(time_steps, gt_deltas_trimmed[:, 0], 'b-', label='GT Delta X', linewidth=2)
                plt.plot(time_steps, gt_deltas_trimmed[:, 1], 'g-', label='GT Delta Y', linewidth=2)
                plt.plot(time_steps, pred_deltas[:, 0], 'r--', label='Pred Delta X', linewidth=2)
                plt.plot(time_steps, pred_deltas[:, 1], 'm--', label='Pred Delta Y', linewidth=2)
                plt.xlabel('Frame')
                plt.ylabel('Delta (m)')
                plt.title('Frame-to-Frame Deltas')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig('final_model_trajectory_test.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"\\nTrajectory plot saved as: final_model_trajectory_test.png")
                
            except Exception as e:
                print(f"Plot creation failed: {e}")
    
    # Test 3: Compare with old model
    print(f"\\n3. COMPARISON WITH OLD MODEL")
    print("-" * 40)
    
    old_model_path = 'multiscale_light_best_model.pth'
    if os.path.exists(old_model_path):
        old_checkpoint = torch.load(old_model_path, map_location=device, weights_only=False)
        model.load_state_dict(old_checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            old_outputs = model(images=images[:, :10], camera_ids=camera_ids)
            old_pred_deltas = old_outputs['delta_poses'][0].cpu().numpy()
        
        old_max_diff = np.abs(np.diff(old_pred_deltas, axis=0)).max()
        old_std = np.mean(np.std(old_pred_deltas[:, :3], axis=0))
        
        print(f"Model comparison:")
        print(f"  Old model - Max difference: {old_max_diff:.8f}, Mean std: {old_std:.8f}")
        print(f"  New model - Max difference: {max_diff:.8f}, Mean std: {mean_std:.8f}")
        
        if max_diff > old_max_diff * 1000:
            print(f"  MASSIVE IMPROVEMENT in frame diversity!")
        elif max_diff > old_max_diff * 10:
            print(f"  Major improvement in frame diversity")
        elif max_diff > old_max_diff:
            print(f"  Improvement in frame diversity")
        else:
            print(f"  Similar diversity")
    
    print(f"\\n" + "=" * 60)
    print("FINAL MODEL TEST COMPLETE")
    print("=" * 60)
    print("SUMMARY:")
    if mean_std > 0.005:
        print("EXCELLENT frame diversity - model produces varying predictions")
    elif mean_std > 0.002:
        print("GOOD frame diversity - model avoids constant predictions")
    else:
        print("MODERATE frame diversity - some improvement needed")
    
    if max_diff > 0.1:
        print("STRONG frame-to-frame variation")
    elif max_diff > 0.01:
        print("MODERATE frame-to-frame variation")
    else:
        print("LOW frame-to-frame variation")


if __name__ == "__main__":
    test_final_model()