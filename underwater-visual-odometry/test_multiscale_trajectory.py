"""
Test Multi-Scale Trained Model on Trajectory Prediction

This script loads the trained multi-scale model and tests it on real trajectory data
to see if it now predicts curved trajectories instead of straight lines.
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer.multiscale_uw_transvo import create_multiscale_model


def load_test_sequence(csv_file, start_idx=1000, sequence_length=20, img_size=192):
    """Load a test sequence from the dataset"""
    
    df = pd.read_csv(csv_file)
    sequence_data = df.iloc[start_idx:start_idx + sequence_length]
    
    images = []
    true_deltas = []
    
    for _, row in sequence_data.iterrows():
        # Load image
        img_path = row['cam0_path']
        if pd.isna(img_path):
            img = np.zeros((img_size, img_size, 3), dtype=np.float32)
        else:
            if not os.path.exists(str(img_path)):
                img_path = os.path.join(".", str(img_path))
            
            if os.path.exists(str(img_path)):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (img_size, img_size))
                        img = img.astype(np.float32) / 255.0
                    else:
                        img = np.zeros((img_size, img_size, 3), dtype=np.float32)
                except:
                    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
            else:
                img = np.zeros((img_size, img_size, 3), dtype=np.float32)
        
        images.append(img)
        
        # True delta pose
        delta_pose = np.array([
            float(row.get('delta_x', 0.0)), float(row.get('delta_y', 0.0)), float(row.get('delta_z', 0.0)),
            float(row.get('delta_roll', 0.0)), float(row.get('delta_pitch', 0.0)), float(row.get('delta_yaw', 0.0))
        ], dtype=np.float32)
        true_deltas.append(delta_pose)
    
    # Convert to tensors
    images = np.stack(images)
    images = torch.tensor(images).permute(0, 3, 1, 2)  # [seq_len, 3, H, W]
    true_deltas = torch.tensor(np.stack(true_deltas))  # [seq_len, 6]
    
    # Add batch and camera dimensions: [1, seq_len, 1, 3, H, W]
    images = images.unsqueeze(0).unsqueeze(2)
    
    return images, true_deltas


def test_multiscale_model():
    """Test the trained multi-scale model"""
    
    print("Testing Multi-Scale Trained Model")
    print("=" * 50)
    
    # Load model configuration (should match training)
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
    print(f"Device: {device}")
    
    # Load trained model
    model_path = 'multiscale_light_best_model.pth'
    if not os.path.exists(model_path):
        print(f"ERROR: Trained model not found at {model_path}")
        print("Make sure training completed successfully!")
        return
    
    print(f"Loading trained model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model and load weights
    model = create_multiscale_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Validation loss during training: {checkpoint['val_loss']:.6f}")
    
    # Load test data
    csv_file = "data/processed/training_dataset/training_data.csv"
    if not os.path.exists(csv_file):
        print(f"ERROR: Test data not found at {csv_file}")
        return
    
    # Test on multiple sequences
    test_sequences = [
        {'start': 1000, 'length': 20, 'name': 'Sequence 1'},
        {'start': 2000, 'length': 20, 'name': 'Sequence 2'},
        {'start': 3000, 'length': 20, 'name': 'Sequence 3'}
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Scale Model: Trajectory Prediction vs Ground Truth', fontsize=16)
    
    for i, test_seq in enumerate(test_sequences):
        print(f"\nTesting {test_seq['name']} (frames {test_seq['start']}-{test_seq['start']+test_seq['length']})")
        
        # Load test sequence
        images, true_deltas = load_test_sequence(
            csv_file, 
            start_idx=test_seq['start'], 
            sequence_length=test_seq['length'],
            img_size=config['img_size']
        )
        
        images = images.to(device)
        camera_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        
        # Predict trajectories
        predicted_trajectories = []
        
        with torch.no_grad():
            # Process in overlapping windows of 10 frames
            for start_frame in range(0, test_seq['length'] - 9, 5):  # Stride of 5
                end_frame = min(start_frame + 10, test_seq['length'])
                window_images = images[:, start_frame:end_frame]
                
                if window_images.size(1) < 10:
                    # Pad if needed
                    pad_size = 10 - window_images.size(1)
                    padding = torch.zeros(1, pad_size, 1, 3, config['img_size'], config['img_size'], device=device)
                    window_images = torch.cat([window_images, padding], dim=1)
                
                # Predict
                outputs = model(images=window_images, camera_ids=camera_ids)
                pred_deltas = outputs['delta_poses'][0]  # [seq_len, 6]
                
                # Take only the non-padded predictions
                actual_len = min(10, end_frame - start_frame)
                predicted_trajectories.append(pred_deltas[:actual_len])
        
        # Combine predictions (take first 10 frames to match model output)
        actual_length = min(10, test_seq['length'])
        true_deltas_truncated = true_deltas[:actual_length]
        
        if predicted_trajectories:
            pred_deltas_combined = predicted_trajectories[0][:actual_length]
        else:
            pred_deltas_combined = torch.zeros_like(true_deltas_truncated)
        
        # Convert to numpy
        true_deltas_np = true_deltas_truncated.cpu().numpy()
        pred_deltas_np = pred_deltas_combined.cpu().numpy()
        
        # Compute accumulated trajectories
        true_trajectory = np.cumsum(true_deltas_np, axis=0)
        pred_trajectory = np.cumsum(pred_deltas_np, axis=0)
        
        # Plot XY trajectory
        ax1 = axes[0, i]
        ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-o', 
                label='Ground Truth', linewidth=2, markersize=4)
        ax1.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r-s', 
                label='Multi-Scale Prediction', linewidth=2, markersize=3)
        ax1.plot(true_trajectory[0, 0], true_trajectory[0, 1], 'go', 
                label='Start', markersize=8)
        ax1.plot(true_trajectory[-1, 0], true_trajectory[-1, 1], 'ro', 
                label='End', markersize=8)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title(f'{test_seq["name"]}: XY Trajectory')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # Plot trajectory components over time
        ax2 = axes[1, i]
        time_steps = np.arange(len(true_trajectory))
        
        ax2.plot(time_steps, true_trajectory[:, 0], 'b-', label='GT X', linewidth=2)
        ax2.plot(time_steps, true_trajectory[:, 1], 'g-', label='GT Y', linewidth=2)
        ax2.plot(time_steps, pred_trajectory[:, 0], 'r--', label='Pred X', linewidth=2)
        ax2.plot(time_steps, pred_trajectory[:, 1], 'm--', label='Pred Y', linewidth=2)
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Position (m)')
        ax2.set_title(f'{test_seq["name"]}: Position vs Time')
        ax2.legend()
        ax2.grid(True)
        
        # Compute metrics
        position_error = np.linalg.norm(pred_trajectory - true_trajectory, axis=1)
        mean_error = np.mean(position_error)
        final_error = position_error[-1]
        
        print(f"  Mean position error: {mean_error:.6f}m")
        print(f"  Final position error: {final_error:.6f}m")
        print(f"  Trajectory length: {np.sum(np.linalg.norm(np.diff(true_trajectory, axis=0), axis=1)):.6f}m")
        
        # Check if prediction is straight line
        if len(pred_trajectory) > 2:
            # Compute curvature (simplified)
            pred_diffs = np.diff(pred_trajectory, axis=0)
            pred_angles = np.arctan2(pred_diffs[:, 1], pred_diffs[:, 0])
            angle_changes = np.diff(pred_angles)
            angle_changes = np.abs(np.arctan2(np.sin(angle_changes), np.cos(angle_changes)))
            mean_curvature = np.mean(angle_changes)
            
            print(f"  Predicted trajectory curvature: {mean_curvature:.6f} rad")
            
            if mean_curvature < 0.01:
                print(f"  WARNING: Prediction appears to be nearly straight!")
            else:
                print(f"  SUCCESS: Prediction shows curvature!")
    
    plt.tight_layout()
    plt.savefig('multiscale_trajectory_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n" + "="*50)
    print("TRAJECTORY TEST COMPLETED!")
    print("="*50)
    print("Results saved to: multiscale_trajectory_test_results.png")
    print("\nKey things to check in the plots:")
    print("1. Are predicted trajectories curved (not straight lines)?")
    print("2. Do predictions follow general direction of ground truth?")
    print("3. Are position errors reasonable (< 0.1m for short sequences)?")
    print("\nIf predictions are still straight lines, we need stronger supervision!")


if __name__ == "__main__":
    test_multiscale_model()