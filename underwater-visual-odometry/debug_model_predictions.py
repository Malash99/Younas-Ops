#!/usr/bin/env python3
"""
Debug Model Predictions
Check what the model is actually predicting vs ground truth deltas
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO

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

def debug_model_predictions():
    """Debug what the model is actually predicting"""
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'ultra_conservative_best_model.pth'
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']['model']
    model = UltraConservativeModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load Bag 0 data
    df = pd.read_csv('data/processed/training_dataset/training_data_filtered.csv')
    bag0_data = df[df['bag_name'] == 'ariel_2023-12-21-14-24-42_0'].sort_values('timestamp').reset_index(drop=True)
    
    print("DEBUGGING MODEL PREDICTIONS")
    print("=" * 60)
    print(f"Loaded model with config: {config}")
    print(f"Bag 0 frames: {len(bag0_data)}")
    print()
    
    # Test on first 50 samples to see what model predicts
    print("FIRST 50 PREDICTIONS vs GROUND TRUTH:")
    print("Frame   GT_X        GT_Y        GT_Z      |  PRED_X     PRED_Y     PRED_Z    | X_Error   Y_Error")
    print("-" * 95)
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for i in range(min(50, len(bag0_data) - 3)):
            try:
                # Get window data
                window_data = bag0_data.iloc[i:i+3]
                
                # Create dummy images
                images_list = [torch.randn(3, 224, 224) * 0.1 for _ in range(3)]
                images = torch.stack(images_list).unsqueeze(0).unsqueeze(2).to(device)
                
                camera_ids_batch = torch.tensor([[0]], device=device)
                camera_mask = torch.tensor([[False]], device=device)
                
                # Forward pass
                pred = model(images, camera_ids_batch, camera_mask, 3)
                
                if torch.isnan(pred).any():
                    continue
                
                # Get first prediction from window
                pred_delta = pred[0, 0].cpu().numpy()
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
                
                # Print comparison
                x_error = abs(pred_delta[0] - gt_delta[0])
                y_error = abs(pred_delta[1] - gt_delta[1])
                
                print(f"{i:3d}   {gt_delta[0]:8.6f}  {gt_delta[1]:8.6f}  {gt_delta[2]:8.6f} | {pred_delta[0]:8.6f} {pred_delta[1]:8.6f} {pred_delta[2]:8.6f} | {x_error:.6f} {y_error:.6f}")
                
            except Exception as e:
                continue
    
    if len(predictions) == 0:
        print("No valid predictions!")
        return
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    print()
    print("ANALYSIS:")
    print(f"Valid predictions: {len(predictions)}")
    print()
    
    print("GROUND TRUTH STATISTICS:")
    print(f"  X: mean={ground_truths[:, 0].mean():.6f}, std={ground_truths[:, 0].std():.6f}, range=[{ground_truths[:, 0].min():.6f}, {ground_truths[:, 0].max():.6f}]")
    print(f"  Y: mean={ground_truths[:, 1].mean():.6f}, std={ground_truths[:, 1].std():.6f}, range=[{ground_truths[:, 1].min():.6f}, {ground_truths[:, 1].max():.6f}]")
    print(f"  Z: mean={ground_truths[:, 2].mean():.6f}, std={ground_truths[:, 2].std():.6f}, range=[{ground_truths[:, 2].min():.6f}, {ground_truths[:, 2].max():.6f}]")
    print()
    
    print("PREDICTION STATISTICS:")
    print(f"  X: mean={predictions[:, 0].mean():.6f}, std={predictions[:, 0].std():.6f}, range=[{predictions[:, 0].min():.6f}, {predictions[:, 0].max():.6f}]")
    print(f"  Y: mean={predictions[:, 1].mean():.6f}, std={predictions[:, 1].std():.6f}, range=[{predictions[:, 1].min():.6f}, {predictions[:, 1].max():.6f}]")
    print(f"  Z: mean={predictions[:, 2].mean():.6f}, std={predictions[:, 2].std():.6f}, range=[{predictions[:, 2].min():.6f}, {predictions[:, 2].max():.6f}]")
    print()
    
    print("PREDICTION ERRORS:")
    x_errors = np.abs(predictions[:, 0] - ground_truths[:, 0])
    y_errors = np.abs(predictions[:, 1] - ground_truths[:, 1])
    z_errors = np.abs(predictions[:, 2] - ground_truths[:, 2])
    
    print(f"  X errors: mean={x_errors.mean():.6f}, max={x_errors.max():.6f}")
    print(f"  Y errors: mean={y_errors.mean():.6f}, max={y_errors.max():.6f}")
    print(f"  Z errors: mean={z_errors.mean():.6f}, max={z_errors.max():.6f}")
    print()
    
    # Check if model is just predicting mean values
    pred_x_var = predictions[:, 0].var()
    pred_y_var = predictions[:, 1].var()
    gt_x_var = ground_truths[:, 0].var()
    gt_y_var = ground_truths[:, 1].var()
    
    print("VARIANCE ANALYSIS (detecting if model predicts constant values):")
    print(f"  Ground Truth X variance: {gt_x_var:.8f}")
    print(f"  Predicted X variance:    {pred_x_var:.8f} {'(CONSTANT!)' if pred_x_var < 1e-8 else ''}")
    print(f"  Ground Truth Y variance: {gt_y_var:.8f}")
    print(f"  Predicted Y variance:    {pred_y_var:.8f} {'(CONSTANT!)' if pred_y_var < 1e-8 else ''}")
    print()
    
    if pred_x_var < 1e-8 or pred_y_var < 1e-8:
        print("PROBLEM DETECTED:")
        print("   Model is predicting nearly CONSTANT values!")
        print("   This explains why trajectories are straight lines.")
        print("   The model learned to predict the MEAN delta, not actual motion.")
        print()
        print("LIKELY CAUSES:")
        print("   1. Learning rate too small - model can't learn variations")
        print("   2. Loss function issue - not penalizing constant predictions")
        print("   3. Model capacity too small - can't represent variations")
        print("   4. Training data preprocessing issue")
        print("   5. Optimizer issue - not updating parameters properly")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Delta X comparison
    plt.subplot(2, 3, 1)
    plt.plot(ground_truths[:, 0], 'b-', label='Ground Truth X', alpha=0.7)
    plt.plot(predictions[:, 0], 'r--', label='Predicted X', alpha=0.7)
    plt.title('Delta X: Prediction vs Ground Truth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Delta Y comparison
    plt.subplot(2, 3, 2)
    plt.plot(ground_truths[:, 1], 'b-', label='Ground Truth Y', alpha=0.7)
    plt.plot(predictions[:, 1], 'r--', label='Predicted Y', alpha=0.7)
    plt.title('Delta Y: Prediction vs Ground Truth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Delta Z comparison
    plt.subplot(2, 3, 3)
    plt.plot(ground_truths[:, 2], 'b-', label='Ground Truth Z', alpha=0.7)
    plt.plot(predictions[:, 2], 'r--', label='Predicted Z', alpha=0.7)
    plt.title('Delta Z: Prediction vs Ground Truth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error plots
    plt.subplot(2, 3, 4)
    plt.plot(x_errors, 'r-', alpha=0.7)
    plt.title('X Prediction Errors')
    plt.ylabel('Absolute Error (m)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.plot(y_errors, 'g-', alpha=0.7)
    plt.title('Y Prediction Errors')
    plt.ylabel('Absolute Error (m)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    plt.plot(z_errors, 'b-', alpha=0.7)
    plt.title('Z Prediction Errors')
    plt.ylabel('Absolute Error (m)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_prediction_debug.png', dpi=300, bbox_inches='tight')
    print("Debug visualization saved as: model_prediction_debug.png")
    plt.close()

if __name__ == '__main__':
    debug_model_predictions()