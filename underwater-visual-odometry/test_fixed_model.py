#!/usr/bin/env python3
"""
Test Fixed Model
Check if the new model with proper learning rate produces variations
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO

class FixedModel(nn.Module):
    """Same model as used in fixed training"""
    
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

def test_fixed_model():
    """Test if the fixed model produces variations instead of constants"""
    
    # Load the fixed model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'fixed_training_best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Fixed model not found: {model_path}")
        return
    
    print("TESTING FIXED MODEL")
    print("=" * 50)
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']['model']
    model = FixedModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded fixed model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.8f}")
    print()
    
    # Load Bag 0 data
    df = pd.read_csv('data/processed/training_dataset/training_data_filtered.csv')
    bag0_data = df[df['bag_name'] == 'ariel_2023-12-21-14-24-42_0'].sort_values('timestamp').reset_index(drop=True)
    
    print("TESTING FIRST 30 PREDICTIONS:")
    print("Frame   GT_X        GT_Y        GT_Z      |  PRED_X     PRED_Y     PRED_Z    | X_Error   Y_Error")
    print("-" * 95)
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for i in range(min(30, len(bag0_data) - 3)):
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
    print("COMPARISON: OLD vs FIXED MODEL")
    print("=" * 50)
    
    # Calculate variances
    pred_x_var = predictions[:, 0].var()
    pred_y_var = predictions[:, 1].var()
    gt_x_var = ground_truths[:, 0].var()
    gt_y_var = ground_truths[:, 1].var()
    
    print("VARIANCE ANALYSIS:")
    print(f"  Ground Truth X variance: {gt_x_var:.8f}")
    print(f"  Predicted X variance:    {pred_x_var:.8f}")
    print(f"  Ground Truth Y variance: {gt_y_var:.8f}")
    print(f"  Predicted Y variance:    {pred_y_var:.8f}")
    print()
    
    # Compare with old model results
    old_pred_x_var = 0.00000000  # From previous debug
    old_pred_y_var = 0.00000000  # From previous debug
    
    print("IMPROVEMENT CHECK:")
    x_improvement = pred_x_var / old_pred_x_var if old_pred_x_var > 1e-10 else float('inf')
    y_improvement = pred_y_var / old_pred_y_var if old_pred_y_var > 1e-10 else float('inf')
    
    print(f"X Variance Improvement: {x_improvement:.1f}x better" if x_improvement != float('inf') else "X Variance: INFINITELY better (was constant)")
    print(f"Y Variance Improvement: {y_improvement:.1f}x better" if y_improvement != float('inf') else "Y Variance: INFINITELY better (was constant)")
    print()
    
    # Check if still constant
    if pred_x_var < 1e-8 and pred_y_var < 1e-8:
        print("RESULT: Model is STILL predicting constants!")
        print("NEXT STEPS:")
        print("  1. Learning rate may still be too small")
        print("  2. Need stronger variation penalty in loss")
        print("  3. May need more training epochs")
        print("  4. Consider data normalization")
    else:
        print("SUCCESS: Model is now learning variations!")
        print("NEXT STEPS:")
        print("  1. Test full trajectory prediction")
        print("  2. Compare with previous straight-line results")
        print("  3. Fine-tune for better accuracy")
    
    # Calculate prediction statistics
    print()
    print("PREDICTION STATISTICS:")
    print(f"X: mean={predictions[:, 0].mean():.6f}, std={predictions[:, 0].std():.6f}, range=[{predictions[:, 0].min():.6f}, {predictions[:, 0].max():.6f}]")
    print(f"Y: mean={predictions[:, 1].mean():.6f}, std={predictions[:, 1].std():.6f}, range=[{predictions[:, 1].min():.6f}, {predictions[:, 1].max():.6f}]")
    print(f"Z: mean={predictions[:, 2].mean():.6f}, std={predictions[:, 2].std():.6f}, range=[{predictions[:, 2].min():.6f}, {predictions[:, 2].max():.6f}]")

if __name__ == '__main__':
    test_fixed_model()