#!/usr/bin/env python3
"""
Debug the model predictions to understand why we still get straight lines
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders

def debug_model_predictions():
    """Debug what the model is actually predicting"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        'sequence_length': 3,
        'image_size': 224,
        'csv_path': 'data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv',
        'data_root': 'data/processed/visual_odometry_dataset',
        'overlap_frames': 1,
        'test_bags': ['ariel_2023-12-21-14-28-22_4']
    }
    
    # Load model
    model, loss_fn = create_tsformer_vo(
        sequence_length=config['sequence_length'],
        pretrained=True,
        freeze_backbone=False,
        image_size=config['image_size']
    )
    
    checkpoint_path = Path('experiments/3_TSFormer_seq3_advanced_loss/checkpoint_best.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load data
    data_loaders = create_data_loaders(
        csv_path=config['csv_path'],
        data_root=config['data_root'],
        sequence_length=config['sequence_length'],
        overlap_frames=config['overlap_frames'],
        image_size=config['image_size'],
        batch_size=1,
        test_bags=config['test_bags'],
        num_workers=0,
        camera='cam0'
    )
    
    test_loader = data_loaders['test_loader']
    
    print("=== DEBUGGING MODEL PREDICTIONS ===")
    
    # Collect predictions and ground truth
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 50:  # Check first 50 samples
                break
                
            images = batch['images'].to(device)
            poses = batch['poses'].to(device)
            
            pred_poses = model(images)
            
            predictions.append(pred_poses.cpu().numpy()[0])
            ground_truths.append(poses.cpu().numpy()[0])
            
            if i < 10:  # Print first 10
                print(f"Sample {i+1}:")
                print(f"  Predicted: [{pred_poses[0, 0].item():.6f}, {pred_poses[0, 1].item():.6f}, {pred_poses[0, 2].item():.6f}, {pred_poses[0, 3].item():.6f}, {pred_poses[0, 4].item():.6f}, {pred_poses[0, 5].item():.6f}]")
                print(f"  GT:        [{poses[0, 0].item():.6f}, {poses[0, 1].item():.6f}, {poses[0, 2].item():.6f}, {poses[0, 3].item():.6f}, {poses[0, 4].item():.6f}, {poses[0, 5].item():.6f}]")
                print()
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    print("=== PREDICTION ANALYSIS ===")
    print(f"Total samples analyzed: {len(predictions)}")
    print()
    
    # Check if predictions are identical/constant
    print("Prediction Statistics:")
    for i, axis in enumerate(['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw']):
        pred_values = predictions[:, i]
        print(f"{axis}:")
        print(f"  Mean: {np.mean(pred_values):.6f}")
        print(f"  Std:  {np.std(pred_values):.6f}")
        print(f"  Min:  {np.min(pred_values):.6f}")
        print(f"  Max:  {np.max(pred_values):.6f}")
        print(f"  Unique values: {len(np.unique(np.round(pred_values, 6)))}")
        print()
    
    # Check if model is predicting the same value for everything
    unique_predictions = np.unique(predictions.round(6), axis=0)
    print(f"Number of unique prediction vectors: {len(unique_predictions)}")
    
    if len(unique_predictions) <= 5:
        print("WARNING: Model is predicting very few unique values!")
        print("Unique predictions:")
        for i, pred in enumerate(unique_predictions):
            print(f"  {i+1}: {pred}")
    
    # Compare with ground truth variation
    print("Ground Truth Statistics:")
    for i, axis in enumerate(['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw']):
        gt_values = ground_truths[:, i]
        print(f"{axis}:")
        print(f"  Mean: {np.mean(gt_values):.6f}")
        print(f"  Std:  {np.std(gt_values):.6f}")
        print(f"  Min:  {np.min(gt_values):.6f}")
        print(f"  Max:  {np.max(gt_values):.6f}")
        print()
    
    # Visualize prediction distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    labels = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw']
    
    for i in range(6):
        axes[i].hist(predictions[:, i], bins=20, alpha=0.7, label='Predictions', color='red')
        axes[i].hist(ground_truths[:, i], bins=20, alpha=0.7, label='Ground Truth', color='blue')
        axes[i].set_title(f'{labels[i]} Distribution')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_distribution_debug.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Check if model weights are actually changing
    print("=== MODEL BEHAVIOR ANALYSIS ===")
    
    # Test with different inputs
    test_input = torch.randn(1, 3, 3, 224, 224).to(device)
    with torch.no_grad():
        test_output1 = model(test_input)
        test_output2 = model(test_input)  # Same input
        
        # Different random input
        test_input2 = torch.randn(1, 3, 3, 224, 224).to(device)
        test_output3 = model(test_input2)
    
    print(f"Same input twice - difference: {torch.abs(test_output1 - test_output2).max().item():.8f}")
    print(f"Different inputs - difference: {torch.abs(test_output1 - test_output3).max().item():.8f}")
    
    if torch.abs(test_output1 - test_output3).max().item() < 1e-6:
        print("🚨 CRITICAL: Model is outputting identical values for different inputs!")
        print("This suggests the model has collapsed or is not properly trained.")

if __name__ == "__main__":
    debug_model_predictions()