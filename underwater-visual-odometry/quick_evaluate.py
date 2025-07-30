#!/usr/bin/env python3
"""
Quick Model Evaluation for UW-TransVO
Automatically detects model architecture from checkpoint
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time
import json

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.datasets import UnderwaterVODataset
from training.losses import PoseLoss

def infer_model_config_from_checkpoint(checkpoint_path):
    """Infer model configuration from checkpoint structure"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Infer d_model from embedding dimensions
    if 'vision_transformer.cls_token' in state_dict:
        d_model = state_dict['vision_transformer.cls_token'].shape[-1]
    else:
        d_model = 768  # Default fallback
    
    # Infer num_layers from the number of transformer blocks
    num_layers = 0
    for key in state_dict.keys():
        if 'vision_transformer.blocks.' in key:
            layer_num = int(key.split('.')[2])
            num_layers = max(num_layers, layer_num + 1)
    
    # Infer num_heads from attention weight dimensions
    num_heads = 12  # Default fallback
    for key in state_dict.keys():
        if 'attn.qkv.weight' in key:
            qkv_dim = state_dict[key].shape[0]
            num_heads = qkv_dim // (d_model * 3)
            break
    
    print(f"Inferred model config from checkpoint:")
    print(f"  d_model: {d_model}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")
    
    # Create config based on inferred values
    config = {
        'img_size': 224,
        'patch_size': 16,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'max_cameras': 4,
        'max_seq_len': 2,
        'dropout': 0.1,
        'use_imu': False,
        'use_pressure': False,
        'uncertainty_estimation': True
    }
    
    return config

def quick_evaluate_model(checkpoint_path, data_csv, max_samples=50):
    """Quick model evaluation"""
    print("UW-TransVO Quick Model Evaluation")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Infer model config from checkpoint
    model_config = infer_model_config_from_checkpoint(checkpoint_path)
    
    # Create and load model
    print("\\nLoading model...")
    model = UWTransVO(**model_config).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create evaluation dataset
    print("\\nLoading evaluation dataset...")
    eval_dataset = UnderwaterVODataset(
        data_csv=data_csv,
        data_root='.',
        camera_ids=[0, 1, 2, 3],
        sequence_length=2,
        img_size=224,
        use_imu=False,
        use_pressure=False,
        augmentation=False,
        split='val',
        max_samples=max_samples
    )
    
    dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"Evaluation dataset: {len(eval_dataset)} samples")
    
    # Evaluation metrics
    criterion = PoseLoss(translation_weight=1.0, rotation_weight=10.0)
    
    translation_errors = []
    rotation_errors = []
    predictions = []
    ground_truths = []
    total_loss = 0.0
    num_batches = 0
    
    print("\\nRunning evaluation...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            images = batch['images'].to(device)
            camera_ids = batch['camera_ids'].to(device)
            camera_mask = batch['camera_mask'].to(device)
            pose_target = batch['pose_target'].to(device)
            
            # Forward pass
            output = model(
                images=images,
                camera_ids=camera_ids,
                camera_mask=camera_mask
            )
            
            # Calculate loss
            loss_dict = criterion(output['pose'], pose_target)
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
            
            # Store predictions and targets
            pred_poses = output['pose'].cpu().numpy()
            true_poses = pose_target.cpu().numpy()
            
            for i in range(pred_poses.shape[0]):
                pred_pose = pred_poses[i]
                true_pose = true_poses[i]
                
                # Calculate errors
                trans_error = np.linalg.norm(pred_pose[:3] - true_pose[:3])
                rot_error = np.linalg.norm(pred_pose[3:] - true_pose[3:])
                
                translation_errors.append(trans_error)
                rotation_errors.append(rot_error)
                predictions.append(pred_pose)
                ground_truths.append(true_pose)
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss_dict['total_loss'].item():.6f}")
    
    eval_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    
    # Calculate summary statistics
    translation_errors = np.array(translation_errors)
    rotation_errors = np.array(rotation_errors)
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    print("\\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"Evaluation Time: {eval_time:.2f} seconds")
    print(f"Samples Evaluated: {len(translation_errors)}")
    print(f"Average Loss: {avg_loss:.6f}")
    
    print("\\nTRANSLATION PERFORMANCE:")
    print(f"  Mean Error:   {np.mean(translation_errors):.6f} m")
    print(f"  Median Error: {np.median(translation_errors):.6f} m")
    print(f"  Std Dev:      {np.std(translation_errors):.6f} m")
    print(f"  Min Error:    {np.min(translation_errors):.6f} m")
    print(f"  Max Error:    {np.max(translation_errors):.6f} m")
    print(f"  RMSE:         {np.sqrt(np.mean(translation_errors**2)):.6f} m")
    
    print("\\nROTATION PERFORMANCE:")
    print(f"  Mean Error:   {np.mean(rotation_errors):.6f} rad ({np.mean(rotation_errors)*180/np.pi:.2f}°)")
    print(f"  Median Error: {np.median(rotation_errors):.6f} rad ({np.median(rotation_errors)*180/np.pi:.2f}°)")
    print(f"  Std Dev:      {np.std(rotation_errors):.6f} rad ({np.std(rotation_errors)*180/np.pi:.2f}°)")
    print(f"  Min Error:    {np.min(rotation_errors):.6f} rad ({np.min(rotation_errors)*180/np.pi:.2f}°)")
    print(f"  Max Error:    {np.max(rotation_errors):.6f} rad ({np.max(rotation_errors)*180/np.pi:.2f}°)")
    print(f"  RMSE:         {np.sqrt(np.mean(rotation_errors**2)):.6f} rad ({np.sqrt(np.mean(rotation_errors**2))*180/np.pi:.2f}°)")
    
    print("\\nACCURACY ANALYSIS:")
    trans_under_01 = np.sum(translation_errors < 0.1)
    trans_under_005 = np.sum(translation_errors < 0.05)
    rot_under_01 = np.sum(rotation_errors < 0.1)
    rot_under_005 = np.sum(rotation_errors < 0.05)
    total = len(translation_errors)
    
    print(f"  Translation < 0.05m: {trans_under_005}/{total} ({trans_under_005/total*100:.1f}%)")
    print(f"  Translation < 0.10m: {trans_under_01}/{total} ({trans_under_01/total*100:.1f}%)")
    print(f"  Rotation < 0.05rad:  {rot_under_005}/{total} ({rot_under_005/total*100:.1f}%)")
    print(f"  Rotation < 0.10rad:  {rot_under_01}/{total} ({rot_under_01/total*100:.1f}%)")
    
    # Create simple visualization
    print("\\nGenerating visualization...")
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('UW-TransVO Model Evaluation Results', fontsize=16, color='white')
    
    # Translation error histogram
    axes[0,0].hist(translation_errors, bins=20, color='cyan', alpha=0.7)
    axes[0,0].set_title('Translation Error Distribution', color='white')
    axes[0,0].set_xlabel('Translation Error (m)', color='white')
    axes[0,0].set_ylabel('Frequency', color='white')
    axes[0,0].grid(True, alpha=0.3)
    
    # Rotation error histogram
    axes[0,1].hist(rotation_errors, bins=20, color='orange', alpha=0.7)
    axes[0,1].set_title('Rotation Error Distribution', color='white')
    axes[0,1].set_xlabel('Rotation Error (rad)', color='white')
    axes[0,1].set_ylabel('Frequency', color='white')
    axes[0,1].grid(True, alpha=0.3)
    
    # Translation prediction scatter
    axes[1,0].scatter(ground_truths[:, 0], predictions[:, 0], c='red', alpha=0.6, s=20, label='X')
    axes[1,0].scatter(ground_truths[:, 1], predictions[:, 1], c='green', alpha=0.6, s=20, label='Y')
    axes[1,0].scatter(ground_truths[:, 2], predictions[:, 2], c='blue', alpha=0.6, s=20, label='Z')
    axes[1,0].plot([-1, 1], [-1, 1], 'white', linestyle='--', alpha=0.8)
    axes[1,0].set_title('Translation Predictions vs Ground Truth', color='white')
    axes[1,0].set_xlabel('Ground Truth (m)', color='white')
    axes[1,0].set_ylabel('Predicted (m)', color='white')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Rotation prediction scatter
    axes[1,1].scatter(ground_truths[:, 3], predictions[:, 3], c='cyan', alpha=0.6, s=20, label='Roll')
    axes[1,1].scatter(ground_truths[:, 4], predictions[:, 4], c='magenta', alpha=0.6, s=20, label='Pitch')
    axes[1,1].scatter(ground_truths[:, 5], predictions[:, 5], c='yellow', alpha=0.6, s=20, label='Yaw')
    axes[1,1].plot([-1, 1], [-1, 1], 'white', linestyle='--', alpha=0.8)
    axes[1,1].set_title('Rotation Predictions vs Ground Truth', color='white')
    axes[1,1].set_xlabel('Ground Truth (rad)', color='white')
    axes[1,1].set_ylabel('Predicted (rad)', color='white')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print("Visualization saved as 'evaluation_results.png'")
    
    # Save results summary
    results = {
        'model_info': {
            'checkpoint': str(checkpoint_path),
            'parameters': int(sum(p.numel() for p in model.parameters())),
            'config': model_config
        },
        'evaluation_summary': {
            'samples_evaluated': int(len(translation_errors)),
            'evaluation_time_seconds': float(eval_time),
            'average_loss': float(avg_loss),
            'translation_metrics': {
                'mean_error_m': float(np.mean(translation_errors)),
                'median_error_m': float(np.median(translation_errors)),
                'rmse_m': float(np.sqrt(np.mean(translation_errors**2))),
                'std_dev_m': float(np.std(translation_errors))
            },
            'rotation_metrics': {
                'mean_error_rad': float(np.mean(rotation_errors)),
                'mean_error_deg': float(np.mean(rotation_errors) * 180 / np.pi),
                'median_error_rad': float(np.median(rotation_errors)),
                'rmse_rad': float(np.sqrt(np.mean(rotation_errors**2))),
                'std_dev_rad': float(np.std(rotation_errors))
            },
            'accuracy_thresholds': {
                'translation_under_0.05m_percent': float(trans_under_005/total*100),
                'translation_under_0.10m_percent': float(trans_under_01/total*100),
                'rotation_under_0.05rad_percent': float(rot_under_005/total*100),
                'rotation_under_0.10rad_percent': float(rot_under_01/total*100)
            }
        }
    }
    
    with open('evaluation_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results summary saved as 'evaluation_summary.json'")
    print("\\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    
    return results

if __name__ == '__main__':
    checkpoint_path = 'checkpoints/test_4cam_seq2_vision/best_model.pth'
    data_csv = 'data/processed/training_dataset/training_data.csv'
    
    results = quick_evaluate_model(checkpoint_path, data_csv, max_samples=100)