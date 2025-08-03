#!/usr/bin/env python3
"""
Minimal training test - no web dashboard
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data.sub_trajectory_dataset import create_sub_trajectory_dataloaders
from models.transformer import UWTransVO
from training.trajectory_losses import TrajectoryAwareLoss

class SubTrajectoryModel(nn.Module):
    def __init__(self, base_model_config):
        super().__init__()
        self.base_model = UWTransVO(**base_model_config)
        
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

def main():
    print("Minimal Sub-Trajectory Training Test")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Configuration
    config = {
        'img_size': 224,
        'patch_size': 16,
        'd_model': 768,
        'num_heads': 1,
        'num_layers': 6,
        'max_cameras': 3,
        'max_seq_len': 5,
        'dropout': 0.1,
        'use_imu': False,
        'use_pressure': False,
        'uncertainty_estimation': True
    }
    
    print("\n1. Creating model...")
    model = SubTrajectoryModel(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if torch.cuda.is_available():
        print(f"GPU memory after model: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print("\n2. Creating dataset...")
    try:
        train_loader, val_loader = create_sub_trajectory_dataloaders(
            train_csv='data/processed/training_dataset/training_data.csv',
            val_csv='data/processed/training_dataset/training_data.csv',
            sub_trajectory_length=5,
            overlap=2,
            camera_ids=[0, 1, 2],
            batch_size=1,
            num_workers=0,
            max_samples_train=3,  # Tiny test
            max_samples_val=1
        )
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Dataset creation failed: {e}")
        return
    
    print("\n3. Testing forward pass...")
    model.eval()
    
    try:
        for batch_idx, batch in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            
            # Move to device
            images = batch['images'].to(device)
            camera_ids = batch['camera_ids'].to(device)
            camera_mask = batch['camera_mask'].to(device)
            pose_targets = batch['pose_targets'].to(device)
            accumulated_targets = batch['accumulated_poses'].to(device)
            
            print(f"  Images shape: {images.shape}")
            print(f"  Pose targets shape: {pose_targets.shape}")
            
            if torch.cuda.is_available():
                print(f"  GPU memory before forward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Forward pass
            with torch.no_grad():
                sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
                predictions = model(images, camera_ids, camera_mask, sub_traj_length)
                print(f"  Predictions shape: {predictions.shape}")
                
                # Test loss
                criterion = TrajectoryAwareLoss()
                loss_dict = criterion(predictions, pose_targets, accumulated_targets)
                print(f"  Loss: {loss_dict['total_loss'].item():.6f}")
                print(f"  ATE Loss: {loss_dict['ate_loss'].item():.6f}")
                print(f"  Final Drift: {loss_dict['final_position_error'].item():.4f}m")
            
            if torch.cuda.is_available():
                print(f"  GPU memory after forward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            if batch_idx >= 0:  # Test only first batch
                break
                
        print("\nSUCCESS! All components working.")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()