#!/usr/bin/env python3
"""
Debug NaN Issue - Find Root Cause
Investigate exactly what's causing NaN in training
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.sub_trajectory_dataset import create_sub_trajectory_dataloaders
from training.trajectory_losses import TrajectoryAwareLoss

def debug_model_forward():
    """Debug model forward pass for NaN sources"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Debugging on {device}")
    
    # Simple config
    config = {
        'img_size': 224,
        'patch_size': 16,
        'd_model': 384,  # Smaller model
        'num_heads': 6,   # Fewer heads
        'num_layers': 3,  # Fewer layers
        'max_cameras': 1,
        'max_seq_len': 5,
        'dropout': 0.0,   # No dropout
        'use_imu': False,
        'use_pressure': False,
        'uncertainty_estimation': False  # Disable uncertainty
    }
    
    print("🔧 Creating simplified model...")
    model = UWTransVO(**config).to(device)
    model.train()
    
    # Check initial parameters
    print("📊 Checking initial parameters...")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ NaN in initial parameter: {name}")
        if torch.isinf(param).any():
            print(f"❌ Inf in initial parameter: {name}")
        param_stats = {
            'mean': param.mean().item(),
            'std': param.std().item(),
            'min': param.min().item(),
            'max': param.max().item()
        }
        if abs(param_stats['mean']) > 10 or param_stats['std'] > 10:
            print(f"⚠️  Large parameter values in {name}: {param_stats}")
    
    print("✅ Parameter check complete")
    
    # Create simple loss
    criterion = TrajectoryAwareLoss(
        translation_weight=1.0,
        rotation_weight=1.0,  # Reduced
        ate_weight=1.0,       # Reduced
        consistency_weight=0.1,  # Much smaller
        smoothness_weight=0.1    # Much smaller
    )
    
    # Load one batch
    print("📂 Loading data...")
    try:
        train_loader, _ = create_sub_trajectory_dataloaders(
            train_csv='data/processed/training_dataset/training_data_filtered.csv',
            val_csv='data/processed/training_dataset/training_data_filtered.csv',
            sub_trajectory_length=5,
            overlap=2,
            camera_ids=[0],
            batch_size=1,
            num_workers=0,
            max_samples_train=10,  # Only 10 samples for debugging
            max_samples_val=5
        )
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    print("🔍 Testing batches one by one...")
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"\n--- BATCH {batch_idx} ---")
        
        # Move to device
        images = batch['images'].to(device)
        camera_ids = batch['camera_ids'].to(device)
        camera_mask = batch['camera_mask'].to(device)
        pose_targets = batch['pose_targets'].to(device)
        accumulated_targets = batch['accumulated_poses'].to(device)
        
        # Check input data
        print("📊 Input data check:")
        for name, tensor in [('images', images), ('pose_targets', pose_targets), ('accumulated_targets', accumulated_targets)]:
            if torch.isnan(tensor).any():
                print(f"❌ NaN in {name}")
                continue
            if torch.isinf(tensor).any():
                print(f"❌ Inf in {name}")
                continue
            stats = {
                'shape': list(tensor.shape),
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item()
            }
            print(f"  {name}: {stats}")
        
        sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
        
        try:
            # Forward pass with detailed monitoring
            print("🔄 Forward pass...")
            
            # Step 1: Model forward
            with torch.no_grad():  # No gradients for debugging
                predictions = model(images, camera_ids, camera_mask, sub_traj_length)
            
            print(f"📊 Predictions: shape={predictions['pose'].shape}, mean={predictions['pose'].mean().item():.6f}")
            
            if torch.isnan(predictions['pose']).any():
                print("❌ NaN in model predictions!")
                break
            
            # Step 2: Loss computation
            print("🧮 Computing loss...")
            loss_dict = criterion(predictions['pose'], pose_targets, accumulated_targets)
            
            print("📊 Loss components:")
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    val = value.item() if value.numel() == 1 else value.mean().item()
                    print(f"  {key}: {val:.8f}")
                    
                    if torch.isnan(value).any():
                        print(f"❌ NaN detected in {key}!")
                        
                        # Debug this loss component
                        if key == 'ate_loss':
                            print("🔍 Debugging ATE loss...")
                            pred_acc = loss_dict.get('predicted_accumulated')
                            target_acc = loss_dict.get('target_accumulated')
                            if pred_acc is not None and target_acc is not None:
                                print(f"  Pred acc range: [{pred_acc.min().item():.6f}, {pred_acc.max().item():.6f}]")
                                print(f"  Target acc range: [{target_acc.min().item():.6f}, {target_acc.max().item():.6f}]")
                                diff = pred_acc - target_acc
                                print(f"  Diff range: [{diff.min().item():.6f}, {diff.max().item():.6f}]")
                        return batch_idx  # Return problematic batch index
            
            if torch.isnan(loss_dict['total_loss']):
                print(f"❌ Total loss is NaN at batch {batch_idx}!")
                return batch_idx
            
            print(f"✅ Batch {batch_idx} OK")
            
        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")
            return batch_idx
        
        if batch_idx >= 5:  # Test first 6 batches
            break
    
    print("✅ All tested batches passed!")
    return None

def test_simplified_architecture():
    """Test with even simpler architecture"""
    
    print("\n🧪 Testing minimal architecture...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ultra-minimal config
    config = {
        'img_size': 224,
        'patch_size': 32,  # Larger patches = fewer tokens
        'd_model': 192,    # Very small
        'num_heads': 3,    # Minimal heads
        'num_layers': 2,   # Minimal layers
        'max_cameras': 1,
        'max_seq_len': 3,  # Very short sequences
        'dropout': 0.0,
        'use_imu': False,
        'use_pressure': False,
        'uncertainty_estimation': False
    }
    
    model = UWTransVO(**config).to(device)
    print(f"📊 Minimal model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy data
    batch_size = 1
    seq_len = 3
    images = torch.randn(batch_size, seq_len, 1, 3, 224, 224, device=device) * 0.1  # Small values
    camera_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    camera_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
    
    try:
        with torch.no_grad():
            output = model(images, camera_ids, camera_mask)
        print(f"✅ Minimal model works! Output shape: {output['pose'].shape}")
        return True
    except Exception as e:
        print(f"❌ Even minimal model fails: {e}")
        return False

if __name__ == '__main__':
    print("DEBUGGING NaN ISSUE")
    print("=" * 50)
    
    # Test 1: Find problematic batch
    problematic_batch = debug_model_forward()
    
    if problematic_batch is not None:
        print(f"\n🎯 Found problematic batch: {problematic_batch}")
    
    # Test 2: Try minimal architecture
    minimal_works = test_simplified_architecture()
    
    print(f"\n📋 SUMMARY:")
    print(f"  Problematic batch: {problematic_batch}")
    print(f"  Minimal model works: {minimal_works}")
    
    if minimal_works and problematic_batch is not None:
        print("💡 RECOMMENDATION: Use simpler architecture or investigate loss function")
    elif not minimal_works:
        print("💡 RECOMMENDATION: Fundamental model architecture issue")
    else:
        print("💡 RECOMMENDATION: Check data quality or loss function")