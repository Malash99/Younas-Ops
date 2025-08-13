#!/usr/bin/env python3
"""
Debug script to identify training issues step by step
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import UnderwaterVODataset

def test_dataset_loading():
    """Test if dataset loads correctly"""
    print("=== Testing Dataset Loading ===")
    
    try:
        dataset = UnderwaterVODataset(
            csv_path="data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv",
            data_root="data/processed/visual_odometry_dataset",
            sequence_length=8,
            mode='train',
            test_bags=['ariel_2023-12-21-14-28-22_4'],
            camera='cam0'
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Number of windows: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test loading first sample
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Images shape: {sample['images'].shape}")
            print(f"Images dtype: {sample['images'].dtype}")
            print(f"Poses shape: {sample['poses'].shape}")
            print(f"Poses dtype: {sample['poses'].dtype}")
            print(f"Image value range: [{sample['images'].min():.3f}, {sample['images'].max():.3f}]")
            
            return sample
        else:
            print("ERROR: Dataset is empty!")
            return None
            
    except Exception as e:
        print(f"ERROR in dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_creation():
    """Test if model creates correctly"""
    print("\n=== Testing Model Creation ===")
    
    try:
        model, loss_fn = create_tsformer_vo(
            sequence_length=8,
            pretrained=True,
            freeze_backbone=False,
            image_size=224
        )
        
        print("Model created successfully!")
        print(f"Model parameters: {model.get_num_parameters():,}")
        return model, loss_fn
        
    except Exception as e:
        print(f"ERROR in model creation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_single_forward_pass(model, sample):
    """Test single forward pass"""
    print("\n=== Testing Forward Pass ===")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = model.to(device)
        model.eval()
        
        # Prepare input
        images = sample['images'].unsqueeze(0).to(device)  # Add batch dimension
        poses = sample['poses'].unsqueeze(0).to(device)
        
        print(f"Input images shape: {images.shape}")
        print(f"Input poses shape: {poses.shape}")
        print(f"Images on device: {images.device}")
        print(f"Images dtype: {images.dtype}")
        
        # Check image value range
        print(f"Image values - min: {images.min():.3f}, max: {images.max():.3f}, mean: {images.mean():.3f}")
        
        # Forward pass
        with torch.no_grad():
            print("Starting forward pass...")
            pred_poses = model(images)
            print(f"Forward pass successful!")
            print(f"Output shape: {pred_poses.shape}")
            print(f"Output dtype: {pred_poses.dtype}")
            print(f"Sample prediction: {pred_poses[0]}")
            
        return True
        
    except Exception as e:
        print(f"ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing(model, dataset):
    """Test batch processing"""
    print("\n=== Testing Batch Processing ===")
    
    try:
        from torch.utils.data import DataLoader
        
        # Create small batch loader
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        batch = next(iter(loader))
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Batch poses shape: {batch['poses'].shape}")
        
        images = batch['images'].to(device)
        poses = batch['poses'].to(device)
        
        with torch.no_grad():
            pred_poses = model(images)
            print(f"Batch forward pass successful!")
            print(f"Batch output shape: {pred_poses.shape}")
            
        return True
        
    except Exception as e:
        print(f"ERROR in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("TSformer Training Debug Script")
    print("=" * 50)
    
    # Test 1: Dataset loading
    sample = test_dataset_loading()
    if sample is None:
        return
    
    # Test 2: Model creation
    model, loss_fn = test_model_creation()
    if model is None:
        return
        
    # Test 3: Single forward pass
    success = test_single_forward_pass(model, sample)
    if not success:
        return
        
    # Test 4: Batch processing
    dataset = UnderwaterVODataset(
        csv_path="data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv",
        data_root="data/processed/visual_odometry_dataset",
        sequence_length=8,
        mode='train',
        test_bags=['ariel_2023-12-21-14-28-22_4'],
        camera='cam0'
    )
    
    success = test_batch_processing(model, dataset)
    
    if success:
        print("\n" + "=" * 50)
        print("✅ All tests passed! Training should work.")
    else:
        print("\n" + "=" * 50)
        print("❌ Tests failed. Check errors above.")

if __name__ == "__main__":
    main()