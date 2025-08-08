#!/usr/bin/env python3
"""
Quick Start Script for TSformer Visual Odometry

This script provides a simple way to test the complete TSformer-VO pipeline
with your underwater dataset.

Author: Underwater Visual Odometry Research Team
Date: January 2025
"""

import torch
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are available."""
    required_packages = [
        'torch', 'torchvision', 'transformers', 'numpy', 'pandas', 
        'opencv-python', 'PIL', 'sklearn', 'matplotlib', 'tensorboard'
    ]
    
    missing_packages = []
    
    try:
        import torch
        import torchvision
        import transformers
        import numpy
        import pandas
        import cv2
        from PIL import Image
        import sklearn
        import matplotlib
        import tensorboard
        print("All required packages are available")
    except ImportError as e:
        print(f"Missing package: {e}")
        print("\nPlease install requirements:")
        print("pip install -r requirements_tsformer.txt")
        return False
    
    return True

def test_dataset():
    """Test the dataset loading."""
    print("\n" + "="*60)
    print("TESTING DATASET LOADING")
    print("="*60)
    
    try:
        from datasets.underwater_vo_dataset import create_data_loaders
        
        # Create data loaders with small batch size for testing
        data_loaders = create_data_loaders(
            csv_path="data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv",
            data_root="data/processed/visual_odometry_dataset",
            sequence_length=4,  # Smaller for testing
            overlap_frames=2,   # Proper overlap 
            batch_size=2,
            test_bags=["ariel_2023-12-21-14-28-22_4"]
        )
        
        print("Dataset loading successful")
        
        # Test a batch
        train_loader = data_loaders['train_loader']
        for batch in train_loader:
            print(f"Batch shape: images={batch['images'].shape}, poses={batch['poses'].shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        return False

def test_model():
    """Test the TSformer model."""
    print("\n" + "="*60)
    print("TESTING TSFORMER MODEL")
    print("="*60)
    
    try:
        from models.tsformer_vo import create_tsformer_vo
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create model
        model, loss_fn = create_tsformer_vo(sequence_length=4, pretrained=True)
        model = model.to(device)
        
        print(f"Model created successfully")
        print(f"  Parameters: {model.get_num_trainable_parameters():,}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 4
        dummy_images = torch.randn(batch_size, seq_len, 3, 224, 224).to(device)
        dummy_poses = torch.randn(batch_size, 6).to(device)
        
        with torch.no_grad():
            pred_poses = model(dummy_images)
            loss, loss_dict = loss_fn(pred_poses, dummy_poses)
            
        print(f"Forward pass successful")
        print(f"  Input: {dummy_images.shape}")
        print(f"  Output: {pred_poses.shape}")
        print(f"  Loss: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

def run_quick_training():
    """Run a quick training test."""
    print("\n" + "="*60)
    print("QUICK TRAINING TEST (2 epochs)")
    print("="*60)
    
    try:
        import subprocess
        import sys
        
        # Run training for just 2 epochs as a test
        cmd = [
            sys.executable, "train_tsformer.py",
            "--num_epochs", "2",
            "--batch_size", "4",
            "--sequence_length", "4",
            "--log_interval", "5",
            "--output_dir", "test_experiment"
        ]
        
        print("Running command:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("Quick training test successful")
            return True
        else:
            print("Quick training test failed")
            return False
            
    except Exception as e:
        print(f"Training test failed: {e}")
        return False

def main():
    """Main quick start function."""
    print("TSformer Visual Odometry - Quick Start")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("models").exists() or not Path("datasets").exists():
        print("Please run this script from the project root directory")
        print("  Current directory should contain 'models' and 'datasets' folders")
        return
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    # Step 2: Test dataset loading  
    if not test_dataset():
        print("\nDataset test failed. Please check your dataset paths.")
        return
    
    # Step 3: Test model
    if not test_model():
        print("\nModel test failed. Please check your installation.")
        return
    
    print("\nALL TESTS PASSED!")
    print("\nYou can now:")
    print("1. Train the full model:")
    print("   python train_tsformer.py")
    print("\n2. Evaluate a trained model:")
    print("   python evaluate_tsformer.py --checkpoint path/to/checkpoint.pth")
    print("\n3. Run quick training test:")
    response = input("\nWould you like to run a quick 2-epoch training test? (y/n): ")
    if response.lower().startswith('y'):
        run_quick_training()

if __name__ == "__main__":
    main()