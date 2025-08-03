#!/usr/bin/env python3
"""
Simple test of sub-trajectory dataset and model
No web dashboard - just console output
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data.sub_trajectory_dataset import create_sub_trajectory_dataloaders
from models.transformer import UWTransVO
from training.trajectory_losses import TrajectoryAwareLoss

def test_dataset():
    """Test the sub-trajectory dataset"""
    print("Testing Sub-Trajectory Dataset...")
    
    try:
        train_loader, val_loader = create_sub_trajectory_dataloaders(
            train_csv='data/processed/training_dataset/training_data.csv',
            val_csv='data/processed/training_dataset/training_data.csv',
            sub_trajectory_length=5,
            overlap=2,
            camera_ids=[0, 1, 2],
            batch_size=1,
            num_workers=0,
            max_samples_train=5,  # Very small test
            max_samples_val=2
        )
        
        print(f"✅ Dataset created successfully")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
        
        # Test loading one batch
        for batch_idx, batch in enumerate(train_loader):
            print(f"\n✅ Batch {batch_idx} loaded:")
            print(f"   Images shape: {batch['images'].shape}")
            print(f"   Pose targets shape: {batch['pose_targets'].shape}")
            print(f"   Accumulated poses shape: {batch['accumulated_poses'].shape}")
            
            # Test memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"   GPU memory: {memory_used:.2f} GB")
            
            if batch_idx >= 0:  # Test only first batch
                break
                
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def test_model():
    """Test the model with sub-trajectory data"""
    print("\nTesting Model...")
    
    try:
        # Create model
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
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        # Test basic model creation
        base_model = UWTransVO(**config).to(device)
        print(f"✅ Base model created: {sum(p.numel() for p in base_model.parameters()):,} parameters")
        
        # Test with dummy data
        batch_size = 1
        seq_len = 5
        num_cameras = 3
        
        # Create dummy batch
        dummy_images = torch.randn(batch_size, seq_len, num_cameras, 3, 224, 224).to(device)
        dummy_camera_ids = torch.tensor([0, 1, 2]).to(device)
        dummy_camera_mask = torch.tensor([False, False, False]).to(device)
        
        print(f"   Dummy input shape: {dummy_images.shape}")
        
        # Test memory before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1024**3
            print(f"   Memory before forward: {memory_before:.2f} GB")
        
        # Test forward pass (frame by frame)
        base_model.eval()
        with torch.no_grad():
            predictions = []
            for t in range(seq_len - 1):
                frame_pair = torch.stack([dummy_images[:, t], dummy_images[:, t+1]], dim=1)
                output = base_model(
                    images=frame_pair,
                    camera_ids=dummy_camera_ids,
                    camera_mask=dummy_camera_mask
                )
                predictions.append(output['pose'])
            
            predictions = torch.stack(predictions, dim=1)
            print(f"✅ Forward pass successful: {predictions.shape}")
            
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**3
                print(f"   Memory after forward: {memory_after:.2f} GB")
                print(f"   Memory increase: {memory_after - memory_before:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss():
    """Test the trajectory loss function"""
    print("\nTesting Loss Function...")
    
    try:
        criterion = TrajectoryAwareLoss()
        
        # Create dummy data
        batch_size = 1
        seq_len = 4  # 5 frames = 4 predictions
        
        predictions = torch.randn(batch_size, seq_len, 6)
        pose_targets = torch.randn(batch_size, seq_len, 6)
        accumulated_targets = torch.randn(batch_size, seq_len, 6)
        
        loss_dict = criterion(predictions, pose_targets, accumulated_targets)
        
        print(f"✅ Loss calculation successful:")
        print(f"   Total loss: {loss_dict['total_loss'].item():.6f}")
        print(f"   Translation loss: {loss_dict['translation_loss'].item():.6f}")
        print(f"   ATE loss: {loss_dict['ate_loss'].item():.6f}")
        print(f"   Final drift: {loss_dict['final_position_error'].item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        return False

def main():
    print("Sub-Trajectory System Test")
    print("=" * 40)
    
    # Test components individually
    dataset_ok = test_dataset()
    model_ok = test_model()
    loss_ok = test_loss()
    
    print(f"\n" + "=" * 40)
    print("Test Results:")
    print(f"Dataset: {'✅ PASS' if dataset_ok else '❌ FAIL'}")
    print(f"Model: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Loss: {'✅ PASS' if loss_ok else '❌ FAIL'}")
    
    if all([dataset_ok, model_ok, loss_ok]):
        print("\n🎉 All tests passed! Ready for training.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == '__main__':
    main()