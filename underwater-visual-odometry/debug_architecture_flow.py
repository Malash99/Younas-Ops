"""
Debug Architecture Flow - Test if model can produce different outputs per frame
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer.multiscale_uw_transvo import create_multiscale_model

def test_architecture_flow():
    """Test if the model architecture can produce frame-specific outputs"""
    
    print("=" * 60)
    print("ARCHITECTURE FLOW DEBUG")
    print("=" * 60)
    
    # Model configuration
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
    model = create_multiscale_model(config).to(device)
    model.eval()
    
    print(f"1. MODEL CREATED - Parameters: {model.count_parameters():,}")
    
    # Test 1: Identical inputs should produce identical outputs
    print(f"\n2. TEST 1: Identical images across sequence")
    batch_size, seq_len = 1, 5
    
    # Create identical images
    identical_images = torch.randn(1, 1, 3, config['img_size'], config['img_size']).to(device)
    images = identical_images.repeat(batch_size, seq_len, 1, 1, 1, 1)
    camera_ids = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(images=images, camera_ids=camera_ids)
        pred_deltas = outputs['delta_poses'][0].cpu().numpy()
        
        print(f"   Input shape: {images.shape}")
        print(f"   Output shape: {pred_deltas.shape}")
        print(f"   Frame-to-frame predictions:")
        for i in range(seq_len):
            print(f"     Frame {i}: X={pred_deltas[i, 0]:.6f}, Y={pred_deltas[i, 1]:.6f}")
        
        # Check if predictions are identical (this SHOULD happen for identical inputs)
        frame_diffs = np.diff(pred_deltas, axis=0)
        max_diff = np.abs(frame_diffs).max()
        print(f"   Max difference between frames: {max_diff:.8f}")
        if max_diff < 1e-6:
            print(f"   EXPECTED: Identical inputs produce identical outputs")
        else:
            print(f"   UNEXPECTED: Identical inputs produce different outputs")
    
    # Test 2: Different inputs should produce different outputs
    print(f"\n3. TEST 2: Different images across sequence")
    
    # Create different images with significant visual differences
    images_different = torch.randn(batch_size, seq_len, 1, 3, config['img_size'], config['img_size']).to(device)
    # Add increasing intensity per frame
    for i in range(seq_len):
        images_different[:, i] = images_different[:, i] + i * 0.5
    
    with torch.no_grad():
        outputs = model(images=images_different, camera_ids=camera_ids)
        pred_deltas = outputs['delta_poses'][0].cpu().numpy()
        
        print(f"   Input shape: {images_different.shape}")
        print(f"   Output shape: {pred_deltas.shape}")
        print(f"   Frame-to-frame predictions:")
        for i in range(seq_len):
            print(f"     Frame {i}: X={pred_deltas[i, 0]:.6f}, Y={pred_deltas[i, 1]:.6f}")
        
        # Check if predictions are different (this SHOULD happen for different inputs)
        frame_diffs = np.diff(pred_deltas, axis=0)
        max_diff = np.abs(frame_diffs).max()
        print(f"   Max difference between frames: {max_diff:.8f}")
        if max_diff > 1e-4:
            print(f"   EXPECTED: Different inputs produce different outputs")
        else:
            print(f"   PROBLEM: Different inputs produce identical outputs!")
    
    # Test 3: Load trained model and test
    print(f"\n4. TEST 3: Trained model behavior")
    
    model_path = 'multiscale_light_best_model.pth'
    if os.path.exists(model_path):
        print(f"   Loading trained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            # Test with different inputs
            outputs = model(images=images_different, camera_ids=camera_ids)
            pred_deltas = outputs['delta_poses'][0].cpu().numpy()
            
            print(f"   Trained model predictions:")
            for i in range(seq_len):
                print(f"     Frame {i}: X={pred_deltas[i, 0]:.6f}, Y={pred_deltas[i, 1]:.6f}")
            
            # Check frame variance
            frame_diffs = np.diff(pred_deltas, axis=0)
            max_diff = np.abs(frame_diffs).max()
            print(f"   Max difference between frames: {max_diff:.8f}")
            
            if max_diff > 1e-4:
                print(f"   Trained model produces varying predictions")
            else:
                print(f"   TRAINED MODEL PROBLEM: Produces identical predictions!")
                print(f"   This explains the straight-line trajectories!")
    else:
        print(f"   Trained model not found at {model_path}")
    
    print(f"\n" + "=" * 60)
    print("ARCHITECTURE FLOW DEBUG COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_architecture_flow()