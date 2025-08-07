"""
Test the Fixed Anti-Collapse Model

This script tests whether the fixed model produces varying frame-specific predictions
instead of constant values across all frames.
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer.multiscale_uw_transvo import create_multiscale_model

def test_fixed_model():
    """Test if the fixed model produces frame-specific outputs"""
    
    print("=" * 60)
    print("TESTING FIXED ANTI-COLLAPSE MODEL")
    print("=" * 60)
    
    # Model configuration (should match training)
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
    
    # Load the fixed model
    model_path = 'fixed_anti_collapse_model.pth'
    if not os.path.exists(model_path):
        print(f"ERROR: Fixed model not found at {model_path}")
        return
    
    print(f"Loading fixed model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully! Validation loss during training: {checkpoint['val_loss']:.6f}")
    
    # Test with different visual inputs
    print(f"\\nTEST: Different images across sequence")
    batch_size, seq_len = 1, 10
    
    # Create different images with significant visual differences
    images = torch.randn(batch_size, seq_len, 1, 3, config['img_size'], config['img_size']).to(device)
    # Add increasing intensity per frame to create visual differences
    for i in range(seq_len):
        images[:, i] = images[:, i] + i * 0.3
    
    camera_ids = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(images=images, camera_ids=camera_ids)
        pred_deltas = outputs['delta_poses'][0].cpu().numpy()  # [seq_len, 6]
    
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {pred_deltas.shape}")
    print(f"\\nFrame-by-frame predictions:")
    for i in range(seq_len):
        print(f"  Frame {i}: X={pred_deltas[i, 0]:8.6f}, Y={pred_deltas[i, 1]:8.6f}, Z={pred_deltas[i, 2]:8.6f}")
    
    # Analyze frame variation
    frame_diffs = np.diff(pred_deltas, axis=0)
    max_diff = np.abs(frame_diffs).max()
    std_per_axis = np.std(pred_deltas, axis=0)
    
    print(f"\\nVARIATION ANALYSIS:")
    print(f"  Max difference between consecutive frames: {max_diff:.8f}")
    print(f"  Standard deviation per axis: X={std_per_axis[0]:.6f}, Y={std_per_axis[1]:.6f}, Z={std_per_axis[2]:.6f}")
    print(f"  Mean std across translation axes: {np.mean(std_per_axis[:3]):.6f}")
    
    # Check if the model produces varying predictions
    if max_diff > 1e-4:
        print(f"\\nSUCCESS: Fixed model produces VARYING predictions!")
        print(f"  The model now generates different predictions for different frames.")
        if np.mean(std_per_axis[:3]) > 0.001:
            print(f"  Good diversity: Mean translation std = {np.mean(std_per_axis[:3]):.6f}")
        else:
            print(f"  Low diversity: Mean translation std = {np.mean(std_per_axis[:3]):.6f}")
    else:
        print(f"\\nPROBLEM: Fixed model still produces CONSTANT predictions!")
        print(f"  All frames have nearly identical outputs.")
        
    # Test trajectory shape
    trajectory = np.cumsum(pred_deltas[:, :3], axis=0)
    trajectory_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
    curvature_ratio = trajectory_length / (direct_distance + 1e-8)
    
    print(f"\\nTRAJECTORY ANALYSIS:")
    print(f"  Start position: [{trajectory[0, 0]:8.6f}, {trajectory[0, 1]:8.6f}, {trajectory[0, 2]:8.6f}]")
    print(f"  End position:   [{trajectory[-1, 0]:8.6f}, {trajectory[-1, 1]:8.6f}, {trajectory[-1, 2]:8.6f}]")
    print(f"  Trajectory length: {trajectory_length:.6f}m")
    print(f"  Direct distance:   {direct_distance:.6f}m")
    print(f"  Curvature ratio:   {curvature_ratio:.4f}")
    
    if curvature_ratio > 1.05:
        print(f"  CURVED trajectory detected!")
    elif curvature_ratio > 1.01:
        print(f"  ~ Slightly curved trajectory")
    else:
        print(f"  - Nearly straight trajectory")
    
    # Compare with old multiscale model if available
    old_model_path = 'multiscale_light_best_model.pth'
    if os.path.exists(old_model_path):
        print(f"\\nCOMPARISON WITH OLD MODEL:")
        print(f"Loading old model from {old_model_path}")
        
        old_checkpoint = torch.load(old_model_path, map_location=device, weights_only=False)
        model.load_state_dict(old_checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            old_outputs = model(images=images, camera_ids=camera_ids)
            old_pred_deltas = old_outputs['delta_poses'][0].cpu().numpy()
        
        old_max_diff = np.abs(np.diff(old_pred_deltas, axis=0)).max()
        old_std = np.mean(np.std(old_pred_deltas[:, :3], axis=0))
        
        print(f"  Old model max frame difference: {old_max_diff:.8f}")
        print(f"  Old model mean std: {old_std:.8f}")
        print(f"  New model max frame difference: {max_diff:.8f}")
        print(f"  New model mean std: {np.mean(std_per_axis[:3]):.8f}")
        
        improvement_ratio = max_diff / (old_max_diff + 1e-10)
        print(f"  Improvement ratio: {improvement_ratio:.2f}x")
        
        if improvement_ratio > 10:
            print(f"  MAJOR IMPROVEMENT: New model is {improvement_ratio:.1f}x more diverse!")
        elif improvement_ratio > 2:
            print(f"  Good improvement: New model is {improvement_ratio:.1f}x more diverse")
        else:
            print(f"  - Limited improvement: Only {improvement_ratio:.1f}x more diverse")
    
    print(f"\\n" + "=" * 60)
    print("FIXED MODEL TEST COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    test_fixed_model()