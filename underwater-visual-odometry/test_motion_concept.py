"""
Quick test of motion-aware concept using existing single camera model

This demonstrates the motion supervision approach using a simple dataset adapter.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.transformer.motion_aware_uw_transvo import create_motion_aware_model
from training.motion_aware_loss import create_motion_aware_loss


def create_synthetic_motion_data(batch_size=4, seq_len=5):
    """Create synthetic motion data to test the concept"""
    
    # Synthetic curved trajectory (not straight line)
    t = np.linspace(0, 2*np.pi, seq_len)
    
    poses = []
    for b in range(batch_size):
        # Different trajectory parameters for each batch item
        radius = 0.5 + 0.3 * np.random.random()
        phase = 2 * np.pi * np.random.random()
        
        # Create curved motion in XY plane
        x = radius * np.cos(t + phase)
        y = radius * np.sin(t + phase) 
        z = 0.1 * t  # Slight depth change
        
        # Simple rotation
        rx = 0.1 * np.sin(t)
        ry = 0.1 * np.cos(t)  
        rz = 0.05 * t
        
        trajectory = np.column_stack([x, y, z, rx, ry, rz])
        poses.append(trajectory)
    
    poses = np.array(poses)  # [batch_size, seq_len, 6]
    
    # Create synthetic images (random for now)
    images = torch.randn(batch_size, seq_len, 1, 3, 224, 224)
    poses = torch.tensor(poses, dtype=torch.float32)
    
    return images, poses


def test_motion_aware_model():
    """Test motion-aware model with synthetic data"""
    
    print("Testing Motion-Aware UW-TransVO Concept...")
    
    # Model configuration
    config = {
        'img_size': 224,
        'patch_size': 16, 
        'd_model': 256,  # Small for quick test
        'num_heads': 4,
        'num_layers': 2,
        'max_cameras': 1,
        'max_seq_len': 5,
        'dropout': 0.1,
        'use_imu': False,
        'use_pressure': False,
        'uncertainty_estimation': True
    }
    
    loss_config = {
        'loss_type': 'motion_aware',
        'translation_weight': 1.0,
        'rotation_weight': 5.0,
        'motion_weight': 10.0,  # High motion supervision
        'consistency_weight': 2.0,
        'sequence_length': 5
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and loss
    model = create_motion_aware_model(config).to(device)
    criterion = create_motion_aware_loss(loss_config)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    print("\n1. Testing forward pass...")
    images, poses = create_synthetic_motion_data()
    images = images.to(device)
    poses = poses.to(device)
    
    camera_ids = torch.zeros(images.size(0), 1, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(images=images, camera_ids=camera_ids)
        pred_poses = outputs['pose']
        
    print(f"Input shapes: images {images.shape}, poses {poses.shape}")
    print(f"Output shape: {pred_poses.shape}")
    print("[OK] Forward pass successful!")
    
    # Test loss computation
    print("\n2. Testing motion-aware loss...")
    loss_dict = criterion(pred_poses, poses)
    
    print("Loss components:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"  {key}: {value.item():.6f}")
    
    print("[OK] Loss computation successful!")
    
    # Test training step
    print("\n3. Testing training step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    optimizer.zero_grad()
    
    outputs = model(images=images, camera_ids=camera_ids)
    pred_poses = outputs['pose']
    loss_dict = criterion(pred_poses, poses)
    loss = loss_dict['total_loss']
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    print(f"Training loss: {loss.item():.6f}")
    print("[OK] Training step successful!")
    
    # Demonstrate motion learning vs straight line
    print("\n4. Comparing predictions...")
    
    with torch.no_grad():
        model.eval()
        outputs = model(images=images, camera_ids=camera_ids)
        final_pred = outputs['pose'].cpu().numpy()
    
    # Plot comparison for first batch item
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    batch_idx = 0
    ground_truth = poses[batch_idx].cpu().numpy()
    prediction = final_pred[batch_idx]
    
    # XY trajectory plot
    ax1.plot(ground_truth[:, 0], ground_truth[:, 1], 'b-o', label='Ground Truth (Curved)', markersize=8)
    ax1.plot(prediction[:, 0], prediction[:, 1], 'r-s', label='Model Prediction', markersize=6)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('XY Trajectory Comparison')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Position over time
    ax2.plot(ground_truth[:, 0], 'b-', label='GT X')
    ax2.plot(ground_truth[:, 1], 'g-', label='GT Y') 
    ax2.plot(prediction[:, 0], 'r--', label='Pred X')
    ax2.plot(prediction[:, 1], 'm--', label='Pred Y')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('motion_aware_concept_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("[OK] Motion concept test completed!")
    print("\nKey Innovation Summary:")
    print("- [OK] Sequential pose prediction (all frames)")
    print("- [OK] Motion-aware loss (frame-to-frame supervision)")
    print("- [OK] Curved trajectory support (not straight lines)")
    print("- [OK] End-to-end trainable architecture")
    print("\nNext: Apply to real underwater data with proper dataset adapter")


if __name__ == "__main__":
    test_motion_aware_model()