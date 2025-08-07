"""
Test Scale and Direction Fixed Model

This script tests our new model with scale prediction and direction consistency
to see if it fixes the 5.3x magnitude and backward/forward direction issues.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer.multiscale_uw_transvo import create_multiscale_model

class TestDataset(Dataset):
    """Simple dataset for testing"""
    
    def __init__(self, csv_file, sequence_length=10, img_size=192, max_sequences=5):
        self.csv_file = csv_file
        self.sequence_length = sequence_length
        self.img_size = img_size
        
        # Load data
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} frames from {csv_file}")
        
        # Get first few sequences with good motion
        self.sequences = []
        for i in range(0, min(len(self.df) - sequence_length + 1, max_sequences * 10), 10):
            seq_data = self.df.iloc[i:i + sequence_length]
            deltas = seq_data[['delta_x', 'delta_y', 'delta_z']].values
            if np.sum(np.std(deltas, axis=0)) > 0.002:
                self.sequences.append(i)
                if len(self.sequences) >= max_sequences:
                    break
        
        print(f"Using {len(self.sequences)} test sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        start_idx = self.sequences[idx]
        sequence_data = self.df.iloc[start_idx:start_idx + self.sequence_length]
        
        images = []
        delta_poses = []
        
        for _, row in sequence_data.iterrows():
            # Handle image loading with robust fallbacks
            img_path = row['cam0_path']
            if pd.isna(img_path):
                img = np.random.rand(self.img_size, self.img_size, 3) * 0.1
            else:
                if not os.path.exists(str(img_path)):
                    img_path = os.path.join(".", str(img_path))
                
                if os.path.exists(str(img_path)):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (self.img_size, self.img_size))
                            img = img.astype(np.float32) / 255.0
                        else:
                            img = np.random.rand(self.img_size, self.img_size, 3) * 0.1
                    except:
                        img = np.random.rand(self.img_size, self.img_size, 3) * 0.1
                else:
                    img = np.random.rand(self.img_size, self.img_size, 3) * 0.1
            
            images.append(img.astype(np.float32))
            
            # Extract delta pose
            delta_pose = np.array([
                float(row.get('delta_x', 0.0)), float(row.get('delta_y', 0.0)), float(row.get('delta_z', 0.0)),
                float(row.get('delta_roll', 0.0)), float(row.get('delta_pitch', 0.0)), float(row.get('delta_yaw', 0.0))
            ], dtype=np.float32)
            delta_poses.append(delta_pose)
        
        # Convert to tensors
        images = np.stack(images)
        images = torch.tensor(images).permute(0, 3, 1, 2)  # [seq_len, 3, H, W]
        delta_poses = torch.tensor(np.stack(delta_poses))  # [seq_len, 6]
        
        # Add camera dimension and transpose: [1, seq_len, 3, H, W]
        images = images.unsqueeze(1).transpose(0, 1)
        
        return images, delta_poses, start_idx


def poses_to_trajectory(delta_poses, start_pos=np.array([0.0, 0.0, 0.0])):
    """Convert delta poses to cumulative trajectory"""
    trajectory = [start_pos.copy()]
    current_pos = start_pos.copy()
    
    for delta in delta_poses:
        current_pos = current_pos + delta[:3]  # Only translation
        trajectory.append(current_pos.copy())
    
    return np.array(trajectory)


def trajectory_analysis(pred_traj, gt_traj):
    """Analyze trajectory characteristics"""
    def trajectory_length(traj):
        return np.sum([np.linalg.norm(traj[i+1] - traj[i]) for i in range(len(traj)-1)])
    
    def trajectory_curvature(traj):
        if len(traj) < 3:
            return 0.0
        
        # Calculate curvature using three consecutive points
        curvatures = []
        for i in range(1, len(traj)-1):
            p1, p2, p3 = traj[i-1], traj[i], traj[i+1]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Skip if vectors are too small
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
                
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            
            curvatures.append(angle)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    pred_length = trajectory_length(pred_traj)
    gt_length = trajectory_length(gt_traj)
    
    pred_curvature = trajectory_curvature(pred_traj)
    gt_curvature = trajectory_curvature(gt_traj)
    
    return {
        'pred_length': pred_length,
        'gt_length': gt_length,
        'length_ratio': pred_length / (gt_length + 1e-8),
        'pred_curvature': pred_curvature,
        'gt_curvature': gt_curvature,
        'curvature_ratio': pred_curvature / (gt_curvature + 1e-8)
    }


def test_scale_direction_model():
    """Test our improved model with scale and direction fixes"""
    print("=" * 70)
    print("TESTING SCALE & DIRECTION FIXED MODEL")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration
    config = {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'max_seq_len': 10,
        'uncertainty_estimation': False,
        'image_size': 192
    }
    
    # Create model
    model = create_multiscale_model(config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load trained weights if available
    model_path = 'scale_direction_fixed_model.pth'
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"SUCCESS: Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        except Exception as e:
            print(f"WARNING: Could not load trained model: {e}")
            print("Using fresh model for testing")
    else:
        print("WARNING: No trained model found, using fresh model")
    
    # Create test dataset
    test_dataset = TestDataset(
        'data/processed/training_dataset/training_data.csv',
        sequence_length=10,
        max_sequences=3
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Test model
    model.eval()
    results = []
    
    print("\n" + "="*50)
    print("MODEL PREDICTIONS vs GROUND TRUTH")
    print("="*50)
    
    with torch.no_grad():
        for batch_idx, (images, gt_poses, start_idx) in enumerate(test_loader):
            images = images.to(device)
            gt_poses = gt_poses.to(device)
            
            # Forward pass with camera IDs
            batch_size, seq_len = images.shape[1:3]
            camera_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            
            try:
                outputs = model(images, camera_ids)
                
                # Get predictions
                pred_deltas = outputs['delta_poses'].cpu().numpy()[0]  # [seq_len, 6]
                gt_deltas = gt_poses.cpu().numpy()[0]  # [seq_len, 6]
                
                # Get scale information if available
                scale_info = ""
                if 'delta_scales' in outputs:
                    scales = outputs['delta_scales'].cpu().numpy()[0]  # [seq_len]
                    scale_info = f"Scale factors: {scales.mean():.3f} ± {scales.std():.3f}"
                
                # Convert to trajectories
                pred_traj = poses_to_trajectory(pred_deltas)
                gt_traj = poses_to_trajectory(gt_deltas)
                
                # Analyze trajectories
                analysis = trajectory_analysis(pred_traj, gt_traj)
                
                # Store results
                results.append({
                    'seq_idx': batch_idx,
                    'pred_deltas': pred_deltas,
                    'gt_deltas': gt_deltas,
                    'pred_traj': pred_traj,
                    'gt_traj': gt_traj,
                    'analysis': analysis,
                    'scale_info': scale_info
                })
                
                print(f"\nSequence {batch_idx + 1}:")
                print(f"  Start frame: {start_idx.item()}")
                
                # Frame-by-frame comparison (first few frames)
                print("  Frame-by-frame deltas (X, Y, Z):")
                for i in range(min(5, len(pred_deltas))):
                    pred_xyz = pred_deltas[i, :3]
                    gt_xyz = gt_deltas[i, :3]
                    error = np.linalg.norm(pred_xyz - gt_xyz)
                    print(f"    Frame {i}: GT=[{gt_xyz[0]:+.6f}, {gt_xyz[1]:+.6f}, {gt_xyz[2]:+.6f}]")
                    print(f"             Pred=[{pred_xyz[0]:+.6f}, {pred_xyz[1]:+.6f}, {pred_xyz[2]:+.6f}] Error={error:.6f}")
                
                # Magnitude analysis
                pred_magnitudes = np.linalg.norm(pred_deltas[:, :3], axis=1)
                gt_magnitudes = np.linalg.norm(gt_deltas[:, :3], axis=1)
                magnitude_ratio = pred_magnitudes.mean() / (gt_magnitudes.mean() + 1e-8)
                
                print(f"  Magnitude Analysis:")
                print(f"    GT mean magnitude: {gt_magnitudes.mean():.6f}")
                print(f"    Pred mean magnitude: {pred_magnitudes.mean():.6f}")
                print(f"    Ratio (Pred/GT): {magnitude_ratio:.3f} {'✅ GOOD' if 0.5 < magnitude_ratio < 2.0 else '❌ BAD'}")
                
                # Direction analysis
                pred_direction = np.sign(pred_deltas[:, 0].mean())  # X direction
                gt_direction = np.sign(gt_deltas[:, 0].mean())
                direction_match = pred_direction == gt_direction
                
                print(f"  Direction Analysis:")
                print(f"    GT X direction: {'+' if gt_direction > 0 else '-'} ({'forward' if gt_direction > 0 else 'backward'})")
                print(f"    Pred X direction: {'+' if pred_direction > 0 else '-'} ({'forward' if pred_direction > 0 else 'backward'})")
                print(f"    Direction match: {'✅ CORRECT' if direction_match else '❌ WRONG'}")
                
                # Trajectory analysis
                print(f"  Trajectory Analysis:")
                print(f"    GT length: {analysis['gt_length']:.4f}m")
                print(f"    Pred length: {analysis['pred_length']:.4f}m")
                print(f"    Length ratio: {analysis['length_ratio']:.3f}")
                print(f"    Curvature ratio: {analysis['curvature_ratio']:.3f}")
                
                if scale_info:
                    print(f"  {scale_info}")
                
            except Exception as e:
                print(f"❌ Error processing sequence {batch_idx}: {e}")
                continue
    
    if not results:
        print("❌ No successful predictions to analyze")
        return
    
    # Create comprehensive visualization
    print(f"\n{'='*50}")
    print("CREATING TRAJECTORY VISUALIZATION")
    print("="*50)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: X-Y trajectories
    ax = axes[0, 0]
    for i, result in enumerate(results):
        pred_traj = result['pred_traj']
        gt_traj = result['gt_traj']
        
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', linewidth=2, alpha=0.7, label=f'GT {i+1}' if i == 0 else "")
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, alpha=0.7, label=f'Pred {i+1}' if i == 0 else "")
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], color='green', s=100, marker='o', zorder=5)
        ax.scatter(pred_traj[0, 0], pred_traj[0, 1], color='orange', s=100, marker='s', zorder=5)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('X-Y Trajectories\n(Green=GT start, Orange=Pred start)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: X-Z trajectories
    ax = axes[0, 1]
    for i, result in enumerate(results):
        pred_traj = result['pred_traj']
        gt_traj = result['gt_traj']
        
        ax.plot(gt_traj[:, 0], gt_traj[:, 2], 'b-', linewidth=2, alpha=0.7, label=f'GT {i+1}' if i == 0 else "")
        ax.plot(pred_traj[:, 0], pred_traj[:, 2], 'r--', linewidth=2, alpha=0.7, label=f'Pred {i+1}' if i == 0 else "")
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_title('X-Z Trajectories (Side View)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 3: Frame-by-frame deltas (X component)
    ax = axes[0, 2]
    for i, result in enumerate(results):
        pred_deltas = result['pred_deltas']
        gt_deltas = result['gt_deltas']
        frames = np.arange(len(pred_deltas))
        
        ax.plot(frames, gt_deltas[:, 0], 'b-', linewidth=2, alpha=0.7, label=f'GT X {i+1}' if i == 0 else "")
        ax.plot(frames, pred_deltas[:, 0], 'r--', linewidth=2, alpha=0.7, label=f'Pred X {i+1}' if i == 0 else "")
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('X Delta (meters/frame)')
    ax.set_title('X Motion per Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot 4: Magnitude comparison
    ax = axes[1, 0]
    for i, result in enumerate(results):
        pred_deltas = result['pred_deltas']
        gt_deltas = result['gt_deltas']
        frames = np.arange(len(pred_deltas))
        
        pred_mags = np.linalg.norm(pred_deltas[:, :3], axis=1)
        gt_mags = np.linalg.norm(gt_deltas[:, :3], axis=1)
        
        ax.plot(frames, gt_mags, 'b-', linewidth=2, alpha=0.7, label=f'GT Mag {i+1}' if i == 0 else "")
        ax.plot(frames, pred_mags, 'r--', linewidth=2, alpha=0.7, label=f'Pred Mag {i+1}' if i == 0 else "")
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Motion Magnitude (meters/frame)')
    ax.set_title('Motion Magnitude per Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative error
    ax = axes[1, 1]
    for i, result in enumerate(results):
        pred_traj = result['pred_traj']
        gt_traj = result['gt_traj']
        
        errors = np.linalg.norm(pred_traj - gt_traj, axis=1)
        cumulative_error = np.cumsum(errors)
        frames = np.arange(len(errors))
        
        ax.plot(frames, cumulative_error, linewidth=2, label=f'Seq {i+1}', alpha=0.7)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Cumulative Error (meters)')
    ax.set_title('Cumulative Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax = axes[1, 2]
    
    # Calculate summary statistics
    magnitude_ratios = []
    direction_matches = []
    length_ratios = []
    
    for result in results:
        pred_deltas = result['pred_deltas']
        gt_deltas = result['gt_deltas']
        
        pred_mag = np.linalg.norm(pred_deltas[:, :3], axis=1).mean()
        gt_mag = np.linalg.norm(gt_deltas[:, :3], axis=1).mean()
        magnitude_ratios.append(pred_mag / (gt_mag + 1e-8))
        
        pred_dir = np.sign(pred_deltas[:, 0].mean())
        gt_dir = np.sign(gt_deltas[:, 0].mean())
        direction_matches.append(1.0 if pred_dir == gt_dir else 0.0)
        
        length_ratios.append(result['analysis']['length_ratio'])
    
    metrics = ['Magnitude\nRatio', 'Direction\nMatch', 'Length\nRatio']
    values = [np.mean(magnitude_ratios), np.mean(direction_matches), np.mean(length_ratios)]
    targets = [1.0, 1.0, 1.0]  # Target values
    
    x_pos = np.arange(len(metrics))
    bars = ax.bar(x_pos, values, alpha=0.7, color=['blue', 'green', 'orange'])
    ax.bar(x_pos, targets, alpha=0.3, color='red', width=0.3, label='Target')
    
    ax.set_ylabel('Value')
    ax.set_title('Summary Metrics')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('scale_direction_model_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    print(f"Tested {len(results)} sequences")
    print(f"Average magnitude ratio: {np.mean(magnitude_ratios):.3f} (target: 1.0)")
    print(f"Direction accuracy: {np.mean(direction_matches)*100:.1f}% (target: 100%)")
    print(f"Average length ratio: {np.mean(length_ratios):.3f} (target: 1.0)")
    
    # Determine overall status
    mag_good = 0.5 < np.mean(magnitude_ratios) < 2.0
    dir_good = np.mean(direction_matches) > 0.5
    
    print(f"\n🎯 SCALE PROBLEM: {'✅ SIGNIFICANTLY IMPROVED' if mag_good else '❌ NEEDS MORE WORK'}")
    print(f"🧭 DIRECTION PROBLEM: {'✅ FIXED' if dir_good else '❌ NEEDS MORE WORK'}")
    
    if mag_good and dir_good:
        print(f"\n🎉 SUCCESS: Both scale and direction issues are resolved!")
    elif mag_good:
        print(f"\n✅ PARTIAL SUCCESS: Scale fixed, direction needs work")
    elif dir_good:
        print(f"\n✅ PARTIAL SUCCESS: Direction fixed, scale needs work")
    else:
        print(f"\n⚠️ Both issues need more training/tuning")
    
    print(f"\n📊 Visualization saved as: scale_direction_model_test_results.png")
    print("="*70)


if __name__ == '__main__':
    test_scale_direction_model()