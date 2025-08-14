#!/usr/bin/env python3
"""
Evaluate Multi-Scale SE(3) TSformer-VO Model
Generate trajectory visualization plots comparing predictions vs ground truth
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders


def load_model_checkpoint(checkpoint_path, sequence_length=8):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model (will automatically use multi-scale loss since it's default)
    model, loss_fn = create_tsformer_vo(
        sequence_length=sequence_length,
        pretrained=True,
        freeze_backbone=False,
        image_size=224,
        use_multi_scale_loss=True  # This was the trained configuration
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
    
    return model, device


def predict_trajectory(model, data_loader, device, max_batches=50):
    """Generate trajectory predictions"""
    print("Generating predictions...")
    
    predictions = []
    ground_truths = []
    frame_indices = []
    bag_names = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if batch_idx >= max_batches:  # Limit for faster evaluation
                break
                
            # Move data to device
            images = batch['images'].to(device)
            poses = batch['poses'].to(device)  # Single frame deltas
            
            # Get predictions
            pred_poses = model(images)
            
            # Store results
            predictions.extend(pred_poses.cpu().numpy())
            ground_truths.extend(poses.cpu().numpy())
            
            # Get frame info for matching with CSV
            if 'frame_indices' in batch:
                frame_indices.extend(batch['frame_indices'])
            if 'bag_name' in batch:
                bag_names.extend(batch['bag_name'])
    
    return np.array(predictions), np.array(ground_truths), frame_indices, bag_names


def accumulate_trajectory_se3(deltas, gt_trajectory, frame_indices, bag_names, csv_path):
    """Accumulate pose deltas starting from actual world coordinates"""
    from scipy.spatial.transform import Rotation as R
    
    # Load CSV to get the actual starting position
    df = pd.read_csv(csv_path)
    
    # Find the starting world coordinate for the first prediction
    if len(frame_indices) > 0 and len(bag_names) > 0:
        first_frame_list = frame_indices[0]
        first_bag = bag_names[0]
        
        if isinstance(first_frame_list, list) and len(first_frame_list) > 0:
            first_frame = first_frame_list[0]  # Start of sequence
            
            # Find starting position in CSV
            mask = (df['bag_name'] == first_bag) & (df['frame_index'] == first_frame)
            matching_rows = df[mask]
            
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                # Use actual world coordinates as starting point
                initial_world_pos = np.array([
                    row['world_x'], row['world_y'], row['world_z'],
                    0.0, 0.0, 0.0  # Start with zero orientation for simplicity
                ])
                print(f"Starting prediction from world coordinates: [{initial_world_pos[0]:.3f}, {initial_world_pos[1]:.3f}, {initial_world_pos[2]:.3f}]")
            else:
                initial_world_pos = np.zeros(6)
                print("Warning: Could not find starting world coordinates, using origin")
        else:
            initial_world_pos = np.zeros(6)
    else:
        initial_world_pos = np.zeros(6)
    
    # Initialize trajectory with actual world starting position
    trajectory = [initial_world_pos.copy()]
    current_T = pose_to_se3_matrix(initial_world_pos)
    
    # Accumulate deltas from the real starting position
    for delta in deltas:
        # Convert delta to SE(3) matrix
        delta_T = pose_to_se3_matrix(delta)
        
        # Compose: T_new = T_current @ T_delta
        current_T = current_T @ delta_T
        
        # Convert back to pose and store
        pose = se3_matrix_to_pose(current_T)
        trajectory.append(pose.copy())
    
    return np.array(trajectory)


def pose_to_se3_matrix(pose):
    """Convert 6DOF pose to SE(3) transformation matrix"""
    from scipy.spatial.transform import Rotation as R
    
    translation = pose[:3]
    rotation = pose[3:]  # Euler angles
    
    # Convert Euler to rotation matrix
    rot_matrix = R.from_euler('xyz', rotation).as_matrix()
    
    # Create SE(3) matrix
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = translation
    
    return T


def se3_matrix_to_pose(T):
    """Convert SE(3) transformation matrix to 6DOF pose"""
    from scipy.spatial.transform import Rotation as R
    
    # Extract translation
    translation = T[:3, 3]
    
    # Extract rotation and convert to Euler angles
    rotation_matrix = T[:3, :3]
    rotation = R.from_matrix(rotation_matrix).as_euler('xyz')
    
    # Combine into 6DOF pose
    pose = np.concatenate([translation, rotation])
    
    return pose


def get_ground_truth_trajectory_from_csv(csv_path, frame_indices, bag_names):
    """Load ground truth trajectory from CSV matching prediction frames"""
    print(f"Loading ground truth trajectory from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Build trajectory by matching frame indices and bag names
    trajectory_points = []
    
    for i, (frame_idx_list, bag_name) in enumerate(zip(frame_indices, bag_names)):
        if isinstance(frame_idx_list, list) and len(frame_idx_list) > 0:
            # Use the last frame index from the sequence
            frame_idx = frame_idx_list[-1]
        else:
            continue
            
        # Find matching row in CSV
        mask = (df['bag_name'] == bag_name) & (df['frame_index'] == frame_idx)
        matching_rows = df[mask]
        
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            trajectory_points.append([
                row['world_x'],
                row['world_y'], 
                row['world_z']
            ])
        else:
            print(f"Warning: No match found for bag {bag_name}, frame {frame_idx}")
    
    if len(trajectory_points) == 0:
        print("ERROR: No ground truth points found!")
        # Fallback: use first bag
        bag_name = df['bag_name'].iloc[0] 
        df_bag = df[df['bag_name'] == bag_name].head(200)
        trajectory = np.column_stack([
            df_bag['world_x'].values,
            df_bag['world_y'].values,  
            df_bag['world_z'].values
        ])
        print(f"Fallback: Using {len(trajectory)} points from bag {bag_name}")
        return trajectory
    
    trajectory = np.array(trajectory_points)
    
    print(f"Ground truth trajectory: {len(trajectory)} points")
    print(f"Trajectory range: X[{trajectory[:, 0].min():.2f}, {trajectory[:, 0].max():.2f}] "
          f"Y[{trajectory[:, 1].min():.2f}, {trajectory[:, 1].max():.2f}] "
          f"Z[{trajectory[:, 2].min():.2f}, {trajectory[:, 2].max():.2f}]")
    
    return trajectory


def create_trajectory_plots(pred_trajectory, gt_trajectory, output_dir):
    """Create comprehensive trajectory visualization plots"""
    print("Creating trajectory plots...")
    
    # 1. 3D Trajectory Plot
    fig = plt.figure(figsize=(15, 10))
    
    # Main 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
             'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
             'r--', linewidth=2, label='Multi-Scale Prediction', alpha=0.8)
    ax1.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], gt_trajectory[0, 2], 
                c='green', s=100, label='Start', marker='o')
    ax1.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], gt_trajectory[-1, 2], 
                c='red', s=100, label='End', marker='s')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')  
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison (Global Frame)\nMulti-Scale SE(3) Loss - Should Show Infinity Shape')
    ax1.legend()
    ax1.grid(True)
    
    # 2. XY Top View
    ax2 = fig.add_subplot(222)
    ax2.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', linewidth=2, 
             label='Ground Truth', alpha=0.8)
    ax2.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=2, 
             label='Multi-Scale Prediction', alpha=0.8)
    ax2.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], c='green', s=100, 
                label='Start', marker='o')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (XY Plane)\nShould Show Infinity/Figure-8 Pattern')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. XZ Side View  
    ax3 = fig.add_subplot(223)
    ax3.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], 'b-', linewidth=2, 
             label='Ground Truth', alpha=0.8)
    ax3.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], 'r--', linewidth=2, 
             label='Multi-Scale Prediction', alpha=0.8)
    ax3.scatter(gt_trajectory[0, 0], gt_trajectory[0, 2], c='green', s=100, 
                label='Start', marker='o')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (XZ Plane)')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Error Analysis
    ax4 = fig.add_subplot(224)
    
    # Compute trajectory errors
    min_len = min(len(pred_trajectory), len(gt_trajectory))
    pred_trunc = pred_trajectory[:min_len]
    gt_trunc = gt_trajectory[:min_len]
    
    errors = np.linalg.norm(pred_trunc - gt_trunc, axis=1)
    cumulative_error = np.cumsum(errors)
    
    ax4.plot(errors, 'r-', linewidth=2, label=f'Point Error (Mean: {errors.mean():.3f}m)')
    ax4.plot(cumulative_error, 'b--', linewidth=2, label='Cumulative Error')
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Error (m)')
    ax4.set_title('Trajectory Error Analysis')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'trajectory_comparison_multi_scale_se3.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory plot saved: {plot_path}")
    
    # Calculate and return metrics
    metrics = {
        'mean_error': float(errors.mean()),
        'max_error': float(errors.max()),
        'final_error': float(errors[-1]),
        'trajectory_length_gt': float(np.sum(np.linalg.norm(np.diff(gt_trunc, axis=0), axis=1))),
        'trajectory_length_pred': float(np.sum(np.linalg.norm(np.diff(pred_trunc, axis=0), axis=1))),
        'scale_ratio': float(np.sum(np.linalg.norm(np.diff(pred_trunc, axis=0), axis=1)) / 
                           np.sum(np.linalg.norm(np.diff(gt_trunc, axis=0), axis=1)))
    }
    
    return metrics


def main():
    """Main evaluation function"""
    print("="*60)
    print("MULTI-SCALE SE(3) TSFORMER-VO EVALUATION")
    print("="*60)
    
    # Paths
    checkpoint_path = "experiments/tsformer_vo_multi_scale/checkpoint_best.pth"
    csv_path = "data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr_clean.csv"
    output_dir = "evaluation_results_4_TSFormer_seq8_multi_scale_se3_loss"
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Make sure you've completed training first!")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # 1. Load model
        model, device = load_model_checkpoint(checkpoint_path, sequence_length=8)
        
        # 2. Create data loader for evaluation (using validation set)
        print("\nCreating data loaders...")
        data_loaders = create_data_loaders(
            csv_path=csv_path,
            data_root="data/processed/visual_odometry_dataset",
            sequence_length=8,
            overlap_frames=4,
            image_size=224,
            batch_size=4,
            test_bags=["ariel_2023-12-21-14-28-22_4"],  # Same as training
            num_workers=2,
            camera='cam0'
        )
        
        # Use validation set for evaluation
        val_loader = data_loaders['val_loader']
        
        # 3. Generate predictions
        pred_deltas, gt_deltas, frame_indices, bag_names = predict_trajectory(model, val_loader, device, max_batches=50)
        
        # 4. Load ground truth trajectory from CSV (world coordinates)
        print("Loading ground truth trajectory from world coordinates...")
        gt_trajectory = get_ground_truth_trajectory_from_csv(csv_path, frame_indices, bag_names)
        
        # 5. Accumulate prediction trajectory starting from real world coordinates
        print("Accumulating predictions with SE(3) composition from real world start...")
        pred_trajectory = accumulate_trajectory_se3(pred_deltas, gt_trajectory, frame_indices, bag_names, csv_path)
        
        # 6. Create visualization plots
        metrics = create_trajectory_plots(pred_trajectory[:, :3], gt_trajectory, output_dir)
        
        # 7. Save evaluation results
        results = {
            'model_info': {
                'checkpoint': checkpoint_path,
                'sequence_length': 8,
                'loss_type': 'multi_scale_se3',
                'architecture': 'TSformer-VO'
            },
            'metrics': metrics,
            'trajectory_stats': {
                'prediction_points': len(pred_trajectory),
                'ground_truth_points': len(gt_trajectory),
                'prediction_scale': float(np.linalg.norm(pred_trajectory[-1, :3] - pred_trajectory[0, :3])),
                'ground_truth_scale': float(np.linalg.norm(gt_trajectory[-1] - gt_trajectory[0]))
            }
        }
        
        results_path = Path(output_dir) / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 8. Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        print(f"Results saved in: {output_dir}/")
        print(f"\nKey Metrics:")
        print(f"  Mean Trajectory Error: {metrics['mean_error']:.3f} m")
        print(f"  Max Trajectory Error:  {metrics['max_error']:.3f} m") 
        print(f"  Scale Ratio (Pred/GT): {metrics['scale_ratio']:.3f}")
        print(f"  Final Error:           {metrics['final_error']:.3f} m")
        
        print(f"\nTrajectory Lengths:")
        print(f"  Ground Truth: {metrics['trajectory_length_gt']:.2f} m")
        print(f"  Prediction:   {metrics['trajectory_length_pred']:.2f} m")
        
        if metrics['scale_ratio'] > 0.5:
            print(f"\n✅ SCALE IMPROVEMENT: Multi-scale loss achieved {metrics['scale_ratio']:.2f}x scale ratio!")
            print("This is a significant improvement over previous straight-line predictions!")
        else:
            print(f"\n⚠️  Scale ratio still low: {metrics['scale_ratio']:.2f}")
            print("Consider increasing lambda2 weight in multi-scale loss.")
            
        print(f"\nVisualization: {output_dir}/trajectory_comparison_multi_scale_se3.png")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()