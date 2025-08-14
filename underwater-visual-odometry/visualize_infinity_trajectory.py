#!/usr/bin/env python3
"""
Visualize the Infinity-Shaped Trajectory
Show both ground truth and predictions in proper global coordinates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders
from scipy.spatial.transform import Rotation as R


def load_model(checkpoint_path):
    """Load trained model"""
    print(f"Loading model from {checkpoint_path}")
    
    model, loss_fn = create_tsformer_vo(
        sequence_length=8,
        pretrained=True,
        freeze_backbone=False,
        use_multi_scale_loss=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device


def get_complete_ground_truth_trajectory(csv_path, bag_name):
    """Get complete ground truth trajectory for a bag"""
    print(f"Loading complete trajectory for bag: {bag_name}")
    
    df = pd.read_csv(csv_path)
    bag_df = df[df['bag_name'] == bag_name].copy()
    bag_df = bag_df.sort_values('frame_index').reset_index(drop=True)
    
    # Remove rows with NaN world coordinates
    valid_mask = bag_df[['world_x', 'world_y', 'world_z']].notna().all(axis=1)
    bag_df_clean = bag_df[valid_mask].copy()
    
    print(f"Total frames: {len(bag_df)}, Valid world coordinates: {len(bag_df_clean)}")
    
    if len(bag_df_clean) == 0:
        print("ERROR: No valid world coordinates found!")
        return np.array([]), bag_df
    
    # Extract world coordinates
    trajectory = np.column_stack([
        bag_df_clean['world_x'].values,
        bag_df_clean['world_y'].values,
        bag_df_clean['world_z'].values
    ])
    
    print(f"Ground truth: {len(trajectory)} points")
    print(f"X range: [{trajectory[:, 0].min():.3f}, {trajectory[:, 0].max():.3f}]")
    print(f"Y range: [{trajectory[:, 1].min():.3f}, {trajectory[:, 1].max():.3f}]")
    print(f"Z range: [{trajectory[:, 2].min():.3f}, {trajectory[:, 2].max():.3f}]")
    
    return trajectory, bag_df_clean


def predict_on_bag(model, device, csv_path, bag_name, max_sequences=100):
    """Generate predictions for a specific bag"""
    print(f"Generating predictions for bag: {bag_name}")
    
    # Create data loader for just this bag
    data_loaders = create_data_loaders(
        csv_path=csv_path,
        data_root="data/processed/visual_odometry_dataset",
        sequence_length=8,
        overlap_frames=1,  # More overlap for denser predictions
        image_size=224,
        batch_size=1,
        test_bags=[],  # Don't reserve any bags for test
        num_workers=0,
        camera='cam0'
    )
    
    # Find the loader that contains our target bag
    for loader_name, loader in [('train', data_loaders['train_loader']), 
                               ('val', data_loaders['val_loader']), 
                               ('test', data_loaders['test_loader'])]:
        
        predictions = []
        frame_info = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_sequences:
                    break
                    
                # Check if this batch is from our target bag
                if batch['bag_name'][0] == bag_name:
                    images = batch['images'].to(device)
                    
                    # Get prediction
                    pred_poses = model(images)
                    predictions.append(pred_poses[0].cpu().numpy())
                    frame_info.append({
                        'frame_indices': batch['frame_indices'][0],
                        'bag_name': batch['bag_name'][0]
                    })
        
        if len(predictions) > 0:
            print(f"Found {len(predictions)} predictions in {loader_name} set")
            return np.array(predictions), frame_info
    
    print("No predictions found for this bag!")
    return np.array([]), []


def accumulate_predictions_from_real_start(predictions, frame_info, csv_path):
    """Accumulate predictions starting from real world coordinates"""
    if len(predictions) == 0:
        return np.array([])
    
    # Load CSV to find starting position
    df = pd.read_csv(csv_path)
    
    # Get the first frame info
    first_info = frame_info[0]
    first_frame_list = first_info['frame_indices']
    bag_name = first_info['bag_name']
    
    if isinstance(first_frame_list, list) and len(first_frame_list) > 0:
        first_frame = first_frame_list[0]
    else:
        first_frame = first_frame_list
    
    # Find starting world coordinates
    mask = (df['bag_name'] == bag_name) & (df['frame_index'] == first_frame)
    matching_rows = df[mask]
    
    if len(matching_rows) > 0:
        start_row = matching_rows.iloc[0]
        start_pos = np.array([
            start_row['world_x'], start_row['world_y'], start_row['world_z'],
            0.0, 0.0, 0.0  # Zero initial orientation
        ])
        print(f"Starting prediction from: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
    else:
        start_pos = np.zeros(6)
        print("Warning: Using origin as start")
    
    # Accumulate trajectory
    trajectory = [start_pos[:3].copy()]  # Only store XYZ
    current_T = pose_to_se3_matrix(start_pos)
    
    for delta in predictions:
        delta_T = pose_to_se3_matrix(delta)
        current_T = current_T @ delta_T
        
        # Extract position
        pos = current_T[:3, 3]
        trajectory.append(pos.copy())
    
    return np.array(trajectory)


def pose_to_se3_matrix(pose):
    """Convert 6DOF pose to SE(3) transformation matrix"""
    translation = pose[:3]
    rotation = pose[3:]
    
    # Convert Euler to rotation matrix
    rot_matrix = R.from_euler('xyz', rotation).as_matrix()
    
    # Create SE(3) matrix
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = translation
    
    return T


def create_infinity_visualization(gt_trajectory, pred_trajectory, bag_name, output_dir):
    """Create visualization showing the infinity shape"""
    print("Creating infinity trajectory visualization...")
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Large 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
             'b-', linewidth=3, label='Ground Truth (Infinity)', alpha=0.8)
    
    if len(pred_trajectory) > 0:
        ax1.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
                 'r--', linewidth=2, label='Multi-Scale Prediction', alpha=0.8)
    
    ax1.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], gt_trajectory[0, 2], 
                c='green', s=200, label='Start', marker='o')
    ax1.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], gt_trajectory[-1, 2], 
                c='red', s=200, label='End', marker='s')
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_zlabel('Z (m)', fontsize=12)
    ax1.set_title(f'3D Infinity Trajectory - {bag_name}\nMulti-Scale SE(3) Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    # 2. XY Top View - This should show the infinity/figure-8 shape
    ax2 = fig.add_subplot(222)
    ax2.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', linewidth=3, 
             label='Ground Truth (Infinity)', alpha=0.8)
    
    if len(pred_trajectory) > 0:
        ax2.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=2, 
                 label='Multi-Scale Prediction', alpha=0.8)
    
    ax2.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], c='green', s=150, 
                label='Start', marker='o')
    ax2.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], c='red', s=150, 
                label='End', marker='s')
    
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('Top View - Infinity/Figure-8 Pattern', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. XZ Side View
    ax3 = fig.add_subplot(223)
    ax3.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], 'b-', linewidth=3, 
             label='Ground Truth', alpha=0.8)
    
    if len(pred_trajectory) > 0:
        ax3.plot(pred_trajectory[:, 0], pred_trajectory[:, 2], 'r--', linewidth=2, 
                 label='Multi-Scale Prediction', alpha=0.8)
    
    ax3.scatter(gt_trajectory[0, 0], gt_trajectory[0, 2], c='green', s=150, 
                label='Start', marker='o')
    ax3.set_xlabel('X (m)', fontsize=12)
    ax3.set_ylabel('Z (m)', fontsize=12)
    ax3.set_title('Side View (XZ Plane)', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True)
    
    # 4. Trajectory metrics
    ax4 = fig.add_subplot(224)
    
    # Calculate some trajectory statistics
    gt_distances = np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1)
    gt_cumulative = np.concatenate([[0], np.cumsum(gt_distances)])
    
    if len(pred_trajectory) > 0:
        min_len = min(len(pred_trajectory), len(gt_trajectory))
        pred_trunc = pred_trajectory[:min_len]
        gt_trunc = gt_trajectory[:min_len]
        
        errors = np.linalg.norm(pred_trunc - gt_trunc, axis=1)
        ax4.plot(errors, 'r-', linewidth=2, label=f'Point Error (Mean: {errors.mean():.2f}m)')
        ax4.set_ylabel('Error (m)', fontsize=12)
        
        print(f"Trajectory Metrics:")
        print(f"  Mean Error: {errors.mean():.3f} m")
        print(f"  Max Error: {errors.max():.3f} m")
        print(f"  Final Error: {errors[-1]:.3f} m")
    else:
        ax4.plot(gt_cumulative, 'b-', linewidth=2, label='Ground Truth Distance')
        ax4.set_ylabel('Cumulative Distance (m)', fontsize=12)
    
    ax4.set_xlabel('Frame Index', fontsize=12)
    ax4.set_title('Trajectory Analysis', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f'infinity_trajectory_{bag_name.replace("-", "_")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Infinity trajectory plot saved: {plot_path}")
    
    return plot_path


def main():
    """Main visualization function"""
    print("="*60)
    print("INFINITY TRAJECTORY VISUALIZATION")
    print("Multi-Scale SE(3) TSformer-VO Evaluation")
    print("="*60)
    
    # Paths
    checkpoint_path = "experiments/tsformer_vo_multi_scale/checkpoint_best.pth"
    csv_path = "data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr_clean.csv"
    output_dir = "evaluation_results_4_TSFormer_seq8_multi_scale_se3_loss"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Choose bag with largest trajectory range (should have infinity shape)
    bag_name = "ariel_2023-12-21-14-27-27_3"  # This has good X,Y range
    print(f"Using bag: {bag_name}")
    
    # Load complete ground truth trajectory
    gt_trajectory, bag_df = get_complete_ground_truth_trajectory(csv_path, bag_name)
    
    # Load model and generate predictions
    model, device = load_model(checkpoint_path)
    predictions, frame_info = predict_on_bag(model, device, csv_path, bag_name, max_sequences=150)
    
    # Accumulate predictions from real starting position
    if len(predictions) > 0:
        pred_trajectory = accumulate_predictions_from_real_start(predictions, frame_info, csv_path)
        print(f"Generated {len(pred_trajectory)} prediction points")
    else:
        pred_trajectory = np.array([])
        print("No predictions generated")
    
    # Create visualization
    plot_path = create_infinity_visualization(gt_trajectory, pred_trajectory, bag_name, output_dir)
    
    print("\n" + "="*60)
    print("INFINITY TRAJECTORY VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"Ground Truth Points: {len(gt_trajectory)}")
    print(f"Prediction Points: {len(pred_trajectory) if len(pred_trajectory) > 0 else 0}")
    print(f"Visualization saved: {plot_path}")
    
    if len(pred_trajectory) > 0:
        # Calculate trajectory lengths
        gt_length = np.sum(np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1))
        pred_length = np.sum(np.linalg.norm(np.diff(pred_trajectory, axis=0), axis=1))
        print(f"Ground Truth Length: {gt_length:.2f} m")
        print(f"Prediction Length: {pred_length:.2f} m")
        print(f"Length Ratio: {pred_length/gt_length:.3f}")
    
    print(f"\nLook for the infinity/figure-8 pattern in the XY top view!")


if __name__ == "__main__":
    main()