#!/usr/bin/env python3
"""
Generate corrected trajectory plots using TRUE world coordinates
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders

def load_model(model_path, config):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, loss_fn = create_tsformer_vo(
        sequence_length=config['sequence_length'],
        pretrained=True,
        freeze_backbone=False,
        image_size=config['image_size']
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device

def get_model_predictions(model, device, data_loader):
    """Get model predictions for a dataset"""
    predictions = []
    metadata = []
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['images'].to(device)
            pred_poses = model(images)
            predictions.append(pred_poses.cpu().numpy())
            
            metadata.append({
                'bag_name': batch['bag_name'][0],
                'frame_indices': batch['frame_indices'][0].tolist()
            })
    
    return np.concatenate(predictions, axis=0), metadata

def get_true_world_coordinates(csv_path, metadata):
    """Get true world coordinates for the predicted frames"""
    df = pd.read_csv(csv_path)
    
    world_coords = []
    for meta in metadata:
        bag_name = meta['bag_name']
        frame_indices = meta['frame_indices']
        
        # Get the last frame's world coordinates (what we're predicting)
        last_frame = frame_indices[-1]
        frame_data = df[(df['bag_name'] == bag_name) & (df['frame_index'] == last_frame)]
        
        if len(frame_data) > 0:
            row = frame_data.iloc[0]
            world_coords.append([row['world_x'], row['world_y'], row['world_z']])
        else:
            world_coords.append([np.nan, np.nan, np.nan])
    
    return np.array(world_coords)

def accumulate_deltas(deltas):
    """Accumulate delta poses to create trajectory"""
    trajectory = [np.zeros(6)]
    current_pose = np.zeros(6)
    
    for delta in deltas:
        current_pose[:3] += delta[:3]  # Simple accumulation for translation
        current_pose[3:] += delta[3:]  # Simple accumulation for rotation
        trajectory.append(current_pose.copy())
    
    return np.array(trajectory)

def plot_trajectories(pred_trajectory, true_world_coords, dataset_name, output_dir):
    """Create trajectory comparison plots"""
    # Remove NaN values from true coordinates
    valid_mask = ~np.isnan(true_world_coords).any(axis=1)
    true_coords_clean = true_world_coords[valid_mask]
    
    if len(true_coords_clean) == 0:
        print(f"No valid world coordinates for {dataset_name}")
        return
    
    print(f"Plotting {dataset_name}: {len(pred_trajectory)} predicted vs {len(true_coords_clean)} true points")
    
    # Create 2D projection plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # XY plane
    axes[0].plot(true_coords_clean[:, 0], true_coords_clean[:, 1], 'b-', linewidth=3, 
                label='TRUE Ground Truth (World Coordinates)', alpha=0.8)
    axes[0].plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', linewidth=2, 
                label='Model Prediction (Accumulated Deltas)', alpha=0.8)
    axes[0].scatter(true_coords_clean[0, 0], true_coords_clean[0, 1], c='green', s=100, marker='o', label='Start')
    axes[0].scatter(true_coords_clean[-1, 0], true_coords_clean[-1, 1], c='red', s=100, marker='s', label='End')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('XY Plane (Top View)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axis('equal')
    
    # XZ plane
    axes[1].plot(true_coords_clean[:, 0], true_coords_clean[:, 2], 'b-', linewidth=3, 
                label='TRUE Ground Truth (World Coordinates)', alpha=0.8)
    axes[1].plot(pred_trajectory[:, 0], pred_trajectory[:, 2], 'r--', linewidth=2, 
                label='Model Prediction (Accumulated Deltas)', alpha=0.8)
    axes[1].scatter(true_coords_clean[0, 0], true_coords_clean[0, 2], c='green', s=100, marker='o', label='Start')
    axes[1].scatter(true_coords_clean[-1, 0], true_coords_clean[-1, 2], c='red', s=100, marker='s', label='End')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    axes[1].set_title('XZ Plane (Side View)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].axis('equal')
    
    # YZ plane
    axes[2].plot(true_coords_clean[:, 1], true_coords_clean[:, 2], 'b-', linewidth=3, 
                label='TRUE Ground Truth (World Coordinates)', alpha=0.8)
    axes[2].plot(pred_trajectory[:, 1], pred_trajectory[:, 2], 'r--', linewidth=2, 
                label='Model Prediction (Accumulated Deltas)', alpha=0.8)
    axes[2].scatter(true_coords_clean[0, 1], true_coords_clean[0, 2], c='green', s=100, marker='o', label='Start')
    axes[2].scatter(true_coords_clean[-1, 1], true_coords_clean[-1, 2], c='red', s=100, marker='s', label='End')
    axes[2].set_xlabel('Y (m)')
    axes[2].set_ylabel('Z (m)')
    axes[2].set_title('YZ Plane (Front View)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].axis('equal')
    
    plt.suptitle(f'CORRECTED Trajectory Comparison - {dataset_name} Dataset\\nBlue=TRUE World Coordinates, Red=Model Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'corrected_2d_projections_{dataset_name.lower()}.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: corrected_2d_projections_{dataset_name.lower()}.png")

def main():
    """Main function"""
    config = {
        'csv_path': 'data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv',
        'data_root': 'data/processed/visual_odometry_dataset',
        'sequence_length': 3,
        'overlap_frames': 1,
        'image_size': 224,
        'test_bags': ['ariel_2023-12-21-14-28-22_4']
    }
    
    # Model path
    model_path = Path('experiments/2_TSFormer_seq3_frozen_balanced_loss_consistency/checkpoint_best.pth')
    output_dir = Path('evaluation_results_2_TSFormer_seq3_frozen_balanced_loss_consistency')
    
    print("Loading model...")
    model, device = load_model(model_path, config)
    
    print("Loading data...")
    data_loaders = create_data_loaders(
        csv_path=config['csv_path'],
        data_root=config['data_root'],
        sequence_length=config['sequence_length'],
        overlap_frames=config['overlap_frames'],
        image_size=config['image_size'],
        batch_size=1,
        test_bags=config['test_bags'],
        num_workers=0,
        camera='cam0'
    )
    
    # Process each dataset
    for dataset_name, data_loader in [('Training', data_loaders['train_loader']), 
                                     ('Validation', data_loaders['val_loader']),
                                     ('Test', data_loaders['test_loader'])]:
        if len(data_loader) == 0:
            continue
            
        print(f"\\nProcessing {dataset_name} dataset...")
        
        # Get model predictions
        predictions, metadata = get_model_predictions(model, device, data_loader)
        
        # Get true world coordinates
        true_world_coords = get_true_world_coordinates(config['csv_path'], metadata)
        
        # Accumulate predicted deltas to create trajectory
        pred_trajectory = accumulate_deltas(predictions)
        
        # Create plots
        plot_trajectories(pred_trajectory, true_world_coords, dataset_name, output_dir)

if __name__ == "__main__":
    main()