"""
Debug Model Architecture and Inputs

Analyzes the current model structure, inputs, and outputs to identify
why predictions are straight lines in the opposite direction.
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer.multiscale_uw_transvo import create_multiscale_model


def debug_model_architecture():
    """Debug the model architecture and data flow"""
    
    print("=" * 60)
    print("MODEL ARCHITECTURE DEBUG")
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
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_multiscale_model(config).to(device)
    
    print("1. MODEL STRUCTURE:")
    print(f"   Total parameters: {model.count_parameters():,}")
    print(f"   Input image size: {config['img_size']}x{config['img_size']}")
    print(f"   Sequence length: {config['max_seq_len']}")
    print(f"   Feature dimension: {config['d_model']}")
    
    # Print model architecture summary
    print("\n2. ARCHITECTURE COMPONENTS:")
    for name, module in model.named_children():
        print(f"   - {name}: {type(module).__name__}")
    
    # Create dummy input to trace through the model
    print("\n3. INPUT TENSOR SHAPES:")
    batch_size = 2
    seq_len = 10
    num_cameras = 1
    
    # Input shapes
    images = torch.randn(batch_size, seq_len, num_cameras, 3, config['img_size'], config['img_size']).to(device)
    camera_ids = torch.zeros(batch_size, num_cameras, dtype=torch.long).to(device)
    
    print(f"   Images input: {images.shape}")
    print(f"   Camera IDs: {camera_ids.shape}")
    
    # Forward pass with intermediate outputs
    print("\n4. FORWARD PASS ANALYSIS:")
    model.eval()
    with torch.no_grad():
        # Hook to capture intermediate outputs
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
                elif isinstance(output, dict):
                    activations[name] = {k: v.detach() if isinstance(v, torch.Tensor) else v 
                                       for k, v in output.items()}
            return hook
        
        # Register hooks
        model.vision_transformer.register_forward_hook(get_activation('vision_transformer'))
        model.pose_head.register_forward_hook(get_activation('pose_head'))
        
        # Forward pass
        outputs = model(images=images, camera_ids=camera_ids)
        
        print(f"   Vision Transformer output: {activations['vision_transformer'].shape}")
        print(f"   Final outputs: {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in outputs.items()]}")
        
        # Analyze predictions
        if 'delta_poses' in outputs:
            delta_poses = outputs['delta_poses']
            print(f"   Delta poses shape: {delta_poses.shape}")
            print(f"   Delta poses range: [{delta_poses.min():.6f}, {delta_poses.max():.6f}]")
            print(f"   Delta poses mean: {delta_poses.mean():.6f}")
            print(f"   Delta poses std: {delta_poses.std():.6f}")
    
    return model, config


def debug_data_loading():
    """Debug the data loading and preprocessing"""
    
    print("\n" + "=" * 60)
    print("DATA LOADING DEBUG")
    print("=" * 60)
    
    # Load some real data
    csv_file = "data/processed/training_dataset/training_data.csv"
    if not os.path.exists(csv_file):
        print("ERROR: CSV file not found")
        return
    
    df = pd.read_csv(csv_file)
    print(f"1. DATASET INFO:")
    print(f"   Total frames: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Analyze ground truth deltas
    delta_columns = ['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw']
    
    print(f"\n2. GROUND TRUTH DELTA ANALYSIS:")
    for col in delta_columns:
        if col in df.columns:
            values = df[col].values
            print(f"   {col}: range=[{values.min():.6f}, {values.max():.6f}], mean={values.mean():.6f}, std={values.std():.6f}")
    
    # Load a specific sequence for analysis
    print(f"\n3. SAMPLE SEQUENCE ANALYSIS:")
    start_idx = 1000
    seq_len = 10
    sequence = df.iloc[start_idx:start_idx + seq_len]
    
    print(f"   Sequence frames {start_idx}-{start_idx + seq_len}:")
    for i, (_, row) in enumerate(sequence.iterrows()):
        delta_x = row.get('delta_x', 0.0)
        delta_y = row.get('delta_y', 0.0)
        print(f"     Frame {i}: delta_x={delta_x:.6f}, delta_y={delta_y:.6f}")
    
    # Check image paths
    print(f"\n4. IMAGE PATH ANALYSIS:")
    sample_paths = sequence['cam0_path'].iloc[:3]
    for i, path in enumerate(sample_paths):
        exists = os.path.exists(str(path)) if not pd.isna(path) else False
        alt_exists = os.path.exists(os.path.join(".", str(path))) if not pd.isna(path) else False
        print(f"   Sample {i}: {path}")
        print(f"            Exists: {exists}, Alt path exists: {alt_exists}")
    
    # Compute accumulated trajectory for this sequence
    deltas = sequence[delta_columns].values.astype(float)
    accumulated = np.cumsum(deltas, axis=0)
    
    print(f"\n5. ACCUMULATED TRAJECTORY:")
    print(f"   Start position: [{accumulated[0, 0]:.6f}, {accumulated[0, 1]:.6f}]")
    print(f"   End position: [{accumulated[-1, 0]:.6f}, {accumulated[-1, 1]:.6f}]")
    print(f"   Total displacement: {np.linalg.norm(accumulated[-1, :3]):.6f}m")
    
    # Plot this sequence
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(accumulated[:, 0], accumulated[:, 1], 'b-o', label='Ground Truth')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Ground Truth XY Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(1, 3, 2)
    plt.plot(deltas[:, 0], 'b-', label='Delta X')
    plt.plot(deltas[:, 1], 'r-', label='Delta Y')
    plt.xlabel('Frame')
    plt.ylabel('Delta (m)')
    plt.title('Frame-to-Frame Deltas')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    magnitudes = np.linalg.norm(deltas[:, :3], axis=1)
    plt.plot(magnitudes, 'g-o', label='Motion Magnitude')
    plt.xlabel('Frame')
    plt.ylabel('Magnitude (m)')
    plt.title('Motion Magnitude per Frame')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('data_analysis_debug.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return deltas, accumulated


def debug_prediction_vs_ground_truth():
    """Compare model predictions with ground truth in detail"""
    
    print("\n" + "=" * 60)
    print("PREDICTION vs GROUND TRUTH DEBUG")
    print("=" * 60)
    
    # Load trained model
    model_path = 'multiscale_light_best_model.pth'
    if not os.path.exists(model_path):
        print("ERROR: Trained model not found")
        return
    
    config = {
        'img_size': 192, 'd_model': 256, 'num_heads': 4, 'num_layers': 3,
        'max_cameras': 1, 'max_seq_len': 10, 'dropout': 0.1, 'uncertainty_estimation': False
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_multiscale_model(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test sequence
    csv_file = "data/processed/training_dataset/training_data.csv"
    df = pd.read_csv(csv_file)
    
    start_idx = 1000
    seq_len = 10
    sequence = df.iloc[start_idx:start_idx + seq_len]
    
    # Load images (simplified - zeros for now to focus on architecture)
    images = torch.zeros(1, seq_len, 1, 3, config['img_size'], config['img_size']).to(device)
    camera_ids = torch.zeros(1, 1, dtype=torch.long).to(device)
    
    # Get ground truth
    gt_deltas = sequence[['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw']].values.astype(float)
    gt_trajectory = np.cumsum(gt_deltas, axis=0)
    
    print(f"1. GROUND TRUTH ANALYSIS:")
    print(f"   GT deltas shape: {gt_deltas.shape}")
    print(f"   GT trajectory range: X=[{gt_trajectory[:, 0].min():.6f}, {gt_trajectory[:, 0].max():.6f}]")
    print(f"                       Y=[{gt_trajectory[:, 1].min():.6f}, {gt_trajectory[:, 1].max():.6f}]")
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(images=images, camera_ids=camera_ids)
        pred_deltas = outputs['delta_poses'][0].cpu().numpy()  # [seq_len, 6]
    
    pred_trajectory = np.cumsum(pred_deltas, axis=0)
    
    print(f"\n2. MODEL PREDICTION ANALYSIS:")
    print(f"   Pred deltas shape: {pred_deltas.shape}")
    print(f"   Pred deltas range: X=[{pred_deltas[:, 0].min():.6f}, {pred_deltas[:, 0].max():.6f}]")
    print(f"                     Y=[{pred_deltas[:, 1].min():.6f}, {pred_deltas[:, 1].max():.6f}]")
    print(f"   Pred trajectory range: X=[{pred_trajectory[:, 0].min():.6f}, {pred_trajectory[:, 0].max():.6f}]")
    print(f"                         Y=[{pred_trajectory[:, 1].min():.6f}, {pred_trajectory[:, 1].max():.6f}]")
    
    # Detailed comparison
    print(f"\n3. FRAME-BY-FRAME COMPARISON:")
    print(f"   {'Frame':<5} {'GT_X':<10} {'GT_Y':<10} {'Pred_X':<10} {'Pred_Y':<10} {'Ratio_X':<10} {'Ratio_Y':<10}")
    print(f"   {'-'*65}")
    
    for i in range(seq_len):
        gt_x, gt_y = gt_deltas[i, 0], gt_deltas[i, 1]
        pred_x, pred_y = pred_deltas[i, 0], pred_deltas[i, 1]
        
        ratio_x = pred_x / gt_x if abs(gt_x) > 1e-8 else float('inf')
        ratio_y = pred_y / gt_y if abs(gt_y) > 1e-8 else float('inf')
        
        print(f"   {i:<5} {gt_x:<10.6f} {gt_y:<10.6f} {pred_x:<10.6f} {pred_y:<10.6f} {ratio_x:<10.2f} {ratio_y:<10.2f}")
    
    # Direction analysis
    print(f"\n4. DIRECTION ANALYSIS:")
    gt_direction = np.arctan2(gt_trajectory[-1, 1] - gt_trajectory[0, 1], 
                             gt_trajectory[-1, 0] - gt_trajectory[0, 0])
    pred_direction = np.arctan2(pred_trajectory[-1, 1] - pred_trajectory[0, 1], 
                               pred_trajectory[-1, 0] - pred_trajectory[0, 0])
    
    direction_diff = abs(gt_direction - pred_direction)
    if direction_diff > np.pi:
        direction_diff = 2 * np.pi - direction_diff
    
    print(f"   Ground truth direction: {gt_direction * 180 / np.pi:.1f} degrees")
    print(f"   Predicted direction: {pred_direction * 180 / np.pi:.1f} degrees")
    print(f"   Direction difference: {direction_diff * 180 / np.pi:.1f} degrees")
    
    if direction_diff > np.pi / 2:
        print(f"   *** OPPOSITE DIRECTION CONFIRMED! ***")
    
    # Visual comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-o', label='Ground Truth', markersize=6)
    plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r-s', label='Prediction', markersize=4)
    plt.plot(gt_trajectory[0, 0], gt_trajectory[0, 1], 'go', markersize=10, label='Start')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Comparison')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(1, 3, 2)
    frames = np.arange(seq_len)
    plt.plot(frames, gt_deltas[:, 0], 'b-', label='GT Delta X', linewidth=2)
    plt.plot(frames, gt_deltas[:, 1], 'g-', label='GT Delta Y', linewidth=2)
    plt.plot(frames, pred_deltas[:, 0], 'r--', label='Pred Delta X', linewidth=2)
    plt.plot(frames, pred_deltas[:, 1], 'm--', label='Pred Delta Y', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Delta (m)')
    plt.title('Frame-to-Frame Deltas')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    ratios_x = [pred_deltas[i, 0] / gt_deltas[i, 0] if abs(gt_deltas[i, 0]) > 1e-8 else 0 for i in range(seq_len)]
    ratios_y = [pred_deltas[i, 1] / gt_deltas[i, 1] if abs(gt_deltas[i, 1]) > 1e-8 else 0 for i in range(seq_len)]
    
    plt.plot(frames, ratios_x, 'r-o', label='Pred/GT Ratio X')
    plt.plot(frames, ratios_y, 'm-s', label='Pred/GT Ratio Y')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=-1, color='k', linestyle='--', alpha=0.5, label='Opposite Direction')
    plt.xlabel('Frame')
    plt.ylabel('Prediction/GroundTruth Ratio')
    plt.title('Prediction Ratios (Negative = Opposite Direction)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('prediction_debug_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("COMPREHENSIVE MODEL DEBUG ANALYSIS")
    print("This will identify why predictions are straight lines in opposite direction")
    
    # Debug model architecture
    model, config = debug_model_architecture()
    
    # Debug data loading
    deltas, trajectory = debug_data_loading()
    
    # Debug predictions vs ground truth
    debug_prediction_vs_ground_truth()
    
    print("\n" + "="*60)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*60)
    print("Files created:")
    print("- data_analysis_debug.png")
    print("- prediction_debug_analysis.png")
    print("\nKey findings will be shown in the detailed output above.")