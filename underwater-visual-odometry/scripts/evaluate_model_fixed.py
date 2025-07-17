#!/usr/bin/env python3
"""
Fixed evaluation script for trained model.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Import from existing scripts
import sys
sys.path.append('/app/scripts')
from data_loader import UnderwaterVODataset
from models.baseline_cnn import create_model
from coordinate_transforms import integrate_trajectory, compute_trajectory_error
from visualization import (plot_trajectory_3d, plot_trajectory_2d, 
                          plot_errors_over_time, plot_training_history)


def evaluate_model(model_dir, data_dir):
    """Quick evaluation function."""
    print("Starting evaluation...")
    
    # Create output directory
    output_dir = os.path.join(model_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create and load model
    print("Loading model...")
    model = create_model(
        model_type=config.get('model_type', 'baseline'),
        input_shape=(224, 224, 6)
    )
    
    # Build model with dummy input
    dummy_input = tf.zeros((1, 224, 224, 6))
    _ = model(dummy_input, training=False)
    
    # Load weights
    weights_path = os.path.join(model_dir, 'best_model.h5')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, 'final_model.h5')
    
    model.load_weights(weights_path)
    print(f"Loaded weights from: {weights_path}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = UnderwaterVODataset(data_dir, image_size=(224, 224))
    dataset.load_data()
    
    # Quick evaluation on one sequence
    print("\nEvaluating on a test sequence...")
    sequence_length = 100
    start_idx = 100  # Skip first 100 frames
    
    predictions = []
    ground_truth = []
    
    print("Making predictions...")
    for i in tqdm(range(start_idx, start_idx + sequence_length - 1)):
        # Load images
        img1 = dataset.load_image(i)
        img2 = dataset.load_image(i + 1)
        
        # Predict
        img_pair = np.concatenate([img1, img2], axis=-1)
        img_batch = np.expand_dims(img_pair, axis=0)
        pred = model.predict(img_batch, verbose=0)[0]
        predictions.append(pred)
        
        # Ground truth
        gt = dataset.get_relative_pose(i, i + 1)
        gt_pose = np.concatenate([gt['translation'], gt['rotation']])
        ground_truth.append(gt_pose)
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Integrate trajectories
    print("Integrating trajectories...")
    initial_pose = {
        'position': np.array([0, 0, 0]),
        'quaternion': np.array([1, 0, 0, 0])
    }
    
    pred_trajectory = integrate_trajectory(initial_pose, predictions)
    gt_trajectory = integrate_trajectory(initial_pose, ground_truth)
    
    # Compute errors
    errors = compute_trajectory_error(pred_trajectory, gt_trajectory)
    
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(f"Absolute Trajectory Error (ATE): {errors['ate']:.4f} ± {errors['ate_std']:.4f} m")
    print(f"Relative Pose Error (Trans): {errors['rpe_trans']:.4f} ± {errors['rpe_trans_std']:.4f} m")
    print(f"Relative Pose Error (Rot): {np.degrees(errors['rpe_rot']):.2f} ± {np.degrees(errors['rpe_rot_std']):.2f} deg")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 3D trajectory
    trajectories = {
        'Ground Truth': gt_trajectory,
        'Predicted': pred_trajectory
    }
    
    plot_trajectory_3d(
        trajectories,
        title="3D Trajectory Comparison",
        save_path=os.path.join(output_dir, 'trajectory_3d.png')
    )
    
    # 2D trajectory
    plot_trajectory_2d(
        trajectories,
        view='xy',
        title="Top-Down View",
        save_path=os.path.join(output_dir, 'trajectory_xy.png')
    )
    
    # Errors over time
    trans_errors = np.linalg.norm(predictions[:, :3] - ground_truth[:, :3], axis=1)
    rot_errors = np.degrees(np.linalg.norm(predictions[:, 3:] - ground_truth[:, 3:], axis=1))
    
    plot_errors_over_time(
        {
            'Translation Error': trans_errors,
            'Rotation Error (deg)': rot_errors
        },
        title="Prediction Errors Over Time",
        save_path=os.path.join(output_dir, 'errors.png')
    )
    
    # Plot training history
    history_path = os.path.join(model_dir, 'history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        plot_training_history(
            history,
            save_path=os.path.join(output_dir, 'training_history.png')
        )
    
    # Save numerical results
    results = {
        'ate': float(errors['ate']),
        'ate_std': float(errors['ate_std']),
        'rpe_trans': float(errors['rpe_trans']),
        'rpe_trans_std': float(errors['rpe_trans_std']),
        'rpe_rot_deg': float(np.degrees(errors['rpe_rot'])),
        'rpe_rot_std_deg': float(np.degrees(errors['rpe_rot_std'])),
        'sequence_length': sequence_length,
        'start_frame': start_idx
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone! Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - trajectory_3d.png")
    print("  - trajectory_xy.png") 
    print("  - errors.png")
    print("  - training_history.png")
    print("  - results.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, 
                        default='/app/output/models/model_20250708_023343')
    parser.add_argument('--data_dir', type=str, 
                        default='/app/data/raw')
    
    args = parser.parse_args()
    evaluate_model(args.model_dir, args.data_dir)