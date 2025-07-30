#!/usr/bin/env python3
"""
Evaluate trained model and generate visualizations of results.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

from data_loader import UnderwaterVODataset
from models.baseline_cnn import create_model
from models.losses import PoseLoss, HuberPoseLoss, GeometricLoss
from coordinate_transforms import integrate_trajectory, compute_trajectory_error
from visualization import (plot_trajectory_3d, plot_trajectory_2d, 
                          plot_errors_over_time, plot_training_history,
                          visualize_image_pairs)


class ModelEvaluator:
    """Evaluate trained visual odometry models."""
    
    def __init__(self, model_dir: str, data_dir: str):
        """
        Initialize evaluator.
        
        Args:
            model_dir: Directory containing trained model
            data_dir: Directory containing dataset
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.output_dir = os.path.join(model_dir, 'evaluation')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load config
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            print("Warning: config.json not found, using defaults")
            self.config = {
                'model_type': 'baseline',
                'image_height': 224,
                'image_width': 224,
                'loss_type': 'huber',
                'trans_weight': 1.0,
                'rot_weight': 1.0
            }
    
    def load_model(self):
        """Load the trained model."""
        print("Loading model...")
        
        # Create model
        self.model = create_model(
            model_type=self.config.get('model_type', 'baseline'),
            input_shape=(
                self.config.get('image_height', 224),
                self.config.get('image_width', 224),
                6
            )
        )
        
        # Try to load best weights first, then final weights
        best_weights = os.path.join(self.model_dir, 'best_model.h5')
        final_weights = os.path.join(self.model_dir, 'final_model.h5')
        
        if os.path.exists(best_weights):
            print(f"Loading best model weights from: {best_weights}")
            self.model.load_weights(best_weights)
        elif os.path.exists(final_weights):
            print(f"Loading final model weights from: {final_weights}")
            self.model.load_weights(final_weights)
        else:
            raise FileNotFoundError("No model weights found!")
        
        # Build model by calling it with dummy input
        dummy_input = tf.zeros((1, 
                               self.config.get('image_height', 224),
                               self.config.get('image_width', 224),
                               6))
        _ = self.model(dummy_input, training=False)
        
        print("Model loaded successfully!")
    
    def load_dataset(self):
        """Load the dataset."""
        print("Loading dataset...")
        
        self.dataset = UnderwaterVODataset(
            self.data_dir,
            sequence_length=2,
            image_size=(
                self.config.get('image_height', 224),
                self.config.get('image_width', 224)
            )
        )
        self.dataset.load_data()
        
        # Get validation indices
        _, val_indices = self.dataset.train_val_split(
            val_ratio=self.config.get('val_ratio', 0.2)
        )
        
        return val_indices
    
    def evaluate_on_sequences(self, num_sequences=5, sequence_length=100):
        """Evaluate model on continuous sequences and visualize trajectories."""
        print(f"\nEvaluating on {num_sequences} sequences of length {sequence_length}...")
        
        results = []
        
        # Get total number of frames
        total_frames = len(self.dataset.poses_df) - 1
        
        for seq_idx in range(num_sequences):
            # Random starting point
            if total_frames > sequence_length:
                start_idx = np.random.randint(0, total_frames - sequence_length)
            else:
                start_idx = 0
                sequence_length = total_frames
            
            print(f"\nSequence {seq_idx + 1}: frames {start_idx} to {start_idx + sequence_length}")
            
            # Get ground truth trajectory
            gt_poses = []
            predictions = []
            
            # Initial pose
            initial_pose = {
                'position': np.array([0, 0, 0]),
                'quaternion': np.array([1, 0, 0, 0])
            }
            
            # Predict frame by frame
            for i in tqdm(range(start_idx, start_idx + sequence_length - 1), 
                         desc=f"Processing sequence {seq_idx + 1}"):
                # Load images
                img1 = self.dataset.load_image(i)
                img2 = self.dataset.load_image(i + 1)
                
                # Stack images
                img_pair = np.concatenate([img1, img2], axis=-1)
                img_batch = np.expand_dims(img_pair, axis=0)
                
                # Predict
                pred = self.model.predict(img_batch, verbose=0)[0]
                predictions.append(pred)
                
                # Get ground truth
                gt = self.dataset.get_relative_pose(i, i + 1)
                gt_pose = np.concatenate([gt['translation'], gt['rotation']])
                gt_poses.append(gt_pose)
            
            # Convert to numpy arrays
            predictions = np.array(predictions)
            gt_poses = np.array(gt_poses)
            
            # Integrate trajectories
            pred_trajectory = integrate_trajectory(initial_pose, predictions)
            gt_trajectory = integrate_trajectory(initial_pose, gt_poses)
            
            # Compute errors
            errors = compute_trajectory_error(pred_trajectory, gt_trajectory)
            
            # Save results
            result = {
                'sequence_idx': seq_idx,
                'start_frame': start_idx,
                'length': sequence_length,
                'pred_trajectory': pred_trajectory,
                'gt_trajectory': gt_trajectory,
                'errors': errors,
                'predictions': predictions,
                'ground_truth': gt_poses
            }
            results.append(result)
            
            # Visualize trajectory
            trajectories = {
                'Ground Truth': gt_trajectory,
                'Predicted': pred_trajectory
            }
            
            # 3D plot
            plot_trajectory_3d(
                trajectories,
                title=f"Sequence {seq_idx + 1} - 3D Trajectory",
                save_path=os.path.join(self.output_dir, f'trajectory_3d_seq{seq_idx}.png')
            )
            
            # 2D plots
            for view in ['xy', 'xz', 'yz']:
                plot_trajectory_2d(
                    trajectories,
                    view=view,
                    title=f"Sequence {seq_idx + 1} - {view.upper()} View",
                    save_path=os.path.join(self.output_dir, f'trajectory_{view}_seq{seq_idx}.png')
                )
            
            # Error over time
            trans_errors = np.linalg.norm(predictions[:, :3] - gt_poses[:, :3], axis=1)
            rot_errors = np.linalg.norm(predictions[:, 3:] - gt_poses[:, 3:], axis=1)
            
            plot_errors_over_time(
                {
                    'Translation Error': trans_errors,
                    'Rotation Error': rot_errors
                },
                title=f"Sequence {seq_idx + 1} - Errors",
                save_path=os.path.join(self.output_dir, f'errors_seq{seq_idx}.png')
            )
            
            # Print errors
            print(f"  ATE: {errors['ate']:.4f} ± {errors['ate_std']:.4f} m")
            print(f"  RPE (trans): {errors['rpe_trans']:.4f} ± {errors['rpe_trans_std']:.4f} m")
            print(f"  RPE (rot): {errors['rpe_rot']:.4f} ± {errors['rpe_rot_std']:.4f} rad")
        
        return results
    
    def create_summary_report(self, results):
        """Create a summary report of all evaluations."""
        print("\nCreating summary report...")
        
        # Aggregate errors
        all_ate = [r['errors']['ate'] for r in results]
        all_rpe_trans = [r['errors']['rpe_trans'] for r in results]
        all_rpe_rot = [r['errors']['rpe_rot'] for r in results]
        
        # Create summary statistics
        summary = {
            'num_sequences': len(results),
            'ate_mean': np.mean(all_ate),
            'ate_std': np.std(all_ate),
            'rpe_trans_mean': np.mean(all_rpe_trans),
            'rpe_trans_std': np.std(all_rpe_trans),
            'rpe_rot_mean': np.mean(all_rpe_rot),
            'rpe_rot_std': np.std(all_rpe_rot)
        }
        
        # Save summary
        with open(os.path.join(self.output_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create summary plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # ATE boxplot
        ax = axes[0]
        ax.boxplot(all_ate)
        ax.set_ylabel('ATE (m)')
        ax.set_title(f'Absolute Trajectory Error\nMean: {summary["ate_mean"]:.4f} ± {summary["ate_std"]:.4f} m')
        ax.grid(True, alpha=0.3)
        
        # RPE Translation boxplot
        ax = axes[1]
        ax.boxplot(all_rpe_trans)
        ax.set_ylabel('RPE Translation (m)')
        ax.set_title(f'Relative Pose Error (Translation)\nMean: {summary["rpe_trans_mean"]:.4f} ± {summary["rpe_trans_std"]:.4f} m')
        ax.grid(True, alpha=0.3)
        
        # RPE Rotation boxplot
        ax = axes[2]
        ax.boxplot(np.degrees(all_rpe_rot))
        ax.set_ylabel('RPE Rotation (deg)')
        ax.set_title(f'Relative Pose Error (Rotation)\nMean: {np.degrees(summary["rpe_rot_mean"]):.2f} ± {np.degrees(summary["rpe_rot_std"]):.2f} deg')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Model Performance Summary', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_summary.png'), dpi=150)
        plt.close()
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Evaluated on {summary['num_sequences']} sequences")
        print(f"\nAbsolute Trajectory Error (ATE):")
        print(f"  Mean: {summary['ate_mean']:.4f} m")
        print(f"  Std:  {summary['ate_std']:.4f} m")
        print(f"\nRelative Pose Error - Translation:")
        print(f"  Mean: {summary['rpe_trans_mean']:.4f} m")
        print(f"  Std:  {summary['rpe_trans_std']:.4f} m")
        print(f"\nRelative Pose Error - Rotation:")
        print(f"  Mean: {np.degrees(summary['rpe_rot_mean']):.2f}°")
        print(f"  Std:  {np.degrees(summary['rpe_rot_std']):.2f}°")
        print("="*60)
        
        return summary
    
    def visualize_predictions(self, num_samples=10):
        """Visualize sample predictions with images."""
        print(f"\nVisualizing {num_samples} sample predictions...")
        
        # Get random indices
        total_frames = len(self.dataset.poses_df) - 1
        sample_indices = np.random.choice(total_frames, size=min(num_samples, total_frames), replace=False)
        
        image_pairs = []
        predictions = []
        ground_truth = []
        
        for idx in sample_indices:
            # Load images
            img1 = self.dataset.load_image(idx)
            img2 = self.dataset.load_image(idx + 1)
            image_pairs.append((img1, img2))
            
            # Get prediction
            img_pair = np.concatenate([img1, img2], axis=-1)
            img_batch = np.expand_dims(img_pair, axis=0)
            pred = self.model.predict(img_batch, verbose=0)[0]
            predictions.append(pred)
            
            # Get ground truth
            gt = self.dataset.get_relative_pose(idx, idx + 1)
            gt_pose = np.concatenate([gt['translation'], gt['rotation']])
            ground_truth.append(gt_pose)
        
        # Visualize
        visualize_image_pairs(
            image_pairs,
            predictions=np.array(predictions),
            ground_truth=np.array(ground_truth),
            save_dir=os.path.join(self.output_dir, 'sample_predictions')
        )
    
    def plot_training_curves(self):
        """Plot training history if available."""
        history_path = os.path.join(self.model_dir, 'history.json')
        
        if os.path.exists(history_path):
            print("\nPlotting training curves...")
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            plot_training_history(
                history,
                save_path=os.path.join(self.output_dir, 'training_history.png')
            )
        else:
            print("Warning: training history not found")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        # Load model and dataset
        self.load_model()
        val_indices = self.load_dataset()
        
        # Plot training curves
        self.plot_training_curves()
        
        # Evaluate on sequences
        results = self.evaluate_on_sequences(num_sequences=5, sequence_length=100)
        
        # Create summary report
        self.create_summary_report(results)
        
        # Visualize sample predictions
        self.visualize_predictions(num_samples=10)
        
        print(f"\nEvaluation complete! Results saved to: {self.output_dir}")
        print("\nGenerated outputs:")
        print("  - trajectory_*.png: Predicted vs ground truth trajectories")
        print("  - errors_*.png: Error plots over time")
        print("  - performance_summary.png: Overall performance statistics")
        print("  - training_history.png: Training curves")
        print("  - sample_predictions/: Sample image pairs with predictions")
        print("  - evaluation_summary.json: Numerical results")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained visual odometry model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--data_dir', type=str, default='/app/data/raw',
                        help='Path to dataset directory')
    
    args = parser.parse_args()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return
    
    # Run evaluation
    evaluator = ModelEvaluator(args.model_dir, args.data_dir)
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()