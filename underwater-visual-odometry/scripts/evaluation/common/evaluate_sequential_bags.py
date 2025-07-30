#!/usr/bin/env python3
"""
Evaluate trained model on sequential bags 0-3 and compare predicted vs actual trajectory.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import cv2

from models.baseline_cnn import create_model
from coordinate_transforms import integrate_trajectory, compute_trajectory_error
from visualization import plot_trajectory_3d, plot_trajectory_2d


class SequentialBagEvaluator:
    """Evaluate model on sequential bags (0-3)."""
    
    def __init__(self, model_dir: str, processed_data_dir: str):
        """
        Initialize evaluator.
        
        Args:
            model_dir: Directory containing trained model
            processed_data_dir: Directory containing processed bag data
        """
        self.model_dir = model_dir
        self.processed_data_dir = processed_data_dir
        self.output_dir = os.path.join(model_dir, 'sequential_evaluation')
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
        
        # Build model by calling it with dummy input BEFORE loading weights
        dummy_input = tf.zeros((1, 
                               self.config.get('image_height', 224),
                               self.config.get('image_width', 224),
                               6))
        _ = self.model(dummy_input, training=False)
        
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
        
        print("Model loaded successfully!")
    
    def load_image(self, bag_idx: int, frame_idx: int):
        """Load and preprocess image."""
        img_path = os.path.join(
            self.processed_data_dir, 
            f'bag_{bag_idx}', 
            'images', 
            f'frame_{frame_idx:06d}.png'
        )
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img_size = (self.config.get('image_height', 224), 
                   self.config.get('image_width', 224))
        img = cv2.resize(img, img_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_poses_data(self):
        """Load poses data from all sequential bags."""
        print("Loading poses data from bags 0-3...")
        
        all_poses = []
        bag_boundaries = [0]  # Track where each bag starts/ends
        
        for bag_idx in range(4):  # bags 0-3
            poses_path = os.path.join(
                self.processed_data_dir,
                f'bag_{bag_idx}',
                'extracted_poses.csv'
            )
            
            if not os.path.exists(poses_path):
                raise FileNotFoundError(f"Poses file not found: {poses_path}")
            
            df = pd.read_csv(poses_path)
            all_poses.append(df)
            bag_boundaries.append(bag_boundaries[-1] + len(df))
            
            print(f"  Bag {bag_idx}: {len(df)} poses")
        
        # Concatenate all poses
        self.poses_df = pd.concat(all_poses, ignore_index=True)
        self.bag_boundaries = bag_boundaries
        
        print(f"Total poses: {len(self.poses_df)}")
        return self.poses_df
    
    def get_actual_relative_pose(self, global_idx1: int, global_idx2: int):
        """Get actual relative pose between two global frame indices."""
        # Get poses
        pose1 = self.poses_df.iloc[global_idx1]
        pose2 = self.poses_df.iloc[global_idx2]
        
        # Extract positions and orientations
        pos1 = np.array([pose1['x'], pose1['y'], pose1['z']])
        pos2 = np.array([pose2['x'], pose2['y'], pose2['z']])
        
        quat1 = np.array([pose1['qw'], pose1['qx'], pose1['qy'], pose1['qz']])
        quat2 = np.array([pose2['qw'], pose2['qx'], pose2['qy'], pose2['qz']])
        
        # Compute relative transformation
        # Translation
        rel_pos = pos2 - pos1
        
        # Rotation (simplified - just use quaternion difference)
        # In practice, you'd compute proper relative rotation
        rel_quat = quat2 - quat1
        
        return np.concatenate([rel_pos, rel_quat])
    
    def global_to_bag_frame(self, global_idx: int):
        """Convert global frame index to (bag_idx, frame_idx)."""
        for bag_idx in range(len(self.bag_boundaries) - 1):
            if self.bag_boundaries[bag_idx] <= global_idx < self.bag_boundaries[bag_idx + 1]:
                frame_idx = global_idx - self.bag_boundaries[bag_idx]
                return bag_idx, frame_idx
        
        raise ValueError(f"Global index {global_idx} out of bounds")
    
    def evaluate_sequential_trajectory(self):
        """Evaluate model on full sequential trajectory across bags 0-3."""
        print("Evaluating on sequential trajectory across bags 0-3...")
        
        # Load poses data
        self.load_poses_data()
        
        # We'll predict from frame 0 to frame N-1 (last possible pair)
        total_frames = len(self.poses_df)
        max_predictions = total_frames - 1
        
        print(f"Making {max_predictions} sequential predictions...")
        
        predictions = []
        ground_truth = []
        
        # Process frame by frame
        for global_idx in tqdm(range(max_predictions), desc="Processing frames"):
            try:
                # Convert global indices to bag/frame indices
                bag1, frame1 = self.global_to_bag_frame(global_idx)
                bag2, frame2 = self.global_to_bag_frame(global_idx + 1)
                
                # Load images
                img1 = self.load_image(bag1, frame1)
                img2 = self.load_image(bag2, frame2)
                
                # Stack images for model input
                img_pair = np.concatenate([img1, img2], axis=-1)
                img_batch = np.expand_dims(img_pair, axis=0)
                
                # Predict relative pose
                pred = self.model.predict(img_batch, verbose=0)[0]
                predictions.append(pred)
                
                # Get ground truth relative pose
                gt = self.get_actual_relative_pose(global_idx, global_idx + 1)
                ground_truth.append(gt)
                
            except Exception as e:
                print(f"Error processing frame {global_idx}: {e}")
                continue
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        print(f"Successfully processed {len(predictions)} frame pairs")
        
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
        
        # Save results
        results = {
            'predictions': predictions.tolist(),
            'ground_truth': ground_truth.tolist(),
            'pred_trajectory': pred_trajectory,
            'gt_trajectory': gt_trajectory,
            'errors': errors,
            'bag_boundaries': self.bag_boundaries,
            'total_frames': total_frames
        }
        
        # Save results to JSON
        with open(os.path.join(self.output_dir, 'sequential_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'errors': errors,
                'bag_boundaries': self.bag_boundaries,
                'total_frames': total_frames,
                'num_predictions': len(predictions)
            }
            json.dump(json_results, f, indent=2)
        
        return results
    
    def plot_results(self, results):
        """Plot predicted vs actual trajectory."""
        print("Creating visualizations...")
        
        pred_trajectory = results['pred_trajectory']
        gt_trajectory = results['gt_trajectory']
        bag_boundaries = results['bag_boundaries']
        
        trajectories = {
            'Ground Truth': gt_trajectory,
            'Predicted': pred_trajectory
        }
        
        # Create enhanced trajectory plot with bag boundaries
        fig = plt.figure(figsize=(15, 10))
        
        # 3D plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Plot trajectories
        for name, traj in trajectories.items():
            positions = np.array([p['position'] for p in traj])
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    label=name, linewidth=2)
        
        # Mark bag boundaries
        for i, boundary in enumerate(bag_boundaries[1:-1], 1):  # Skip first and last
            if boundary < len(gt_trajectory):
                pos = gt_trajectory[boundary]['position']
                ax1.scatter(pos[0], pos[1], pos[2], 
                           color='red', s=100, marker='o',
                           label=f'Bag {i} start' if i == 1 else "")
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory - Sequential Bags 0-3')
        ax1.legend()
        ax1.grid(True)
        
        # XY plot
        ax2 = fig.add_subplot(222)
        for name, traj in trajectories.items():
            positions = np.array([p['position'] for p in traj])
            ax2.plot(positions[:, 0], positions[:, 1], label=name, linewidth=2)
        
        # Mark bag boundaries
        for i, boundary in enumerate(bag_boundaries[1:-1], 1):
            if boundary < len(gt_trajectory):
                pos = gt_trajectory[boundary]['position']
                ax2.scatter(pos[0], pos[1], color='red', s=50, marker='o')
                ax2.annotate(f'Bag {i}', (pos[0], pos[1]), 
                           xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY View - Sequential Trajectory')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # Error plots
        predictions = results['predictions']
        ground_truth = results['ground_truth']
        
        trans_errors = np.linalg.norm(predictions[:, :3] - ground_truth[:, :3], axis=1)
        rot_errors = np.linalg.norm(predictions[:, 3:] - ground_truth[:, 3:], axis=1)
        
        # Translation error over time
        ax3 = fig.add_subplot(223)
        ax3.plot(trans_errors, label='Translation Error', color='blue')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Translation Error (m)')
        ax3.set_title('Translation Error Over Time')
        ax3.grid(True)
        
        # Add vertical lines at bag boundaries
        for boundary in bag_boundaries[1:-1]:
            if boundary < len(trans_errors):
                ax3.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        
        # Rotation error over time  
        ax4 = fig.add_subplot(224)
        ax4.plot(rot_errors, label='Rotation Error', color='orange')
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Rotation Error (rad)')
        ax4.set_title('Rotation Error Over Time')
        ax4.grid(True)
        
        # Add vertical lines at bag boundaries
        for boundary in bag_boundaries[1:-1]:
            if boundary < len(rot_errors):
                ax4.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sequential_trajectory_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate detailed trajectory plot
        plt.figure(figsize=(12, 8))
        
        for name, traj in trajectories.items():
            positions = np.array([p['position'] for p in traj])
            plt.plot(positions[:, 0], positions[:, 1], label=name, linewidth=2)
        
        # Mark start and end points
        start_pos = np.array(gt_trajectory[0]['position'])
        end_pos = np.array(gt_trajectory[-1]['position'])
        
        plt.scatter(start_pos[0], start_pos[1], color='green', s=100, 
                   marker='s', label='Start', zorder=5)
        plt.scatter(end_pos[0], end_pos[1], color='red', s=100, 
                   marker='s', label='End', zorder=5)
        
        # Mark bag boundaries with different colors
        colors = ['red', 'orange', 'yellow']
        for i, boundary in enumerate(bag_boundaries[1:-1], 1):
            if boundary < len(gt_trajectory):
                pos = gt_trajectory[boundary]['position']
                plt.scatter(pos[0], pos[1], color=colors[i-1], s=80, 
                           marker='o', label=f'Bag {i} start', zorder=4)
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Sequential Trajectory: Bags 0-3 (XY View)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.savefig(os.path.join(self.output_dir, 'sequential_trajectory_detailed.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved!")
    
    def print_summary(self, results):
        """Print evaluation summary."""
        errors = results['errors']
        
        print("\n" + "="*60)
        print("SEQUENTIAL EVALUATION SUMMARY")
        print("="*60)
        print(f"Processed bags: 0, 1, 2, 3")
        print(f"Total frames: {results['total_frames']}")
        print(f"Total predictions: {len(results['predictions'])}")
        print(f"\nBag boundaries: {results['bag_boundaries']}")
        
        print(f"\nTrajectory Errors:")
        print(f"  Absolute Trajectory Error (ATE): {errors['ate']:.4f} ± {errors['ate_std']:.4f} m")
        print(f"  Relative Pose Error (Translation): {errors['rpe_trans']:.4f} ± {errors['rpe_trans_std']:.4f} m")
        print(f"  Relative Pose Error (Rotation): {np.degrees(errors['rpe_rot']):.2f} ± {np.degrees(errors['rpe_rot_std']):.2f} deg")
        
        print(f"\nOutput saved to: {self.output_dir}")
        print("  - sequential_trajectory_analysis.png: Full analysis")
        print("  - sequential_trajectory_detailed.png: Detailed XY view")
        print("  - sequential_results.json: Numerical results")
        print("="*60)
    
    def run_evaluation(self):
        """Run complete sequential evaluation."""
        # Load model
        self.load_model()
        
        # Evaluate on sequential trajectory
        results = self.evaluate_sequential_trajectory()
        
        # Create visualizations
        self.plot_results(results)
        
        # Print summary
        self.print_summary(results)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on sequential bags 0-3')
    parser.add_argument('--model_dir', type=str, 
                       default='/app/output/models/model_20250708_023343',
                       help='Path to trained model directory')
    parser.add_argument('--data_dir', type=str, default='/app/data/processed',
                       help='Path to processed data directory')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return
    
    # Run evaluation
    evaluator = SequentialBagEvaluator(args.model_dir, args.data_dir)
    results = evaluator.run_evaluation()


if __name__ == "__main__":
    main()