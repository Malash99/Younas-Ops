#!/usr/bin/env python3
"""
Simple trajectory plotting without complex error computation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import cv2

from models.baseline.baseline_cnn import create_model


class SimpleTrajectoryPlotter:
    """Simple trajectory plotting."""
    
    def __init__(self, model_dir: str, processed_data_dir: str, subsample_rate: int = 20):
        self.model_dir = model_dir
        self.processed_data_dir = processed_data_dir
        self.subsample_rate = subsample_rate
        self.output_dir = os.path.join(model_dir, 'simple_trajectory_plot')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load config
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import json
                self.config = json.load(f)
        else:
            self.config = {'image_height': 224, 'image_width': 224}
    
    def load_model(self):
        """Load the trained model."""
        print("Loading model...")
        
        self.model = create_model(
            model_type=self.config.get('model_type', 'baseline'),
            input_shape=(224, 224, 6)
        )
        
        # Build model first
        dummy_input = tf.zeros((1, 224, 224, 6))
        _ = self.model(dummy_input, training=False)
        
        # Load weights
        best_weights = os.path.join(self.model_dir, 'best_model.h5')
        if os.path.exists(best_weights):
            self.model.load_weights(best_weights)
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError("No model weights found!")
    
    def load_image(self, bag_idx: int, frame_idx: int):
        """Load and preprocess image."""
        img_path = os.path.join(
            self.processed_data_dir, 
            f'bag_{bag_idx}', 
            'images', 
            f'frame_{frame_idx:06d}.png'
        )
        
        if not os.path.exists(img_path):
            return None
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_poses_data(self):
        """Load poses data from all sequential bags."""
        print("Loading poses data from bags 0-3...")
        
        all_poses = []
        bag_boundaries = [0]
        
        for bag_idx in range(4):
            poses_path = os.path.join(
                self.processed_data_dir,
                f'bag_{bag_idx}',
                'extracted_poses.csv'
            )
            
            df = pd.read_csv(poses_path)
            all_poses.append(df)
            bag_boundaries.append(bag_boundaries[-1] + len(df))
            
            print(f"  Bag {bag_idx}: {len(df)} poses")
        
        self.poses_df = pd.concat(all_poses, ignore_index=True)
        self.bag_boundaries = bag_boundaries
        
        print(f"Total poses: {len(self.poses_df)}")
        return self.poses_df
    
    def global_to_bag_frame(self, global_idx: int):
        """Convert global frame index to (bag_idx, frame_idx)."""
        for bag_idx in range(len(self.bag_boundaries) - 1):
            if self.bag_boundaries[bag_idx] <= global_idx < self.bag_boundaries[bag_idx + 1]:
                frame_idx = global_idx - self.bag_boundaries[bag_idx]
                return bag_idx, frame_idx
        
        raise ValueError(f"Global index {global_idx} out of bounds")
    
    def run_evaluation(self):
        """Run simple trajectory comparison."""
        print("Running simple trajectory evaluation...")
        
        self.load_model()
        self.load_poses_data()
        
        # Create subsampled indices
        total_frames = len(self.poses_df)
        subsampled_indices = list(range(0, total_frames - self.subsample_rate, self.subsample_rate))
        
        print(f"Processing {len(subsampled_indices)} frame pairs (subsampled 1:{self.subsample_rate})...")
        
        predictions = []
        actual_positions = []
        predicted_positions = []
        frame_indices = []
        
        # Simple integration - just accumulate relative motions
        current_pred_pos = np.array([0., 0., 0.])
        
        for i, global_idx in tqdm(enumerate(subsampled_indices), total=len(subsampled_indices), desc="Processing"):
            if global_idx + self.subsample_rate >= len(self.poses_df):
                continue
                
            try:
                # Get actual positions for ground truth trajectory
                pose_start = self.poses_df.iloc[global_idx]
                pose_end = self.poses_df.iloc[global_idx + self.subsample_rate]
                
                actual_start = np.array([pose_start['x'], pose_start['y'], pose_start['z']])
                actual_end = np.array([pose_end['x'], pose_end['y'], pose_end['z']])
                
                # For first frame, initialize predicted position to match actual
                if i == 0:
                    current_pred_pos = actual_start.copy()
                
                # Load images for prediction
                bag1, frame1 = self.global_to_bag_frame(global_idx)
                bag2, frame2 = self.global_to_bag_frame(global_idx + self.subsample_rate)
                
                img1 = self.load_image(bag1, frame1)
                img2 = self.load_image(bag2, frame2)
                
                if img1 is None or img2 is None:
                    continue
                
                # Predict relative motion
                img_pair = np.concatenate([img1, img2], axis=-1)
                img_batch = np.expand_dims(img_pair, axis=0)
                
                pred = self.model.predict(img_batch, verbose=0)[0]
                
                # Simple integration: add translation component
                rel_translation = pred[:3]
                current_pred_pos += rel_translation
                
                # Store results
                predictions.append(pred)
                actual_positions.append(actual_end)
                predicted_positions.append(current_pred_pos.copy())
                frame_indices.append(global_idx)
                
            except Exception as e:
                print(f"Error processing frame {global_idx}: {e}")
                continue
        
        actual_positions = np.array(actual_positions)
        predicted_positions = np.array(predicted_positions)
        
        print(f"Successfully processed {len(actual_positions)} frame pairs")
        
        # Create visualization
        self.plot_trajectories(actual_positions, predicted_positions, frame_indices)
        
        return {
            'actual_positions': actual_positions,
            'predicted_positions': predicted_positions,
            'frame_indices': frame_indices,
            'bag_boundaries': self.bag_boundaries
        }
    
    def plot_trajectories(self, actual_positions, predicted_positions, frame_indices):
        """Plot actual vs predicted trajectories."""
        print("Creating trajectory plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # XY trajectory
        ax = axes[0, 0]
        ax.plot(actual_positions[:, 0], actual_positions[:, 1], 'b-', 
               label='Ground Truth', linewidth=2, alpha=0.8)
        ax.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'r--', 
               label='Predicted', linewidth=2, alpha=0.8)
        
        # Mark start and end
        ax.scatter(actual_positions[0, 0], actual_positions[0, 1], 
                  color='green', s=100, marker='s', label='Start', zorder=5)
        ax.scatter(actual_positions[-1, 0], actual_positions[-1, 1], 
                  color='red', s=100, marker='s', label='End', zorder=5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'XY Trajectory (Subsampled 1:{self.subsample_rate})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # XZ trajectory
        ax = axes[0, 1]
        ax.plot(actual_positions[:, 0], actual_positions[:, 2], 'b-', 
               label='Ground Truth', linewidth=2, alpha=0.8)
        ax.plot(predicted_positions[:, 0], predicted_positions[:, 2], 'r--', 
               label='Predicted', linewidth=2, alpha=0.8)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title('XZ Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # YZ trajectory
        ax = axes[1, 0]
        ax.plot(actual_positions[:, 1], actual_positions[:, 2], 'b-', 
               label='Ground Truth', linewidth=2, alpha=0.8)
        ax.plot(predicted_positions[:, 1], predicted_positions[:, 2], 'r--', 
               label='Predicted', linewidth=2, alpha=0.8)
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title('YZ Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Position errors over time
        ax = axes[1, 1]
        position_errors = np.linalg.norm(predicted_positions - actual_positions, axis=1)
        ax.plot(range(len(position_errors)), position_errors, 'g-', linewidth=2)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Position Error (m)')
        ax.set_title('Position Error Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(position_errors)
        std_error = np.std(position_errors)
        ax.text(0.05, 0.95, f'Mean Error: {mean_error:.3f}m\\nStd Error: {std_error:.3f}m', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'trajectory_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed XY plot
        plt.figure(figsize=(12, 10))
        plt.plot(actual_positions[:, 0], actual_positions[:, 1], 'b-', 
                label='Ground Truth', linewidth=3, alpha=0.8)
        plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'r--', 
                label='Predicted', linewidth=3, alpha=0.8)
        
        # Mark start and end
        plt.scatter(actual_positions[0, 0], actual_positions[0, 1], 
                   color='green', s=150, marker='s', label='Start', zorder=5)
        plt.scatter(actual_positions[-1, 0], actual_positions[-1, 1], 
                   color='red', s=150, marker='s', label='End', zorder=5)
        
        # Add bag boundaries if available
        # Mark every 50th point for reference
        for i in range(0, len(actual_positions), 50):
            plt.scatter(actual_positions[i, 0], actual_positions[i, 1], 
                       color='blue', s=30, alpha=0.6, zorder=3)
            plt.scatter(predicted_positions[i, 0], predicted_positions[i, 1], 
                       color='red', s=30, alpha=0.6, zorder=3)
        
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        plt.title(f'Sequential Trajectory Comparison - Bags 0-3\\n(Subsampled 1:{self.subsample_rate}, {len(actual_positions)} points)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Add error statistics
        mean_error = np.mean(position_errors)
        std_error = np.std(position_errors)
        max_error = np.max(position_errors)
        
        plt.text(0.02, 0.98, f'Position Error Statistics:\\nMean: {mean_error:.3f} m\\nStd: {std_error:.3f} m\\nMax: {max_error:.3f} m', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=10)
        
        plt.savefig(os.path.join(self.output_dir, 'detailed_xy_trajectory.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {self.output_dir}")
        print(f"Mean position error: {mean_error:.3f} m")
        print(f"Std position error: {std_error:.3f} m")
        print(f"Max position error: {max_error:.3f} m")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple trajectory plotting')
    parser.add_argument('--model_dir', type=str, 
                       default='/app/output/models/model_20250708_023343',
                       help='Path to trained model directory')
    parser.add_argument('--data_dir', type=str, default='/app/data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--subsample', type=int, default=20,
                       help='Subsample rate (take every Nth frame)')
    
    args = parser.parse_args()
    
    plotter = SimpleTrajectoryPlotter(args.model_dir, args.data_dir, args.subsample)
    results = plotter.run_evaluation()


if __name__ == "__main__":
    main()