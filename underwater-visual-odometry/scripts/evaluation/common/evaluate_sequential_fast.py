#!/usr/bin/env python3
"""
Fast evaluation on sequential bags with subsampling for quicker results.
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


class FastSequentialEvaluator:
    """Fast evaluation with subsampling."""
    
    def __init__(self, model_dir: str, processed_data_dir: str, subsample_rate: int = 10):
        """
        Initialize evaluator.
        
        Args:
            model_dir: Directory containing trained model
            processed_data_dir: Directory containing processed bag data
            subsample_rate: Take every Nth frame (default: 10)
        """
        self.model_dir = model_dir
        self.processed_data_dir = processed_data_dir
        self.subsample_rate = subsample_rate
        self.output_dir = os.path.join(model_dir, 'fast_sequential_evaluation')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load config
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'model_type': 'baseline',
                'image_height': 224,
                'image_width': 224,
            }
    
    def load_model(self):
        """Load the trained model."""
        print("Loading model...")
        
        self.model = create_model(
            model_type=self.config.get('model_type', 'baseline'),
            input_shape=(
                self.config.get('image_height', 224),
                self.config.get('image_width', 224),
                6
            )
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
    
    def get_actual_relative_pose(self, global_idx1: int, global_idx2: int):
        """Get actual relative pose between two global frame indices."""
        pose1 = self.poses_df.iloc[global_idx1]
        pose2 = self.poses_df.iloc[global_idx2]
        
        pos1 = np.array([pose1['x'], pose1['y'], pose1['z']])
        pos2 = np.array([pose2['x'], pose2['y'], pose2['z']])
        
        # Simple relative translation
        rel_pos = pos2 - pos1
        
        # For rotation, use simple difference approximation
        # This is a simplified approach - proper SE(3) computation would be more complex
        quat1 = np.array([pose1['qw'], pose1['qx'], pose1['qy'], pose1['qz']])
        quat2 = np.array([pose2['qw'], pose2['qx'], pose2['qy'], pose2['qz']])
        
        # Convert to rotation vector approximation (small angle assumption)
        rel_rot = (quat2[1:4] - quat1[1:4]) * 2  # Simplified quaternion difference
        
        return np.concatenate([rel_pos, rel_rot])
    
    def global_to_bag_frame(self, global_idx: int):
        """Convert global frame index to (bag_idx, frame_idx)."""
        for bag_idx in range(len(self.bag_boundaries) - 1):
            if self.bag_boundaries[bag_idx] <= global_idx < self.bag_boundaries[bag_idx + 1]:
                frame_idx = global_idx - self.bag_boundaries[bag_idx]
                return bag_idx, frame_idx
        
        raise ValueError(f"Global index {global_idx} out of bounds")
    
    def evaluate_subsampled_trajectory(self):
        """Evaluate model on subsampled sequential trajectory."""
        print(f"Evaluating on subsampled trajectory (every {self.subsample_rate} frames)...")
        
        self.load_poses_data()
        
        # Create subsampled indices
        total_frames = len(self.poses_df)
        subsampled_indices = list(range(0, total_frames - 1, self.subsample_rate))
        
        print(f"Processing {len(subsampled_indices)} frame pairs...")
        
        predictions = []
        ground_truth = []
        frame_indices = []
        
        # Batch processing for speed
        batch_size = 32
        
        for i in tqdm(range(0, len(subsampled_indices), batch_size), desc="Processing batches"):
            batch_indices = subsampled_indices[i:i+batch_size]
            batch_imgs = []
            batch_gt = []
            valid_indices = []
            
            # Prepare batch
            for global_idx in batch_indices:
                if global_idx + self.subsample_rate >= len(self.poses_df):
                    continue
                    
                try:
                    bag1, frame1 = self.global_to_bag_frame(global_idx)
                    bag2, frame2 = self.global_to_bag_frame(global_idx + self.subsample_rate)
                    
                    img1 = self.load_image(bag1, frame1)
                    img2 = self.load_image(bag2, frame2)
                    
                    if img1 is None or img2 is None:
                        continue
                    
                    img_pair = np.concatenate([img1, img2], axis=-1)
                    batch_imgs.append(img_pair)
                    
                    gt = self.get_actual_relative_pose(global_idx, global_idx + self.subsample_rate)
                    batch_gt.append(gt)
                    valid_indices.append(global_idx)
                    
                except Exception as e:
                    continue
            
            # Predict batch
            if len(batch_imgs) > 0:
                batch_imgs = np.array(batch_imgs)
                batch_preds = self.model.predict(batch_imgs, verbose=0)
                
                predictions.extend(batch_preds)
                ground_truth.extend(batch_gt)
                frame_indices.extend(valid_indices)
        
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
        
        return {
            'predictions': predictions,
            'ground_truth': ground_truth,
            'pred_trajectory': pred_trajectory,
            'gt_trajectory': gt_trajectory,
            'errors': errors,
            'bag_boundaries': self.bag_boundaries,
            'frame_indices': frame_indices,
            'subsample_rate': self.subsample_rate
        }
    
    def plot_results(self, results):
        """Plot predicted vs actual trajectory."""
        print("Creating visualizations...")
        
        pred_trajectory = results['pred_trajectory']
        gt_trajectory = results['gt_trajectory']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract positions
        pred_positions = np.array([p['position'] for p in pred_trajectory])
        gt_positions = np.array([p['position'] for p in gt_trajectory])
        
        # XY trajectory
        ax1.plot(gt_positions[:, 0], gt_positions[:, 1], 'b-', label='Ground Truth', linewidth=2)
        ax1.plot(pred_positions[:, 0], pred_positions[:, 1], 'r--', label='Predicted', linewidth=2)
        ax1.scatter(gt_positions[0, 0], gt_positions[0, 1], color='green', s=100, marker='s', label='Start', zorder=5)
        ax1.scatter(gt_positions[-1, 0], gt_positions[-1, 1], color='red', s=100, marker='s', label='End', zorder=5)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'XY Trajectory (Subsampled 1:{self.subsample_rate})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # XZ trajectory
        ax2.plot(gt_positions[:, 0], gt_positions[:, 2], 'b-', label='Ground Truth', linewidth=2)
        ax2.plot(pred_positions[:, 0], pred_positions[:, 2], 'r--', label='Predicted', linewidth=2)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('XZ Trajectory')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Translation errors
        trans_errors = np.linalg.norm(results['predictions'][:, :3] - results['ground_truth'][:, :3], axis=1)
        ax3.plot(trans_errors, 'b-', linewidth=1)
        ax3.set_xlabel('Frame Index')
        ax3.set_ylabel('Translation Error (m)')
        ax3.set_title('Translation Error Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Rotation errors
        rot_errors = np.linalg.norm(results['predictions'][:, 3:] - results['ground_truth'][:, 3:], axis=1)
        ax4.plot(rot_errors, 'r-', linewidth=1)
        ax4.set_xlabel('Frame Index')
        ax4.set_ylabel('Rotation Error (rad)')
        ax4.set_title('Rotation Error Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fast_trajectory_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualization saved!")
    
    def print_summary(self, results):
        """Print evaluation summary."""
        errors = results['errors']
        
        print("\n" + "="*60)
        print("FAST SEQUENTIAL EVALUATION SUMMARY")
        print("="*60)
        print(f"Subsample rate: 1:{self.subsample_rate}")
        print(f"Processed frame pairs: {len(results['predictions'])}")
        
        print(f"\nTrajectory Errors:")
        print(f"  ATE: {errors['ate']:.4f} ± {errors['ate_std']:.4f} m")
        print(f"  RPE (trans): {errors['rpe_trans']:.4f} ± {errors['rpe_trans_std']:.4f} m")
        print(f"  RPE (rot): {np.degrees(errors['rpe_rot']):.2f} ± {np.degrees(errors['rpe_rot_std']):.2f} deg")
        
        print(f"\nOutput saved to: {self.output_dir}")
        print("="*60)
    
    def run_evaluation(self):
        """Run complete fast evaluation."""
        self.load_model()
        results = self.evaluate_subsampled_trajectory()
        self.plot_results(results)
        self.print_summary(results)
        return results


def main():
    parser = argparse.ArgumentParser(description='Fast evaluation on sequential bags')
    parser.add_argument('--model_dir', type=str, 
                       default='/app/output/models/model_20250708_023343',
                       help='Path to trained model directory')
    parser.add_argument('--data_dir', type=str, default='/app/data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--subsample', type=int, default=10,
                       help='Subsample rate (take every Nth frame)')
    
    args = parser.parse_args()
    
    evaluator = FastSequentialEvaluator(args.model_dir, args.data_dir, args.subsample)
    results = evaluator.run_evaluation()


if __name__ == "__main__":
    main()