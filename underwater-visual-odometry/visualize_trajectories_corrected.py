#!/usr/bin/env python3
"""
CORRECTED Visualize TSformer-VO Predictions vs Ground Truth

Creates proper trajectory visualizations using:
- Ground Truth: Real world coordinates from CSV
- Predictions: Properly accumulated SE(3) transformations from pose deltas
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.transform import Rotation as R

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrajectoryVisualizerCorrected:
    """Properly visualize predicted vs ground truth trajectories"""
    
    def __init__(self, model_path, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Load model
        print("Loading trained model...")
        self.model, self.loss_fn = create_tsformer_vo(
            sequence_length=config['sequence_length'],
            pretrained=config['pretrained'],
            freeze_backbone=config['freeze_backbone'],
            image_size=config['image_size']
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        # Load CSV to get world coordinates
        print("Loading CSV for world coordinates...")
        self.df = pd.read_csv(config['csv_path'])
        print(f"CSV loaded: {len(self.df)} frames")
        
        # Load data loaders
        print("Loading datasets...")
        data_loaders = create_data_loaders(
            csv_path=config['csv_path'],
            data_root=config['data_root'],
            sequence_length=config['sequence_length'],
            overlap_frames=config['overlap_frames'],
            image_size=config['image_size'],
            batch_size=1,  # Use batch_size=1 for prediction
            test_bags=config['test_bags'],
            num_workers=0,  # Avoid multiprocessing issues
            camera=config['camera']
        )
        
        self.train_loader = data_loaders['train_loader']
        self.val_loader = data_loaders['val_loader'] 
        self.test_loader = data_loaders['test_loader']
        
    def get_ground_truth_trajectory(self, data_loader, dataset_name=""):
        """Extract ground truth trajectory from world coordinates in CSV"""
        print(f"Extracting {dataset_name} ground truth from world coordinates...")
        
        # Get frame indices from dataset
        frame_indices = []
        for batch in data_loader:
            # Get the frame ID from the dataset
            # We need to access the underlying dataset to get frame indices
            pass
        
        # Alternative approach: extract by bag names
        if dataset_name.lower() == "test":
            # Filter by test bags
            test_data = self.df[self.df['bag_name'].isin(self.config['test_bags'])]
        elif dataset_name.lower() == "train":
            # Filter by train bags (all except test)
            train_bags = [bag for bag in self.df['bag_name'].unique() 
                         if bag not in self.config['test_bags']]
            train_data = self.df[self.df['bag_name'].isin(train_bags)]
            # Take a subset for cleaner visualization
            test_data = train_data.iloc[:len(data_loader) * self.config['sequence_length']]
        else:  # validation
            # Similar to train but different subset
            train_bags = [bag for bag in self.df['bag_name'].unique() 
                         if bag not in self.config['test_bags']]
            all_train_data = self.df[self.df['bag_name'].isin(train_bags)]
            # Take validation subset (different from train)
            val_start = len(self.train_loader) * self.config['sequence_length']
            val_end = val_start + len(data_loader) * self.config['sequence_length']
            test_data = all_train_data.iloc[val_start:val_end]
        
        # Extract world coordinates
        world_positions = test_data[['world_x', 'world_y', 'world_z']].values
        world_orientations = test_data[['world_qx', 'world_qy', 'world_qz', 'world_qw']].values
        
        print(f"  {dataset_name} GT trajectory shape: {world_positions.shape}")
        
        return world_positions, world_orientations
    
    def predict_and_accumulate_trajectory(self, data_loader, max_samples=None, dataset_name=""):
        """Predict pose deltas and properly accumulate them using SE(3)"""
        print(f"Predicting and accumulating {dataset_name} trajectory...")
        
        predictions = []
        sample_count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if max_samples and sample_count >= max_samples:
                    break
                    
                images = batch['images'].to(self.device)
                pred_poses = self.model(images)
                
                # Get pose deltas for the sequence
                pose_deltas = pred_poses.cpu().numpy()  # (1, 6) for batch_size=1
                predictions.append(pose_deltas[0])  # Remove batch dimension
                
                sample_count += 1
                
                if (sample_count) % 50 == 0:
                    print(f"  Processed {sample_count}/{min(max_samples or len(data_loader), len(data_loader))} samples")
        
        predictions = np.array(predictions)  # (N, 6)
        print(f"  {dataset_name} predictions shape: {predictions.shape}")
        
        # Now properly accumulate using SE(3) transformations
        accumulated_trajectory = self.accumulate_se3_trajectory(predictions)
        
        return accumulated_trajectory, predictions
    
    def accumulate_se3_trajectory(self, pose_deltas):
        """Properly accumulate pose deltas using SE(3) transformations"""
        print("  Accumulating SE(3) transformations...")
        
        # Initialize trajectory
        trajectory = np.zeros((len(pose_deltas) + 1, 3))
        
        # Current transformation matrix
        current_T = np.eye(4)
        trajectory[0] = current_T[:3, 3]  # Initial position (origin)
        
        for i, delta in enumerate(pose_deltas):
            # Extract delta translation and rotation
            dt_pos = delta[:3]  # [dx, dy, dz]
            dt_rot = delta[3:]  # [droll, dpitch, dyaw]
            
            # Create delta transformation matrix
            delta_T = self.pose_delta_to_se3(dt_pos, dt_rot)
            
            # Compose transformations: T_new = T_current * T_delta
            current_T = current_T @ delta_T
            
            # Extract position
            trajectory[i + 1] = current_T[:3, 3]
        
        return trajectory
    
    def pose_delta_to_se3(self, translation, rotation):
        """Convert pose delta to SE(3) transformation matrix"""
        # Create rotation matrix from Euler angles
        rot_matrix = R.from_euler('xyz', rotation).as_matrix()
        
        # Create SE(3) transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = translation
        
        return T
    
    def quaternion_to_position(self, quat_data):
        """Convert quaternion data to positions (if needed)"""
        # This is just for reference - we use world_x, world_y, world_z directly
        pass
    
    def compute_trajectory_metrics(self, pred_traj, gt_traj):
        """Compute trajectory comparison metrics"""
        # Ensure same length
        min_len = min(len(pred_traj), len(gt_traj))
        pred_traj = pred_traj[:min_len]
        gt_traj = gt_traj[:min_len]
        
        # Align trajectories at start
        pred_aligned = pred_traj - pred_traj[0]
        gt_aligned = gt_traj - gt_traj[0]
        
        # Compute metrics
        mse = mean_squared_error(gt_aligned.flatten(), pred_aligned.flatten())
        mae = mean_absolute_error(gt_aligned.flatten(), pred_aligned.flatten())
        
        # Endpoint error
        endpoint_error = np.linalg.norm(pred_aligned[-1] - gt_aligned[-1])
        
        # Path length ratio
        pred_length = np.sum(np.linalg.norm(np.diff(pred_aligned, axis=0), axis=1))
        gt_length = np.sum(np.linalg.norm(np.diff(gt_aligned, axis=0), axis=1))
        length_ratio = pred_length / gt_length if gt_length > 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'endpoint_error': endpoint_error,
            'path_length_ratio': length_ratio,
            'pred_length': pred_length,
            'gt_length': gt_length
        }
    
    def plot_corrected_trajectory_comparison(self, pred_traj, gt_traj, title, save_path=None):
        """Create corrected trajectory comparison plot matching reference style"""
        
        # Ensure same length for comparison
        min_len = min(len(pred_traj), len(gt_traj))
        pred_traj = pred_traj[:min_len]
        gt_traj = gt_traj[:min_len]
        
        # Create figure with 3 subplots (XY, XZ, YZ planes)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'CORRECTED Trajectory Comparison - {title}\\nBlue=TRUE World Coordinates, Red=Model Predictions', 
                     fontsize=14, fontweight='bold')
        
        # XY Plane (Top View)
        ax1.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', linewidth=3, 
                label='TRUE Ground Truth (World Coordinates)', alpha=0.8)
        ax1.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, 
                label='Model Prediction (Accumulated Deltas)', alpha=0.8)
        
        # Mark start and end
        ax1.scatter(*gt_traj[0, :2], color='green', s=100, label='Start', marker='o', edgecolor='black')
        ax1.scatter(*gt_traj[-1, :2], color='blue', s=100, label='End', marker='s', edgecolor='black')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('XY Plane (Top View)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # XZ Plane (Side View)
        ax2.plot(gt_traj[:, 0], gt_traj[:, 2], 'b-', linewidth=3, 
                label='TRUE Ground Truth (World Coordinates)', alpha=0.8)
        ax2.plot(pred_traj[:, 0], pred_traj[:, 2], 'r--', linewidth=2, 
                label='Model Prediction (Accumulated Deltas)', alpha=0.8)
        
        ax2.scatter(*gt_traj[0, [0,2]], color='green', s=100, label='Start', marker='o', edgecolor='black')
        ax2.scatter(*gt_traj[-1, [0,2]], color='blue', s=100, label='End', marker='s', edgecolor='black')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('XZ Plane (Side View)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # YZ Plane (Front View)
        ax3.plot(gt_traj[:, 1], gt_traj[:, 2], 'b-', linewidth=3, 
                label='TRUE Ground Truth (World Coordinates)', alpha=0.8)
        ax3.plot(pred_traj[:, 1], pred_traj[:, 2], 'r--', linewidth=2, 
                label='Model Prediction (Accumulated Deltas)', alpha=0.8)
        
        ax3.scatter(*gt_traj[0, [1,2]], color='green', s=100, label='Start', marker='o', edgecolor='black')
        ax3.scatter(*gt_traj[-1, [1,2]], color='blue', s=100, label='End', marker='s', edgecolor='black')
        
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('YZ Plane (Front View)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Corrected plot saved: {save_path}")
        
        return fig
    
    def generate_corrected_visualizations(self, output_dir="corrected_visualizations"):
        """Generate corrected trajectory visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Generating CORRECTED visualizations in: {output_dir}")
        
        results = {}
        
        # Process each dataset
        datasets = [
            ("train", self.train_loader, 100),  # Limit train samples for cleaner viz
            ("val", self.val_loader, None),
            ("test", self.test_loader, None)
        ]
        
        for dataset_name, data_loader, max_samples in datasets:
            print(f"\\n{'='*50}")
            print(f"Processing {dataset_name.upper()} dataset")
            print(f"{'='*50}")
            
            # Get ground truth from world coordinates
            gt_positions, gt_orientations = self.get_ground_truth_trajectory(data_loader, dataset_name)
            
            # Get model predictions and accumulate properly
            pred_trajectory, pred_deltas = self.predict_and_accumulate_trajectory(
                data_loader, max_samples, dataset_name
            )
            
            # Ensure same length for comparison
            min_len = min(len(pred_trajectory), len(gt_positions))
            pred_trajectory = pred_trajectory[:min_len]
            gt_trajectory = gt_positions[:min_len]
            
            # Compute metrics
            metrics = self.compute_trajectory_metrics(pred_trajectory, gt_trajectory)
            
            # Store results
            results[dataset_name] = {
                'pred_traj': pred_trajectory,
                'gt_traj': gt_trajectory,
                'metrics': metrics
            }
            
            # Create corrected plot
            save_path = output_dir / f'corrected_{dataset_name}_trajectory_comparison.png'
            self.plot_corrected_trajectory_comparison(
                pred_trajectory, gt_trajectory, 
                f"{dataset_name.title()} Dataset", 
                save_path
            )
            
            # Print metrics
            print(f"{dataset_name.title()} Metrics:")
            print(f"  Endpoint Error: {metrics['endpoint_error']:.3f}m")
            print(f"  Path Length Ratio: {metrics['path_length_ratio']:.3f}")
            print(f"  MSE: {metrics['mse']:.2e}")
        
        # Save metrics
        metrics_summary = {f'{k}_metrics': v['metrics'] for k, v in results.items()}
        metrics_path = output_dir / 'corrected_trajectory_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\\nCorrected metrics saved: {metrics_path}")
        
        # Print summary
        print(f"\\n{'='*60}")
        print("CORRECTED TRAJECTORY EVALUATION SUMMARY")
        print(f"{'='*60}")
        for dataset_name, data in results.items():
            metrics = data['metrics']
            print(f"{dataset_name:>8} - Endpoint Error: {metrics['endpoint_error']:.4f}m, "
                  f"Path Length Ratio: {metrics['path_length_ratio']:.3f}, "
                  f"MSE: {metrics['mse']:.2e}")
        
        print(f"\\n{'='*60}")
        print("VISUALIZATION COMPLETE!")
        print(f"{'='*60}")
        print("Generated corrected files:")
        print("  - corrected_train_trajectory_comparison.png")
        print("  - corrected_val_trajectory_comparison.png") 
        print("  - corrected_test_trajectory_comparison.png")
        print("  - corrected_trajectory_metrics.json")
        print("\\nThese should now show the proper infinity-loop patterns!")
        
        return results

def main():
    # Configuration (adjust paths as needed)
    config = {
        'csv_path': 'data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr_clean.csv',
        'data_root': 'data/processed/visual_odometry_dataset',
        'test_bags': ['ariel_2023-12-21-14-28-22_4'],
        'sequence_length': 8,
        'overlap_frames': 1,
        'image_size': 224,
        'camera': 'cam0',
        'pretrained': True,
        'freeze_backbone': False
    }
    
    # Model path (use best checkpoint)
    model_path = "experiments/tsformer_vo/checkpoint_best.pth"
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        print("Please make sure training is completed and checkpoint exists.")
        return
    
    # Create visualizer and generate corrected plots
    visualizer = TrajectoryVisualizerCorrected(model_path, config)
    results = visualizer.generate_corrected_visualizations()

if __name__ == "__main__":
    main()