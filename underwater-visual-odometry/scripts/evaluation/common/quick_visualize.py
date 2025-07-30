#!/usr/bin/env python3
"""
Quick visualization of model outputs - shows live predictions on image sequences.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf

from data_loader import UnderwaterVODataset
from models.baseline_cnn import create_model
from coordinate_transforms import integrate_trajectory


class LiveVisualizer:
    """Visualize model predictions in real-time."""
    
    def __init__(self, model_path, data_dir, start_frame=0):
        self.model_path = model_path
        self.data_dir = data_dir
        self.start_frame = start_frame
        
        # Load model
        print("Loading model...")
        self.model = create_model('baseline')
        self.model.load_weights(model_path)
        
        # Load dataset
        print("Loading dataset...")
        self.dataset = UnderwaterVODataset(data_dir)
        self.dataset.load_data()
        
        # Initialize trajectory
        self.initial_pose = {
            'position': np.array([0, 0, 0]),
            'quaternion': np.array([1, 0, 0, 0])
        }
        
        self.pred_trajectory = [self.initial_pose['position'].copy()]
        self.gt_trajectory = [self.initial_pose['position'].copy()]
        
        self.current_frame = start_frame
        
    def predict_and_update(self):
        """Make prediction and update trajectories."""
        if self.current_frame >= len(self.dataset.poses_df) - 1:
            return False
        
        # Load images
        img1 = self.dataset.load_image(self.current_frame)
        img2 = self.dataset.load_image(self.current_frame + 1)
        
        # Make prediction
        img_pair = np.concatenate([img1, img2], axis=-1)
        img_batch = np.expand_dims(img_pair, axis=0)
        
        pred = self.model.predict(img_batch, verbose=0)[0]
        
        # Get ground truth
        gt = self.dataset.get_relative_pose(self.current_frame, self.current_frame + 1)
        gt_pose = np.concatenate([gt['translation'], gt['rotation']])
        
        # Update trajectories (simplified - just using translation for visualization)
        self.pred_trajectory.append(self.pred_trajectory[-1] + pred[:3])
        self.gt_trajectory.append(self.gt_trajectory[-1] + gt_pose[:3])
        
        self.current_frame += 1
        
        return True, img1, img2, pred, gt_pose
    
    def create_visualization(self):
        """Create interactive visualization."""
        # Set up the figure
        fig = plt.figure(figsize=(16, 8))
        
        # Image displays
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        
        # Trajectory plot
        ax3 = plt.subplot(2, 3, 3)
        
        # Error plots
        ax4 = plt.subplot(2, 3, 4)
        ax5 = plt.subplot(2, 3, 5)
        
        # Text info
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Initialize plots
        im1 = ax1.imshow(np.zeros((224, 224, 3)))
        im2 = ax2.imshow(np.zeros((224, 224, 3)))
        ax1.set_title('Frame t')
        ax2.set_title('Frame t+1')
        ax1.axis('off')
        ax2.axis('off')
        
        # Trajectory lines
        pred_line, = ax3.plot([], [], 'r-', label='Predicted', linewidth=2)
        gt_line, = ax3.plot([], [], 'g-', label='Ground Truth', linewidth=2)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('Trajectory (XY View)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Error tracking
        self.trans_errors = []
        self.rot_errors = []
        
        trans_line, = ax4.plot([], [], 'b-', linewidth=2)
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Translation Error (m)')
        ax4.set_title('Translation Error')
        ax4.grid(True, alpha=0.3)
        
        rot_line, = ax5.plot([], [], 'r-', linewidth=2)
        ax5.set_xlabel('Frame')
        ax5.set_ylabel('Rotation Error (deg)')
        ax5.set_title('Rotation Error')
        ax5.grid(True, alpha=0.3)
        
        # Info text
        info_text = ax6.text(0.1, 0.5, '', fontsize=12, verticalalignment='center')
        
        def update(frame):
            result = self.predict_and_update()
            
            if not result:
                return
            
            _, img1, img2, pred, gt = result
            
            # Update images
            im1.set_data(img1)
            im2.set_data(img2)
            
            # Update trajectory
            pred_traj = np.array(self.pred_trajectory)
            gt_traj = np.array(self.gt_trajectory)
            
            pred_line.set_data(pred_traj[:, 0], pred_traj[:, 1])
            gt_line.set_data(gt_traj[:, 0], gt_traj[:, 1])
            
            # Auto-scale trajectory plot
            all_x = np.concatenate([pred_traj[:, 0], gt_traj[:, 0]])
            all_y = np.concatenate([pred_traj[:, 1], gt_traj[:, 1]])
            margin = 0.5
            ax3.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax3.set_ylim(all_y.min() - margin, all_y.max() + margin)
            
            # Calculate errors
            trans_error = np.linalg.norm(pred[:3] - gt[:3])
            rot_error = np.degrees(np.linalg.norm(pred[3:] - gt[3:]))
            
            self.trans_errors.append(trans_error)
            self.rot_errors.append(rot_error)
            
            # Update error plots
            frames = list(range(len(self.trans_errors)))
            trans_line.set_data(frames, self.trans_errors)
            rot_line.set_data(frames, self.rot_errors)
            
            ax4.set_xlim(0, max(10, len(frames)))
            ax4.set_ylim(0, max(0.1, max(self.trans_errors) * 1.1))
            
            ax5.set_xlim(0, max(10, len(frames)))
            ax5.set_ylim(0, max(1, max(self.rot_errors) * 1.1))
            
            # Update info text
            info = f"Frame: {self.current_frame}\n\n"
            info += f"Predicted:\n"
            info += f"  Trans: [{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}]\n"
            info += f"  Rot:   [{np.degrees(pred[3]):.1f}°, {np.degrees(pred[4]):.1f}°, {np.degrees(pred[5]):.1f}°]\n\n"
            info += f"Ground Truth:\n"
            info += f"  Trans: [{gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f}]\n"
            info += f"  Rot:   [{np.degrees(gt[3]):.1f}°, {np.degrees(gt[4]):.1f}°, {np.degrees(gt[5]):.1f}°]\n\n"
            info += f"Error:\n"
            info += f"  Trans: {trans_error:.4f} m\n"
            info += f"  Rot:   {rot_error:.2f}°"
            
            info_text.set_text(info)
            
            return im1, im2, pred_line, gt_line, trans_line, rot_line, info_text
        
        # Create animation
        anim = FuncAnimation(fig, update, interval=100, blit=False)
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Quick visualization of model predictions')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model weights (.h5 file)')
    parser.add_argument('--data_dir', type=str, default='/app/data/raw',
                        help='Path to dataset directory')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Starting frame index')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        # Try to find it in the latest model directory
        latest_model_dir = max(glob.glob('/app/output/models/model_*'), key=os.path.getmtime)
        model_path = os.path.join(latest_model_dir, 'best_model.h5')
        if os.path.exists(model_path):
            args.model = model_path
            print(f"Using model: {args.model}")
        else:
            print(f"Error: Model file not found: {args.model}")
            return
    
    # Create visualizer
    visualizer = LiveVisualizer(args.model, args.data_dir, args.start_frame)
    visualizer.create_visualization()


if __name__ == "__main__":
    import glob
    main()