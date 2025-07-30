"""
Visualization utilities for underwater visual odometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from typing import List, Dict, Optional, Tuple


def plot_trajectory_3d(trajectories: Dict[str, np.ndarray], 
                      title: str = "3D Trajectory",
                      save_path: Optional[str] = None):
    """
    Plot 3D trajectories.
    
    Args:
        trajectories: Dictionary of trajectories, each with shape (N, 7)
                     where columns are [x, y, z, qw, qx, qy, qz]
        title: Plot title
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (label, traj) in enumerate(trajectories.items()):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=colors[i % len(colors)], 
                label=label, 
                linewidth=2)
        
        # Mark start and end
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                  color=colors[i % len(colors)], 
                  marker='o', s=100, label=f'{label} start')
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                  color=colors[i % len(colors)], 
                  marker='s', s=100, label=f'{label} end')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Equal aspect ratio
    max_range = np.array([
        trajectories[list(trajectories.keys())[0]][:, 0].max() - trajectories[list(trajectories.keys())[0]][:, 0].min(),
        trajectories[list(trajectories.keys())[0]][:, 1].max() - trajectories[list(trajectories.keys())[0]][:, 1].min(),
        trajectories[list(trajectories.keys())[0]][:, 2].max() - trajectories[list(trajectories.keys())[0]][:, 2].min()
    ]).max() / 2.0
    
    mid_x = (trajectories[list(trajectories.keys())[0]][:, 0].max() + trajectories[list(trajectories.keys())[0]][:, 0].min()) * 0.5
    mid_y = (trajectories[list(trajectories.keys())[0]][:, 1].max() + trajectories[list(trajectories.keys())[0]][:, 1].min()) * 0.5
    mid_z = (trajectories[list(trajectories.keys())[0]][:, 2].max() + trajectories[list(trajectories.keys())[0]][:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_trajectory_2d(trajectories: Dict[str, np.ndarray],
                      view: str = 'xy',
                      title: Optional[str] = None,
                      save_path: Optional[str] = None):
    """
    Plot 2D projection of trajectories.
    
    Args:
        trajectories: Dictionary of trajectories
        view: Which plane to plot ('xy', 'xz', 'yz')
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    view_indices = {
        'xy': (0, 1, 'X (m)', 'Y (m)'),
        'xz': (0, 2, 'X (m)', 'Z (m)'),
        'yz': (1, 2, 'Y (m)', 'Z (m)')
    }
    
    idx1, idx2, xlabel, ylabel = view_indices[view]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (label, traj) in enumerate(trajectories.items()):
        plt.plot(traj[:, idx1], traj[:, idx2], 
                color=colors[i % len(colors)], 
                label=label, 
                linewidth=2)
        
        # Mark start and end
        plt.scatter(traj[0, idx1], traj[0, idx2], 
                   color=colors[i % len(colors)], 
                   marker='o', s=100, zorder=5)
        plt.scatter(traj[-1, idx1], traj[-1, idx2], 
                   color=colors[i % len(colors)], 
                   marker='s', s=100, zorder=5)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title or f'Trajectory {view.upper()} View')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_errors_over_time(errors: Dict[str, np.ndarray],
                         timestamps: Optional[np.ndarray] = None,
                         title: str = "Errors over Time",
                         save_path: Optional[str] = None):
    """
    Plot error metrics over time.
    
    Args:
        errors: Dictionary of error arrays
        timestamps: Optional timestamp array
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    if timestamps is None:
        timestamps = np.arange(len(list(errors.values())[0]))
    
    # Translation errors
    ax1 = axes[0]
    for label, error in errors.items():
        if 'trans' in label.lower():
            ax1.plot(timestamps, error, label=label, linewidth=2)
    
    ax1.set_ylabel('Translation Error (m)')
    ax1.set_title('Translation Errors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rotation errors
    ax2 = axes[1]
    for label, error in errors.items():
        if 'rot' in label.lower():
            ax2.plot(timestamps, error, label=label, linewidth=2)
    
    ax2.set_ylabel('Rotation Error (rad)')
    ax2.set_xlabel('Time (s)' if timestamps is not None else 'Frame')
    ax2.set_title('Rotation Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_training_history(history: Dict[str, List[float]],
                         save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    ax = axes[0, 0]
    if 'loss' in history:
        ax.plot(history['loss'], label='Train')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Validation')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Translation error
    ax = axes[0, 1]
    if 'translation_error' in history:
        ax.plot(history['translation_error'], label='Train')
    if 'val_translation_error' in history:
        ax.plot(history['val_translation_error'], label='Validation')
    ax.set_title('Translation Error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotation error
    ax = axes[1, 0]
    if 'rotation_error' in history:
        ax.plot(history['rotation_error'], label='Train')
    if 'val_rotation_error' in history:
        ax.plot(history['val_rotation_error'], label='Validation')
    ax.set_title('Rotation Error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (rad)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[1, 1]
    if 'translation_rmse' in history:
        ax.plot(history['translation_rmse'], label='Trans RMSE (Train)')
    if 'val_translation_rmse' in history:
        ax.plot(history['val_translation_rmse'], label='Trans RMSE (Val)')
    if 'rotation_rmse' in history:
        ax.plot(history['rotation_rmse'], label='Rot RMSE (Train)')
    if 'val_rotation_rmse' in history:
        ax.plot(history['val_rotation_rmse'], label='Rot RMSE (Val)')
    ax.set_title('RMSE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training History')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def visualize_image_pairs(image_pairs: List[Tuple[np.ndarray, np.ndarray]],
                         predictions: Optional[np.ndarray] = None,
                         ground_truth: Optional[np.ndarray] = None,
                         save_dir: Optional[str] = None):
    """
    Visualize image pairs with optional pose predictions.
    
    Args:
        image_pairs: List of (img1, img2) tuples
        predictions: Optional predicted poses
        ground_truth: Optional ground truth poses
        save_dir: Directory to save visualizations
    """
    for i, (img1, img2) in enumerate(image_pairs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display images
        axes[0].imshow(img1)
        axes[0].set_title('Frame t')
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].set_title('Frame t+1')
        axes[1].axis('off')
        
        # Add pose information
        if predictions is not None or ground_truth is not None:
            info_text = ""
            if ground_truth is not None:
                info_text += f"GT: t=[{ground_truth[i][0]:.3f}, {ground_truth[i][1]:.3f}, {ground_truth[i][2]:.3f}] "
                info_text += f"r=[{ground_truth[i][3]:.3f}, {ground_truth[i][4]:.3f}, {ground_truth[i][5]:.3f}]\n"
            if predictions is not None:
                info_text += f"Pred: t=[{predictions[i][0]:.3f}, {predictions[i][1]:.3f}, {predictions[i][2]:.3f}] "
                info_text += f"r=[{predictions[i][3]:.3f}, {predictions[i][4]:.3f}, {predictions[i][5]:.3f}]"
            
            plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10)
        
        plt.suptitle(f'Image Pair {i}')
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'pair_{i:04d}.png'), dpi=150)
            plt.close()
        else:
            plt.show()
            break  # Only show first pair in interactive mode


def create_video_from_trajectory(images_dir: str,
                               trajectory: np.ndarray,
                               output_path: str,
                               fps: int = 10):
    """
    Create a video visualization of the trajectory.
    
    Args:
        images_dir: Directory containing image sequence
        trajectory: Trajectory array with shape (N, 7)
        output_path: Path to save output video
        fps: Frames per second
    """
    # Get image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    # Read first image to get dimensions
    first_img = cv2.imread(os.path.join(images_dir, image_files[0]))
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    # Create trajectory plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    for i, img_file in enumerate(image_files[:len(trajectory)]):
        # Read image
        img = cv2.imread(os.path.join(images_dir, img_file))
        
        # Create trajectory plot up to current frame
        ax.clear()
        ax.plot(trajectory[:i+1, 0], trajectory[:i+1, 1], 'b-', linewidth=2)
        ax.scatter(trajectory[i, 0], trajectory[i, 1], c='red', s=100, zorder=5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Trajectory (Frame {i})')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Convert plot to image
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        plot_img = cv2.resize(plot_img, (width, height))
        
        # Combine images
        combined = np.hstack([img, plot_img])
        
        # Write frame
        out.write(combined)
    
    out.release()
    plt.close(fig)
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization functions...")
    
    # Create dummy trajectory
    t = np.linspace(0, 4*np.pi, 100)
    trajectory = np.column_stack([
        np.sin(t),
        np.cos(t),
        t / 10,
        np.ones_like(t),  # qw
        np.zeros_like(t),  # qx
        np.zeros_like(t),  # qy
        np.zeros_like(t)   # qz
    ])
    
    # Test 3D plot
    plot_trajectory_3d({'Test': trajectory}, title="Test 3D Trajectory")
    
    # Test 2D plot
    plot_trajectory_2d({'Test': trajectory}, view='xy', title="Test XY Trajectory")
    
    # Test error plot
    errors = {
        'trans_error': np.random.normal(0.1, 0.02, 100),
        'rot_error': np.random.normal(0.05, 0.01, 100)
    }
    plot_errors_over_time(errors)
    
    print("Visualization tests completed!")