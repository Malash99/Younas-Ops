"""
Training script for Motion-Aware UW-TransVO

This script trains the model with frame-to-frame motion supervision
to help it learn visual motion patterns instead of predicting straight lines.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer.motion_aware_uw_transvo import create_motion_aware_model
from training.motion_aware_loss import create_motion_aware_loss
from data.datasets import UnderwaterVODataset
from data.sub_trajectory_dataset import SubTrajectoryDataset
from training.trajectory_losses import calculate_trajectory_metrics


def create_motion_sequence_dataset(csv_file: str, sequence_length: int = 5) -> SubTrajectoryDataset:
    """Create dataset with sequential frames for motion learning"""
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} total frames from {csv_file}")
    
    # Create sequence-based sub-trajectories
    dataset = SubTrajectoryDataset(
        data_csv=csv_file,
        data_root="data/processed/training_dataset",
        sub_trajectory_length=sequence_length,
        overlap=sequence_length - 1,  # High overlap for dense supervision
        camera_ids=[0],  # Single camera for now
        img_size=224,
        use_imu=False,
        use_pressure=False,
        augmentation=True,
        split='train',
        max_samples=None
    )
    
    print(f"Created {len(dataset)} sequential sub-trajectories of length {sequence_length}")
    return dataset


def train_motion_aware_model():
    """Train the motion-aware model with sequence supervision"""
    
    # Configuration
    config = {
        'img_size': 224,
        'patch_size': 16,
        'd_model': 384,  # Smaller for faster training
        'num_heads': 6,
        'num_layers': 4,
        'max_cameras': 1,
        'max_seq_len': 5,
        'dropout': 0.1,
        'use_imu': False,
        'use_pressure': False,
        'uncertainty_estimation': True
    }
    
    loss_config = {
        'loss_type': 'motion_aware',
        'translation_weight': 1.0,
        'rotation_weight': 5.0,
        'motion_weight': 10.0,  # High weight on motion learning
        'consistency_weight': 2.0,
        'sequence_length': 5
    }
    
    # Training parameters
    batch_size = 8  # Small batch for sequences
    learning_rate = 5e-5  # Conservative learning rate
    num_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in create_motion_aware_model(config).parameters()):,}")
    
    # Create dataset
    csv_file = "data/processed/training_dataset/training_data.csv"
    if not os.path.exists(csv_file):
        print(f"ERROR: Training data not found at {csv_file}")
        return
    
    print("Creating motion sequence dataset...")
    dataset = create_motion_sequence_dataset(csv_file, sequence_length=config['max_seq_len'])
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Single threaded for debugging
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model and loss
    print("Creating motion-aware model...")
    model = create_motion_aware_model(config).to(device)
    criterion = create_motion_aware_loss(loss_config)
    
    # Optimizer with gradient clipping
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # Training tracking
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_motion_loss': [],
        'val_motion_loss': [],
        'train_ate': [],
        'val_ate': []
    }
    
    print("Starting motion-aware training...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_losses = []
        train_motion_losses = []
        train_ates = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        
        for batch_idx, batch in enumerate(train_pbar):
            try:
                # Unpack batch (sequence data)
                if isinstance(batch, dict):
                    images = batch['images'].to(device)  # [batch, seq_len, 1, 3, H, W]
                    poses = batch['poses'].to(device)    # [batch, seq_len, 6]
                    camera_ids = torch.zeros(images.size(0), 1, dtype=torch.long).to(device)
                else:
                    images, poses = batch
                    images = images.to(device)
                    poses = poses.to(device)
                    camera_ids = torch.zeros(images.size(0), images.size(2), dtype=torch.long).to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                outputs = model(
                    images=images,
                    camera_ids=camera_ids,
                    camera_mask=None
                )
                
                pred_poses = outputs['pose']  # [batch, seq_len, 6]
                
                # Compute motion-aware loss
                loss_dict = criterion(pred_poses, poses)
                loss = loss_dict['total_loss']
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}! Skipping...")
                    continue
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Compute metrics (use last frame for ATE)
                with torch.no_grad():
                    pred_last = pred_poses[:, -1]  # [batch, 6]
                    target_last = poses[:, -1]     # [batch, 6]
                    
                    metrics = calculate_trajectory_metrics(pred_last, target_last)
                    
                train_losses.append(loss.item())
                train_motion_losses.append(loss_dict.get('motion_translation_loss', 0).item())
                train_ates.append(metrics['ate_mean'])
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f"{loss.item():.6f}",
                    'Motion': f"{loss_dict.get('motion_translation_loss', 0).item():.6f}",
                    'ATE': f"{metrics['ate_mean']:.6f}",
                    'LR': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Validation phase
        model.eval()
        val_losses = []
        val_motion_losses = []
        val_ates = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
            
            for batch in val_pbar:
                try:
                    # Unpack batch
                    if isinstance(batch, dict):
                        images = batch['images'].to(device)
                        poses = batch['poses'].to(device)
                        camera_ids = torch.zeros(images.size(0), 1, dtype=torch.long).to(device)
                    else:
                        images, poses = batch
                        images = images.to(device)
                        poses = poses.to(device)
                        camera_ids = torch.zeros(images.size(0), images.size(2), dtype=torch.long).to(device)
                    
                    # Forward pass
                    outputs = model(
                        images=images,
                        camera_ids=camera_ids,
                        camera_mask=None
                    )
                    
                    pred_poses = outputs['pose']
                    
                    # Compute loss
                    loss_dict = criterion(pred_poses, poses)
                    loss = loss_dict['total_loss']
                    
                    if not torch.isnan(loss):
                        # Compute metrics
                        pred_last = pred_poses[:, -1]
                        target_last = poses[:, -1]
                        metrics = calculate_trajectory_metrics(pred_last, target_last)
                        
                        val_losses.append(loss.item())
                        val_motion_losses.append(loss_dict.get('motion_translation_loss', 0).item())
                        val_ates.append(metrics['ate_mean'])
                        
                        val_pbar.set_postfix({
                            'Loss': f"{loss.item():.6f}",
                            'Motion': f"{loss_dict.get('motion_translation_loss', 0).item():.6f}",
                            'ATE': f"{metrics['ate_mean']:.6f}"
                        })
                        
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Update learning rate
        scheduler.step()
        
        # Compute epoch averages
        if train_losses and val_losses:
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            avg_train_motion = np.mean(train_motion_losses)
            avg_val_motion = np.mean(val_motion_losses)
            avg_train_ate = np.mean(train_ates)
            avg_val_ate = np.mean(val_ates)
            
            # Save training history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['train_motion_loss'].append(avg_train_motion)
            training_history['val_motion_loss'].append(avg_val_motion)
            training_history['train_ate'].append(avg_train_ate)
            training_history['val_ate'].append(avg_val_ate)
            
            epoch_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{num_epochs} RESULTS")
            print(f"{'='*60}")
            print(f"Time: {epoch_time:.1f}s")
            print(f"TRAIN  | Loss: {avg_train_loss:.6f} | Motion: {avg_train_motion:.6f} | ATE: {avg_train_ate:.6f}")
            print(f"VAL    | Loss: {avg_val_loss:.6f} | Motion: {avg_val_motion:.6f} | ATE: {avg_val_ate:.6f}")
            print(f"LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': config,
                    'loss_config': loss_config
                }, 'motion_aware_best_model.pth')
                print(f"*** NEW BEST MODEL SAVED! Val Loss: {avg_val_loss:.6f}")
            
            print(f"{'='*60}\n")
    
    # Save final training results
    with open('motion_aware_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(training_history)
    
    print("Motion-aware training completed!")
    return training_history


def plot_training_curves(history):
    """Plot training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Motion loss curves
    ax2.plot(epochs, history['train_motion_loss'], 'b-', label='Train Motion Loss')
    ax2.plot(epochs, history['val_motion_loss'], 'r-', label='Val Motion Loss')
    ax2.set_title('Motion Loss (Key Innovation)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Motion Loss')
    ax2.legend()
    ax2.grid(True)
    
    # ATE curves
    ax3.plot(epochs, history['train_ate'], 'b-', label='Train ATE')
    ax3.plot(epochs, history['val_ate'], 'r-', label='Val ATE')
    ax3.set_title('Absolute Trajectory Error')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('ATE (m)')
    ax3.legend()
    ax3.grid(True)
    
    # Combined view
    ax4.plot(epochs, np.array(history['val_loss']) / np.max(history['val_loss']), 'r-', label='Val Loss (norm)')
    ax4.plot(epochs, np.array(history['val_motion_loss']) / np.max(history['val_motion_loss']), 'g-', label='Val Motion (norm)')
    ax4.plot(epochs, np.array(history['val_ate']) / np.max(history['val_ate']), 'b-', label='Val ATE (norm)')
    ax4.set_title('Normalized Metrics')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Normalized Value')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('motion_aware_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Starting Motion-Aware UW-TransVO Training")
    print("This training focuses on learning frame-to-frame motion patterns")
    print("instead of predicting straight-line trajectories.\n")
    
    history = train_motion_aware_model()
    
    print("\nTraining completed!")
    print("Key files created:")
    print("- motion_aware_best_model.pth")
    print("- motion_aware_training_history.json") 
    print("- motion_aware_training_curves.png")