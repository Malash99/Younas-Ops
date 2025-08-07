#!/usr/bin/env python3
"""
LOCAL TRAJECTORY WINDOWS TRAINING
Fix wrong direction by learning local trajectory shapes while keeping shuffling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.sub_trajectory_dataset import create_sub_trajectory_dataloaders

class TrajectoryAwareLoss(nn.Module):
    """Loss that learns local trajectory shapes while keeping shuffling"""
    
    def __init__(self, trajectory_weight=2.0, smoothness_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.trajectory_weight = trajectory_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(self, predictions, pose_targets, accumulated_targets):
        """
        predictions: [batch, seq_len-1, 6] - delta predictions
        pose_targets: [batch, seq_len-1, 6] - delta targets  
        accumulated_targets: [batch, seq_len-1, 6] - cumulative positions
        """
        
        # 1. Standard delta loss (frame-to-frame)
        translation_loss = self.mse(predictions[:, :, :3], pose_targets[:, :, :3])
        rotation_loss = self.mse(predictions[:, :, 3:], pose_targets[:, :, 3:])
        delta_loss = translation_loss + rotation_loss
        
        # 2. TRAJECTORY SHAPE LOSS - Convert deltas to local cumulative positions
        # Cumulative positions within each window (relative to start)
        pred_cumulative = torch.cumsum(predictions, dim=1)  # [batch, seq_len-1, 6]
        target_cumulative = torch.cumsum(pose_targets, dim=1)  # [batch, seq_len-1, 6]
        
        # Loss on trajectory shape (cumulative positions within window)
        trajectory_loss = self.mse(pred_cumulative[:, :, :3], target_cumulative[:, :, :3])
        trajectory_rot_loss = self.mse(pred_cumulative[:, :, 3:], target_cumulative[:, :, 3:])
        
        # 3. SMOOTHNESS LOSS - Penalize sudden direction changes
        if predictions.shape[1] > 1:
            pred_velocity_changes = torch.diff(predictions[:, :, :3], dim=1)  # [batch, seq_len-2, 3]
            target_velocity_changes = torch.diff(pose_targets[:, :, :3], dim=1)
            smoothness_loss = self.mse(pred_velocity_changes, target_velocity_changes)
        else:
            smoothness_loss = torch.tensor(0.0, device=predictions.device)
        
        # 4. DIRECTION CONSISTENCY LOSS - Ensure overall direction is correct
        # Final position in window should be in correct direction
        final_pred_pos = pred_cumulative[:, -1, :3]  # [batch, 3]
        final_target_pos = target_cumulative[:, -1, :3]  # [batch, 3]
        direction_loss = self.mse(final_pred_pos, final_target_pos)
        
        # Combine losses
        total_loss = (
            delta_loss + 
            self.trajectory_weight * (trajectory_loss + trajectory_rot_loss) +
            self.smoothness_weight * smoothness_loss +
            direction_loss
        )
        
        # Calculate drift metric (final position error)
        final_drift = torch.mean(torch.norm(final_pred_pos - final_target_pos, dim=-1))
        
        return {
            'total_loss': total_loss,
            'delta_loss': delta_loss,
            'trajectory_loss': trajectory_loss,
            'smoothness_loss': smoothness_loss,
            'direction_loss': direction_loss,
            'translation_loss': translation_loss,
            'rotation_loss': rotation_loss,
            'ate_loss': trajectory_loss,  # Use trajectory loss as ATE
            'consistency_loss': smoothness_loss,
            'final_position_error': final_drift,
            'predicted_accumulated': pred_cumulative,
            'target_accumulated': target_cumulative
        }

class TrajectoryWindowModel(nn.Module):
    """Model for trajectory window training"""
    
    def __init__(self, config):
        super().__init__()
        self.base_model = UWTransVO(**config)
        
        # Normalization parameters
        self.register_buffer('delta_mean', torch.zeros(6))
        self.register_buffer('delta_std', torch.ones(6))
        
    def set_normalization(self, delta_mean, delta_std):
        """Set normalization parameters"""
        self.delta_mean.copy_(delta_mean)
        self.delta_std.copy_(delta_std)
        
    def forward(self, images, camera_ids, camera_mask, sub_traj_length):
        batch_size, seq_len, num_cameras, C, H, W = images.shape
        all_predictions = []
        
        for t in range(seq_len - 1):
            frame_pair = torch.stack([images[:, t], images[:, t+1]], dim=1)
            output = self.base_model(
                images=frame_pair,
                camera_ids=camera_ids,
                camera_mask=camera_mask
            )
            all_predictions.append(output['pose'])
        
        predictions = torch.stack(all_predictions, dim=1)
        return predictions
    
    def denormalize_predictions(self, normalized_preds):
        """Convert normalized predictions back to real scale"""
        return normalized_preds * self.delta_std + self.delta_mean

def compute_dataset_normalization(train_loader):
    """Compute normalization parameters from training data"""
    print("Computing dataset normalization parameters...")
    
    all_deltas = []
    
    for batch in tqdm(train_loader, desc="Computing normalization"):
        pose_targets = batch['pose_targets']  # [batch, seq_len-1, 6]
        all_deltas.append(pose_targets.view(-1, 6))
    
    all_deltas = torch.cat(all_deltas, dim=0)  # [N, 6]
    
    delta_mean = torch.mean(all_deltas, dim=0)
    delta_std = torch.std(all_deltas, dim=0) + 1e-8  # Avoid division by zero
    
    print(f"Delta means: {delta_mean.numpy()}")
    print(f"Delta stds:  {delta_std.numpy()}")
    print(f"Normalization will map deltas to ~N(0,1) distribution")
    
    return delta_mean, delta_std

class TrajectoryWindowTrainer:
    def __init__(self, config, train_loader):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Clear screen and show banner
        os.system('cls' if os.name == 'nt' else 'clear')
        self.print_banner()
        
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        
        print("\\nInitializing TRAJECTORY WINDOW Model...")
        self.model = TrajectoryWindowModel(config['model']).to(self.device)
        
        # Compute normalization parameters
        self.delta_mean, self.delta_std = compute_dataset_normalization(train_loader)
        self.model.set_normalization(self.delta_mean, self.delta_std)
        
        # Trajectory-aware loss with strong weights
        self.criterion = TrajectoryAwareLoss(
            trajectory_weight=config['training']['trajectory_weight'],
            smoothness_weight=config['training']['smoothness_weight']
        )
        
        # Optimizer with good learning rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=2
        )
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model Parameters: {param_count:,}")
        print(f"Normalization: mean={self.delta_mean.numpy()}, std={self.delta_std.numpy()}")
    
    def print_banner(self):
        banner = """
========================================================
         UW-TransVO TRAJECTORY WINDOW Training
              LOCAL TRAJECTORY LEARNING
========================================================
APPROACH:
  1. Keep data shuffling (prevents memorization)
  2. Learn local trajectory shapes within windows
  3. Predict both deltas AND cumulative positions
  4. Strong trajectory shape penalties
  5. Direction consistency enforcement
========================================================
        """
        print(banner)
    
    def normalize_batch(self, pose_targets):
        """Normalize pose targets to N(0,1)"""
        return (pose_targets - self.delta_mean.to(pose_targets.device)) / self.delta_std.to(pose_targets.device)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_delta_loss = 0.0
        total_trajectory_loss = 0.0
        total_direction_loss = 0.0
        total_drift = 0.0
        successful_batches = 0
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1} Trajectory Training",
            ncols=140
        )
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move to device
                images = batch['images'].to(self.device)
                camera_ids = batch['camera_ids'].to(self.device)
                camera_mask = batch['camera_mask'].to(self.device)
                pose_targets = batch['pose_targets'].to(self.device)
                accumulated_targets = batch['accumulated_poses'].to(self.device)
                
                # NORMALIZE targets
                normalized_targets = self.normalize_batch(pose_targets)
                normalized_accumulated = self.normalize_batch(accumulated_targets)
                
                sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(images, camera_ids, camera_mask, sub_traj_length)
                
                # Calculate trajectory-aware loss
                loss_dict = self.criterion(predictions, normalized_targets, normalized_accumulated)
                
                # Check for NaN BEFORE backward pass
                if torch.isnan(loss_dict['total_loss']) or torch.isinf(loss_dict['total_loss']):
                    print(f"\\nSkipping batch {batch_idx} (NaN/Inf loss: {loss_dict['total_loss'].item()})")
                    continue
                
                # Backward pass
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"\\nSkipping batch {batch_idx} (NaN/Inf gradients)")
                    self.optimizer.zero_grad()
                    continue
                
                # Optimizer step
                self.optimizer.step()
                
                # Accumulate metrics
                batch_loss = loss_dict['total_loss'].item()
                batch_delta_loss = loss_dict['delta_loss'].item()
                batch_traj_loss = loss_dict['trajectory_loss'].item()
                batch_dir_loss = loss_dict['direction_loss'].item()
                batch_drift = loss_dict['final_position_error'].item()
                
                if not (np.isnan(batch_loss) or np.isinf(batch_loss)):
                    total_loss += batch_loss
                    total_delta_loss += batch_delta_loss
                    total_trajectory_loss += batch_traj_loss
                    total_direction_loss += batch_dir_loss
                    total_drift += batch_drift
                    successful_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{batch_loss:.6f}',
                    'Delta': f'{batch_delta_loss:.6f}',
                    'Traj': f'{batch_traj_loss:.6f}',
                    'Dir': f'{batch_dir_loss:.6f}',
                    'Drift': f'{batch_drift:.3f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.3e}',
                    'Success': f'{successful_batches}/{batch_idx+1}'
                })
                
            except Exception as e:
                print(f"\\nError in batch {batch_idx}: {e}")
                continue
        
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            avg_delta_loss = total_delta_loss / successful_batches
            avg_traj_loss = total_trajectory_loss / successful_batches
            avg_dir_loss = total_direction_loss / successful_batches
            avg_drift = total_drift / successful_batches
        else:
            avg_loss = float('inf')
            avg_delta_loss = float('inf')
            avg_traj_loss = float('inf')
            avg_dir_loss = float('inf')
            avg_drift = float('inf')
        
        print(f"\\nSuccessful batches: {successful_batches}/{len(train_loader)}")
        return avg_loss, avg_delta_loss, avg_traj_loss, avg_dir_loss, avg_drift
    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0
        total_delta_loss = 0.0
        total_trajectory_loss = 0.0
        total_direction_loss = 0.0
        total_drift = 0.0
        successful_batches = 0
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", ncols=140)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                try:
                    images = batch['images'].to(self.device)
                    camera_ids = batch['camera_ids'].to(self.device)
                    camera_mask = batch['camera_mask'].to(self.device)
                    pose_targets = batch['pose_targets'].to(self.device)
                    accumulated_targets = batch['accumulated_poses'].to(self.device)
                    
                    # NORMALIZE targets
                    normalized_targets = self.normalize_batch(pose_targets)
                    normalized_accumulated = self.normalize_batch(accumulated_targets)
                    
                    sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
                    
                    predictions = self.model(images, camera_ids, camera_mask, sub_traj_length)
                    loss_dict = self.criterion(predictions, normalized_targets, normalized_accumulated)
                    
                    batch_loss = loss_dict['total_loss'].item()
                    batch_delta_loss = loss_dict['delta_loss'].item()
                    batch_traj_loss = loss_dict['trajectory_loss'].item()
                    batch_dir_loss = loss_dict['direction_loss'].item()
                    batch_drift = loss_dict['final_position_error'].item()
                    
                    if not (np.isnan(batch_loss) or np.isinf(batch_loss)):
                        total_loss += batch_loss
                        total_delta_loss += batch_delta_loss
                        total_trajectory_loss += batch_traj_loss
                        total_direction_loss += batch_dir_loss
                        total_drift += batch_drift
                        successful_batches += 1
                    
                    pbar.set_postfix({
                        'Loss': f'{batch_loss:.6f}',
                        'Delta': f'{batch_delta_loss:.6f}',
                        'Traj': f'{batch_traj_loss:.6f}',
                        'Dir': f'{batch_dir_loss:.6f}',
                        'Drift': f'{batch_drift:.3f}'
                    })
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            avg_delta_loss = total_delta_loss / successful_batches
            avg_traj_loss = total_trajectory_loss / successful_batches
            avg_dir_loss = total_direction_loss / successful_batches
            avg_drift = total_drift / successful_batches
        else:
            avg_loss = float('inf')
            avg_delta_loss = float('inf')
            avg_traj_loss = float('inf')
            avg_dir_loss = float('inf')
            avg_drift = float('inf')
        
        # Update learning rate scheduler
        self.scheduler.step(avg_loss)
        
        return avg_loss, avg_delta_loss, avg_traj_loss, avg_dir_loss, avg_drift
    
    def train(self, train_loader, val_loader, epochs):
        print(f"\\nSTARTING TRAJECTORY WINDOW TRAINING")
        print(f"Epochs: {epochs}")
        print(f"Train Samples: {len(train_loader.dataset)}")
        print(f"Val Samples: {len(val_loader.dataset)}")
        print("=" * 60)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            print(f"\\nEpoch {epoch+1}/{epochs}")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')} | LR: {self.optimizer.param_groups[0]['lr']:.3e}")
            
            # Train
            train_loss, train_delta, train_traj, train_dir, train_drift = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_delta, val_traj, val_dir, val_drift = self.validate(val_loader, epoch)
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\\nEPOCH {epoch+1} RESULTS:")
            print(f"Time: {epoch_time:.1f}s (Total: {total_time/60:.1f}min)")
            print(f"TRAIN  | Loss: {train_loss:.6f} | Delta: {train_delta:.6f} | Traj: {train_traj:.6f} | Dir: {train_dir:.6f} | Drift: {train_drift:.4f}m")
            print(f"VAL    | Loss: {val_loss:.6f} | Delta: {val_delta:.6f} | Traj: {val_traj:.6f} | Dir: {val_dir:.6f} | Drift: {val_drift:.4f}m")
            
            # Save best model
            if val_loss < best_val_loss and val_loss != float('inf'):
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config,
                    'delta_mean': self.delta_mean,
                    'delta_std': self.delta_std
                }, 'trajectory_window_best_model.pth')
                print(f"NEW BEST MODEL SAVED! (Loss: {val_loss:.6f})")
            
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\\nTRAJECTORY WINDOW TRAINING COMPLETED!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return best_val_loss

def main():
    # Trajectory window configuration
    config = {
        'model': {
            'img_size': 224,
            'patch_size': 32,
            'd_model': 256,        # Good size
            'num_heads': 4,        # Good attention
            'num_layers': 3,       # Sufficient depth
            'max_cameras': 1,      
            'max_seq_len': 5,      # Longer sequences for better trajectory learning
            'dropout': 0.1,        
            'use_imu': False,      
            'use_pressure': False,
            'uncertainty_estimation': False
        },
        'training': {
            'epochs': 25,               # More epochs for trajectory learning
            'learning_rate': 5e-4,      # Balanced learning rate
            'batch_size': 4,            # Good batch size
            'trajectory_weight': 3.0,   # Strong trajectory penalty
            'smoothness_weight': 1.0    # Smoothness penalty
        }
    }
    
    print("TRAJECTORY WINDOW Training Configuration:")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Model size: {config['model']['d_model']} dim, {config['model']['num_layers']} layers")
    print(f"  Sequence length: {config['model']['max_seq_len']} frames")
    print(f"  Trajectory weight: {config['training']['trajectory_weight']}")
    print(f"  Key Fix: LOCAL TRAJECTORY LEARNING with shuffling")
    
    try:
        # Create dataloaders with longer sequences
        print("\\nLoading Dataset...")
        train_loader, val_loader = create_sub_trajectory_dataloaders(
            train_csv='data/processed/training_dataset/training_data_filtered.csv',
            val_csv='data/processed/training_dataset/training_data_filtered.csv',
            sub_trajectory_length=config['model']['max_seq_len'],   
            overlap=2,                 # More overlap for trajectory learning
            camera_ids=[0],            # Camera 0 only
            batch_size=config['training']['batch_size'],
            num_workers=0,
            max_samples_train=400,     # More data for trajectory learning
            max_samples_val=80
        )
        
        print(f"Dataset loaded successfully!")
        
        # Create trainer and start training
        trainer = TrajectoryWindowTrainer(config, train_loader)
        best_loss = trainer.train(train_loader, val_loader, config['training']['epochs'])
        
        print(f"\\nTRAJECTORY WINDOW FINAL RESULT: Best validation loss = {best_loss:.6f}")
        print("Trajectory window model saved as: trajectory_window_best_model.pth")
        
    except Exception as e:
        print(f"\\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()