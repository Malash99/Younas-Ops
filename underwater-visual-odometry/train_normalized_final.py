#!/usr/bin/env python3
"""
FINAL FIX: Normalized Training
Normalize delta values to enable proper learning of variations
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

class NormalizedTrajectoryLoss(nn.Module):
    """Loss with normalization and strong variation penalty"""
    
    def __init__(self, variation_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.variation_weight = variation_weight
        
    def forward(self, predictions, pose_targets, accumulated_targets):
        # Main MSE loss on normalized values
        translation_loss = self.mse(predictions[:, :, :3], pose_targets[:, :, :3])
        rotation_loss = self.mse(predictions[:, :, 3:], pose_targets[:, :, 3:])
        mse_loss = translation_loss + rotation_loss
        
        # Strong variation penalty - FORCE model to predict variations
        pred_flat = predictions.view(-1, 6)
        target_flat = pose_targets.view(-1, 6)
        
        # Calculate standard deviations
        pred_std = torch.std(pred_flat, dim=0)
        target_std = torch.std(target_flat, dim=0)
        
        # Penalize when predicted std is much smaller than target std
        variation_loss = self.mse(pred_std, target_std)
        
        # Additional penalty for near-constant predictions
        constant_penalty = torch.mean(torch.exp(-pred_std * 1000))  # Exponential penalty for low std
        
        total_loss = mse_loss + self.variation_weight * (variation_loss + constant_penalty)
        
        # Calculate drift metric
        final_pred = predictions[:, -1, :3]
        final_target = pose_targets[:, -1, :3]
        drift = torch.mean(torch.norm(final_pred - final_target, dim=-1))
        
        return {
            'total_loss': total_loss,
            'translation_loss': translation_loss,
            'rotation_loss': rotation_loss,
            'variation_loss': variation_loss,
            'constant_penalty': constant_penalty,
            'ate_loss': translation_loss,
            'consistency_loss': torch.tensor(0.0, device=total_loss.device),
            'smoothness_loss': torch.tensor(0.0, device=total_loss.device),
            'final_position_error': drift,
            'predicted_accumulated': predictions.cumsum(dim=1),
            'target_accumulated': pose_targets.cumsum(dim=1)
        }

class NormalizedModel(nn.Module):
    """Model with input/output normalization"""
    
    def __init__(self, config):
        super().__init__()
        self.base_model = UWTransVO(**config)
        
        # Normalization parameters (will be set during training)
        self.register_buffer('delta_mean', torch.zeros(6))
        self.register_buffer('delta_std', torch.ones(6))
        
    def set_normalization(self, delta_mean, delta_std):
        """Set normalization parameters"""
        self.delta_mean.copy_(torch.tensor(delta_mean, dtype=torch.float32))
        self.delta_std.copy_(torch.tensor(delta_std, dtype=torch.float32))
        
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

class NormalizedTrainer:
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
        
        print("\\nInitializing NORMALIZED Model...")
        self.model = NormalizedModel(config['model']).to(self.device)
        
        # Compute normalization parameters
        self.delta_mean, self.delta_std = compute_dataset_normalization(train_loader)
        self.model.set_normalization(self.delta_mean, self.delta_std)
        
        # Strong variation-aware loss
        self.criterion = NormalizedTrajectoryLoss(variation_weight=2.0)
        
        # AGGRESSIVE learning rate (much higher than before)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],  # Now 1e-3!
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model Parameters: {param_count:,}")
        print(f"Normalization: mean={self.delta_mean.numpy()}, std={self.delta_std.numpy()}")
    
    def print_banner(self):
        banner = """
========================================================
              UW-TransVO NORMALIZED Training
                  FINAL FIX: Data Normalization
========================================================
FIXES APPLIED:
  1. Delta Normalization: Scale deltas to N(0,1)
  2. Learning Rate: 1e-5 -> 1e-3 (100x increase)
  3. Strong Variation Penalty: Force predictions to vary
  4. Constant Prediction Penalty: Exponential penalty
========================================================
        """
        print(banner)
    
    def normalize_batch(self, pose_targets):
        """Normalize pose targets to N(0,1)"""
        return (pose_targets - self.delta_mean.to(pose_targets.device)) / self.delta_std.to(pose_targets.device)
    
    def denormalize_predictions(self, normalized_preds):
        """Denormalize predictions back to real scale"""
        return normalized_preds * self.delta_std.to(normalized_preds.device) + self.delta_mean.to(normalized_preds.device)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_drift = 0.0
        total_variation_loss = 0.0
        total_constant_penalty = 0.0
        successful_batches = 0
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1} Normalized Training",
            ncols=130
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
                
                # Calculate loss on NORMALIZED values
                loss_dict = self.criterion(predictions, normalized_targets, normalized_accumulated)
                
                # Check for NaN BEFORE backward pass
                if torch.isnan(loss_dict['total_loss']) or torch.isinf(loss_dict['total_loss']):
                    print(f"\\nSkipping batch {batch_idx} (NaN/Inf loss: {loss_dict['total_loss'].item()})")
                    continue
                
                # Backward pass
                loss_dict['total_loss'].backward()
                
                # Reasonable gradient clipping
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"\\nSkipping batch {batch_idx} (NaN/Inf gradients)")
                    self.optimizer.zero_grad()
                    continue
                
                # Optimizer step
                self.optimizer.step()
                
                # Accumulate metrics
                batch_loss = loss_dict['total_loss'].item()
                batch_drift = loss_dict['final_position_error'].item()
                batch_var_loss = loss_dict['variation_loss'].item()
                batch_const_penalty = loss_dict['constant_penalty'].item()
                
                if not (np.isnan(batch_loss) or np.isinf(batch_loss)):
                    total_loss += batch_loss
                    total_drift += batch_drift
                    total_variation_loss += batch_var_loss
                    total_constant_penalty += batch_const_penalty
                    successful_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{batch_loss:.6f}',
                    'Drift': f'{batch_drift:.3f}',
                    'VarLoss': f'{batch_var_loss:.6f}',
                    'ConstPen': f'{batch_const_penalty:.6f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.3e}',
                    'Success': f'{successful_batches}/{batch_idx+1}'
                })
                
            except Exception as e:
                print(f"\\nError in batch {batch_idx}: {e}")
                continue
        
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            avg_drift = total_drift / successful_batches
            avg_var_loss = total_variation_loss / successful_batches
            avg_const_penalty = total_constant_penalty / successful_batches
        else:
            avg_loss = float('inf')
            avg_drift = float('inf')
            avg_var_loss = float('inf')
            avg_const_penalty = float('inf')
        
        print(f"\\nSuccessful batches: {successful_batches}/{len(train_loader)}")
        return avg_loss, avg_drift, avg_var_loss, avg_const_penalty
    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0
        total_drift = 0.0
        total_variation_loss = 0.0
        total_constant_penalty = 0.0
        successful_batches = 0
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", ncols=130)
        
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
                    batch_drift = loss_dict['final_position_error'].item()
                    batch_var_loss = loss_dict['variation_loss'].item()
                    batch_const_penalty = loss_dict['constant_penalty'].item()
                    
                    if not (np.isnan(batch_loss) or np.isinf(batch_loss)):
                        total_loss += batch_loss
                        total_drift += batch_drift
                        total_variation_loss += batch_var_loss
                        total_constant_penalty += batch_const_penalty
                        successful_batches += 1
                    
                    pbar.set_postfix({
                        'Loss': f'{batch_loss:.6f}',
                        'Drift': f'{batch_drift:.3f}',
                        'VarLoss': f'{batch_var_loss:.6f}',
                        'ConstPen': f'{batch_const_penalty:.6f}'
                    })
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            avg_drift = total_drift / successful_batches
            avg_var_loss = total_variation_loss / successful_batches
            avg_const_penalty = total_constant_penalty / successful_batches
        else:
            avg_loss = float('inf')
            avg_drift = float('inf')
            avg_var_loss = float('inf')
            avg_const_penalty = float('inf')
        
        return avg_loss, avg_drift, avg_var_loss, avg_const_penalty
    
    def train(self, train_loader, val_loader, epochs):
        print(f"\\nSTARTING NORMALIZED TRAINING")
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
            train_loss, train_drift, train_var_loss, train_const_pen = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_drift, val_var_loss, val_const_pen = self.validate(val_loader, epoch)
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\\nEPOCH {epoch+1} RESULTS:")
            print(f"Time: {epoch_time:.1f}s (Total: {total_time/60:.1f}min)")
            print(f"TRAIN  | Loss: {train_loss:.8f} | Drift: {train_drift:.4f} | VarLoss: {train_var_loss:.6f} | ConstPen: {train_const_pen:.6f}")
            print(f"VAL    | Loss: {val_loss:.8f} | Drift: {val_drift:.4f} | VarLoss: {val_var_loss:.6f} | ConstPen: {val_const_pen:.6f}")
            
            # Save best model
            if val_loss < best_val_loss and val_loss != float('inf'):
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config,
                    'delta_mean': self.delta_mean,
                    'delta_std': self.delta_std
                }, 'normalized_training_best_model.pth')
                print(f"NEW BEST MODEL SAVED! (Loss: {val_loss:.8f})")
            
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\\nNORMALIZED TRAINING COMPLETED!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.8f}")
        
        return best_val_loss

def main():
    # NORMALIZED configuration with AGGRESSIVE learning rate
    config = {
        'model': {
            'img_size': 224,
            'patch_size': 32,
            'd_model': 256,        # Slightly larger
            'num_heads': 4,        # More heads
            'num_layers': 3,       # More layers
            'max_cameras': 1,      
            'max_seq_len': 3,      
            'dropout': 0.1,        
            'use_imu': False,      
            'use_pressure': False,
            'uncertainty_estimation': False
        },
        'training': {
            'epochs': 20,          # More epochs
            'learning_rate': 1e-3, # AGGRESSIVE: 100x higher than previous
            'batch_size': 4        # Larger batch
        }
    }
    
    print("NORMALIZED Training Configuration:")
    print(f"  Learning rate: {config['training']['learning_rate']} (was: 1e-5)")
    print(f"  Model size: {config['model']['d_model']} dim, {config['model']['num_layers']} layers")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Key Fix: DATA NORMALIZATION + Strong variation penalty")
    
    try:
        # Create dataloaders
        print("\\nLoading Dataset...")
        train_loader, val_loader = create_sub_trajectory_dataloaders(
            train_csv='data/processed/training_dataset/training_data_filtered.csv',
            val_csv='data/processed/training_dataset/training_data_filtered.csv',
            sub_trajectory_length=3,   
            overlap=1,                 
            camera_ids=[0],            # Camera 0 only
            batch_size=config['training']['batch_size'],
            num_workers=0,
            max_samples_train=300,     # More data
            max_samples_val=60
        )
        
        print(f"Dataset loaded successfully!")
        
        # Create trainer and start training
        trainer = NormalizedTrainer(config, train_loader)
        best_loss = trainer.train(train_loader, val_loader, config['training']['epochs'])
        
        print(f"\\nNORMALIZED TRAINING FINAL RESULT: Best validation loss = {best_loss:.8f}")
        print("Normalized model saved as: normalized_training_best_model.pth")
        
    except Exception as e:
        print(f"\\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()