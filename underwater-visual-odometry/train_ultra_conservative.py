#!/usr/bin/env python3
"""
Ultra-Conservative Training - UW-TransVO
Maximum stability, minimal complexity
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Ensure console window is visible on Windows
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.AllocConsole()
        kernel32.SetConsoleTitleW("UW-TransVO Ultra-Conservative Training")
    except:
        pass

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.sub_trajectory_dataset import create_sub_trajectory_dataloaders

class SimpleTrajectoryLoss(nn.Module):
    """Ultra-simple loss function to avoid numerical issues"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, pose_targets, accumulated_targets):
        """Simple MSE loss only"""
        # Only use frame-to-frame pose loss
        translation_loss = self.mse(predictions[:, :, :3], pose_targets[:, :, :3])
        rotation_loss = self.mse(predictions[:, :, 3:], pose_targets[:, :, 3:])
        
        total_loss = translation_loss + rotation_loss
        
        # Calculate simple drift metric
        final_pred = predictions[:, -1, :3]  # Last predicted translation
        final_target = pose_targets[:, -1, :3]  # Last target translation
        drift = torch.mean(torch.norm(final_pred - final_target, dim=-1))
        
        return {
            'total_loss': total_loss,
            'translation_loss': translation_loss,
            'rotation_loss': rotation_loss,
            'ate_loss': translation_loss,  # Same as translation for simplicity
            'consistency_loss': torch.tensor(0.0, device=total_loss.device),
            'smoothness_loss': torch.tensor(0.0, device=total_loss.device),
            'final_position_error': drift,
            'predicted_accumulated': predictions.cumsum(dim=1),  # Simple accumulation
            'target_accumulated': pose_targets.cumsum(dim=1)
        }

class UltraConservativeModel(nn.Module):
    """Minimal wrapper for UW-TransVO"""
    
    def __init__(self, config):
        super().__init__()
        self.base_model = UWTransVO(**config)
        
    def forward(self, images, camera_ids, camera_mask, sub_traj_length):
        batch_size, seq_len, num_cameras, C, H, W = images.shape
        all_predictions = []
        
        # Process pairs sequentially (most stable)
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

class UltraConservativeTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Clear screen and show banner
        os.system('cls' if os.name == 'nt' else 'clear')
        self.print_banner()
        
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        
        print("\nInitializing Ultra-Conservative Model...")
        self.model = UltraConservativeModel(config['model']).to(self.device)
        
        # Ultra-simple loss
        self.criterion = SimpleTrajectoryLoss()
        
        # Ultra-conservative optimizer
        self.optimizer = torch.optim.SGD(  # SGD instead of AdamW for stability
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.0,  # No momentum
            weight_decay=0.0  # No weight decay
        )
        
        # No scheduler - constant learning rate
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model Parameters: {param_count:,}")
        
        # Check for NaN in initial parameters
        nan_params = sum(1 for p in self.model.parameters() if torch.isnan(p).any())
        if nan_params > 0:
            print(f"WARNING: {nan_params} parameter tensors contain NaN values!")
        else:
            print("Model parameters initialized correctly (no NaN values)")
    
    def print_banner(self):
        banner = """
========================================================
            UW-TransVO Ultra-Conservative Training
                Maximum Stability Mode
========================================================
        """
        print(banner)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_drift = 0.0
        successful_batches = 0
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1} Training",
            ncols=100
        )
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move to device
                images = batch['images'].to(self.device)
                camera_ids = batch['camera_ids'].to(self.device)
                camera_mask = batch['camera_mask'].to(self.device)
                pose_targets = batch['pose_targets'].to(self.device)
                accumulated_targets = batch['accumulated_poses'].to(self.device)
                
                sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
                
                # Forward pass - NO mixed precision, pure float32
                self.optimizer.zero_grad()
                predictions = self.model(images, camera_ids, camera_mask, sub_traj_length)
                
                # Calculate loss
                loss_dict = self.criterion(predictions, pose_targets, accumulated_targets)
                
                # Check for NaN BEFORE backward pass
                if torch.isnan(loss_dict['total_loss']) or torch.isinf(loss_dict['total_loss']):
                    print(f"\nSkipping batch {batch_idx} (NaN/Inf loss: {loss_dict['total_loss'].item()})")
                    continue
                
                # Backward pass
                loss_dict['total_loss'].backward()
                
                # Ultra-conservative gradient clipping
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.001)
                
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"\nSkipping batch {batch_idx} (NaN/Inf gradients)")
                    self.optimizer.zero_grad()
                    continue
                
                # Optimizer step
                self.optimizer.step()
                
                # Accumulate metrics
                batch_loss = loss_dict['total_loss'].item()
                batch_drift = loss_dict['final_position_error'].item()
                
                if not (np.isnan(batch_loss) or np.isinf(batch_loss)):
                    total_loss += batch_loss
                    total_drift += batch_drift
                    successful_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{batch_loss:.6f}',
                    'Drift': f'{batch_drift:.3f}m',
                    'Batches': f'{successful_batches}/{batch_idx+1}'
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            avg_drift = total_drift / successful_batches
        else:
            avg_loss = float('inf')
            avg_drift = float('inf')
        
        print(f"\nSuccessful batches: {successful_batches}/{len(train_loader)}")
        return avg_loss, avg_drift
    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0
        total_drift = 0.0
        successful_batches = 0
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", ncols=100)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                try:
                    images = batch['images'].to(self.device)
                    camera_ids = batch['camera_ids'].to(self.device)
                    camera_mask = batch['camera_mask'].to(self.device)
                    pose_targets = batch['pose_targets'].to(self.device)
                    accumulated_targets = batch['accumulated_poses'].to(self.device)
                    
                    sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
                    
                    predictions = self.model(images, camera_ids, camera_mask, sub_traj_length)
                    loss_dict = self.criterion(predictions, pose_targets, accumulated_targets)
                    
                    batch_loss = loss_dict['total_loss'].item()
                    batch_drift = loss_dict['final_position_error'].item()
                    
                    if not (np.isnan(batch_loss) or np.isinf(batch_loss)):
                        total_loss += batch_loss
                        total_drift += batch_drift
                        successful_batches += 1
                    
                    pbar.set_postfix({
                        'Loss': f'{batch_loss:.6f}',
                        'Drift': f'{batch_drift:.3f}m'
                    })
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            avg_drift = total_drift / successful_batches
        else:
            avg_loss = float('inf')
            avg_drift = float('inf')
        
        return avg_loss, avg_drift
    
    def train(self, train_loader, val_loader, epochs):
        print(f"\nSTARTING ULTRA-CONSERVATIVE TRAINING")
        print(f"Epochs: {epochs}")
        print(f"Train Samples: {len(train_loader.dataset)}")
        print(f"Val Samples: {len(val_loader.dataset)}")
        print("=" * 60)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')} | LR: {self.optimizer.param_groups[0]['lr']:.10f}")
            
            # Train
            train_loss, train_drift = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_drift = self.validate(val_loader, epoch)
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\nEPOCH {epoch+1} RESULTS:")
            print(f"Time: {epoch_time:.1f}s (Total: {total_time/60:.1f}min)")
            print(f"TRAIN  | Loss: {train_loss:.8f} | Drift: {train_drift:.4f}m")
            print(f"VAL    | Loss: {val_loss:.8f} | Drift: {val_drift:.4f}m")
            
            # Save best model
            if val_loss < best_val_loss and val_loss != float('inf'):
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, 'ultra_conservative_best_model.pth')
                print(f"NEW BEST MODEL SAVED! (Loss: {val_loss:.8f})")
            
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\nTRAINING COMPLETED!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.8f}")
        
        return best_val_loss

def main():
    import numpy as np
    
    # Ultra-conservative configuration
    config = {
        'model': {
            'img_size': 224,
            'patch_size': 32,      # Larger patches = fewer tokens
            'd_model': 192,        # Much smaller
            'num_heads': 3,        # Minimal heads
            'num_layers': 2,       # Minimal layers
            'max_cameras': 1,      # Single camera
            'max_seq_len': 3,      # Very short sequences
            'dropout': 0.0,        # No dropout
            'use_imu': False,      # No additional sensors
            'use_pressure': False,
            'uncertainty_estimation': False  # No uncertainty
        },
        'training': {
            'epochs': 10,          # Fewer epochs for testing
            'learning_rate': 1e-8, # EXTREMELY small learning rate
            'batch_size': 1        # Single batch
        }
    }
    
    print("Ultra-Conservative Configuration:")
    print(f"  Model size: {config['model']['d_model']} dim, {config['model']['num_layers']} layers")
    print(f"  Sequence length: {config['model']['max_seq_len']} frames")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    
    try:
        # Create dataloaders with minimal sequences
        print("\nLoading Dataset...")
        train_loader, val_loader = create_sub_trajectory_dataloaders(
            train_csv='data/processed/training_dataset/training_data_filtered.csv',
            val_csv='data/processed/training_dataset/training_data_filtered.csv',
            sub_trajectory_length=3,   # Very short sequences
            overlap=1,                 # Minimal overlap
            camera_ids=[0],            # Single camera only
            batch_size=config['training']['batch_size'],
            num_workers=0,
            max_samples_train=100,     # Small dataset for testing
            max_samples_val=20
        )
        
        print(f"Dataset loaded successfully!")
        
        # Create trainer and start training
        trainer = UltraConservativeTrainer(config)
        best_loss = trainer.train(train_loader, val_loader, config['training']['epochs'])
        
        print(f"\nFINAL RESULT: Best validation loss = {best_loss:.8f}")
        print("Model saved as: ultra_conservative_best_model.pth")
        
        # Keep console open
        if sys.platform == "win32":
            input("\nPress Enter to exit...")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        if sys.platform == "win32":
            input("\nPress Enter to exit...")

if __name__ == '__main__':
    main()