#!/usr/bin/env python3
"""
Simple Console Training - Sub-Trajectory with ATE Loss
Clean console output, no web dashboard
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.sub_trajectory_dataset import create_sub_trajectory_dataloaders
from training.trajectory_losses import TrajectoryAwareLoss, calculate_trajectory_metrics

class SubTrajectoryModel(nn.Module):
    def __init__(self, base_model_config):
        super().__init__()
        self.base_model = UWTransVO(**base_model_config)
        
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

class ConsoleTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
            
        self.model = SubTrajectoryModel(config['model']).to(self.device)
        
        self.criterion = TrajectoryAwareLoss(
            translation_weight=1.0,
            rotation_weight=10.0,
            ate_weight=5.0,
            consistency_weight=1.0,
            smoothness_weight=0.5
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['training']['epochs']
        )
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_ate_loss = 0.0
        total_drift = 0.0
        num_batches = len(train_loader)
        
        print(f"\nEpoch {epoch+1} Training:")
        print("-" * 50)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            images = batch['images'].to(self.device)
            camera_ids = batch['camera_ids'].to(self.device)
            camera_mask = batch['camera_mask'].to(self.device)
            pose_targets = batch['pose_targets'].to(self.device)
            accumulated_targets = batch['accumulated_poses'].to(self.device)
            
            sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images, camera_ids, camera_mask, sub_traj_length)
            
            # Calculate loss
            loss_dict = self.criterion(predictions, pose_targets, accumulated_targets)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss_dict['total_loss'].item()
            total_ate_loss += loss_dict['ate_loss'].item()
            total_drift += loss_dict['final_position_error'].item()
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0 or batch_idx == num_batches - 1:
                gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                print(f"  [{batch_idx:4d}/{num_batches}] "
                      f"Loss: {loss_dict['total_loss'].item():.6f} | "
                      f"ATE: {loss_dict['ate_loss'].item():.6f} | "
                      f"Drift: {loss_dict['final_position_error'].item():.4f}m | "
                      f"GPU: {gpu_mem:.2f}GB")
        
        avg_loss = total_loss / num_batches
        avg_ate_loss = total_ate_loss / num_batches
        avg_drift = total_drift / num_batches
        
        return avg_loss, avg_ate_loss, avg_drift
    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0
        total_ate_loss = 0.0
        total_drift = 0.0
        all_metrics = []
        
        print(f"\nEpoch {epoch+1} Validation:")
        print("-" * 50)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['images'].to(self.device)
                camera_ids = batch['camera_ids'].to(self.device)
                camera_mask = batch['camera_mask'].to(self.device)
                pose_targets = batch['pose_targets'].to(self.device)
                accumulated_targets = batch['accumulated_poses'].to(self.device)
                
                sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
                
                predictions = self.model(images, camera_ids, camera_mask, sub_traj_length)
                loss_dict = self.criterion(predictions, pose_targets, accumulated_targets)
                
                total_loss += loss_dict['total_loss'].item()
                total_ate_loss += loss_dict['ate_loss'].item()
                total_drift += loss_dict['final_position_error'].item()
                
                # Calculate detailed metrics
                pred_acc = loss_dict['predicted_accumulated']
                target_acc = loss_dict['target_accumulated']
                
                for i in range(pred_acc.shape[0]):
                    metrics = calculate_trajectory_metrics(pred_acc[i], target_acc[i])
                    all_metrics.append(metrics)
                
                if batch_idx % 25 == 0 or batch_idx == len(val_loader) - 1:
                    print(f"  [{batch_idx:4d}/{len(val_loader)}] "
                          f"Loss: {loss_dict['total_loss'].item():.6f} | "
                          f"ATE: {loss_dict['ate_loss'].item():.6f} | "
                          f"Drift: {loss_dict['final_position_error'].item():.4f}m")
        
        avg_loss = total_loss / len(val_loader)
        avg_ate_loss = total_ate_loss / len(val_loader)
        avg_drift = total_drift / len(val_loader)
        
        # Average detailed metrics
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        else:
            avg_metrics = {}
        
        return avg_loss, avg_ate_loss, avg_drift, avg_metrics
    
    def train(self, train_loader, val_loader, epochs):
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING: {epochs} epochs")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"{'='*60}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_ate, train_drift = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_ate, val_drift, val_metrics = self.validate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{epochs} SUMMARY")
            print(f"{'='*60}")
            print(f"Time: {epoch_time:.1f}s (Total: {total_time/60:.1f}min)")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")
            print(f"")
            print(f"TRAINING:")
            print(f"  Loss: {train_loss:.6f}")
            print(f"  ATE Loss: {train_ate:.6f}")
            print(f"  Avg Drift: {train_drift:.4f}m")
            print(f"")
            print(f"VALIDATION:")
            print(f"  Loss: {val_loss:.6f}")
            print(f"  ATE Loss: {val_ate:.6f}")
            print(f"  Avg Drift: {val_drift:.4f}m")
            
            if val_metrics:
                print(f"  Final Drift: {val_metrics.get('final_drift_m', 0):.4f}m")
                print(f"  Relative Drift: {val_metrics.get('relative_drift_percent', 0):.2f}%")
                print(f"  Trajectory Length: {val_metrics.get('trajectory_length_m', 0):.4f}m")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'config': self.config
                }, 'console_best_model.pth')
                print(f"  ★ NEW BEST MODEL SAVED! (Loss: {val_loss:.6f})")
            
            print(f"{'='*60}")
        
        total_time = time.time() - start_time
        print(f"\nTRAINING COMPLETED!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return best_val_loss

def main():
    config = {
        'model': {
            'img_size': 224,
            'patch_size': 16,
            'd_model': 768,
            'num_heads': 1,
            'num_layers': 6,
            'max_cameras': 3,
            'max_seq_len': 5,
            'dropout': 0.1,
            'use_imu': False,
            'use_pressure': False,
            'uncertainty_estimation': True
        },
        'training': {
            'epochs': 30,  # Reasonable number for testing
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'batch_size': 1
        }
    }
    
    print("Sub-Trajectory Training - Console Version")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Sub-trajectory: 5 frames, 3 cameras")
    
    # Create dataloaders
    train_loader, val_loader = create_sub_trajectory_dataloaders(
        train_csv='data/processed/training_dataset/training_data.csv',
        val_csv='data/processed/training_dataset/training_data.csv',
        sub_trajectory_length=5,
        overlap=2,
        camera_ids=[0, 1, 2],
        batch_size=config['training']['batch_size'],
        num_workers=0,
        max_samples_train=None,  # Full dataset
        max_samples_val=None
    )
    
    # Create trainer and start training
    trainer = ConsoleTrainer(config)
    best_loss = trainer.train(train_loader, val_loader, config['training']['epochs'])
    
    print(f"\nFinal Result: Best validation loss = {best_loss:.6f}")
    print("Model saved as: console_best_model.pth")

if __name__ == '__main__':
    main()