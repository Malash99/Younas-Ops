#!/usr/bin/env python3
"""
Quick Test: Sub-Trajectory Training with ATE-Aware Loss
Fast test to verify the approach works before full training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import time
import json
import threading
import queue
from datetime import datetime
import webbrowser

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.sub_trajectory_dataset import create_sub_trajectory_dataloaders
from training.trajectory_losses import TrajectoryAwareLoss, calculate_trajectory_metrics
from web_training_dashboard import app, socketio, training_data, training_queue

class SubTrajectoryModel(nn.Module):
    """
    Wrapper for UWTransVO to handle sub-trajectory sequences
    """
    
    def __init__(self, base_model_config):
        super().__init__()
        self.base_model = UWTransVO(**base_model_config)
        self.sub_traj_length = None
        
    def forward(self, images, camera_ids, camera_mask, sub_traj_length):
        """
        Forward pass for sub-trajectory
        
        Args:
            images: [batch, sub_traj_len, num_cameras, 3, H, W]
            camera_ids: [num_cameras]
            camera_mask: [num_cameras]
            sub_traj_length: Length of sub-trajectory
            
        Returns:
            predictions: [batch, sub_traj_len-1, 6] relative poses
        """
        batch_size, seq_len, num_cameras, C, H, W = images.shape
        
        # Process each consecutive frame pair
        all_predictions = []
        
        for t in range(seq_len - 1):
            # Get frame pair: t and t+1
            frame_pair = torch.stack([images[:, t], images[:, t+1]], dim=1)  # [batch, 2, num_cameras, 3, H, W]
            
            # Forward through base model
            output = self.base_model(
                images=frame_pair,
                camera_ids=camera_ids,
                camera_mask=camera_mask
            )
            
            all_predictions.append(output['pose'])  # [batch, 6]
        
        # Stack predictions: [batch, seq_len-1, 6]
        predictions = torch.stack(all_predictions, dim=1)
        
        return predictions


class QuickTrainer:
    """Quick trainer for sub-trajectory testing"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = SubTrajectoryModel(config['model']).to(self.device) 
        
        # Create loss function
        self.criterion = TrajectoryAwareLoss(
            translation_weight=1.0,
            rotation_weight=10.0,
            ate_weight=5.0,
            consistency_weight=1.0,
            smoothness_weight=0.5
        )
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['training']['epochs']
        )
        
        print(f"QuickTrainer initialized")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            images = batch['images'].to(self.device)  # [batch, sub_traj_len, num_cameras, 3, H, W]
            camera_ids = batch['camera_ids'].to(self.device)
            camera_mask = batch['camera_mask'].to(self.device)
            pose_targets = batch['pose_targets'].to(self.device)  # [batch, sub_traj_len-1, 6]
            accumulated_targets = batch['accumulated_poses'].to(self.device)  # [batch, sub_traj_len-1, 6]
            
            sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images, camera_ids, camera_mask, sub_traj_length)
            
            # Calculate loss
            loss_dict = self.criterion(predictions, pose_targets, accumulated_targets)
            total_loss += loss_dict['total_loss'].item()
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update web dashboard
            if batch_idx % 5 == 0:  # Update every 5 batches
                metrics = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'total_batches': num_batches,
                    'loss': loss_dict['total_loss'].item(),
                    'translation_loss': loss_dict['translation_loss'].item(),
                    'rotation_loss': loss_dict['rotation_loss'].item(),
                    'ate_loss': loss_dict['ate_loss'].item(),
                    'final_drift_m': loss_dict['final_position_error'].item(),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'gpu_memory': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                }
                
                try:
                    training_queue.put(('batch_update', metrics), block=False)
                except queue.Full:
                    pass
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{num_batches} - "
                      f"Loss: {loss_dict['total_loss'].item():.6f} - "
                      f"ATE: {loss_dict['ate_loss'].item():.6f} - "
                      f"Drift: {loss_dict['final_position_error'].item():.4f}m")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move to device  
                images = batch['images'].to(self.device)
                camera_ids = batch['camera_ids'].to(self.device)
                camera_mask = batch['camera_mask'].to(self.device)
                pose_targets = batch['pose_targets'].to(self.device)
                accumulated_targets = batch['accumulated_poses'].to(self.device)
                
                sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
                
                # Forward pass
                predictions = self.model(images, camera_ids, camera_mask, sub_traj_length)
                
                # Calculate loss
                loss_dict = self.criterion(predictions, pose_targets, accumulated_targets)
                total_loss += loss_dict['total_loss'].item()
                
                # Calculate trajectory metrics for each sample in batch
                pred_acc = loss_dict['predicted_accumulated']
                target_acc = loss_dict['target_accumulated']
                
                for i in range(pred_acc.shape[0]):
                    metrics = calculate_trajectory_metrics(pred_acc[i], target_acc[i])
                    all_metrics.append(metrics)
        
        avg_loss = total_loss / len(val_loader)
        
        # Average metrics
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        else:
            avg_metrics = {}
        
        return avg_loss, avg_metrics
    
    def train(self, train_loader, val_loader, epochs):
        """Full training loop"""
        print(f"Starting quick training for {epochs} epochs...")
        
        # Update global training data
        training_data.update({
            'status': 'training',
            'total_epochs': epochs,
            'total_batches': len(train_loader),
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'dataset_size': len(train_loader.dataset),
            'start_time': datetime.now().isoformat()
        })
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            training_data['current_epoch'] = epoch
            
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update web dashboard
            epoch_data = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            try:
                training_queue.put(('epoch_complete', epoch_data), block=False)
            except queue.Full:
                pass
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            if val_metrics:
                print(f"Val Drift: {val_metrics.get('final_drift_m', 0):.4f}m "
                      f"({val_metrics.get('relative_drift_percent', 0):.1f}%)")
            
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
                }, 'quick_test_best_model.pth')
                print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
        
        training_data['status'] = 'completed'
        print(f"\nQuick training completed!")
        return best_val_loss


def run_web_dashboard():
    """Run the web dashboard in a separate thread"""
    def process_training_updates():
        while True:
            try:
                update_type, data = training_queue.get(timeout=1)
                
                if update_type == 'batch_update':
                    training_data.update(data)
                    training_data['last_update'] = datetime.now().isoformat()
                    socketio.emit('training_update', training_data)
                    
                elif update_type == 'epoch_complete':
                    training_data['losses'].append(data['train_loss'])
                    training_data.update(data)
                    socketio.emit('epoch_complete', data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Dashboard update error: {e}")
    
    # Start background thread for processing updates
    update_thread = threading.Thread(target=process_training_updates, daemon=True)
    update_thread.start()
    
    # Run Flask app
    print("Starting web dashboard at http://localhost:5000")
    try:
        webbrowser.open('http://localhost:5000')
    except:
        pass
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)


def main():
    # Configuration
    config = {
        'model': {
            'img_size': 224,
            'patch_size': 16,
            'd_model': 768,
            'num_heads': 1,
            'num_layers': 6,
            'max_cameras': 3,  # Reduced from 4 to 3
            'max_seq_len': 5,  # Reduced from 8 to 5
            'dropout': 0.1,
            'use_imu': False,
            'use_pressure': False,
            'uncertainty_estimation': True
        },
        'training': {
            'epochs': 10,  # Quick test
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'batch_size': 1  # Very small batch due to memory constraints
        }
    }
    
    print("Sub-Trajectory Training - Quick Test")
    print("=" * 50)
    
    # Create dataloaders
    train_loader, val_loader = create_sub_trajectory_dataloaders(
        train_csv='data/processed/training_dataset/training_data.csv',
        val_csv='data/processed/training_dataset/training_data.csv',
        sub_trajectory_length=5,  # Reduced from 8 to 5
        overlap=2,  # Reduced overlap accordingly
        camera_ids=[0, 1, 2],  # Use only 3 cameras instead of 4
        batch_size=config['training']['batch_size'],
        num_workers=0,  # Avoid multiprocessing issues on Windows
        max_samples_train=50,  # Quick test with limited samples
        max_samples_val=20
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Start web dashboard in background
    dashboard_thread = threading.Thread(target=run_web_dashboard, daemon=True)
    dashboard_thread.start()
    
    # Wait a moment for dashboard to start
    time.sleep(2)
    
    # Create trainer and train
    trainer = QuickTrainer(config)
    best_loss = trainer.train(train_loader, val_loader, config['training']['epochs'])
    
    print(f"\nQuick test completed!")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Model saved: quick_test_best_model.pth")
    print(f"Web dashboard: http://localhost:5000")
    
    # Keep dashboard running
    input("Press Enter to stop...")


if __name__ == '__main__':
    main()