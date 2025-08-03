#!/usr/bin/env python3
"""
Full Sub-Trajectory Training with Fixed Web Dashboard
Production-ready training with trajectory-aware loss
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
import os

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.sub_trajectory_dataset import create_sub_trajectory_dataloaders
from training.trajectory_losses import TrajectoryAwareLoss, calculate_trajectory_metrics

# Import dashboard components
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'underwater-visual-odometry-2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for training state
training_data = {
    'status': 'idle',
    'current_epoch': 0,
    'current_batch': 0,
    'total_epochs': 0,
    'total_batches': 0,
    'losses': [],
    'gpu_memory': [],
    'timestamps': [],
    'model_params': 0,
    'dataset_size': 0,
    'training_speed': [],
    'start_time': None,
    'last_update': None,
    'loss': 0.0,
    'ate_loss': 0.0,
    'final_drift_m': 0.0,
    'lr': 0.0
}

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """API endpoint for training status"""
    return jsonify(training_data)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('training_update', training_data)
    print(f"Dashboard client connected")

class SubTrajectoryModel(nn.Module):
    """Wrapper for UWTransVO to handle sub-trajectory sequences"""
    
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

class FullTrainer:
    """Complete trainer for sub-trajectory training"""
    
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
        
        print(f"FullTrainer initialized")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def update_dashboard(self, data_dict):
        """Update dashboard with training data"""
        training_data.update(data_dict)
        training_data['last_update'] = datetime.now().strftime('%H:%M:%S')
        try:
            socketio.emit('training_update', training_data)
        except:
            pass  # Ignore dashboard errors
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
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
            loss = loss_dict['total_loss'] / gradient_accumulation_steps  # Scale loss
            total_loss += loss_dict['total_loss'].item()
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update dashboard every 10 batches
            if batch_idx % 10 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                self.update_dashboard({
                    'status': 'training',
                    'epoch': epoch,
                    'batch': batch_idx,
                    'total_batches': num_batches,
                    'loss': loss_dict['total_loss'].item(),
                    'translation_loss': loss_dict['translation_loss'].item(),
                    'rotation_loss': loss_dict['rotation_loss'].item(),
                    'ate_loss': loss_dict['ate_loss'].item(),
                    'final_drift_m': loss_dict['final_position_error'].item(),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'gpu_memory': gpu_memory
                })
            
            if batch_idx % 25 == 0:
                print(f"  Batch {batch_idx:4d}/{num_batches} - "
                      f"Loss: {loss_dict['total_loss'].item():.6f} - "
                      f"ATE: {loss_dict['ate_loss'].item():.6f} - "
                      f"Drift: {loss_dict['final_position_error'].item():.4f}m - "
                      f"GPU: {gpu_memory:.2f}GB")
        
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
    
    def save_checkpoint(self, epoch, val_loss, val_metrics, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        filename = 'sub_trajectory_best_model.pth' if is_best else f'sub_trajectory_checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, filename)
        return filename
    
    def train(self, train_loader, val_loader, epochs):
        """Full training loop"""
        print(f"\nStarting full sub-trajectory training for {epochs} epochs...")
        
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
            
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Train
            start_time = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            train_time = time.time() - start_time
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update dashboard
            epoch_data = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr'],
                'train_time': train_time
            }
            
            self.update_dashboard(epoch_data)
            
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Train Time: {train_time:.1f}s")
            
            if val_metrics:
                print(f"  Val Drift: {val_metrics.get('final_drift_m', 0):.4f}m "
                      f"({val_metrics.get('relative_drift_percent', 0):.1f}%)")
                print(f"  Val ATE: {val_metrics.get('ate_rmse', 0):.4f}m")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_metrics)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_filename = self.save_checkpoint(epoch, val_loss, val_metrics, is_best=True)
                print(f"  -> NEW BEST MODEL: {best_filename} (val_loss: {val_loss:.6f})")
            
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")
        
        training_data['status'] = 'completed'
        self.update_dashboard({'status': 'completed'})
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"{'='*60}")
        
        return best_val_loss

def run_dashboard_server():
    """Run the dashboard server in background"""
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Dashboard error: {e}")

def main():
    # Configuration for full training
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
            'epochs': 50,  # Full training
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'batch_size': 1,  # Must use 1 for 4GB VRAM
            'gradient_accumulation_steps': 4  # Simulate batch_size=4
        }
    }
    
    print("Sub-Trajectory Training - FULL VERSION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Sub-trajectory length: 5 frames")
    print(f"  Cameras: 3 (0, 1, 2)")
    
    # Start dashboard server in background thread
    dashboard_thread = threading.Thread(target=run_dashboard_server, daemon=True)
    dashboard_thread.start()
    
    print(f"\nWeb dashboard starting at http://localhost:5000")
    time.sleep(2)  # Give dashboard time to start
    
    try:
        webbrowser.open('http://localhost:5000')
    except:
        print("Could not open browser automatically")
    
    # Create dataloaders
    print(f"\nCreating datasets...")
    train_loader, val_loader = create_sub_trajectory_dataloaders(
        train_csv='data/processed/training_dataset/training_data.csv',
        val_csv='data/processed/training_dataset/training_data.csv',
        sub_trajectory_length=5,
        overlap=2,
        camera_ids=[0, 1, 2],
        batch_size=config['training']['batch_size'],
        num_workers=0,
        max_samples_train=None,  # Use full dataset
        max_samples_val=None     # Use full dataset
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create trainer and train
    trainer = FullTrainer(config)
    best_loss = trainer.train(train_loader, val_loader, config['training']['epochs'])
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Web dashboard still running at http://localhost:5000")
    
    # Keep dashboard running
    input("\nPress Enter to stop dashboard and exit...")

if __name__ == '__main__':
    main()