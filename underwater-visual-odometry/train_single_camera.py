#!/usr/bin/env python3
"""
Single Camera Training Script - UW-TransVO
Features:
- One camera only for simplified training
- Progress bars for visual feedback
- Console window visibility
- Real-time metrics display
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import subprocess

# Ensure console window is visible on Windows
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.AllocConsole()
        kernel32.SetConsoleTitleW("UW-TransVO Training - Single Camera")
    except:
        pass

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.sub_trajectory_dataset import create_sub_trajectory_dataloaders
from training.trajectory_losses import TrajectoryAwareLoss, calculate_trajectory_metrics

class SingleCameraModel(nn.Module):
    """Wrapper for UW-TransVO with single camera input"""
    
    def __init__(self, config):
        super().__init__()
        # Modify config for single camera
        config['max_cameras'] = 1
        self.base_model = UWTransVO(**config)
        
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

class SingleCameraTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Clear screen and show banner
        os.system('cls' if os.name == 'nt' else 'clear')
        self.print_banner()
        
        print(f"🚀 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
            print(f"💾 GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print("\n📦 Initializing Model...")
        self.model = SingleCameraModel(config['model']).to(self.device)
        
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
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"🔢 Model Parameters: {param_count:,}")
        print(f"📊 Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "")
    
    def print_banner(self):
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    UW-TransVO Training                       ║
║                  Single Camera Version                       ║
║                                                              ║
║  🌊 Underwater Visual Odometry with Transformers            ║
║  📹 Single Camera Setup                                      ║
║  🚀 Real-time Progress Tracking                             ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_ate_loss = 0.0
        total_drift = 0.0
        
        # Create progress bar
        pbar = tqdm(
            train_loader, 
            desc=f"🏋️  Epoch {epoch+1} Training",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        for batch_idx, batch in enumerate(pbar):
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
            batch_loss = loss_dict['total_loss'].item()
            batch_ate = loss_dict['ate_loss'].item()
            batch_drift = loss_dict['final_position_error'].item()
            
            total_loss += batch_loss
            total_ate_loss += batch_ate
            total_drift += batch_drift
            
            # Update progress bar
            gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'ATE': f'{batch_ate:.4f}',
                'Drift': f'{batch_drift:.2f}m',
                'GPU': f'{gpu_mem:.1f}GB'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_ate_loss = total_ate_loss / len(train_loader)
        avg_drift = total_drift / len(train_loader)
        
        return avg_loss, avg_ate_loss, avg_drift
    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0
        total_ate_loss = 0.0
        total_drift = 0.0
        all_metrics = []
        
        # Create progress bar for validation
        pbar = tqdm(
            val_loader, 
            desc=f"✅ Epoch {epoch+1} Validation",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(self.device)
                camera_ids = batch['camera_ids'].to(self.device)
                camera_mask = batch['camera_mask'].to(self.device)
                pose_targets = batch['pose_targets'].to(self.device)
                accumulated_targets = batch['accumulated_poses'].to(self.device)
                
                sub_traj_length = batch['metadata']['sub_traj_length'][0].item()
                
                predictions = self.model(images, camera_ids, camera_mask, sub_traj_length)
                loss_dict = self.criterion(predictions, pose_targets, accumulated_targets)
                
                batch_loss = loss_dict['total_loss'].item()
                batch_ate = loss_dict['ate_loss'].item()
                batch_drift = loss_dict['final_position_error'].item()
                
                total_loss += batch_loss
                total_ate_loss += batch_ate
                total_drift += batch_drift
                
                # Calculate detailed metrics
                pred_acc = loss_dict['predicted_accumulated']
                target_acc = loss_dict['target_accumulated']
                
                for i in range(pred_acc.shape[0]):
                    metrics = calculate_trajectory_metrics(pred_acc[i], target_acc[i])
                    all_metrics.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{batch_loss:.4f}',
                    'ATE': f'{batch_ate:.4f}',
                    'Drift': f'{batch_drift:.2f}m'
                })
        
        avg_loss = total_loss / len(val_loader)
        avg_ate_loss = total_ate_loss / len(val_loader)
        avg_drift = total_drift / len(val_loader)
        
        # Average detailed metrics
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        return avg_loss, avg_ate_loss, avg_drift, avg_metrics
    
    def train(self, train_loader, val_loader, epochs):
        print(f"\n{'='*80}")
        print(f"🚀 STARTING TRAINING")
        print(f"📊 Epochs: {epochs}")
        print(f"🏋️  Train Samples: {len(train_loader.dataset)}")
        print(f"✅ Val Samples: {len(val_loader.dataset)}")
        print(f"📹 Camera Setup: Single Camera (ID=0)")
        print(f"⚙️  Model: UW-TransVO (Single Camera)")
        print(f"{'='*80}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            print(f"\n🔄 Epoch {epoch+1}/{epochs}")
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | LR: {self.optimizer.param_groups[0]['lr']:.8f}")
            
            # Train
            train_loss, train_ate, train_drift = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_ate, val_drift, val_metrics = self.validate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\n📈 EPOCH {epoch+1} RESULTS:")
            print(f"{'─'*60}")
            print(f"⏱️  Time: {epoch_time:.1f}s (Total: {total_time/60:.1f}min)")
            print(f"🏋️  TRAIN  | Loss: {train_loss:.6f} | ATE: {train_ate:.6f} | Drift: {train_drift:.4f}m")
            print(f"✅ VAL    | Loss: {val_loss:.6f} | ATE: {val_ate:.6f} | Drift: {val_drift:.4f}m")
            
            if val_metrics:
                print(f"📊 METRICS | Final Drift: {val_metrics.get('final_drift_m', 0):.4f}m | " +
                      f"Relative: {val_metrics.get('relative_drift_percent', 0):.2f}% | " +
                      f"Trajectory: {val_metrics.get('trajectory_length_m', 0):.2f}m")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = 'single_camera_best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'config': self.config
                }, model_path)
                print(f"⭐ NEW BEST MODEL SAVED! ({model_path}) | Loss: {val_loss:.6f}")
            
            print(f"{'─'*60}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
        print(f"🏆 Best Validation Loss: {best_val_loss:.6f}")
        print(f"💾 Model Saved: single_camera_best_model.pth")
        
        return best_val_loss

def main():
    # Configuration for single camera
    config = {
        'model': {
            'img_size': 224,
            'patch_size': 16,
            'd_model': 768,
            'num_heads': 12,      # Full attention heads
            'num_layers': 6,
            'max_cameras': 1,     # Single camera only
            'max_seq_len': 5,
            'dropout': 0.1,
            'use_imu': False,     # Disabled as requested
            'use_pressure': False, # Disabled as requested
            'uncertainty_estimation': True
        },
        'training': {
            'epochs': 50,         # More epochs for single camera
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'batch_size': 2       # Larger batch size for single camera
        }
    }
    
    print("🔧 Configuration:")
    print(f"   📊 Epochs: {config['training']['epochs']}")
    print(f"   📦 Batch Size: {config['training']['batch_size']}")
    print(f"   🎯 Learning Rate: {config['training']['learning_rate']}")
    print(f"   📹 Cameras: 1 (Single Camera)")
    print(f"   🔄 Sub-trajectory Length: 5 frames")
    print(f"   🚫 IMU/Pressure: Disabled")
    
    try:
        # Create dataloaders for single camera
        print("\n📂 Loading Dataset...")
        train_loader, val_loader = create_sub_trajectory_dataloaders(
            train_csv='data/processed/training_dataset/training_data.csv',
            val_csv='data/processed/training_dataset/training_data.csv',
            sub_trajectory_length=5,
            overlap=2,
            camera_ids=[0],  # Single camera only (camera 0)
            batch_size=config['training']['batch_size'],
            num_workers=0,   # Single worker for stability
            max_samples_train=None,  # Use full dataset
            max_samples_val=100      # Limit validation for speed
        )
        
        print(f"✅ Dataset loaded successfully!")
        
        # Create trainer and start training
        trainer = SingleCameraTrainer(config)
        best_loss = trainer.train(train_loader, val_loader, config['training']['epochs'])
        
        print(f"\n🎯 FINAL RESULT: Best validation loss = {best_loss:.6f}")
        
        # Keep console open
        if sys.platform == "win32":
            input("\nPress Enter to exit...")
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        if sys.platform == "win32":
            input("\nPress Enter to exit...")

if __name__ == '__main__':
    main()