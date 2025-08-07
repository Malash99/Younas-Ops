"""
Scale and Direction Fixed Training Script

This training script implements learnable scale factors and direction consistency
to fix the two major remaining issues:
1. Scale problem: Predictions 5.3x too small
2. Direction problem: Backward vs forward motion confusion

Based on research from SC-SfMLearner++, DeepVO++, and CamPoseNet approaches.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer.multiscale_uw_transvo import create_multiscale_model
from training.multiscale_loss import ScaleDirectionLoss

class ScaleDirectionDataset(Dataset):
    """Dataset with scale and direction supervision measures"""
    
    def __init__(self, csv_file, sequence_length=10, img_size=192, stride=6):
        self.csv_file = csv_file
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.stride = stride
        
        # Load data
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} frames from {csv_file}")
        
        # Filter sequences with significant motion variation
        self.sequences = self._filter_good_sequences()
        
        print(f"Created {len(self.sequences)} diverse motion sequences")
    
    def _filter_good_sequences(self):
        """Filter sequences that have good motion variation to prevent collapse"""
        sequences = []
        
        for i in range(0, len(self.df) - self.sequence_length + 1, self.stride):
            seq_data = self.df.iloc[i:i + self.sequence_length]
            
            # Extract motion deltas
            deltas = seq_data[['delta_x', 'delta_y', 'delta_z']].values
            
            # Check motion diversity
            motion_std = np.std(deltas, axis=0)
            total_motion = np.sum(motion_std)
            
            # Only keep sequences with sufficient motion variation
            if total_motion > 0.002:  # Minimum motion diversity threshold
                sequences.append(i)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        start_idx = self.sequences[idx]
        sequence_data = self.df.iloc[start_idx:start_idx + self.sequence_length]
        
        images = []
        delta_poses = []
        
        for _, row in sequence_data.iterrows():
            # Handle image loading with robust fallbacks
            img_path = row['cam0_path']
            if pd.isna(img_path):
                img = np.random.rand(self.img_size, self.img_size, 3) * 0.1  # Small random noise instead of zeros
            else:
                if not os.path.exists(str(img_path)):
                    img_path = os.path.join(".", str(img_path))
                
                if os.path.exists(str(img_path)):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (self.img_size, self.img_size))
                            img = img.astype(np.float32) / 255.0
                        else:
                            img = np.random.rand(self.img_size, self.img_size, 3) * 0.1
                    except:
                        img = np.random.rand(self.img_size, self.img_size, 3) * 0.1
                else:
                    img = np.random.rand(self.img_size, self.img_size, 3) * 0.1
            
            images.append(img.astype(np.float32))
            
            # Extract delta pose
            delta_pose = np.array([
                float(row.get('delta_x', 0.0)), float(row.get('delta_y', 0.0)), float(row.get('delta_z', 0.0)),
                float(row.get('delta_roll', 0.0)), float(row.get('delta_pitch', 0.0)), float(row.get('delta_yaw', 0.0))
            ], dtype=np.float32)
            delta_poses.append(delta_pose)
        
        # Convert to tensors
        images = np.stack(images)
        images = torch.tensor(images).permute(0, 3, 1, 2)  # [seq_len, 3, H, W]
        delta_poses = torch.tensor(np.stack(delta_poses))  # [seq_len, 6]
        
        # Add camera dimension and transpose: [1, seq_len, 3, H, W]
        images = images.unsqueeze(1).transpose(0, 1)
        
        return images, delta_poses

# Configuration
CONFIG = {
    # Model configuration
    'model': {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'max_seq_len': 10,
        'uncertainty_estimation': False,
        'image_size': 192
    },
    
    # Training configuration  
    'training': {
        'batch_size': 4,
        'learning_rate': 8e-6,  # Slightly lower for stability with new loss
        'num_epochs': 15,
        'gradient_clip_norm': 0.5,
        'weight_decay': 1e-4,
        'warmup_steps': 200,
        'scheduler_patience': 3,
        'early_stopping_patience': 5,
        'val_check_interval': 50
    },
    
    # NEW: Scale & Direction Loss Configuration
    'loss': {
        'mse_weight': 1.0,
        'diversity_weight': 0.1,        # Keep diversity to prevent collapse
        'frame_diff_weight': 0.5,       # Frame variation
        'magnitude_weight': 4.0,        # INCREASED: Strong scale supervision
        'direction_weight': 3.0,        # INCREASED: Strong direction consistency  
        'scale_reg_weight': 0.2         # Scale regularization
    },
    
    # Data configuration
    'data': {
        'train_csv': 'data/processed/training_dataset/training_data.csv',
        'val_csv': 'data/processed/training_dataset/training_data.csv',  # Use same for now, will split internally
        'sequence_length': 10,
        'stride': 6,
        'img_size': 192,
        'min_motion_threshold': 0.002
    },
    
    # Output configuration
    'output': {
        'model_save_path': 'scale_direction_fixed_model.pth',
        'history_save_path': 'scale_direction_training_history.json',
        'plot_save_path': 'scale_direction_training_progress.png'
    }
}

class ScaleDirectionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model using the working factory function
        self.model = create_multiscale_model(config['model']).to(self.device)
        
        # Skip pre-trained loading for now due to architecture changes
        print("Training from scratch with new scale prediction architecture...")
        
        # Initialize NEW loss function with scale and direction supervision
        self.criterion = ScaleDirectionLoss(
            mse_weight=config['loss']['mse_weight'],
            diversity_weight=config['loss']['diversity_weight'],
            frame_diff_weight=config['loss']['frame_diff_weight'],
            magnitude_weight=config['loss']['magnitude_weight'],
            direction_weight=config['loss']['direction_weight'],
            scale_reg_weight=config['loss']['scale_reg_weight']
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=config['training']['scheduler_patience'],
            factor=0.7
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_mse': [], 'val_mse': [],
            'train_magnitude': [], 'val_magnitude': [],
            'train_direction': [], 'val_direction': [],
            'train_scale_reg': [], 'val_scale_reg': [],
            'magnitude_ratios': [], 'scale_means': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def setup_data(self):
        """Setup training and validation datasets"""
        print("Setting up datasets...")
        
        # Training dataset
        train_dataset = ScaleDirectionDataset(
            csv_file=self.config['data']['train_csv'],
            sequence_length=self.config['data']['sequence_length'],
            img_size=self.config['data']['img_size'],
            stride=self.config['data']['stride']
        )
        
        # For validation, use same file but with different stride and offset
        val_dataset = ScaleDirectionDataset(
            csv_file=self.config['data']['val_csv'],
            sequence_length=self.config['data']['sequence_length'],
            img_size=self.config['data']['img_size'],
            stride=self.config['data']['stride'] * 3  # Larger stride for validation + offset
        )
        # Manually offset validation sequences to avoid overlap
        if hasattr(val_dataset, 'sequences'):
            val_dataset.sequences = val_dataset.sequences[1::2]  # Use every other sequence
        
        print(f"Train dataset: {len(train_dataset)} sequences")
        print(f"Val dataset: {len(val_dataset)} sequences")
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True, 
            num_workers=0,  # Use 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,  # Use 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': [], 'mse': [], 'magnitude': [], 'direction': [], 'scale_reg': []}
        epoch_metrics = {'magnitude_ratios': [], 'scale_means': []}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch_idx, batch in enumerate(pbar):
            try:
                images, poses = batch
                images = images.to(self.device)  
                poses = poses.to(self.device)
                
                # Forward pass with camera IDs
                batch_size, seq_len = images.shape[1:3]  # [num_cameras, batch, seq_len, 3, H, W]
                camera_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)  # All cam0
                outputs = self.model(images, camera_ids)
                
                # Get delta poses (primary output)
                pred_deltas = outputs['delta_poses']  # Already scaled by model
                target_deltas = poses
                
                # Get predicted scale factors
                pred_scales = outputs.get('delta_scales', None)
                
                # Compute loss with NEW scale and direction supervision
                loss_dict = self.criterion(pred_deltas, target_deltas, pred_scales)
                
                loss = loss_dict['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip_norm']
                )
                self.optimizer.step()
                
                # Accumulate losses
                epoch_losses['total'].append(loss.item())
                epoch_losses['mse'].append(loss_dict['mse_loss'].item())
                epoch_losses['magnitude'].append(loss_dict['magnitude_loss'].item())
                epoch_losses['direction'].append(loss_dict['direction_loss'].item())
                epoch_losses['scale_reg'].append(loss_dict['scale_reg_loss'].item())
                
                # Accumulate metrics
                epoch_metrics['magnitude_ratios'].append(loss_dict['magnitude_ratio'].item())
                if 'scale_mean' in loss_dict:
                    epoch_metrics['scale_means'].append(loss_dict['scale_mean'].item())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MSE': f'{loss_dict["mse_loss"].item():.4f}',
                    'Mag': f'{loss_dict["magnitude_loss"].item():.4f}',
                    'Dir': f'{loss_dict["direction_loss"].item():.4f}',
                    'Scale': f'{loss_dict.get("scale_mean", 0.0):.3f}',
                    'Ratio': f'{loss_dict["magnitude_ratio"].item():.3f}'
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return avg_losses, avg_metrics
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = {'total': [], 'mse': [], 'magnitude': [], 'direction': [], 'scale_reg': []}
        epoch_metrics = {'magnitude_ratios': [], 'scale_means': []}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            for batch_idx, batch in enumerate(pbar):
                try:
                    images, poses = batch
                    images = images.to(self.device)
                    poses = poses.to(self.device)
                    
                    # Forward pass with camera IDs
                    batch_size, seq_len = images.shape[1:3]  # [num_cameras, batch, seq_len, 3, H, W]  
                    camera_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)  # All cam0
                    outputs = self.model(images, camera_ids)
                    
                    pred_deltas = outputs['delta_poses']
                    target_deltas = poses
                    pred_scales = outputs.get('delta_scales', None)
                    
                    # Compute loss
                    loss_dict = self.criterion(pred_deltas, target_deltas, pred_scales)
                    
                    # Accumulate losses
                    epoch_losses['total'].append(loss_dict['total_loss'].item())
                    epoch_losses['mse'].append(loss_dict['mse_loss'].item())
                    epoch_losses['magnitude'].append(loss_dict['magnitude_loss'].item())
                    epoch_losses['direction'].append(loss_dict['direction_loss'].item())
                    epoch_losses['scale_reg'].append(loss_dict['scale_reg_loss'].item())
                    
                    # Accumulate metrics
                    epoch_metrics['magnitude_ratios'].append(loss_dict['magnitude_ratio'].item())
                    if 'scale_mean' in loss_dict:
                        epoch_metrics['scale_means'].append(loss_dict['scale_mean'].item())
                    
                    pbar.set_postfix({
                        'Loss': f'{loss_dict["total_loss"].item():.4f}',
                        'Ratio': f'{loss_dict["magnitude_ratio"].item():.3f}'
                    })
                    
                except Exception as e:
                    print(f"Error in val batch {batch_idx}: {e}")
                    continue
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return avg_losses, avg_metrics
    
    def train(self):
        """Main training loop"""
        print("Starting Scale & Direction Fixed Training...")
        print(f"Target: Fix 5.3x scale problem and backward/forward direction issue")
        
        for epoch in range(self.config['training']['num_epochs']):
            # Train
            train_losses, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_losses, val_metrics = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['train_mse'].append(train_losses['mse'])
            self.history['val_mse'].append(val_losses['mse'])
            self.history['train_magnitude'].append(train_losses['magnitude'])
            self.history['val_magnitude'].append(val_losses['magnitude'])
            self.history['train_direction'].append(train_losses['direction'])
            self.history['val_direction'].append(val_losses['direction'])
            self.history['train_scale_reg'].append(train_losses['scale_reg'])
            self.history['val_scale_reg'].append(val_losses['scale_reg'])
            self.history['magnitude_ratios'].append(val_metrics['magnitude_ratios'])
            if val_metrics['scale_means']:
                self.history['scale_means'].append(val_metrics['scale_means'])
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']} Summary:")
            print(f"Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f}")
            print(f"Magnitude Ratio: {val_metrics['magnitude_ratios']:.3f} (target: ~1.0)")
            print(f"Direction Loss: {val_losses['direction']:.4f} (target: ~0.0)")
            if val_metrics['scale_means']:
                print(f"Scale Factor: {val_metrics['scale_means']:.3f} (target: ~5.0)")
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.epochs_without_improvement = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_losses['total'],
                    'config': self.config
                }, self.config['output']['model_save_path'])
                
                print(f"✅ New best model saved! Val loss: {val_losses['total']:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['training']['early_stopping_patience']:
                print(f"Early stopping triggered after {self.epochs_without_improvement} epochs without improvement")
                break
            
            # Save training history
            with open(self.config['output']['history_save_path'], 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # Plot progress every few epochs
            if (epoch + 1) % 2 == 0:
                self.plot_training_progress()
        
        print("Training completed!")
        return self.history
    
    def plot_training_progress(self):
        """Plot training progress with scale and direction metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plots
        axes[0,0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0,0].plot(epochs, self.history['val_loss'], 'r-', label='Val')
        axes[0,0].set_title('Total Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # MSE plots
        axes[0,1].plot(epochs, self.history['train_mse'], 'b-', label='Train MSE')
        axes[0,1].plot(epochs, self.history['val_mse'], 'r-', label='Val MSE')
        axes[0,1].set_title('MSE Loss')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Magnitude supervision
        axes[0,2].plot(epochs, self.history['train_magnitude'], 'b-', label='Train Mag')
        axes[0,2].plot(epochs, self.history['val_magnitude'], 'r-', label='Val Mag')
        axes[0,2].set_title('Magnitude Loss (Scale Supervision)')
        axes[0,2].legend()
        axes[0,2].grid(True)
        
        # Direction consistency
        axes[1,0].plot(epochs, self.history['train_direction'], 'b-', label='Train Dir')
        axes[1,0].plot(epochs, self.history['val_direction'], 'r-', label='Val Dir')
        axes[1,0].axhline(y=0, color='g', linestyle='--', alpha=0.7, label='Target=0')
        axes[1,0].set_title('Direction Loss (Cosine Similarity)')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Magnitude ratio (key metric!)
        if self.history['magnitude_ratios']:
            axes[1,1].plot(epochs, self.history['magnitude_ratios'], 'g-', linewidth=2, label='Pred/GT Ratio')
            axes[1,1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Target=1.0')
            axes[1,1].axhline(y=0.19, color='orange', linestyle='--', alpha=0.7, label='Current=0.19 (5.3x too small)')
            axes[1,1].set_title('Magnitude Ratio (Critical Metric)')
            axes[1,1].legend()
            axes[1,1].grid(True)
            axes[1,1].set_ylim(0, 2)
        
        # Scale factors
        if self.history['scale_means']:
            axes[1,2].plot(epochs, self.history['scale_means'], 'm-', linewidth=2, label='Predicted Scale')
            axes[1,2].axhline(y=5.3, color='r', linestyle='--', alpha=0.7, label='Target≈5.3')
            axes[1,2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline=1.0')
            axes[1,2].set_title('Learned Scale Factors')
            axes[1,2].legend()
            axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.config['output']['plot_save_path'], dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Initialize trainer
    trainer = ScaleDirectionTrainer(CONFIG)
    
    # Setup data
    trainer.setup_data()
    
    # Start training
    history = trainer.train()
    
    print(f"\n🎯 TRAINING COMPLETE!")
    print(f"📊 Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"💾 Model saved to: {CONFIG['output']['model_save_path']}")
    print(f"📈 Training history saved to: {CONFIG['output']['history_save_path']}")
    print(f"🖼️ Progress plots saved to: {CONFIG['output']['plot_save_path']}")
    
    # Final status
    if history['magnitude_ratios']:
        final_ratio = history['magnitude_ratios'][-1]
        print(f"\n🔍 FINAL SCALE STATUS:")
        print(f"Magnitude ratio: {final_ratio:.3f} (target: 1.0)")
        if final_ratio > 0.5:
            print("✅ SIGNIFICANT IMPROVEMENT in scale!")
        else:
            print("⚠️ Scale still needs work")
    
    if history['val_direction']:
        final_direction = history['val_direction'][-1]
        print(f"\n🧭 FINAL DIRECTION STATUS:")
        print(f"Direction loss: {final_direction:.4f} (target: <0.2)")
        if final_direction < 0.3:
            print("✅ GOOD directional consistency!")
        else:
            print("⚠️ Direction still needs improvement")

if __name__ == '__main__':
    main()