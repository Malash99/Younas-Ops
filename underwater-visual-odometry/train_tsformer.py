#!/usr/bin/env python3
"""
TSformer Visual Odometry Training Script

Complete training pipeline for transformer-based underwater visual odometry.
Includes transfer learning, windowed sequences, and proper evaluation.

Author: Underwater Visual Odometry Research Team
Date: January 2025
"""

import os
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders


class TSformerTrainer:
    """TSformer Visual Odometry Trainer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # GPU memory optimization
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Enable memory optimization
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Create model and loss
        self.model, self.loss_fn = create_tsformer_vo(
            sequence_length=config['sequence_length'],
            pretrained=config['pretrained'],
            freeze_backbone=config['freeze_backbone'],
            image_size=config['image_size']
        )
        self.model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.setup_optimizer()
        
        # Create data loaders
        self.setup_data()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        print(f"Model parameters: {self.model.get_num_trainable_parameters():,}")
        
    def setup_logging(self):
        """Setup tensorboard logging."""
        log_dir = self.output_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        print(f"Tensorboard logs: {log_dir}")
        
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
    def setup_data(self):
        """Setup data loaders."""
        data_loaders = create_data_loaders(
            csv_path=self.config['csv_path'],
            data_root=self.config['data_root'],
            sequence_length=self.config['sequence_length'],
            overlap_frames=self.config['overlap_frames'],
            image_size=self.config['image_size'],
            batch_size=self.config['batch_size'],
            test_bags=self.config['test_bags'],
            num_workers=self.config['num_workers'],
            camera=self.config['camera']
        )
        
        self.train_loader = data_loaders['train_loader']
        self.val_loader = data_loaders['val_loader']
        self.test_loader = data_loaders['test_loader']
        
        print(f"Data loaders created:")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        # Handle both old and new loss component names
        if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
            epoch_metrics = {'single_step_loss': [], 'multi_step_loss': [], 'chain_consistency_loss': []}
        else:  # Original loss
            epoch_metrics = {'geodesic_loss': [], 'consistency_loss': [], 'magnitude_loss': []}
        
        # Gradient accumulation for effective larger batch size
        accumulate_steps = self.config.get('accumulate_steps', 1)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['images'].to(self.device)  # (B, T, C, H, W)
            poses = batch['poses'].to(self.device)    # (B, 6) - single frame deltas
            
            # Check if we have relative poses (for multi-scale loss)
            if 'relative_poses' in batch:
                relative_poses = batch['relative_poses'].to(self.device)  # (B, 6) - full sequence relative pose
            else:
                relative_poses = poses  # Fallback to single frame poses
            
            # Forward pass
            if batch_idx % accumulate_steps == 0:
                self.optimizer.zero_grad()
                
            pred_poses = self.model(images)
            
            # Calculate loss (check if multi-scale loss)
            if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
                loss, loss_dict = self.loss_fn(pred_poses, poses, relative_poses)
            else:  # Original loss
                loss, loss_dict = self.loss_fn(pred_poses, poses)
            loss = loss / accumulate_steps  # Scale loss for accumulation
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulate_steps
            if (batch_idx + 1) % accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            # Clear cache periodically to prevent memory buildup
            if batch_idx % 50 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Track metrics
            epoch_losses.append(loss_dict['total_loss'])
            
            # Handle different loss component names
            if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
                epoch_metrics['single_step_loss'].append(loss_dict['single_step_loss'])
                epoch_metrics['multi_step_loss'].append(loss_dict['multi_step_loss'])
                epoch_metrics['chain_consistency_loss'].append(loss_dict['chain_consistency_loss'])
            else:  # Original loss
                epoch_metrics['geodesic_loss'].append(loss_dict['geodesic_loss'])
                epoch_metrics['consistency_loss'].append(loss_dict['consistency_loss'])
                epoch_metrics['magnitude_loss'].append(loss_dict['magnitude_loss'])
            
            # Log batch metrics
            if batch_idx % self.config['log_interval'] == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss_dict['total_loss'], step)
                
                # Handle different loss types for logging
                if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
                    print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}: "
                          f"Loss={loss_dict['total_loss']:.6f}, SingleStep={loss_dict['single_step_loss']:.6f}, "
                          f"MultiStep={loss_dict['multi_step_loss']:.6f}, Chain={loss_dict['chain_consistency_loss']:.6f}")
                    
                    self.writer.add_scalar('Train/SingleStepLoss', loss_dict['single_step_loss'], step)
                    self.writer.add_scalar('Train/MultiStepLoss', loss_dict['multi_step_loss'], step)
                    self.writer.add_scalar('Train/ChainLoss', loss_dict['chain_consistency_loss'], step)
                else:  # Original loss
                    print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}: "
                          f"Loss={loss_dict['total_loss']:.6f}, Geodesic={loss_dict['geodesic_loss']:.6f}, "
                          f"Consistency={loss_dict['consistency_loss']:.6f}, Magnitude={loss_dict['magnitude_loss']:.6f}")
                    
                    self.writer.add_scalar('Train/GeodesicLoss', loss_dict['geodesic_loss'], step)
                    self.writer.add_scalar('Train/ConsistencyLoss', loss_dict['consistency_loss'], step)
                    self.writer.add_scalar('Train/MagnitudeLoss', loss_dict['magnitude_loss'], step)
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        
        if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
            avg_comp1 = np.mean(epoch_metrics['single_step_loss'])
            avg_comp2 = np.mean(epoch_metrics['multi_step_loss'])
            avg_comp3 = np.mean(epoch_metrics['chain_consistency_loss'])
        else:  # Original loss
            avg_comp1 = np.mean(epoch_metrics['geodesic_loss'])
            avg_comp2 = np.mean(epoch_metrics['consistency_loss'])
            avg_comp3 = np.mean(epoch_metrics['magnitude_loss'])
        
        return avg_loss, avg_comp1, avg_comp2, avg_comp3
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        val_losses = []
        # Handle both old and new loss component names
        if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
            val_metrics = {'single_step_loss': [], 'multi_step_loss': [], 'chain_consistency_loss': []}
        else:  # Original loss
            val_metrics = {'geodesic_loss': [], 'consistency_loss': [], 'magnitude_loss': []}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                images = batch['images'].to(self.device)
                poses = batch['poses'].to(self.device)
                
                # Check if we have relative poses (for multi-scale loss)
                if 'relative_poses' in batch:
                    relative_poses = batch['relative_poses'].to(self.device)
                else:
                    relative_poses = poses
                
                # Forward pass
                pred_poses = self.model(images)
                
                # Calculate loss (check if multi-scale loss)
                if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
                    loss, loss_dict = self.loss_fn(pred_poses, poses, relative_poses)
                else:  # Original loss
                    loss, loss_dict = self.loss_fn(pred_poses, poses)
                
                # Track metrics
                val_losses.append(loss_dict['total_loss'])
                
                # Handle different loss component names
                if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
                    val_metrics['single_step_loss'].append(loss_dict['single_step_loss'])
                    val_metrics['multi_step_loss'].append(loss_dict['multi_step_loss'])
                    val_metrics['chain_consistency_loss'].append(loss_dict['chain_consistency_loss'])
                else:  # Original loss
                    val_metrics['geodesic_loss'].append(loss_dict['geodesic_loss'])
                    val_metrics['consistency_loss'].append(loss_dict['consistency_loss'])
                    val_metrics['magnitude_loss'].append(loss_dict['magnitude_loss'])
        
        # Calculate averages
        avg_val_loss = np.mean(val_losses)
        
        if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
            avg_comp1 = np.mean(val_metrics['single_step_loss'])
            avg_comp2 = np.mean(val_metrics['multi_step_loss'])
            avg_comp3 = np.mean(val_metrics['chain_consistency_loss'])
        else:  # Original loss
            avg_comp1 = np.mean(val_metrics['geodesic_loss'])
            avg_comp2 = np.mean(val_metrics['consistency_loss'])
            avg_comp3 = np.mean(val_metrics['magnitude_loss'])
        
        return avg_val_loss, avg_comp1, avg_comp2, avg_comp3
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded: epoch {self.current_epoch}, best_val_loss={self.best_val_loss:.6f}")
    
    def train(self):
        """Main training loop."""
        print(f"\\nStarting training for {self.config['num_epochs']} epochs...")
        print("="*80)
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_loss, train_geodesic, train_consistency, train_magnitude = self.train_epoch()
            
            # Validate
            val_loss, val_geodesic, val_consistency, val_magnitude = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Logging
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} ({epoch_time:.1f}s):")
            
            # Handle different loss types for logging
            if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
                print(f"  Train Loss: {train_loss:.6f} (SingleStep: {train_geodesic:.6f}, MultiStep: {train_consistency:.6f}, Chain: {train_magnitude:.6f})")
                print(f"  Val Loss:   {val_loss:.6f} (SingleStep: {val_geodesic:.6f}, MultiStep: {val_consistency:.6f}, Chain: {val_magnitude:.6f})")
            else:  # Original loss
                print(f"  Train Loss: {train_loss:.6f} (Geodesic: {train_geodesic:.6f}, Consistency: {train_consistency:.6f}, Magnitude: {train_magnitude:.6f})")
                print(f"  Val Loss:   {val_loss:.6f} (Geodesic: {val_geodesic:.6f}, Consistency: {val_consistency:.6f}, Magnitude: {val_magnitude:.6f})")
            
            print(f"  LR: {current_lr:.2e}")
            
            # Tensorboard logging
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
            self.writer.add_scalar('Epoch/LearningRate', current_lr, epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best)
            
            # Early stopping
            if current_lr < 1e-7:
                print("Learning rate too small, stopping training.")
                break
        
        print("\\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Final evaluation on test set
        self.evaluate_test()
        
        # Plot training curves
        self.plot_training_curves()
    
    def evaluate_test(self):
        """Evaluate on test set."""
        print("\\nEvaluating on test set...")
        
        # Load best model
        best_checkpoint = self.output_dir / 'checkpoint_best.pth'
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        test_losses = []
        # Handle both old and new loss component names
        if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
            test_metrics = {'single_step_loss': [], 'multi_step_loss': [], 'chain_consistency_loss': []}
        else:  # Original loss
            test_metrics = {'geodesic_loss': [], 'consistency_loss': [], 'magnitude_loss': []}
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['images'].to(self.device)
                poses = batch['poses'].to(self.device)
                
                # Check if we have relative poses (for multi-scale loss)
                if 'relative_poses' in batch:
                    relative_poses = batch['relative_poses'].to(self.device)
                else:
                    relative_poses = poses
                
                pred_poses = self.model(images)
                
                # Calculate loss (check if multi-scale loss)
                if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
                    loss, loss_dict = self.loss_fn(pred_poses, poses, relative_poses)
                else:  # Original loss
                    loss, loss_dict = self.loss_fn(pred_poses, poses)
                
                test_losses.append(loss_dict['total_loss'])
                
                # Handle different loss component names
                if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
                    test_metrics['single_step_loss'].append(loss_dict['single_step_loss'])
                    test_metrics['multi_step_loss'].append(loss_dict['multi_step_loss'])
                    test_metrics['chain_consistency_loss'].append(loss_dict['chain_consistency_loss'])
                else:  # Original loss
                    test_metrics['geodesic_loss'].append(loss_dict['geodesic_loss'])
                    test_metrics['consistency_loss'].append(loss_dict['consistency_loss'])
                    test_metrics['magnitude_loss'].append(loss_dict['magnitude_loss'])
                
                # Store predictions for analysis
                predictions.append(pred_poses.cpu().numpy())
                ground_truths.append(poses.cpu().numpy())
        
        # Calculate test metrics
        avg_test_loss = np.mean(test_losses)
        
        if hasattr(self.loss_fn, 'λ2'):  # Multi-scale loss
            avg_comp1 = np.mean(test_metrics['single_step_loss'])
            avg_comp2 = np.mean(test_metrics['multi_step_loss'])
            avg_comp3 = np.mean(test_metrics['chain_consistency_loss'])
            
            print(f"Test Results:")
            print(f"  Test Loss: {avg_test_loss:.6f}")
            print(f"  Single Step Loss: {avg_comp1:.6f}")
            print(f"  Multi Step Loss: {avg_comp2:.6f}")
            print(f"  Chain Consistency Loss: {avg_comp3:.6f}")
            
            # Save test results
            test_results = {
                'test_loss': avg_test_loss,
                'single_step_loss': avg_comp1,
                'multi_step_loss': avg_comp2,
                'chain_consistency_loss': avg_comp3,
                'predictions': np.concatenate(predictions, axis=0).tolist(),
                'ground_truth': np.concatenate(ground_truths, axis=0).tolist()
            }
        else:  # Original loss
            avg_comp1 = np.mean(test_metrics['geodesic_loss'])
            avg_comp2 = np.mean(test_metrics['consistency_loss'])
            avg_comp3 = np.mean(test_metrics['magnitude_loss'])
            
            print(f"Test Results:")
            print(f"  Test Loss: {avg_test_loss:.6f}")
            print(f"  Geodesic Loss: {avg_comp1:.6f}")
            print(f"  Consistency Loss: {avg_comp2:.6f}")
            print(f"  Magnitude Loss: {avg_comp3:.6f}")
            
            # Save test results
            test_results = {
                'test_loss': avg_test_loss,
                'geodesic_loss': avg_comp1,
                'consistency_loss': avg_comp2,
                'magnitude_loss': avg_comp3,
                'predictions': np.concatenate(predictions, axis=0).tolist(),
                'ground_truth': np.concatenate(ground_truths, axis=0).tolist()
            }
        
        results_path = self.output_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"Test results saved: {results_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('TSformer-VO Training Curves')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TSformer Visual Odometry")
    
    # Data
    parser.add_argument('--csv_path', type=str, 
                       default='data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv',
                       help='Path to dataset CSV')
    parser.add_argument('--data_root', type=str,
                       default='data/processed/visual_odometry_dataset', 
                       help='Root directory containing images')
    parser.add_argument('--test_bags', nargs='+', 
                       default=['ariel_2023-12-21-14-28-22_4'],
                       help='Bags to reserve for testing')
    
    # Model
    parser.add_argument('--sequence_length', type=int, default=8,
                       help='Number of frames per sequence')
    parser.add_argument('--overlap_frames', type=int, default=1,
                       help='Frame overlap between windows')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--camera', type=str, default='cam0',
                       help='Which camera to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ViT backbone')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                       help='Freeze ViT backbone parameters')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--accumulate_steps', type=int, default=2,
                       help='Gradient accumulation steps for larger effective batch size')
    
    # Logging
    parser.add_argument('--output_dir', type=str, default='experiments/tsformer_vo',
                       help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Convert to config dict
    config = vars(args)
    
    print("TSformer Visual Odometry Training")
    print("="*50)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*50)
    
    # Create trainer
    trainer = TSformerTrainer(config)
    
    # Resume if requested
    if args.resume and Path(args.resume).exists():
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()