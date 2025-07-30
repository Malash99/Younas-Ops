"""
Main training pipeline for UW-TransVO

Handles:
- Multi-camera configuration training
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Model checkpointing
- Tensorboard logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from models.transformer import UWTransVO
from .losses import create_loss_function
from data.preprocessing import DataStandardizer


class UWTransVOTrainer:
    """
    Comprehensive trainer for UW-TransVO model
    
    Supports multiple camera configurations, mixed precision training,
    and extensive logging for research experiments
    """
    
    def __init__(
        self,
        model: UWTransVO,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str,
        experiment_name: str
    ):
        """
        Args:
            model: UW-TransVO model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Training device
            checkpoint_dir: Directory for saving checkpoints
            experiment_name: Name of the experiment
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_name = experiment_name
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.checkpoint_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._setup_training_components()
        self._setup_logging()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Data standardizer
        self.data_standardizer = DataStandardizer()
        
        print(f"Trainer initialized for experiment: {experiment_name}")
        print(f"Model parameters: {model.count_parameters():,}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
    def _setup_training_components(self):
        """Initialize optimizer, scheduler, loss function, etc."""
        
        # Loss function
        self.criterion = create_loss_function(self.config['loss'])
        
        # Optimizer
        optimizer_config = self.config['optimizer']
        if optimizer_config['type'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_config['type'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
        
        # Learning rate scheduler
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        # Mixed precision training (essential for 4GB GPU)
        self.use_amp = self.config.get('mixed_precision', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training enabled for GPU memory efficiency")
        
        # Gradient accumulation
        self.accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
    def _setup_logging(self):
        """Initialize logging components"""
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Save configuration
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {}
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            loss_dict = self.train_step(batch)
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value)
            
            # Logging
            if batch_idx % self.config.get('log_interval', 50) == 0:
                self._log_batch(batch_idx, num_batches, loss_dict)
            
            self.step += 1
        
        # Average losses over epoch
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        
        # Move batch to device
        images = batch['images'].to(self.device)
        camera_ids = batch['camera_ids'].to(self.device)
        camera_mask = batch['camera_mask'].to(self.device)
        pose_target = batch['pose_target'].to(self.device)
        
        # Optional sensor data
        imu_data = batch.get('imu_data')
        pressure_data = batch.get('pressure_data')
        
        if imu_data is not None:
            imu_data = imu_data.to(self.device)
        if pressure_data is not None:
            pressure_data = pressure_data.to(self.device)
        
        # Forward pass with mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast():
                predictions = self.model(
                    images=images,
                    camera_ids=camera_ids,
                    camera_mask=camera_mask,
                    imu_data=imu_data,
                    pressure_data=pressure_data
                )
                
                # Compute loss
                loss_dict = self.criterion(predictions['pose'], pose_target)
                
                loss = loss_dict['total_loss'] / self.accumulation_steps
        else:
            predictions = self.model(
                images=images,
                camera_ids=camera_ids,
                camera_mask=camera_mask,
                imu_data=imu_data,
                pressure_data=pressure_data
            )
            
            # Compute loss
            if isinstance(self.criterion, nn.Module):
                if 'uncertainty' in predictions:
                    loss_dict = self.criterion(
                        predictions['pose'], 
                        pose_target,
                        predictions.get('uncertainty')
                    )
                else:
                    loss_dict = self.criterion(predictions['pose'], pose_target)
            else:
                loss_dict = self.criterion(predictions, pose_target)
            
            loss = loss_dict['total_loss'] / self.accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.step + 1) % self.accumulation_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        # Convert to float for logging
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in loss_dict.items()}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = {}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                images = batch['images'].to(self.device)
                camera_ids = batch['camera_ids'].to(self.device)
                camera_mask = batch['camera_mask'].to(self.device)
                pose_target = batch['pose_target'].to(self.device)
                
                # Optional sensor data
                imu_data = batch.get('imu_data')
                pressure_data = batch.get('pressure_data')
                
                if imu_data is not None:
                    imu_data = imu_data.to(self.device)
                if pressure_data is not None:
                    pressure_data = pressure_data.to(self.device)
                
                # Forward pass
                predictions = self.model(
                    images=images,
                    camera_ids=camera_ids,
                    camera_mask=camera_mask,
                    imu_data=imu_data,
                    pressure_data=pressure_data
                )
                
                # Compute loss
                loss_dict = self.criterion(predictions['pose'], pose_target)
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    if key not in val_losses:
                        val_losses[key] = []
                    val_losses[key].append(value.item() if isinstance(value, torch.Tensor) else value)
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
        
        return avg_losses
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['training']['epochs']} epochs...")
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total_loss'])
                else:
                    self.scheduler.step()
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            self._log_epoch(epoch, train_losses, val_losses, epoch_time)
            
            # Save checkpoint
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
            
            self._save_checkpoint(epoch, train_losses, val_losses, is_best)
            
            # Store losses
            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Save final results
        self._save_training_results()
        
        self.writer.close()
    
    def _log_batch(self, batch_idx: int, num_batches: int, loss_dict: Dict[str, float]):
        """Log batch results"""
        if batch_idx % self.config.get('log_interval', 50) == 0:
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {self.epoch:3d} [{batch_idx:4d}/{num_batches:4d}] "
                  f"Loss: {loss_dict['total_loss']:.6f} "
                  f"Trans: {loss_dict.get('translation_loss', 0):.6f} "
                  f"Rot: {loss_dict.get('rotation_loss', 0):.6f} "
                  f"LR: {lr:.2e}")
    
    def _log_epoch(
        self, 
        epoch: int, 
        train_losses: Dict[str, float], 
        val_losses: Dict[str, float],
        epoch_time: float
    ):
        """Log epoch results"""
        lr = self.optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch:3d} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {lr:.2e}")
        print(f"  Train Loss: {train_losses['total_loss']:.6f}")
        print(f"  Val Loss: {val_losses['total_loss']:.6f}")
        
        # Tensorboard logging
        for key, value in train_losses.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        for key, value in val_losses.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        self.writer.add_scalar('Learning_Rate', lr, epoch)
        self.writer.add_scalar('Epoch_Time', epoch_time, epoch)
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        train_losses: Dict[str, float], 
        val_losses: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  New best model saved! Val loss: {val_losses['total_loss']:.6f}")
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints(keep_last=3)
    
    def _cleanup_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoint files"""
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        checkpoint_files.sort()
        
        if len(checkpoint_files) > keep_last:
            for old_checkpoint in checkpoint_files[:-keep_last]:
                old_checkpoint.unlink()
    
    def _save_training_results(self):
        """Save training results and plots"""
        results = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.train_losses),
            'model_parameters': self.model.count_parameters()
        }
        
        results_path = self.checkpoint_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training results saved to {results_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from epoch {self.epoch}")


def create_trainer(
    model: UWTransVO,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    checkpoint_dir: str,
    experiment_name: str
) -> UWTransVOTrainer:
    """Factory function to create trainer"""
    return UWTransVOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name
    )