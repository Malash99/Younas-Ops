#!/usr/bin/env python3
"""
Stable Training Script for Underwater VIO
==========================================

This script provides a more stable training approach that starts with simple MSE loss
and gradually transitions to complex SE(3) geodesic loss to prevent NaN issues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime
import os

# Import our models and losses
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.underwater_vio_lstm import UnderwaterVIODataset, create_model
from models.vio_losses import create_loss_function, create_simple_loss_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StableVIOTrainer:
    def __init__(self, 
                 csv_path, 
                 data_root,
                 sequence_length=10,
                 batch_size=8,
                 learning_rate=1e-4,
                 num_epochs=100,
                 validation_split=0.2,
                 device=None,
                 save_dir="checkpoints"):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset and dataloaders
        logger.info("Loading dataset...")
        full_dataset = UnderwaterVIODataset(
            csv_path=csv_path, 
            data_root=data_root, 
            sequence_length=sequence_length,
            use_ground_truth_only=True
        )
        
        # Split dataset
        train_size = int((1 - validation_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        # Model and optimizer
        self.model = create_model().to(self.device)
        self.init_model_weights()
        
        # Start with simple MSE loss for stability
        self.simple_criterion = nn.MSELoss()
        self.complex_criterion = create_loss_function()
        self.use_complex_loss = False  # Start with simple loss
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5,
            eps=1e-8
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5, verbose=True
        )
        
        # Training tracking
        self.num_epochs = num_epochs
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logger.info(f"Using device: {self.device}")
    
    def init_model_weights(self):
        """Initialize model weights carefully"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)  # Smaller gain for stability
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param, gain=0.5)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param, gain=0.5)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)
        
        self.model.apply(init_weights)
        logger.info("Model weights initialized with conservative values")
    
    def get_current_criterion(self, epoch):
        """Get the appropriate loss function for current epoch"""
        # Switch to complex loss after 20 epochs of stable simple loss training
        if epoch > 20 and not self.use_complex_loss:
            logger.info("Switching to complex SE(3) geodesic loss...")
            self.use_complex_loss = True
        
        return self.complex_criterion if self.use_complex_loss else self.simple_criterion
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        valid_batches = 0
        
        criterion = self.get_current_criterion(epoch)
        loss_name = "Complex" if self.use_complex_loss else "Simple MSE"
        
        with tqdm(self.train_loader, desc=f"Training ({loss_name})") as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(self.device)
                imu = batch['imu'].to(self.device)
                targets = batch['pose'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(images, imu)
                
                # Compute loss based on current criterion
                if self.use_complex_loss:
                    loss, loss_dict = criterion(predictions, targets, images)
                    current_loss = loss.item() if not torch.isnan(loss) else float('inf')
                else:
                    loss = criterion(predictions, targets)
                    current_loss = loss.item()
                
                # Skip batch if loss is NaN or too large
                if torch.isnan(loss) or torch.isinf(loss) or current_loss > 100:
                    logger.warning(f"Skipping batch {batch_idx} due to unstable loss: {current_loss}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Conservative gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += current_loss
                valid_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.6f}',
                    'avg': f'{total_loss/max(valid_batches, 1):.6f}'
                })
        
        avg_loss = total_loss / max(valid_batches, 1)
        if valid_batches < len(self.train_loader) * 0.5:
            logger.warning(f"Only {valid_batches}/{len(self.train_loader)} batches were valid!")
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        valid_batches = 0
        
        criterion = self.get_current_criterion(epoch)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation") as pbar:
                for batch in pbar:
                    images = batch['images'].to(self.device)
                    imu = batch['imu'].to(self.device)
                    targets = batch['pose'].to(self.device)
                    
                    predictions = self.model(images, imu)
                    
                    if self.use_complex_loss:
                        loss, loss_dict = criterion(predictions, targets, images)
                        current_loss = loss.item() if not torch.isnan(loss) else float('inf')
                    else:
                        loss = criterion(predictions, targets)
                        current_loss = loss.item()
                    
                    if not (torch.isnan(loss) or torch.isinf(loss) or current_loss > 100):
                        total_loss += current_loss
                        valid_batches += 1
                        
                        all_predictions.append(predictions.cpu())
                        all_targets.append(targets.cpu())
                    
                    pbar.set_postfix({'val_loss': f'{current_loss:.6f}'})
        
        avg_loss = total_loss / max(valid_batches, 1)
        
        # Analyze predictions
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            pred_std = torch.std(all_predictions, dim=(0, 1))
            target_std = torch.std(all_targets, dim=(0, 1))
            
            logger.info(f"Prediction std: {pred_std.numpy()}")
            logger.info(f"Target std: {target_std.numpy()}")
            
            if torch.any(pred_std < 0.01):
                logger.warning("Low prediction variance detected!")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'use_complex_loss': self.use_complex_loss
        }
        
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pt')
            logger.info(f"Saved best model at epoch {epoch}")
    
    def train(self):
        logger.info("Starting stable training...")
        logger.info("Phase 1: Simple MSE loss for stable initialization")
        logger.info("Phase 2: Complex SE(3) geodesic loss after epoch 20")
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            logger.info(f"Train Loss: {train_loss:.6f}")
            logger.info(f"Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        logger.info(f"Training completed! Best validation loss: {self.best_val_loss:.6f}")
        return self.model


def main():
    # Configuration
    config = {
        'csv_path': r'data\processed\visual_odometry_dataset\visual_odometry_dataset_kalibr_clean.csv',
        'data_root': r'data\processed\visual_odometry_dataset',
        'sequence_length': 10,
        'batch_size': 4,
        'learning_rate': 5e-5,  # More conservative learning rate
        'num_epochs': 100,
        'validation_split': 0.2,
        'save_dir': 'checkpoints/stable_underwater_vio'
    }
    
    # Create trainer and start training
    trainer = StableVIOTrainer(**config)
    trained_model = trainer.train()
    
    # Save final configuration
    with open(Path(config['save_dir']) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Stable training completed successfully!")


if __name__ == "__main__":
    main()