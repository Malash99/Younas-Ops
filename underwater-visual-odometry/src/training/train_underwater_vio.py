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
from models.vio_losses import create_loss_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VIOTrainer:
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
        
        # Model, loss, and optimizer
        self.model = create_model().to(self.device)
        
        # Initialize model weights more carefully
        self.init_model_weights()
        
        self.criterion = create_loss_function()
        
        # Use more conservative optimizer settings
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5,
            eps=1e-8,  # Prevent division by zero
            betas=(0.9, 0.999)  # Standard values
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
        """Initialize model weights more carefully to prevent NaN"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Xavier normal initialization for linear layers
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # Kaiming normal for conv layers (better for ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                # Careful LSTM initialization
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1 (common practice)
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)
        logger.info("Model weights initialized")
    
    def check_gradients(self):
        """Check for NaN or inf gradients"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.warning(f"NaN/Inf gradient detected in {name}")
                    param.grad.zero_()  # Zero out bad gradients
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        loss_components = {'se3': 0.0, 'consistency': 0.0, 'photometric': 0.0, 'huber': 0.0}
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(self.device)
                imu = batch['imu'].to(self.device)
                targets = batch['pose'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(images, imu)
                
                # Compute loss
                loss, loss_dict = self.criterion(predictions, targets, images)
                
                # Check for NaN loss before backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Check gradients before optimization
                self.check_gradients()
                
                # Gradient clipping with more conservative value
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                for key in loss_components:
                    if key in loss_dict and isinstance(loss_dict[key], torch.Tensor):
                        loss_components[key] += loss_dict[key].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'se3': f'{loss_dict.get("se3", 0):.6f}' if isinstance(loss_dict.get("se3"), torch.Tensor) else '0.0'
                })
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return avg_loss, loss_components
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        loss_components = {'se3': 0.0, 'consistency': 0.0, 'photometric': 0.0, 'huber': 0.0}
        num_batches = len(self.val_loader)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation") as pbar:
                for batch in pbar:
                    images = batch['images'].to(self.device)
                    imu = batch['imu'].to(self.device)
                    targets = batch['pose'].to(self.device)
                    
                    # Forward pass
                    predictions = self.model(images, imu)
                    
                    # Compute loss
                    loss, loss_dict = self.criterion(predictions, targets, images)
                    
                    total_loss += loss.item()
                    for key in loss_components:
                        if key in loss_dict and isinstance(loss_dict[key], torch.Tensor):
                            loss_components[key] += loss_dict[key].item()
                    
                    # Store predictions and targets for analysis
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
                    
                    pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        # Analyze predictions to detect constant output issue
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        pred_std = torch.std(all_predictions, dim=(0, 1))
        target_std = torch.std(all_targets, dim=(0, 1))
        
        logger.info(f"Prediction std: {pred_std.numpy()}")
        logger.info(f"Target std: {target_std.numpy()}")
        
        # Warning if predictions have very low variance (constant prediction issue)
        if torch.any(pred_std < 0.01):
            logger.warning("Low prediction variance detected - possible constant prediction issue!")
        
        return avg_loss, loss_components, all_predictions, all_targets
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pt')
            logger.info(f"Saved best model at epoch {epoch}")
    
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        if len(self.train_losses) > 1:
            plt.plot(self.train_losses[1:], label='Train (from epoch 2)')
            plt.plot(self.val_losses[1:], label='Validation (from epoch 2)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss (excluding first epoch)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        logger.info("Starting training...")
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
            
            # Training
            train_loss, train_components = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_components, predictions, targets = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            logger.info(f"Train Loss: {train_loss:.6f}")
            logger.info(f"Val Loss: {val_loss:.6f}")
            logger.info(f"Train Components: {train_components}")
            logger.info(f"Val Components: {val_components}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Plot losses every 5 epochs
            if epoch % 5 == 0:
                self.plot_losses()
        
        logger.info(f"Training completed! Best validation loss: {self.best_val_loss:.6f}")
        return self.model


def main():
    # Configuration
    config = {
        'csv_path': r'data\processed\visual_odometry_dataset\visual_odometry_dataset_kalibr_clean.csv',
        'data_root': r'data\processed\visual_odometry_dataset',
        'sequence_length': 10,
        'batch_size': 4,  # Reduced for memory efficiency
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'validation_split': 0.2,
        'save_dir': 'checkpoints/underwater_vio'
    }
    
    # Create trainer and start training
    trainer = VIOTrainer(**config)
    trained_model = trainer.train()
    
    # Save final configuration
    with open(Path(config['save_dir']) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()