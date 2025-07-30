#!/usr/bin/env python3
"""
Training script for UW-TransVO

Quick start script to train the model with specified configuration.
Use this for single experiments or testing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO, create_uw_transvo_model
from data.datasets import UnderwaterVODataset, create_dataloaders
from training.trainer import UWTransVOTrainer
from training.losses import create_loss_function


def create_default_config():
    """Create default configuration for quick start"""
    return {
        'experiment_name': 'quad_seq2_vision',
        'model': {
            'img_size': 224,
            'patch_size': 16,
            'd_model': 384,  # Optimized for GTX 1050 4GB
            'num_heads': 6,  # Optimized for GTX 1050 4GB
            'num_layers': 4, # Optimized for GTX 1050 4GB
            'max_cameras': 4,
            'max_seq_len': 2,
            'dropout': 0.1,
            'use_imu': False,
            'use_pressure': False,
            'uncertainty_estimation': True
        },
        'data': {
            'data_csv': 'data/processed/training_dataset/training_data.csv',
            'data_root': '.',
            'camera_ids': [0, 1, 2, 3],
            'sequence_length': 2,
            'img_size': 224,
            'use_imu': False,
            'use_pressure': False,
            'augmentation': True,
            'max_samples': None  # Set to small number for testing, None for full dataset
        },
        'training': {
            'epochs': 50,
            'batch_size': 4,  # Optimized for GTX 1050 4GB
            'mixed_precision': True,
            'gradient_accumulation_steps': 4,  # Effective batch size = 4 * 4 = 16
            'log_interval': 10
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'betas': [0.9, 0.999]
        },
        'scheduler': {
            'type': 'cosine',
            'min_lr': 1e-6
        },
        'loss': {
            'loss_type': 'pose',
            'translation_weight': 1.0,
            'rotation_weight': 10.0,
            'base_loss': 'mse'
        }
    }


def setup_data_splits(data_csv: str, data_root: str):
    """
    Create train/val/test splits based on bag names
    
    Args:
        data_csv: Path to the main data CSV
        data_root: Root directory for data
        
    Returns:
        Tuple of (train_config, val_config, test_config)
    """
    
    # For now, use the same CSV and rely on bag-based splitting
    # In production, you'd create separate CSV files for each split
    
    train_config = {
        'data_csv': data_csv,
        'data_root': data_root,
        'split': 'train'
    }
    
    val_config = {
        'data_csv': data_csv,
        'data_root': data_root, 
        'split': 'val'
    }
    
    test_config = {
        'data_csv': data_csv,
        'data_root': data_root,
        'split': 'test'
    }
    
    return train_config, val_config, test_config


def main():
    parser = argparse.ArgumentParser(description='Train UW-TransVO model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_csv', type=str, 
                       default='data/processed/training_dataset/training_data.csv',
                       help='Path to training data CSV')
    parser.add_argument('--data_root', type=str,
                       default='data/processed/training_dataset', 
                       help='Root directory for data')
    parser.add_argument('--cameras', type=str, default='0,1,2,3',
                       help='Comma-separated camera IDs')
    parser.add_argument('--sequence_length', type=int, default=2,
                       help='Sequence length (2 or 4)')
    parser.add_argument('--use_imu', action='store_true',
                       help='Use IMU data')
    parser.add_argument('--use_pressure', action='store_true', 
                       help='Use pressure data')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, 
                       help='Resume from checkpoint')
    parser.add_argument('--test_run', action='store_true',
                       help='Run with small dataset for testing')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
        # Override with command line arguments
        camera_ids = [int(x) for x in args.cameras.split(',')]
        config['data'].update({
            'data_csv': args.data_csv,
            'data_root': args.data_root,
            'camera_ids': camera_ids,
            'sequence_length': args.sequence_length,
            'use_imu': args.use_imu,
            'use_pressure': args.use_pressure
        })
        
        config['model'].update({
            'max_cameras': len(camera_ids),
            'max_seq_len': args.sequence_length,
            'use_imu': args.use_imu,
            'use_pressure': args.use_pressure
        })
        
        config['training'].update({
            'batch_size': args.batch_size,
            'epochs': args.epochs
        })
        
        config['optimizer']['lr'] = args.lr
        
        # Update experiment name
        cam_str = 'mono' if len(camera_ids) == 1 else f'{len(camera_ids)}cam'
        modality_str = 'vision'
        if args.use_imu:
            modality_str += '_imu'
        if args.use_pressure:
            modality_str += '_pressure'
        
        config['experiment_name'] = f"{cam_str}_seq{args.sequence_length}_{modality_str}"
    
    # Test run with small dataset
    if args.test_run:
        config['data']['max_samples'] = 100
        config['training']['epochs'] = 3
        config['training']['log_interval'] = 2
        config['experiment_name'] = f"test_{config['experiment_name']}"
        print("TEST: Running test mode with small dataset")
    
    print(f"Starting experiment: {config['experiment_name']}")
    print(f"Cameras: {config['data']['camera_ids']}")
    print(f"Sequence length: {config['data']['sequence_length']}")
    print(f"Modalities: Vision{'+ IMU' if config['model']['use_imu'] else ''}{'+ Pressure' if config['model']['use_pressure'] else ''}")
    
    # Create model
    print("Creating model...")
    model = create_uw_transvo_model(config['model'])
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create datasets and dataloaders
    print("Loading datasets...")
    
    # Setup data configurations
    base_data_config = config['data'].copy()
    
    train_config = {**base_data_config, 'split': 'train', 'augmentation': True}
    val_config = {**base_data_config, 'split': 'val', 'augmentation': False}
    test_config = {**base_data_config, 'split': 'test', 'augmentation': False}
    
    # For now, create simple datasets
    # TODO: Implement proper train/val/test splitting
    train_dataset = UnderwaterVODataset(**train_config)
    val_dataset = UnderwaterVODataset(**val_config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create trainer
    print("Initializing trainer...")
    trainer = UWTransVOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=config['experiment_name']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save current state
        trainer._save_checkpoint(
            trainer.epoch, 
            {'total_loss': 0.0}, 
            {'total_loss': trainer.best_val_loss},
            is_best=False
        )
        print("Checkpoint saved")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()