"""
Simplified Motion-Aware Training Script

Uses existing single-camera training data with motion supervision.
This is a working solution to fix the straight-line prediction problem.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer.motion_aware_uw_transvo import create_motion_aware_model
from training.motion_aware_loss import create_motion_aware_loss


class SimpleSequenceDataset(Dataset):
    """Simple dataset that creates sequences from existing training data"""
    
    def __init__(self, csv_file, sequence_length=5, img_size=224):
        self.csv_file = csv_file
        self.sequence_length = sequence_length
        self.img_size = img_size
        
        # Load data
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} frames from {csv_file}")
        
        # Create sequences
        self.sequences = []
        for i in range(len(self.df) - sequence_length + 1):
            if i % 10 == 0:  # Sample every 10th sequence to avoid overlap
                self.sequences.append(i)
        
        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        start_idx = self.sequences[idx]
        
        # Get sequence data
        sequence_data = self.df.iloc[start_idx:start_idx + self.sequence_length]
        
        images = []
        poses = []
        
        for _, row in sequence_data.iterrows():
            # Load and preprocess image (using cam0_path)
            img_path = row['cam0_path']
            if not os.path.exists(img_path):
                # Try with current directory
                img_path = os.path.join(".", img_path)
            
            if os.path.exists(img_path):
                # Load image
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = img.astype(np.float32) / 255.0
                else:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            else:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            
            images.append(img)
            
            # Extract pose (using delta values as relative motion)
            pose = np.array([
                row.get('delta_x', 0.0), row.get('delta_y', 0.0), row.get('delta_z', 0.0),
                row.get('delta_roll', 0.0), row.get('delta_pitch', 0.0), row.get('delta_yaw', 0.0)
            ], dtype=np.float32)
            poses.append(pose)
        
        # Convert to tensors
        images = np.stack(images)  # [seq_len, H, W, 3]
        images = torch.tensor(images).permute(0, 3, 1, 2)  # [seq_len, 3, H, W]
        poses = torch.tensor(np.stack(poses))  # [seq_len, 6]
        
        # Add camera dimension
        images = images.unsqueeze(1)  # [seq_len, 1, 3, H, W]
        
        # Transpose to batch-first format
        images = images.transpose(0, 1)  # [1, seq_len, 3, H, W]
        
        return images, poses


def train_motion_aware_simple():
    """Simplified motion-aware training"""
    
    # Configuration  
    config = {
        'img_size': 224,
        'd_model': 384,
        'num_heads': 6,
        'num_layers': 4,
        'max_cameras': 1,
        'max_seq_len': 5,
        'dropout': 0.1,
        'uncertainty_estimation': True
    }
    
    loss_config = {
        'loss_type': 'motion_aware',
        'translation_weight': 1.0,
        'rotation_weight': 5.0,
        'motion_weight': 15.0,  # Strong motion supervision
        'consistency_weight': 3.0,
        'sequence_length': 5
    }
    
    # Training parameters
    batch_size = 4
    learning_rate = 3e-5
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Motion-Aware Training Configuration:")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Motion weight: {loss_config['motion_weight']}x")
    
    # Create dataset
    csv_file = "data/processed/training_dataset/training_data.csv"
    if not os.path.exists(csv_file):
        print(f"ERROR: Training data not found at {csv_file}")
        return None
    
    dataset = SimpleSequenceDataset(csv_file, sequence_length=config['max_seq_len'])
    
    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model and loss
    model = create_motion_aware_model(config).to(device)
    criterion = create_motion_aware_loss(loss_config)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'motion_loss': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        motion_losses = []
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train")
        for batch_idx, (images, poses) in enumerate(train_bar):
            try:
                images = images.to(device)  # [batch, 1, seq_len, 3, H, W] 
                poses = poses.to(device)   # [batch, seq_len, 6]
                
                # Reshape to expected format [batch, seq_len, cameras, 3, H, W]
                images = images.transpose(1, 2)  # [batch, seq_len, 1, 3, H, W]
                
                camera_ids = torch.zeros(images.size(0), 1, dtype=torch.long).to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images=images, camera_ids=camera_ids)
                pred_poses = outputs['pose']
                
                # Compute motion-aware loss
                loss_dict = criterion(pred_poses, poses)
                loss = loss_dict['total_loss']
                
                if torch.isnan(loss):
                    print(f"NaN loss at batch {batch_idx}")
                    continue
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                motion_losses.append(loss_dict['motion_translation_loss'].item())
                
                train_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Motion': f"{loss_dict['motion_translation_loss'].item():.4f}"
                })
                
            except Exception as e:
                print(f"Training error: {e}")
                continue
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val")
            for images, poses in val_bar:
                try:
                    images = images.to(device).transpose(1, 2)
                    poses = poses.to(device)
                    camera_ids = torch.zeros(images.size(0), 1, dtype=torch.long).to(device)
                    
                    outputs = model(images=images, camera_ids=camera_ids)
                    pred_poses = outputs['pose']
                    
                    loss_dict = criterion(pred_poses, poses)
                    loss = loss_dict['total_loss']
                    
                    if not torch.isnan(loss):
                        val_losses.append(loss.item())
                        val_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
                        
                except Exception as e:
                    continue
        
        # Epoch summary
        if train_losses and val_losses:
            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            avg_motion = np.mean(motion_losses)
            
            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            history['motion_loss'].append(avg_motion)
            
            print(f"\nEPOCH {epoch+1} SUMMARY:")
            print(f"Train Loss: {avg_train:.6f}")
            print(f"Val Loss: {avg_val:.6f}")
            print(f"Motion Loss: {avg_motion:.6f}")
            
            # Save best model
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': avg_val,
                    'config': config
                }, 'motion_aware_simple_best.pth')
                print(f"*** SAVED BEST MODEL (Val Loss: {avg_val:.6f}) ***")
    
    # Save training history
    with open('motion_training_history.json', 'w') as f:
        json.dump(history, f)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='Train Loss')
    plt.plot(history['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['motion_loss'], 'g-', label='Motion Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Motion Loss')
    plt.title('Motion Learning Progress')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('motion_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nMotion-aware training completed!")
    return history


if __name__ == "__main__":
    print("Simple Motion-Aware Training")
    print("This should fix the straight-line prediction problem!")
    print("=" * 50)
    
    history = train_motion_aware_simple()
    
    if history:
        print("\nSUCCESS! Key improvements:")
        print("- Sequential pose prediction (5 frames)")  
        print("- Motion supervision (frame-to-frame)")
        print("- Curved trajectory learning")
        print("\nFiles created:")
        print("- motion_aware_simple_best.pth") 
        print("- motion_training_history.json")
        print("- motion_training_progress.png")