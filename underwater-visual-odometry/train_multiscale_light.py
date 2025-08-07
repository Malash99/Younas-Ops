"""
Memory-Optimized Multi-Scale Training for 4GB GPU

Reduced memory footprint version that still implements multi-scale supervision
but with smaller sequences and model size for 4GB GPU constraints.
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

from models.transformer.multiscale_uw_transvo import create_multiscale_model
from training.multiscale_loss import create_multiscale_loss, analyze_multiscale_predictions


class LightMultiScaleDataset(Dataset):
    """Memory-optimized dataset for 4GB GPU"""
    
    def __init__(self, csv_file, sequence_length=10, img_size=192, stride=8):
        self.csv_file = csv_file
        self.sequence_length = sequence_length
        self.img_size = img_size  # Smaller images
        self.stride = stride
        
        # Load data
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} frames from {csv_file}")
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.df) - sequence_length + 1, stride):
            self.sequences.append(i)
        
        print(f"Created {len(self.sequences)} sequences of length {sequence_length} (stride={stride})")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        start_idx = self.sequences[idx]
        sequence_data = self.df.iloc[start_idx:start_idx + self.sequence_length]
        
        images = []
        delta_poses = []
        
        for _, row in sequence_data.iterrows():
            # Handle NaN values in image paths
            img_path = row['cam0_path']
            if pd.isna(img_path):
                # Create dummy image if path is NaN
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
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
                            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                    except:
                        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                else:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            
            images.append(img)
            
            # Extract delta pose
            delta_pose = np.array([
                float(row.get('delta_x', 0.0)), float(row.get('delta_y', 0.0)), float(row.get('delta_z', 0.0)),
                float(row.get('delta_roll', 0.0)), float(row.get('delta_pitch', 0.0)), float(row.get('delta_yaw', 0.0))
            ], dtype=np.float32)
            delta_poses.append(delta_pose)
        
        # Convert to tensors
        images = np.stack(images)
        images = torch.tensor(images).permute(0, 3, 1, 2)  # [seq_len, 3, H, W]
        delta_poses = torch.tensor(np.stack(delta_poses))
        
        # Add camera dimension
        images = images.unsqueeze(1).transpose(0, 1)  # [1, seq_len, 3, H, W]
        
        return images, delta_poses


def train_multiscale_light():
    """Memory-optimized multi-scale training for 4GB GPU"""
    
    # Reduced configuration for 4GB GPU
    config = {
        'img_size': 192,        # Smaller images (192 vs 224)
        'd_model': 256,         # Smaller model (256 vs 512)
        'num_heads': 4,         # Fewer heads (4 vs 8)
        'num_layers': 3,        # Fewer layers (3 vs 6)
        'max_cameras': 1,
        'max_seq_len': 10,      # Shorter sequences (10 vs 20)
        'dropout': 0.1,
        'uncertainty_estimation': False  # Disable to save memory
    }
    
    loss_config = {
        'loss_type': 'multiscale',  # Standard version (no curriculum)
        'delta_weight': 1.0,
        'short_weight': 3.0,        # 5-frame windows
        'long_weight': 8.0,         # 10-frame total
        'magnitude_weight': 4.0,    # Strong anti-straight-line
        'smoothness_weight': 1.5,
        'translation_weight': 1.0,
        'rotation_weight': 3.0
    }
    
    # Training parameters optimized for memory
    batch_size = 2          # Very small batch
    learning_rate = 3e-5    # Conservative
    num_epochs = 12
    sequence_length = 10    # Shorter sequences
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("MEMORY-OPTIMIZED MULTI-SCALE TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"GPU Memory Optimizations:")
    print(f"  - Image size: 192x192 (vs 224x224)")
    print(f"  - Model size: 256d (vs 512d)")
    print(f"  - Sequence length: {sequence_length} (vs 20)")
    print(f"  - Batch size: {batch_size} (vs 3)")
    print(f"Multi-scale supervision: 1-frame, 5-frame, {sequence_length}-frame")
    print("=" * 60)
    
    # Create dataset
    csv_file = "data/processed/training_dataset/training_data.csv"
    if not os.path.exists(csv_file):
        print(f"ERROR: Training data not found at {csv_file}")
        return None
    
    dataset = LightMultiScaleDataset(
        csv_file, 
        sequence_length=sequence_length,
        img_size=config['img_size'],
        stride=6  # Good overlap
    )
    
    # Train/val split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model and loss
    model = create_multiscale_model(config).to(device)
    criterion = create_multiscale_loss(loss_config)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Enable memory optimization
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False  # Save memory
        torch.backends.cudnn.deterministic = True
    
    # Training tracking
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [], 'val_loss': [],
        'delta_loss': [], 'short_loss': [], 'long_loss': [],
        'magnitude_loss': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Training phase
        model.train()
        train_losses = []
        train_components = {'delta': [], 'short': [], 'long': [], 'magnitude': []}
        
        train_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, delta_poses) in enumerate(train_bar):
            try:
                # Clear cache periodically
                if batch_idx % 50 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                images = images.to(device, non_blocking=True)
                delta_poses = delta_poses.to(device, non_blocking=True)
                
                # Reshape: [batch, seq_len, cameras, 3, H, W]
                images = images.transpose(1, 2)
                
                camera_ids = torch.zeros(images.size(0), 1, dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images=images, camera_ids=camera_ids)
                pred_deltas = outputs['delta_poses']
                
                # Multi-scale loss
                loss_dict = criterion(pred_deltas, delta_poses)
                loss = loss_dict['total_loss']
                
                if torch.isnan(loss):
                    print(f"NaN loss at batch {batch_idx}")
                    continue
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track losses
                train_losses.append(loss.item())
                train_components['delta'].append(loss_dict['delta_loss'].item())
                train_components['short'].append(loss_dict['short_loss'].item()) 
                train_components['long'].append(loss_dict['long_loss'].item())
                train_components['magnitude'].append(loss_dict['magnitude_loss'].item())
                
                # Update progress
                train_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Long': f"{loss_dict['long_loss'].item():.4f}",
                    'Mag': f"{loss_dict['magnitude_loss'].item():.4f}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch {batch_idx}, clearing cache...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Training error: {e}")
                    continue
        
        # Validation phase
        model.eval()
        val_losses = []
        val_components = {'delta': [], 'short': [], 'long': [], 'magnitude': []}
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            
            for images, delta_poses in val_bar:
                try:
                    images = images.to(device, non_blocking=True).transpose(1, 2)
                    delta_poses = delta_poses.to(device, non_blocking=True)
                    camera_ids = torch.zeros(images.size(0), 1, dtype=torch.long, device=device)
                    
                    outputs = model(images=images, camera_ids=camera_ids)
                    pred_deltas = outputs['delta_poses']
                    
                    loss_dict = criterion(pred_deltas, delta_poses)
                    loss = loss_dict['total_loss']
                    
                    if not torch.isnan(loss):
                        val_losses.append(loss.item())
                        val_components['delta'].append(loss_dict['delta_loss'].item())
                        val_components['short'].append(loss_dict['short_loss'].item())
                        val_components['long'].append(loss_dict['long_loss'].item())
                        val_components['magnitude'].append(loss_dict['magnitude_loss'].item())
                        
                        val_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
                        
                except RuntimeError as e:
                    if "out of memory" in str(e) and device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
        
        # Update learning rate
        scheduler.step()
        
        # Epoch summary
        if train_losses and val_losses:
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            avg_train_comp = {k: np.mean(v) for k, v in train_components.items()}
            avg_val_comp = {k: np.mean(v) for k, v in val_components.items()}
            
            # Save history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['delta_loss'].append(avg_val_comp['delta'])
            training_history['short_loss'].append(avg_val_comp['short'])
            training_history['long_loss'].append(avg_val_comp['long'])
            training_history['magnitude_loss'].append(avg_val_comp['magnitude'])
            
            print(f"TRAIN | Total: {avg_train_loss:.6f}")
            print(f"      | Delta: {avg_train_comp['delta']:.6f} | Short: {avg_train_comp['short']:.6f}")
            print(f"      | Long:  {avg_train_comp['long']:.6f} | Mag: {avg_train_comp['magnitude']:.6f}")
            print(f"VAL   | Total: {avg_val_loss:.6f}")
            print(f"      | Delta: {avg_val_comp['delta']:.6f} | Short: {avg_val_comp['short']:.6f}")
            print(f"      | Long:  {avg_val_comp['long']:.6f} | Mag: {avg_val_comp['magnitude']:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': config,
                    'loss_config': loss_config
                }, 'multiscale_light_best_model.pth')
                print(f"*** BEST MODEL SAVED! Val Loss: {avg_val_loss:.6f} ***")
    
    # Save results
    with open('multiscale_light_history.json', 'w') as f:
        history_clean = {}
        for key, value in training_history.items():
            history_clean[key] = [float(x) for x in value]
        json.dump(history_clean, f, indent=2)
    
    # Quick plot
    if len(training_history['train_loss']) > 0:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_history['train_loss'], 'b-', label='Train')
        plt.plot(training_history['val_loss'], 'r-', label='Val')
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(training_history['long_loss'], 'r-', label='Long-term (Anti-Straight)')
        plt.plot(training_history['magnitude_loss'], 'm-', label='Magnitude')
        plt.title('Key Anti-Straight-Line Losses')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('multiscale_light_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\n" + "="*60)
    print("MEMORY-OPTIMIZED MULTI-SCALE TRAINING COMPLETED!")
    print("="*60)
    print("Model trained with multi-scale supervision:")
    print("- Frame-to-frame deltas (local consistency)")
    print("- 5-frame windows (short-term shape)")  
    print("- 10-frame segments (long-term patterns)")
    print("- Strong magnitude loss (anti-straight-line)")
    print("\nFiles created:")
    print("- multiscale_light_best_model.pth")
    print("- multiscale_light_history.json")
    print("- multiscale_light_progress.png")
    
    return training_history


if __name__ == "__main__":
    print("Memory-Optimized Multi-Scale UW-TransVO Training")
    print("Designed for 4GB GPU with anti-straight-line supervision")
    
    history = train_multiscale_light()
    
    if history:
        print("\nSUCCESS! The model should now predict curved trajectories.")
        print("Test it with real trajectory data to verify improvements.")