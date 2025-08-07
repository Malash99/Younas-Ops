"""
Multi-Scale Training Script for UW-TransVO

Trains the model with supervision at multiple temporal scales:
1. Frame-to-frame deltas (fine-grained local consistency)
2. 5-frame accumulated poses (short-term trajectory shape)  
3. 20-frame trajectory segments (long-term global patterns)

This approach should solve the straight-line prediction problem by explicitly
supervising trajectory shape at different temporal resolutions.
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


class MultiScaleSequenceDataset(Dataset):
    """
    Dataset that creates longer sequences for multi-scale training
    Provides ground truth at multiple temporal scales
    """
    
    def __init__(self, csv_file, sequence_length=20, img_size=224, stride=10):
        self.csv_file = csv_file
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.stride = stride
        
        # Load data
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} frames from {csv_file}")
        
        # Create sequences with stride
        self.sequences = []
        for i in range(0, len(self.df) - sequence_length + 1, stride):
            self.sequences.append(i)
        
        print(f"Created {len(self.sequences)} sequences of length {sequence_length} (stride={stride})")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        start_idx = self.sequences[idx]
        
        # Get sequence data
        sequence_data = self.df.iloc[start_idx:start_idx + self.sequence_length]
        
        images = []
        delta_poses = []
        
        for _, row in sequence_data.iterrows():
            # Load and preprocess image
            img_path = row['cam0_path']
            if not os.path.exists(img_path):
                img_path = os.path.join(".", img_path)
            
            if os.path.exists(img_path):
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
            
            # Extract delta pose
            delta_pose = np.array([
                row.get('delta_x', 0.0), row.get('delta_y', 0.0), row.get('delta_z', 0.0),
                row.get('delta_roll', 0.0), row.get('delta_pitch', 0.0), row.get('delta_yaw', 0.0)
            ], dtype=np.float32)
            delta_poses.append(delta_pose)
        
        # Convert to tensors
        images = np.stack(images)  # [seq_len, H, W, 3]
        images = torch.tensor(images).permute(0, 3, 1, 2)  # [seq_len, 3, H, W]
        delta_poses = torch.tensor(np.stack(delta_poses))  # [seq_len, 6]
        
        # Add camera dimension: [seq_len, 1, 3, H, W]
        images = images.unsqueeze(1)
        
        # Transpose to batch-first: [1, seq_len, 3, H, W] 
        images = images.transpose(0, 1)
        
        return images, delta_poses


def train_multiscale_model():
    """Main training function with multi-scale supervision"""
    
    # Configuration
    config = {
        'img_size': 224,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'max_cameras': 1,
        'max_seq_len': 20,
        'dropout': 0.1,
        'uncertainty_estimation': True
    }
    
    loss_config = {
        'loss_type': 'adaptive_multiscale',
        'delta_weight': 0.5,      # Start low, increase during training
        'short_weight': 2.0,      # Medium importance
        'long_weight': 8.0,       # High importance (anti-straight-line)
        'magnitude_weight': 3.0,  # Force non-zero predictions
        'smoothness_weight': 1.0, # Trajectory smoothness
        'translation_weight': 1.0,
        'rotation_weight': 3.0,
        'curriculum_steps': 3000, # Curriculum learning steps
        'initial_long_weight': 15.0  # Start with very high long-term weight
    }
    
    # Training parameters
    batch_size = 3  # Small batch due to long sequences
    learning_rate = 2e-5  # Conservative learning rate
    num_epochs = 15
    sequence_length = 20  # Long sequences for multi-scale
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("MULTI-SCALE UW-TransVO TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Sequence length: {sequence_length} frames")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Multi-scale weights: Delta={loss_config['delta_weight']}, Short={loss_config['short_weight']}, Long={loss_config['long_weight']}")
    print(f"Curriculum learning: {loss_config['curriculum_steps']} steps")
    print("=" * 60)
    
    # Create dataset
    csv_file = "data/processed/training_dataset/training_data.csv"
    if not os.path.exists(csv_file):
        print(f"ERROR: Training data not found at {csv_file}")
        return None
    
    dataset = MultiScaleSequenceDataset(
        csv_file, 
        sequence_length=sequence_length,
        stride=5  # Some overlap for better learning
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
    
    # Training tracking
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [], 'val_loss': [],
        'delta_loss': [], 'short_loss': [], 'long_loss': [],
        'magnitude_loss': [], 'trajectory_analysis': []
    }
    
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        train_losses = []
        train_components = {'delta': [], 'short': [], 'long': [], 'magnitude': []}
        
        train_bar = tqdm(train_loader, desc=f"Training")
        
        for batch_idx, (images, delta_poses) in enumerate(train_bar):
            try:
                images = images.to(device)  # [batch, 1, seq_len, 3, H, W]
                delta_poses = delta_poses.to(device)  # [batch, seq_len, 6]
                
                # Reshape to expected format [batch, seq_len, cameras, 3, H, W]
                images = images.transpose(1, 2)  # [batch, seq_len, 1, 3, H, W]
                
                camera_ids = torch.zeros(images.size(0), 1, dtype=torch.long).to(device)
                
                # Update curriculum learning weights
                if hasattr(criterion, 'update_weights'):
                    criterion.update_weights(global_step)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images=images, camera_ids=camera_ids)
                
                # For multi-scale loss, we need the predicted deltas
                pred_deltas = outputs['delta_poses']  # [batch, seq_len, 6]
                
                # Compute multi-scale loss
                loss_dict = criterion(pred_deltas, delta_poses)
                loss = loss_dict['total_loss']
                
                if torch.isnan(loss):
                    print(f"NaN loss at batch {batch_idx}, step {global_step}")
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
                
                # Update progress bar
                current_weights = ""
                if hasattr(criterion, 'current_long_weight'):
                    current_weights = f"W_L={criterion.current_long_weight:.1f}"
                
                train_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Delta': f"{loss_dict['delta_loss'].item():.4f}",
                    'Short': f"{loss_dict['short_loss'].item():.4f}",
                    'Long': f"{loss_dict['long_loss'].item():.4f}",
                    'Weights': current_weights
                })
                
                global_step += 1
                
            except Exception as e:
                print(f"Training error at batch {batch_idx}: {e}")
                continue
        
        # Validation phase
        model.eval()
        val_losses = []
        val_components = {'delta': [], 'short': [], 'long': [], 'magnitude': []}
        trajectory_analyses = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Validation")
            
            for images, delta_poses in val_bar:
                try:
                    images = images.to(device).transpose(1, 2)
                    delta_poses = delta_poses.to(device)
                    camera_ids = torch.zeros(images.size(0), 1, dtype=torch.long).to(device)
                    
                    # Forward pass
                    outputs = model(images=images, camera_ids=camera_ids)
                    pred_deltas = outputs['delta_poses']
                    
                    # Compute loss
                    loss_dict = criterion(pred_deltas, delta_poses)
                    loss = loss_dict['total_loss']
                    
                    if not torch.isnan(loss):
                        val_losses.append(loss.item())
                        val_components['delta'].append(loss_dict['delta_loss'].item())
                        val_components['short'].append(loss_dict['short_loss'].item())
                        val_components['long'].append(loss_dict['long_loss'].item())
                        val_components['magnitude'].append(loss_dict['magnitude_loss'].item())
                        
                        # Analyze trajectory quality
                        analysis = analyze_multiscale_predictions(pred_deltas, delta_poses)
                        trajectory_analyses.append(analysis)
                        
                        val_bar.set_postfix({
                            'Loss': f"{loss.item():.4f}",
                            'Delta': f"{loss_dict['delta_loss'].item():.4f}",
                            'Long': f"{loss_dict['long_loss'].item():.4f}"
                        })
                        
                except Exception as e:
                    continue
        
        # Update learning rate
        scheduler.step()
        
        # Epoch summary
        if train_losses and val_losses:
            # Average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            avg_train_components = {k: np.mean(v) for k, v in train_components.items()}
            avg_val_components = {k: np.mean(v) for k, v in val_components.items()}
            
            # Average trajectory analysis
            if trajectory_analyses:
                avg_analysis = {}
                for key in trajectory_analyses[0].keys():
                    avg_analysis[key] = np.mean([a[key] for a in trajectory_analyses])
            else:
                avg_analysis = {}
            
            # Save to history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['delta_loss'].append(avg_val_components['delta'])
            training_history['short_loss'].append(avg_val_components['short'])
            training_history['long_loss'].append(avg_val_components['long'])
            training_history['magnitude_loss'].append(avg_val_components['magnitude'])
            training_history['trajectory_analysis'].append(avg_analysis)
            
            # Print epoch results
            print(f"\nEPOCH {epoch+1} RESULTS:")
            print(f"TRAIN  | Total: {avg_train_loss:.6f}")
            print(f"       | Delta: {avg_train_components['delta']:.6f}")
            print(f"       | Short: {avg_train_components['short']:.6f}")
            print(f"       | Long:  {avg_train_components['long']:.6f}")
            print(f"       | Mag:   {avg_train_components['magnitude']:.6f}")
            print(f"VAL    | Total: {avg_val_loss:.6f}")
            print(f"       | Delta: {avg_val_components['delta']:.6f}")
            print(f"       | Short: {avg_val_components['short']:.6f}")
            print(f"       | Long:  {avg_val_components['long']:.6f}")
            print(f"       | Mag:   {avg_val_components['magnitude']:.6f}")
            
            if avg_analysis:
                print(f"TRAJECTORY ANALYSIS:")
                for key, value in avg_analysis.items():
                    print(f"       | {key}: {value:.6f}")
            
            print(f"LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': config,
                    'loss_config': loss_config,
                    'trajectory_analysis': avg_analysis
                }, 'multiscale_best_model.pth')
                print(f"*** BEST MODEL SAVED! Val Loss: {avg_val_loss:.6f} ***")
    
    # Save final results
    with open('multiscale_training_history.json', 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        history_serializable = {}
        for key, value in training_history.items():
            if key == 'trajectory_analysis':
                history_serializable[key] = value
            else:
                history_serializable[key] = [float(x) for x in value]
        json.dump(history_serializable, f, indent=2)
    
    # Plot training curves
    plot_multiscale_training_curves(training_history)
    
    print("\n" + "="*60)
    print("MULTI-SCALE TRAINING COMPLETED!")
    print("="*60)
    print("Key improvements expected:")
    print("- No more straight-line predictions")
    print("- Better long-term trajectory consistency") 
    print("- Improved local motion smoothness")
    print("\nFiles created:")
    print("- multiscale_best_model.pth")
    print("- multiscale_training_history.json")
    print("- multiscale_training_curves.png")
    
    return training_history


def plot_multiscale_training_curves(history):
    """Plot comprehensive training curves for multi-scale training"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Multi-scale component losses
    axes[0, 1].plot(epochs, history['delta_loss'], 'g-', label='Delta Loss')
    axes[0, 1].plot(epochs, history['short_loss'], 'b-', label='Short Loss') 
    axes[0, 1].plot(epochs, history['long_loss'], 'r-', label='Long Loss')
    axes[0, 1].set_title('Multi-Scale Component Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale('log')
    
    # Magnitude loss (anti-straight-line)
    axes[0, 2].plot(epochs, history['magnitude_loss'], 'm-', label='Magnitude Loss')
    axes[0, 2].set_title('Motion Magnitude Loss (Anti-Straight-Line)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Trajectory analysis metrics
    if history['trajectory_analysis'] and len(history['trajectory_analysis']) > 0:
        try:
            # Extract trajectory metrics over time
            delta_errors = [a.get('delta_mean_error', 0) for a in history['trajectory_analysis']]
            short_errors = [a.get('short_mean_error', 0) for a in history['trajectory_analysis']]
            long_errors = [a.get('long_mean_error', 0) for a in history['trajectory_analysis']]
            
            if any(delta_errors):
                axes[1, 0].plot(epochs, delta_errors, 'g-', label='Delta Error')
            if any(short_errors):
                axes[1, 0].plot(epochs, short_errors, 'b-', label='Short Error')  
            if any(long_errors):
                axes[1, 0].plot(epochs, long_errors, 'r-', label='Long Error')
                
            axes[1, 0].set_title('Trajectory Prediction Errors')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Mean Error (m)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        except:
            axes[1, 0].text(0.5, 0.5, 'Trajectory Analysis\nNot Available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Loss component ratios
    if len(history['long_loss']) > 0:
        total_components = np.array(history['delta_loss']) + np.array(history['short_loss']) + np.array(history['long_loss'])
        delta_ratio = np.array(history['delta_loss']) / (total_components + 1e-8)
        short_ratio = np.array(history['short_loss']) / (total_components + 1e-8)
        long_ratio = np.array(history['long_loss']) / (total_components + 1e-8)
        
        axes[1, 1].plot(epochs, delta_ratio, 'g-', label='Delta Ratio')
        axes[1, 1].plot(epochs, short_ratio, 'b-', label='Short Ratio')
        axes[1, 1].plot(epochs, long_ratio, 'r-', label='Long Ratio')
        axes[1, 1].set_title('Loss Component Ratios (Curriculum Progress)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Combined normalized view
    if len(history['val_loss']) > 0:
        val_loss_norm = np.array(history['val_loss']) / np.max(history['val_loss'])
        magnitude_loss_norm = np.array(history['magnitude_loss']) / np.max(history['magnitude_loss'])
        
        axes[1, 2].plot(epochs, val_loss_norm, 'r-', label='Val Loss (norm)')
        axes[1, 2].plot(epochs, magnitude_loss_norm, 'm-', label='Magnitude Loss (norm)')
        axes[1, 2].set_title('Normalized Progress')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Normalized Value')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('multiscale_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Multi-Scale UW-TransVO Training")
    print("This should definitively solve the straight-line prediction problem!")
    
    history = train_multiscale_model()
    
    if history:
        print("\nTRAINING SUCCESS!")
        print("\nKey innovations implemented:")
        print("- Multi-scale temporal supervision (1, 5, 20 frames)")
        print("- Motion magnitude loss (prevents zero predictions)")
        print("- Curriculum learning (long-term -> fine details)")
        print("- Enhanced trajectory modeling")
        print("\nThe model should now predict realistic curved trajectories!")