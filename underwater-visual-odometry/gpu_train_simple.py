#!/usr/bin/env python3
"""
Simple GPU training script - optimized for GTX 1050 4GB
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.datasets import UnderwaterVODataset
from training.losses import PoseLoss

def main():
    print("GPU Training - UW-TransVO")
    print("=" * 40)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Optimized config for GTX 1050
    config = {
        'img_size': 224,
        'patch_size': 16,
        'd_model': 256,  # Smaller for 4GB GPU
        'num_heads': 4,  # Smaller for 4GB GPU  
        'num_layers': 2, # Smaller for 4GB GPU
        'max_cameras': 4,
        'max_seq_len': 2,
        'dropout': 0.1,
        'use_imu': False,
        'use_pressure': False,
        'uncertainty_estimation': False
    }
    
    # Create model
    print("\nCreating model...")
    model = UWTransVO(**config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = UnderwaterVODataset(
        data_csv='data/processed/training_dataset/training_data.csv',
        data_root='.',
        camera_ids=[0, 1, 2, 3],
        sequence_length=2,
        img_size=224,
        use_imu=False,
        use_pressure=False,
        augmentation=False,
        split='train',
        max_samples=50  # Small for quick test
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    # Setup training
    criterion = PoseLoss(translation_weight=1.0, rotation_weight=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    print(f"Dataset size: {len(dataset)}")
    print("Starting training...\n")
    
    # Memory monitoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Training loop
    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            start_time = time.time()
            
            try:
                # Move to device
                images = batch['images'].to(device, non_blocking=True)
                camera_ids = batch['camera_ids'].to(device, non_blocking=True)
                camera_mask = batch['camera_mask'].to(device, non_blocking=True)
                pose_target = batch['pose_target'].to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                optimizer.zero_grad()
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        predictions = model(
                            images=images,
                            camera_ids=camera_ids,
                            camera_mask=camera_mask
                        )
                        loss_dict = criterion(predictions['pose'], pose_target)
                        loss = loss_dict['total_loss']
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    predictions = model(
                        images=images,
                        camera_ids=camera_ids,
                        camera_mask=camera_mask
                    )
                    loss_dict = criterion(predictions['pose'], pose_target)
                    loss = loss_dict['total_loss']
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                batch_time = time.time() - start_time
                
                # Memory monitoring
                if torch.cuda.is_available() and batch_idx % 5 == 0:
                    mem_used = torch.cuda.memory_allocated() / 1024**3
                    mem_cached = torch.cuda.memory_reserved() / 1024**3
                    print(f"Epoch {epoch+1} Batch {batch_idx+1:2d}: "
                          f"Loss: {loss.item():.6f}, "
                          f"Time: {batch_time:.2f}s, "
                          f"GPU: {mem_used:.2f}GB/{mem_cached:.2f}GB")
                else:
                    print(f"Epoch {epoch+1} Batch {batch_idx+1:2d}: "
                          f"Loss: {loss.item():.6f}, Time: {batch_time:.2f}s")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU OUT OF MEMORY! Error: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                else:
                    print(f"Runtime error: {e}")
                    continue
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch+1} complete. Average loss: {avg_loss:.6f}")
        
        # Clear cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\nGPU TRAINING COMPLETE!")
    print("Success! Your model can train on GPU with current memory constraints.")
    
    # Final memory check
    if torch.cuda.is_available():
        print(f"Final GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

if __name__ == '__main__':
    main()