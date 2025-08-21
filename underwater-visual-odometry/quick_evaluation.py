#!/usr/bin/env python3
"""
Quick evaluation of the latest trained model (3_TSFormer_seq3_advanced_loss)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from models.tsformer_vo import create_tsformer_vo
from datasets.underwater_vo_dataset import create_data_loaders

def load_and_check_model():
    """Load and check the latest model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    config = {
        'sequence_length': 3,
        'image_size': 224,
        'csv_path': 'data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv',
        'data_root': 'data/processed/visual_odometry_dataset',
        'overlap_frames': 1,
        'test_bags': ['ariel_2023-12-21-14-28-22_4']
    }
    
    # Load model
    model, loss_fn = create_tsformer_vo(
        sequence_length=config['sequence_length'],
        pretrained=True,
        freeze_backbone=False,
        image_size=config['image_size']
    )
    
    # Load checkpoint
    checkpoint_path = Path('experiments/3_TSFormer_seq3_advanced_loss/checkpoint_best.pth')
    
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print training info
        print(f"\n=== TRAINING RESULTS ===")
        print(f"Epochs trained: {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
        
        if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            
            print(f"Final training loss: {train_losses[-1]:.6f}")
            print(f"Final validation loss: {val_losses[-1]:.6f}")
            
            # Plot training curves
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(train_losses) + 1)
            plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('TSformer-VO Training Progress (Advanced Loss)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            # Save plot
            plt.savefig('training_progress_advanced_loss.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Training curves saved to: training_progress_advanced_loss.png")
            
        model = model.to(device)
        model.eval()
        
        return model, device, config, checkpoint
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None, None, None

def quick_test(model, device, config):
    """Quick test on a few samples"""
    print(f"\n=== QUICK MODEL TEST ===")
    
    # Create test data loader
    data_loaders = create_data_loaders(
        csv_path=config['csv_path'],
        data_root=config['data_root'],
        sequence_length=config['sequence_length'],
        overlap_frames=config['overlap_frames'],
        image_size=config['image_size'],
        batch_size=1,
        test_bags=config['test_bags'],
        num_workers=0,
        camera='cam0'
    )
    
    test_loader = data_loaders['test_loader']
    print(f"Test samples: {len(test_loader)}")
    
    # Test on first few batches
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:  # Test on first 10 samples
                break
                
            images = batch['images'].to(device)
            poses = batch['poses'].to(device)
            
            pred_poses = model(images)
            
            predictions.append(pred_poses.cpu().numpy())
            ground_truths.append(poses.cpu().numpy())
            
            if i < 3:  # Print first 3 predictions
                print(f"Sample {i+1}:")
                print(f"  Predicted: {pred_poses[0].cpu().numpy()}")
                print(f"  Ground Truth: {poses[0].cpu().numpy()}")
                print(f"  Error: {torch.abs(pred_poses - poses)[0].cpu().numpy()}")
    
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    
    # Compute simple metrics
    trans_errors = np.linalg.norm(predictions[:, :3] - ground_truths[:, :3], axis=1)
    rot_errors = np.linalg.norm(predictions[:, 3:] - ground_truths[:, 3:], axis=1)
    
    print(f"\n=== QUICK EVALUATION METRICS ===")
    print(f"Translation Error (10 samples):")
    print(f"  Mean: {np.mean(trans_errors):.4f} m")
    print(f"  Std:  {np.std(trans_errors):.4f} m")
    print(f"  Max:  {np.max(trans_errors):.4f} m")
    
    print(f"Rotation Error (10 samples):")
    print(f"  Mean: {np.mean(rot_errors):.4f} rad ({np.degrees(np.mean(rot_errors)):.1f}°)")
    print(f"  Std:  {np.std(rot_errors):.4f} rad ({np.degrees(np.std(rot_errors)):.1f}°)")
    print(f"  Max:  {np.max(rot_errors):.4f} rad ({np.degrees(np.max(rot_errors)):.1f}°)")

def main():
    print("=== QUICK EVALUATION: TSFormer with Advanced Loss ===")
    
    model, device, config, checkpoint = load_and_check_model()
    
    if model is not None:
        quick_test(model, device, config)
        
        print(f"\n=== SUMMARY ===")
        print(f"✅ Model loaded successfully")
        print(f"✅ Training completed for {checkpoint['epoch']} epochs")
        print(f"✅ Best validation loss: {checkpoint['best_val_loss']:.6f}")
        print(f"✅ Quick test completed")
        print(f"\nFor full evaluation, run:")
        print(f"python evaluate_tsformer_comprehensive.py")
    else:
        print("❌ Failed to load model")

if __name__ == "__main__":
    main()