#!/usr/bin/env python3
"""
Simple training script for the corrected two-frame consecutive model.
Uses fixed coordinate transformations for proper reference frame consistency.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import json

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from data_processing.data_loader_fixed import UnderwaterVODatasetFixed
from models.baseline.baseline_cnn import create_model
from utils.coordinate_transforms_fixed import integrate_trajectory_correct, compute_trajectory_errors_correct


def create_simple_loss(translation_weight=1.0, rotation_weight=1.0):
    """Create simple MSE loss for pose regression."""
    def pose_loss(y_true, y_pred):
        # Translation loss (first 3 components)
        trans_loss = tf.reduce_mean(tf.square(y_true[:, :3] - y_pred[:, :3]))
        
        # Rotation loss (last 3 components)
        rot_loss = tf.reduce_mean(tf.square(y_true[:, 3:] - y_pred[:, 3:]))
        
        return translation_weight * trans_loss + rotation_weight * rot_loss
    
    return pose_loss


def train_corrected_model():
    """Train the model with corrected coordinate transformations."""
    print("="*80)
    print("TRAINING CORRECTED TWO-FRAME CONSECUTIVE MODEL")
    print("="*80)
    
    # Configuration
    config = {
        'data_dir': './data/raw',
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 0.001,
        'image_size': (224, 224),
        'val_ratio': 0.2,
        'patience': 15
    }
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'./output/models/corrected_model_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model will be saved to: {model_dir}")
    
    # Load dataset with FIXED coordinate transformations
    print("\nLoading dataset with CORRECTED coordinate transformations...")
    dataset = UnderwaterVODatasetFixed(
        config['data_dir'],
        sequence_length=2,
        image_size=config['image_size']
    )
    dataset.load_data()
    
    # Validate ground truth consistency
    print("\nValidating ground truth consistency...")
    dataset.validate_ground_truth_consistency(5)
    
    # Create train/val split
    train_indices, val_indices = dataset.train_val_split(config['val_ratio'])
    print(f"\nDataset split: {len(train_indices)} train, {len(val_indices)} val samples")
    
    # Create TF datasets
    print("Creating TensorFlow datasets...")
    train_dataset = dataset.create_tf_dataset(train_indices, batch_size=config['batch_size'], shuffle=True)
    val_dataset = dataset.create_tf_dataset(val_indices, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    print("Creating baseline CNN model...")
    model = create_model('baseline', input_shape=(*config['image_size'], 6))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=create_simple_loss(translation_weight=1.0, rotation_weight=1.0)
    )
    
    # Build model and show summary
    model.build((None, *config['image_size'], 6))
    model.summary()
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nStarting training for {config['epochs']} epochs...")
    print("="*80)
    
    history = model.fit(
        train_dataset,
        epochs=config['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save_weights(os.path.join(model_dir, 'final_model.h5'))
    
    # Save training history
    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f, indent=2)
    
    print(f"\nTraining completed! Model saved to {model_dir}")
    
    # Quick evaluation
    print("\nEvaluating on validation set...")
    val_loss = model.evaluate(val_dataset, verbose=1)
    print(f"Final validation loss: {val_loss:.6f}")
    
    print("="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Model directory: {model_dir}")
    print(f"Key improvements:")
    print("- Uses CORRECTED coordinate transformations")
    print("- Ensures consistent reference frame handling")
    print("- Ground truth validated for reconstruction accuracy")
    print("- Ready for accurate trajectory evaluation")
    
    return model, history, model_dir


if __name__ == "__main__":
    train_corrected_model()