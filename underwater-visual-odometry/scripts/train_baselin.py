"""
Training script for baseline visual odometry model.
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_loader import UnderwaterVODataset
from models.baseline_cnn import create_model
from models.losses import (PoseLoss, HuberPoseLoss, GeometricLoss,
                          translation_error, rotation_error,
                          translation_rmse, rotation_rmse)
from coordinate_transforms import integrate_trajectory, compute_trajectory_error


def create_callbacks(model_dir, patience=10):
    """Create training callbacks."""
    callbacks = [
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=0,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks


def train_model(args):
    """Main training function."""
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(args.output_dir, f'model_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save training configuration
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Loading dataset...")
    # Initialize dataset
    dataset = UnderwaterVODataset(
        args.data_dir,
        sequence_length=2,
        image_size=(args.image_height, args.image_width)
    )
    dataset.load_data()
    
    # Create train/val split
    train_indices, val_indices = dataset.train_val_split(val_ratio=args.val_ratio)
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    # Create TF datasets
    train_dataset = dataset.create_tf_dataset(train_indices, batch_size=args.batch_size, shuffle=True)
    val_dataset = dataset.create_tf_dataset(val_indices, batch_size=args.batch_size, shuffle=False)
    
    print("Creating model...")
    # Create model
    model = create_model(
        model_type=args.model_type,
        input_shape=(args.image_height, args.image_width, 6)
    )
    
    # Choose loss function
    if args.loss_type == 'l1':
        loss_fn = PoseLoss(
            translation_weight=args.trans_weight,
            rotation_weight=args.rot_weight,
            loss_type='l1'
        )
    elif args.loss_type == 'l2':
        loss_fn = PoseLoss(
            translation_weight=args.trans_weight,
            rotation_weight=args.rot_weight,
            loss_type='l2'
        )
    elif args.loss_type == 'huber':
        loss_fn = HuberPoseLoss(
            translation_weight=args.trans_weight,
            rotation_weight=args.rot_weight
        )
    elif args.loss_type == 'geometric':
        loss_fn = GeometricLoss(
            translation_weight=args.trans_weight,
            rotation_weight=args.rot_weight
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    
    # Choose optimizer
    if args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=0.9,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            translation_error,
            rotation_error,
            translation_rmse,
            rotation_rmse
        ]
    )
    
    # Build model
    model.build((None, args.image_height, args.image_width, 6))
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(model_dir, patience=args.patience)
    
    print("Starting training...")
    # Train model
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save_weights(os.path.join(model_dir, 'final_model.h5'))
    
    # Save training history
    history_dict = history.history
    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training completed! Models saved to {model_dir}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    evaluate_model(model, val_dataset, dataset, val_indices)
    
    return model, history


def evaluate_model(model, val_dataset, dataset, val_indices):
    """Evaluate model performance."""
    # Get predictions on validation set
    predictions = []
    ground_truth = []
    
    print("Getting predictions...")
    for img_batch, pose_batch in tqdm(val_dataset):
        pred_batch = model.predict(img_batch, verbose=0)
        predictions.extend(pred_batch)
        ground_truth.extend(pose_batch.numpy())
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Compute per-frame errors
    trans_errors = np.linalg.norm(predictions[:, :3] - ground_truth[:, :3], axis=1)
    rot_errors = np.linalg.norm(predictions[:, 3:] - ground_truth[:, 3:], axis=1)
    
    print("\nPer-frame errors:")
    print(f"Translation - Mean: {np.mean(trans_errors):.4f} m, Std: {np.std(trans_errors):.4f} m")
    print(f"Rotation - Mean: {np.mean(rot_errors):.4f} rad, Std: {np.std(rot_errors):.4f} rad")
    
    # Compute trajectory error for a subset
    if len(val_indices) > 100:
        # Use first 100 consecutive frames for trajectory evaluation
        traj_indices = sorted(val_indices[:100])
        
        # Get initial pose
        initial_pose = {
            'position': np.array([0, 0, 0]),
            'quaternion': np.array([1, 0, 0, 0])
        }
        
        # Integrate predicted trajectory
        pred_trajectory = integrate_trajectory(initial_pose, predictions[:99])
        
        # Integrate ground truth trajectory
        gt_trajectory = integrate_trajectory(initial_pose, ground_truth[:99])
        
        # Compute trajectory errors
        traj_errors = compute_trajectory_error(pred_trajectory, gt_trajectory)
        
        print("\nTrajectory errors (100 frames):")
        print(f"ATE: {traj_errors['ate']:.4f} ± {traj_errors['ate_std']:.4f} m")
        print(f"RPE (trans): {traj_errors['rpe_trans']:.4f} ± {traj_errors['rpe_trans_std']:.4f} m")
        print(f"RPE (rot): {traj_errors['rpe_rot']:.4f} ± {traj_errors['rpe_rot_std']:.4f} rad")


def main():
    parser = argparse.ArgumentParser(description='Train underwater visual odometry model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/app/data/raw',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='/app/output/models',
                        help='Path to save models')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='baseline',
                        choices=['baseline', 'improved'],
                        help='Model architecture to use')
    parser.add_argument('--image_height', type=int, default=224,
                        help='Input image height')
    parser.add_argument('--image_width', type=int, default=224,
                        help='Input image width')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer to use')
    
    # Loss arguments
    parser.add_argument('--loss_type', type=str, default='huber',
                        choices=['l1', 'l2', 'huber', 'geometric'],
                        help='Loss function type')
    parser.add_argument('--trans_weight', type=float, default=1.0,
                        help='Translation loss weight')
    parser.add_argument('--rot_weight', type=float, default=1.0,
                        help='Rotation loss weight')
    
    # Other arguments
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Train model
    train_model(args)


if __name__ == "__main__":
    main()