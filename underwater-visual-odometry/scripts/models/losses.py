"""
Loss functions for visual odometry training.
Includes separate losses for translation and rotation components.
"""

import tensorflow as tf
import numpy as np


class PoseLoss(tf.keras.losses.Loss):
    """Combined loss for 6-DOF pose estimation."""
    
    def __init__(self, translation_weight=1.0, rotation_weight=1.0, loss_type='l1'):
        """
        Initialize pose loss.
        
        Args:
            translation_weight: Weight for translation loss
            rotation_weight: Weight for rotation loss
            loss_type: 'l1' or 'l2' for loss function type
        """
        super(PoseLoss, self).__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.loss_type = loss_type
        
    def call(self, y_true, y_pred):
        """
        Compute pose loss.
        
        Args:
            y_true: Ground truth poses [batch, 6] (3 translation + 3 rotation)
            y_pred: Predicted poses [batch, 6]
            
        Returns:
            Scalar loss value
        """
        # Split translation and rotation
        trans_true, rot_true = y_true[:, :3], y_true[:, 3:]
        trans_pred, rot_pred = y_pred[:, :3], y_pred[:, 3:]
        
        # Compute translation loss
        if self.loss_type == 'l1':
            trans_loss = tf.reduce_mean(tf.abs(trans_true - trans_pred))
        else:  # l2
            trans_loss = tf.reduce_mean(tf.square(trans_true - trans_pred))
        
        # Compute rotation loss
        if self.loss_type == 'l1':
            rot_loss = tf.reduce_mean(tf.abs(rot_true - rot_pred))
        else:  # l2
            rot_loss = tf.reduce_mean(tf.square(rot_true - rot_pred))
        
        # Combine losses
        total_loss = (self.translation_weight * trans_loss + 
                     self.rotation_weight * rot_loss)
        
        return total_loss


class HuberPoseLoss(tf.keras.losses.Loss):
    """Huber loss for robust pose estimation."""
    
    def __init__(self, translation_weight=1.0, rotation_weight=1.0, delta=1.0):
        super(HuberPoseLoss, self).__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.delta = delta
        
    def call(self, y_true, y_pred):
        # Split components
        trans_true, rot_true = y_true[:, :3], y_true[:, 3:]
        trans_pred, rot_pred = y_pred[:, :3], y_pred[:, 3:]
        
        # Huber loss for translation
        trans_diff = tf.abs(trans_true - trans_pred)
        trans_loss = tf.where(
            trans_diff <= self.delta,
            0.5 * tf.square(trans_diff),
            self.delta * trans_diff - 0.5 * tf.square(self.delta)
        )
        trans_loss = tf.reduce_mean(trans_loss)
        
        # Huber loss for rotation
        rot_diff = tf.abs(rot_true - rot_pred)
        rot_loss = tf.where(
            rot_diff <= self.delta,
            0.5 * tf.square(rot_diff),
            self.delta * rot_diff - 0.5 * tf.square(self.delta)
        )
        rot_loss = tf.reduce_mean(rot_loss)
        
        return self.translation_weight * trans_loss + self.rotation_weight * rot_loss


class GeometricLoss(tf.keras.losses.Loss):
    """Geometric loss considering the SE(3) manifold structure."""
    
    def __init__(self, translation_weight=1.0, rotation_weight=1.0):
        super(GeometricLoss, self).__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        
    def call(self, y_true, y_pred):
        # Split components
        trans_true, rot_true = y_true[:, :3], y_true[:, 3:]
        trans_pred, rot_pred = y_pred[:, :3], y_pred[:, 3:]
        
        # Translation loss (standard L2)
        trans_loss = tf.reduce_mean(tf.norm(trans_true - trans_pred, axis=1))
        
        # Rotation loss (geodesic distance approximation)
        # For small angles, geodesic distance ≈ angular distance
        angle_diff = rot_true - rot_pred
        
        # Wrap angles to [-pi, pi]
        angle_diff = tf.atan2(tf.sin(angle_diff), tf.cos(angle_diff))
        
        rot_loss = tf.reduce_mean(tf.norm(angle_diff, axis=1))
        
        return self.translation_weight * trans_loss + self.rotation_weight * rot_loss


class UncertaintyWeightedLoss(tf.keras.losses.Loss):
    """Loss with learned uncertainty weights (homoscedastic uncertainty)."""
    
    def __init__(self):
        super(UncertaintyWeightedLoss, self).__init__()
        # Learnable parameters for uncertainty
        self.log_vars = tf.Variable([0.0, 0.0], trainable=True, name='log_vars')
        
    def call(self, y_true, y_pred):
        # Split components
        trans_true, rot_true = y_true[:, :3], y_true[:, 3:]
        trans_pred, rot_pred = y_pred[:, :3], y_pred[:, 3:]
        
        # Get uncertainty weights
        trans_var = tf.exp(-self.log_vars[0])
        rot_var = tf.exp(-self.log_vars[1])
        
        # Compute losses
        trans_loss = trans_var * tf.reduce_mean(tf.square(trans_true - trans_pred))
        rot_loss = rot_var * tf.reduce_mean(tf.square(rot_true - rot_pred))
        
        # Add regularization term
        reg_loss = tf.reduce_sum(self.log_vars)
        
        return trans_loss + rot_loss + reg_loss


# Metric functions for evaluation
def translation_error(y_true, y_pred):
    """Mean translation error in meters."""
    trans_true = y_true[:, :3]
    trans_pred = y_pred[:, :3]
    return tf.reduce_mean(tf.norm(trans_true - trans_pred, axis=1))


def rotation_error(y_true, y_pred):
    """Mean rotation error in radians."""
    rot_true = y_true[:, 3:]
    rot_pred = y_pred[:, 3:]
    
    # Compute angle difference
    angle_diff = rot_true - rot_pred
    angle_diff = tf.atan2(tf.sin(angle_diff), tf.cos(angle_diff))
    
    return tf.reduce_mean(tf.norm(angle_diff, axis=1))


def translation_rmse(y_true, y_pred):
    """Root mean square translation error."""
    trans_true = y_true[:, :3]
    trans_pred = y_pred[:, :3]
    return tf.sqrt(tf.reduce_mean(tf.square(trans_true - trans_pred)))


def rotation_rmse(y_true, y_pred):
    """Root mean square rotation error."""
    rot_true = y_true[:, 3:]
    rot_pred = y_pred[:, 3:]
    return tf.sqrt(tf.reduce_mean(tf.square(rot_true - rot_pred)))


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 32
    y_true = tf.random.normal((batch_size, 6))
    y_pred = y_true + tf.random.normal((batch_size, 6)) * 0.1
    
    # Test different losses
    losses = {
        'PoseLoss (L1)': PoseLoss(loss_type='l1'),
        'PoseLoss (L2)': PoseLoss(loss_type='l2'),
        'HuberPoseLoss': HuberPoseLoss(),
        'GeometricLoss': GeometricLoss(),
        'UncertaintyWeightedLoss': UncertaintyWeightedLoss()
    }
    
    for name, loss_fn in losses.items():
        loss_val = loss_fn(y_true, y_pred)
        print(f"{name}: {loss_val:.4f}")
    
    # Test metrics
    print("\nTesting metrics...")
    print(f"Translation error: {translation_error(y_true, y_pred):.4f} m")
    print(f"Rotation error: {rotation_error(y_true, y_pred):.4f} rad")
    print(f"Translation RMSE: {translation_rmse(y_true, y_pred):.4f} m")
    print(f"Rotation RMSE: {rotation_rmse(y_true, y_pred):.4f} rad")