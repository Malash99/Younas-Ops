"""
Baseline CNN model for underwater visual odometry.
Simple CNN + regression head for 6-DOF pose estimation.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class BaselineCNN(Model):
    """Simple CNN baseline for visual odometry."""
    
    def __init__(self, input_shape=(224, 224, 6), dropout_rate=0.3):
        """
        Initialize baseline CNN.
        
        Args:
            input_shape: Input image shape (height, width, channels)
                        Note: 6 channels for concatenated image pairs
            dropout_rate: Dropout rate for regularization
        """
        super(BaselineCNN, self).__init__()
        
        # Convolutional backbone
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(2)
        
        self.conv2 = layers.Conv2D(128, 5, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2)
        
        self.conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(2)
        
        self.conv4 = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.bn4 = layers.BatchNormalization()
        
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling2D()
        
        # Fully connected layers
        self.fc1 = layers.Dense(256, activation='relu')
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.fc2 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Separate heads for translation and rotation
        self.translation_head = layers.Dense(3, name='translation')
        self.rotation_head = layers.Dense(3, name='rotation')
        
    def call(self, inputs, training=False):
        """Forward pass."""
        # Convolutional layers
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        
        # Separate predictions for translation and rotation
        translation = self.translation_head(x)
        rotation = self.rotation_head(x)
        
        # Concatenate outputs
        output = tf.concat([translation, rotation], axis=-1)
        
        return output


class ImprovedCNN(Model):
    """Improved CNN with residual connections and attention."""
    
    def __init__(self, input_shape=(224, 224, 6), dropout_rate=0.3):
        super(ImprovedCNN, self).__init__()
        
        # Initial convolution
        self.conv_init = layers.Conv2D(64, 7, strides=2, padding='same')
        self.bn_init = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.pool_init = layers.MaxPooling2D(3, strides=2, padding='same')
        
        # Residual blocks
        self.res_blocks = [
            ResidualBlock(64, stride=1),
            ResidualBlock(128, stride=2),
            ResidualBlock(256, stride=2),
            ResidualBlock(512, stride=2)
        ]
        
        # Spatial attention
        self.attention = SpatialAttention()
        
        # Global pooling and FC layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(512, activation='relu')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.fc2 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Output heads
        self.translation_head = layers.Dense(3, name='translation')
        self.rotation_head = layers.Dense(3, name='rotation')
        
    def call(self, inputs, training=False):
        # Initial convolution
        x = self.conv_init(inputs)
        x = self.bn_init(x, training=training)
        x = self.relu(x)
        x = self.pool_init(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x, training=training)
        
        # Apply spatial attention
        x = self.attention(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # FC layers
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        
        # Predictions
        translation = self.translation_head(x)
        rotation = self.rotation_head(x)
        
        return tf.concat([translation, rotation], axis=-1)


class ResidualBlock(layers.Layer):
    """Residual block with optional downsampling."""
    
    def __init__(self, filters, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Projection shortcut if dimensions don't match
        self.projection = None
        if stride > 1:
            self.projection = layers.Conv2D(filters, 1, strides=stride, padding='same')
            self.bn_proj = layers.BatchNormalization()
        
        self.relu2 = layers.ReLU()
        
    def call(self, inputs, training=False):
        shortcut = inputs
        
        # First conv block
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Apply projection if needed
        if self.projection is not None:
            shortcut = self.projection(inputs)
            shortcut = self.bn_proj(shortcut, training=training)
        
        # Add shortcut
        x = x + shortcut
        x = self.relu2(x)
        
        return x


class SpatialAttention(layers.Layer):
    """Spatial attention module."""
    
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv2D(1, 7, padding='same', activation='sigmoid')
        
    def call(self, inputs):
        # Compute attention across spatial dimensions
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        
        attention = self.conv(concat)
        
        return inputs * attention


def create_model(model_type='baseline', input_shape=(224, 224, 6)):
    """
    Create a visual odometry model.
    
    Args:
        model_type: 'baseline' or 'improved'
        input_shape: Input shape for the model
        
    Returns:
        Compiled model
    """
    if model_type == 'baseline':
        model = BaselineCNN(input_shape)
    elif model_type == 'improved':
        model = ImprovedCNN(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing baseline CNN model...")
    
    # Create model
    model = create_model('baseline')
    
    # Test forward pass
    dummy_input = tf.random.normal((1, 224, 224, 6))
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output.numpy()}")
    
    # Print model summary
    model.build((None, 224, 224, 6))
    model.summary()