# UW-TransVO: Underwater Visual Odometry with Transformers

## Overview

UW-TransVO is a state-of-the-art transformer-based architecture designed for underwater visual odometry. It processes multi-camera underwater imagery to estimate 6-DOF camera poses with high accuracy, addressing the unique challenges of underwater environments such as poor visibility, color distortion, and lighting variations.

## Architecture

🔄 Epoch 1/50
⏰ 01:32:25 | LR: 0.00001000
🏋️  Epoch 1 Training:  26%|██▎      | 167/654 [04:47<15:18,  1.89s/it] , Loss=0.0053, ATE=0.0006, D rift=0.02m, GPU=1.0GB
❌ NaN loss detected at batch 167! Stopping training.
🔧 Try reducing learning rate or checking input data.
🏋️  Epoch 1 Training:  26%|██▎      | 167/654 [04:48<14:02,  1.73s/it] , Loss=0.0053, ATE=0.0006, D rift=0.02m, GPU=1.0GB
✅ Epoch 1 Validation: 100%|███████████████████| 100/100 [01:30<00:00,  1.11it/s] , Loss=0.0305, ATE=0.0057, Drift=0.17m

📈 EPOCH 1 RESULTS:
────────────────────────────────────────────────────────────
⏱️  Time: 379.4s (Total: 6.3min)
🏋️  TRAIN  | Loss: nan | ATE: nan | Drift: nanm
✅ VAL    | Loss: 0.020914 | ATE: 0.003642 | Drift: 0.0958m
📊 METRICS | Final Drift: 0.0958m | Relative: 51.75% | Trajectory: 0.13m
⭐ NEW BEST MODEL SAVED! (single_camera_best_model.pth) | Loss: 0.020914
────────────────────────────────────────────────────────────

🔄 Epoch 2/50
⏰ 01:38:45 | LR: 0.00000999
🏋️  Epoch 2 Training:   5%|▍         | 31/654 [00:54<18:50,  1.81s/it] , Loss=0.0048, ATE=0.0006, D rift=0.07m, GPU=1.0GB
❌ NaN loss detected at batch 31! Stopping training.
🔧 Try reducing learning rate or checking input data.
🏋️  Epoch 2 Training:   5%|▍         | 31/654 [00:55<18:34,  1.79s/it] , Loss=0.0048, ATE=0.0006, D rift=0.07m, GPU=1.0GB
✅ Epoch 2 Validation: 100%|███████████████████| 100/100 [01:29<00:00,  1.11it/s] , Loss=0.0306, ATE=0.0058, Drift=0.16m

📈 EPOCH 2 RESULTS:
────────────────────────────────────────────────────────────
⏱️  Time: 145.3s (Total: 8.8min)
🏋️  TRAIN  | Loss: nan | ATE: nan | Drift: nanm
✅ VAL    | Loss: 0.021083 | ATE: 0.003677 | Drift: 0.0983m
📊 METRICS | Final Drift: 0.0983m | Relative: 52.49% | Trajectory: 0.13m
────────────────────────────────────────────────────────────

🔄 Epoch 3/50
⏰ 01:41:11 | LR: 0.00000996
🏋️  Epoch 3 Training:   2%|▏         | 12/654 [00:20<18:51,  1.76s/it] , Loss=0.0302, ATE=0.0057, D rift=0.21m, GPU=1.0GB
### Model Pipeline

```
Input Images → Enhancement → Feature Extraction → Spatial Attention → 
Temporal Attention → Multi-modal Fusion → Pose Regression → 6-DOF Poses
```

### Layer-by-Layer Breakdown

#### 1. **Underwater Image Enhancement** (`UnderwaterImageEnhancement`)
- **Purpose**: Addresses underwater-specific image degradation
- **Components**:
  - Color correction layers (Conv2d + BatchNorm + ReLU)
  - Contrast enhancement (Conv2d + Sigmoid)
  - Noise reduction (Conv2d filters)
- **Input**: Raw underwater images `[B, 3, H, W]`
- **Output**: Enhanced images `[B, 3, H, W]`

#### 2. **Vision Transformer** (`VisionTransformer`)
- **Purpose**: Extracts patch-based visual features
- **Architecture**:
  - **Patch Embedding**: Converts images to patches `[B, N_patches, D]`
  - **Multi-Head Attention**: 12 heads, 768 dimensions
  - **Transformer Blocks**: 6 layers with residual connections
  - **Class Token**: Global image representation
- **Input**: Enhanced images `[B, 3, 224, 224]`
- **Output**: Global features `[B, 768]`

#### 3. **Positional Encodings**
- **Camera Positional Encoding**: Learnable embeddings for each camera
- **Temporal Positional Encoding**: Sinusoidal encoding for sequence positions
- **Purpose**: Provides spatial and temporal context

#### 4. **Spatial Cross-Camera Attention** (`SpatialCrossCameraAttention`)
- **Purpose**: Fuses information across simultaneous camera views
- **Components**:
  - Multi-head attention across cameras
  - Layer normalization and residual connections
- **Input**: Camera features `[B, N_cameras, D]`
- **Output**: Spatially-attended features `[B, N_cameras, D]`

#### 5. **Temporal Self-Attention** (`TemporalSelfAttention`)
- **Purpose**: Models temporal relationships in image sequences
- **Components**:
  - Multi-head self-attention across time steps
  - Feed-forward network (4x expansion)
  - Layer normalization and residual connections
- **Input**: Temporal features `[B, T, D]`
- **Output**: Temporally-attended features `[B, T, D]`

#### 6. **Multi-Modal Fusion** (`MultiModalFusion`)
- **Purpose**: Integrates visual features with IMU and pressure sensor data
- **Inputs**:
  - Visual features: `[B, D]`
  - IMU data (optional): `[B, T, 6]` (acceleration + gyroscope)
  - Pressure data (optional): `[B, T, 1]`
- **Output**: Fused features `[B, D]`

#### 7. **Pose Regression Head** (`PoseRegressionHead`)
- **Purpose**: Predicts 6-DOF camera poses with uncertainty estimation
- **Components**:
  - Multi-layer perceptron
  - Separate heads for translation and rotation
  - Uncertainty estimation (optional)
- **Input**: Fused features `[B, D]`
- **Output**: 6-DOF poses `[B, 6]` + uncertainties `[B, 6]`

## Input/Output Specifications

### Model Inputs

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `images` | `[B, T, N_cam, 3, H, W]` | Input image sequences |
| `camera_ids` | `[B, N_cam]` | Camera identifier indices |
| `camera_mask` | `[B, N_cam]` | Mask for missing cameras (optional) |
| `imu_data` | `[B, T, 6]` | IMU measurements (optional) |
| `pressure_data` | `[B, T, 1]` | Pressure sensor data (optional) |

**Where:**
- `B`: Batch size
- `T`: Temporal sequence length
- `N_cam`: Number of cameras
- `H, W`: Image height and width (default: 224×224)

### Model Outputs

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `pose` | `[B, 6]` | 6-DOF pose (tx, ty, tz, rx, ry, rz) |
| `uncertainty` | `[B, 6]` | Pose uncertainty estimates (optional) |

**Pose Format:**
- Translation: `[tx, ty, tz]` in meters
- Rotation: `[rx, ry, rz]` in radians (axis-angle representation)

## Model Configurations

### Standard Configuration

```python
config = {
    'img_size': 224,           # Input image size
    'patch_size': 16,          # Vision transformer patch size
    'd_model': 768,            # Feature dimension
    'num_heads': 12,           # Multi-head attention heads
    'num_layers': 6,           # Transformer layers
    'max_cameras': 5,          # Maximum number of cameras
    'max_seq_len': 10,         # Maximum sequence length
    'dropout': 0.1,            # Dropout rate
    'use_imu': True,           # Enable IMU fusion
    'use_pressure': True,      # Enable pressure sensor fusion
    'uncertainty_estimation': True  # Enable uncertainty estimation
}
```

### Single Camera Configuration

```python
config = {
    'img_size': 224,
    'patch_size': 16,
    'd_model': 768,
    'num_heads': 12,
    'num_layers': 6,
    'max_cameras': 1,          # Single camera
    'max_seq_len': 5,
    'dropout': 0.1,
    'use_imu': False,          # Disabled for vision-only
    'use_pressure': False,     # Disabled for vision-only
    'uncertainty_estimation': True
}
```

## Training

### Dataset Requirements

The model expects data in the following format:
- **Training CSV**: Contains image paths, timestamps, and ground truth poses
- **Images**: Preprocessed underwater images (224×224 pixels)
- **Poses**: 6-DOF ground truth trajectories

### Training Scripts

1. **Single Camera Training**: `train_single_camera.py`
   - Simplified setup for single camera
   - Progress bars and visual feedback
   - Console window visibility

2. **Multi-Camera Training**: `train_console_simple.py`
   - Full multi-camera capabilities
   - Sub-trajectory training approach

### Training Features

- **Progress Visualization**: Real-time progress bars with tqdm
- **Metrics Tracking**: Loss, ATE (Absolute Trajectory Error), drift
- **GPU Monitoring**: Memory usage tracking
- **Model Checkpointing**: Automatic best model saving
- **Console Visibility**: Explicit console window management

## Loss Functions

### Trajectory-Aware Loss (`TrajectoryAwareLoss`)

The model uses a sophisticated loss function that combines:

1. **Translation Loss**: L2 loss on position estimates
2. **Rotation Loss**: Angular loss on orientation estimates  
3. **ATE Loss**: Absolute Trajectory Error for global consistency
4. **Consistency Loss**: Temporal smoothness constraints
5. **Smoothness Loss**: Velocity and acceleration regularization

```python
total_loss = (translation_weight * translation_loss + 
              rotation_weight * rotation_loss +
              ate_weight * ate_loss +
              consistency_weight * consistency_loss +
              smoothness_weight * smoothness_loss)
```

## Performance Metrics

The model tracks several key metrics:

- **ATE (Absolute Trajectory Error)**: Global trajectory accuracy
- **Drift**: Cumulative position error over time
- **Final Position Error**: End-point accuracy
- **Relative Drift Percentage**: Drift relative to trajectory length
- **Trajectory Length**: Total distance traveled

## Model Variants

### By Camera Count
- **Single Camera**: Simplified architecture for single viewpoint
- **Dual Camera**: Stereo-like setup with two cameras
- **Multi-Camera**: Full 3-5 camera array processing

### By Modality
- **Vision-Only**: Uses only camera data
- **Multi-Modal**: Integrates IMU and pressure sensors
- **Uncertainty-Aware**: Provides confidence estimates

## Memory and Computational Requirements

### Model Size
- **Parameters**: ~85M parameters (standard config)
- **Memory**: ~4-6 GB GPU memory during training
- **Inference**: ~2-3 GB GPU memory

### Computational Complexity
- **Training**: O(T × N_cam × H × W) per forward pass
- **Inference**: Real-time capable on modern GPUs
- **Attention**: O(N²) complexity in sequence length and camera count

## Getting Started

### Quick Start

```bash
# Single camera training
python train_single_camera.py

# Multi-camera training  
python train_console_simple.py

# Model evaluation
python evaluate_model.py
```

### Data Preparation

1. Extract images from ROS bags
2. Generate training CSV with poses
3. Preprocess images (resize, normalize)
4. Create sub-trajectory sequences

## Research Applications

UW-TransVO is designed for:
- **Autonomous Underwater Vehicles (AUVs)**
- **Underwater robotics navigation**
- **Marine archaeology and surveying**
- **Underwater inspection and monitoring**
- **Scientific underwater exploration**

## Citation

If you use UW-TransVO in your research, please cite:

```bibtex
@article{uw-transvo-2024,
  title={UW-TransVO: Scalable Multi-Camera Underwater Visual Odometry with Transformers},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024}
}
```

## Technical Notes

### Underwater Challenges Addressed
- **Color distortion**: Wavelength-dependent attenuation
- **Poor visibility**: Scattering and absorption
- **Non-uniform lighting**: Artificial lighting effects
- **Motion blur**: Camera and water movement
- **Scale ambiguity**: Lack of familiar objects

### Transformer Advantages
- **Long-range dependencies**: Better temporal modeling
- **Multi-modal fusion**: Natural integration of different sensors
- **Attention visualization**: Interpretable feature importance
- **Scalability**: Handles variable camera counts and sequence lengths

### Implementation Details
- **Framework**: PyTorch
- **Training**: Mixed precision, gradient clipping
- **Optimization**: AdamW with cosine annealing
- **Regularization**: Dropout, weight decay
- **Data augmentation**: Geometric and photometric transforms