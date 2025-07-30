# Scalable Multi-Camera Underwater Visual Odometry with Transformers

## Project Overview

This project develops a novel **scalable multi-camera transformer architecture** for underwater visual odometry that can adaptively work with 1-5 cameras while incorporating IMU and barometer data. The goal is to create a robust, generalizable system that performs well across different camera configurations and can handle sensor failures.

## Key Innovation: Adaptive Multi-Camera Architecture

Unlike traditional fixed-camera setups, our approach dynamically adapts to available cameras, making it practical for real-world underwater robotics where camera failures are common.

## Research Questions

1. **How does camera count affect underwater VO performance?**
2. **What's the optimal camera selection strategy for different underwater conditions?**
3. **How do sequence length and multi-modal fusion impact accuracy?**
4. **Can the model generalize across different underwater trajectories?**

## Architecture: UW-TransVO (Underwater Transformer Visual Odometry)

### Core Components

```
Input: Variable cameras (1-5) × sequence_length (2 or 4) × 224×224×3 + IMU + Pressure

┌─────────────────────────────────────────────────────────────┐
│                    UW-TransVO Architecture                  │
├─────────────────────────────────────────────────────────────┤
│ 1. Underwater Image Enhancement Module                      │
│    ├── Color correction for underwater conditions           │
│    ├── Contrast enhancement                                 │
│    └── Noise reduction                                      │
├─────────────────────────────────────────────────────────────┤
│ 2. Multi-Camera Feature Extraction                         │
│    ├── Shared CNN backbone (ResNet50/EfficientNet)         │
│    ├── Camera-specific positional encoding                 │
│    └── Feature dimension: 768                              │
├─────────────────────────────────────────────────────────────┤
│ 3. Spatial Cross-Camera Attention                          │
│    ├── Attention between simultaneous camera views         │
│    ├── Handles variable number of cameras (1-5)            │
│    └── Camera masking for missing inputs                   │
├─────────────────────────────────────────────────────────────┤
│ 4. Temporal Self-Attention                                 │
│    ├── Attention across time steps (2 or 4 frames)         │
│    ├── Motion-aware attention weights                      │
│    └── Positional encoding for temporal relationships      │
├─────────────────────────────────────────────────────────────┤
│ 5. Multi-Modal Sensor Fusion                               │
│    ├── IMU integration (acceleration + gyroscope)          │
│    ├── Pressure/depth information                          │
│    └── Cross-modal attention between vision and sensors    │
├─────────────────────────────────────────────────────────────┤
│ 6. 6-DOF Pose Regression Head                              │
│    ├── Translation: (Δx, Δy, Δz)                          │
│    ├── Rotation: (Δroll, Δpitch, Δyaw)                    │
│    └── Uncertainty estimation                              │
└─────────────────────────────────────────────────────────────┘

Output: 6-DOF relative pose + uncertainty estimates
```

## Experimental Design

### Dataset Configuration
- **Training Dataset**: Ariel underwater vehicle (5-camera setup)
- **Validation Dataset**: Same trajectory, different time segment
- **Test Dataset 1**: Unseen trajectory from same platform
- **Test Dataset 2**: Additional downloaded dataset (generalizability test)

### Camera Configurations
```python
CAMERA_CONFIGS = {
    'monocular': [0],                    # Single camera
    'stereo_wide': [0, 2],              # Wide baseline stereo
    'stereo_close': [0, 1],             # Close baseline stereo  
    'triple': [0, 1, 2],                # Triple camera setup
    'quad': [0, 1, 2, 3],              # Quad camera setup
    'full': [0, 1, 2, 3, 4]            # Full 5-camera array
}
```

### Sequence Configurations
```python
SEQUENCE_CONFIGS = {
    'short': 2,     # 2-frame sequences
    'long': 4       # 4-frame sequences  
}
```

### Multi-Modal Configurations
```python
MODALITY_CONFIGS = {
    'vision_only': ['cameras'],
    'vision_imu': ['cameras', 'imu'],
    'vision_pressure': ['cameras', 'pressure'],
    'vision_all': ['cameras', 'imu', 'pressure']
}
```

## Complete Experiment Matrix

| Model | Cameras | Sequence | Modality | Total Experiments |
|-------|---------|----------|----------|-------------------|
| UW-TransVO | 6 configs | 2 lengths | 4 modalities | **48 experiments** |

### Ablation Studies
1. **Camera Count Impact**: Performance vs number of cameras
2. **Sequence Length**: 2-frame vs 4-frame sequences
3. **Sensor Fusion**: Vision-only vs multi-modal
4. **Camera Selection**: Optimal camera combinations
5. **Robustness**: Performance with missing cameras

## Loss Functions

### Weighted Multi-Task Loss
```python
L_total = λ_trans * L_translation + λ_rot * L_rotation + λ_uncert * L_uncertainty

Where:
- λ_trans = 1.0      # Translation weight
- λ_rot = 10.0       # Rotation weight (higher due to scale difference)
- λ_uncert = 0.1     # Uncertainty regularization
```

### Individual Loss Components
```python
L_translation = MSE(pred_xyz, gt_xyz)
L_rotation = MSE(pred_rpy, gt_rpy) 
L_uncertainty = KL_divergence(pred_uncertainty, target_uncertainty)
```

## Evaluation Metrics

### Trajectory Metrics
- **ATE (Absolute Trajectory Error)**: Global position accuracy
- **RPE (Relative Pose Error)**: Local motion accuracy
- **Rotation Error**: Angular accuracy in degrees
- **Scale Drift**: Long-term trajectory consistency

### Underwater-Specific Metrics
- **Performance vs Visibility**: Error correlation with image quality
- **Depth Accuracy**: Pressure sensor fusion effectiveness
- **Motion Type Analysis**: Performance on different motion patterns

### Generalizability Metrics
- **Cross-Dataset Performance**: Model trained on dataset A, tested on B
- **Camera Failure Robustness**: Performance with missing cameras
- **Trajectory Length**: Accuracy vs sequence length

## Training Strategy & Data Processing

### **Key Training Innovation: Shuffled Frame Pairs**

Instead of sequential frame pairs (1,2), (2,3), (3,4)..., we use **randomized frame pairs** from across the entire dataset:
- Frame pairs: (12,13), (156,157), (89,90), (234,235)...
- **Benefits:**
  - Forces model to learn **general motion patterns** rather than memorizing sequences
  - **Better generalization** across different motion types
  - **Prevents overfitting** to specific trajectory patterns
  - **More robust** to various underwater conditions

### Training Data Structure
```
data/processed/training_dataset/
├── images/
│   ├── cam0/          # 27,399 images
│   ├── cam1/          # 27,399 images  
│   ├── cam2/          # 27,399 images
│   ├── cam3/          # 27,399 images
│   └── cam4/          # 27,399 images
├── data/
│   ├── training_data.csv      # All modalities + ground truth
│   └── metadata.json          # Dataset configuration
└── splits/
    ├── train_samples.txt      # Bags 0-2 (training)
    ├── val_samples.txt        # Bag 3 (validation)  
    └── test_samples.txt       # Bag 4 (test)
```

### Ground Truth Format
```csv
sample_id,bag_name,timestamp,dt,delta_x,delta_y,delta_z,delta_roll,delta_pitch,delta_yaw,
pose_x,pose_y,pose_z,imu_accel_x,imu_accel_y,imu_accel_z,imu_gyro_x,imu_gyro_y,imu_gyro_z,
pressure,cam0_path,cam1_path,cam2_path,cam3_path,cam4_path
```

## Implementation Plan

### Phase 1: Core Architecture ✅ **COMPLETED**
- ✅ Scalable transformer architecture (1-5 cameras)
- ✅ Multi-camera spatial attention
- ✅ Temporal self-attention 
- ✅ Multi-modal sensor fusion
- ✅ Underwater image enhancement
- ✅ Adaptive data loading with camera masks

### Phase 2: Training Pipeline 🚧 **IN PROGRESS**
- ✅ Shuffled frame pair strategy
- 🔄 Loss functions with uncertainty weighting
- 🔄 Multi-task optimization
- 🔄 Experiment configuration framework
- ⏳ Training loop with multiple camera configs

### Phase 3: Evaluation Framework ⏳ **NEXT**
- ⏳ Trajectory visualization (global frame)
- ⏳ Comprehensive metrics (ATE, RPE)
- ⏳ Ablation study framework
- ⏳ Baseline comparisons

### Phase 4: Generalizability Testing ⏳ **PLANNED**
- ⏳ Cross-dataset validation
- ⏳ Camera failure simulation
- ⏳ Performance analysis
- ⏳ Paper preparation

## Expected Contributions

### Technical Contributions
1. **Novel Architecture**: First scalable multi-camera transformer for underwater VO
2. **Adaptive Attention**: Camera-agnostic attention mechanism
3. **Multi-Modal Fusion**: Effective integration of vision, IMU, and pressure data
4. **Robustness**: Handle sensor failures gracefully

### Scientific Contributions  
1. **Comprehensive Analysis**: Camera count vs performance trade-offs
2. **Generalizability Study**: Cross-dataset validation methodology
3. **Underwater-Specific Insights**: Domain-specific challenges and solutions
4. **Practical Guidelines**: Camera selection strategies for underwater robots

## Publication Strategy

### Target Venues
- **Primary**: ICRA 2025, IROS 2025
- **Secondary**: IEEE Robotics and Automation Letters (RA-L)
- **Domain-specific**: OCEANS Conference, Autonomous Robots Journal

### Paper Structure
```
1. Introduction
   - Underwater VO challenges
   - Multi-camera system motivation
   
2. Related Work
   - Visual odometry methods
   - Transformer architectures
   - Underwater robotics
   
3. Method: UW-TransVO
   - Architecture details
   - Multi-camera adaptation
   - Multi-modal fusion
   
4. Experiments
   - Dataset description
   - Ablation studies
   - Generalizability tests
   
5. Results & Analysis
   - Performance comparison
   - Camera count analysis
   - Failure case studies
   
6. Conclusion & Future Work
```

## Getting Started

### Prerequisites
```bash
pip install torch torchvision transformers
pip install opencv-python numpy pandas matplotlib
pip install scikit-learn tqdm tensorboard
```

### Quick Start
```bash
# 1. Extract training data
python scripts/extract_all_data.py --output_dir data/processed/full_dataset

# 2. Train baseline model (quad cameras, vision only)
python experiments/train_transformer.py --cameras 0,1,2,3 --sequence_length 2 --modality vision_only

# 3. Evaluate and visualize
python experiments/evaluate_model.py --model_path models/quad_seq2_vision.pth --plot_trajectory
```

---

**This approach provides a solid foundation for a high-impact publication in underwater visual odometry, with novel technical contributions and comprehensive experimental validation.**