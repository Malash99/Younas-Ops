# Underwater Visual Odometry

A deep learning-based visual odometry system for underwater environments using CNN models to estimate camera motion from sequential underwater images.

## Project Overview

This project implements a visual odometry system that:
- Extracts sequential images and poses from ROS bag files
- Trains CNN models to predict relative camera motion
- Evaluates trajectory accuracy on sequential underwater data
- Supports attention-based and baseline CNN architectures

## Dataset

- **5 ROS bag files** with synchronized camera images and pose data
- **~5,480 total images** (1,096 per bag) at 20Hz
- **Camera**: AlphaSense driver (`/alphasense_driver_ros/cam0`)
- **Ground truth**: MAVROS odometry (`/mavros/local_position/odom`)
- **Duration**: ~55 seconds per bag

## Project Structure

```
underwater-visual-odometry/
├── data/
│   ├── raw/                    # Original ROS bag files
│   └── processed/              # Extracted images and poses
├── scripts/                    # Organized by functionality
│   ├── data_processing/        # Data extraction & preprocessing
│   ├── models/                 # Model architectures by type
│   │   ├── baseline/           # Baseline CNN models
│   │   └── attention/          # Attention-based models
│   ├── training/               # Training scripts by model type
│   │   ├── baseline/           # Baseline model training
│   │   └── attention/          # Attention model training
│   ├── evaluation/             # Model evaluation tools
│   │   ├── common/             # Shared evaluation scripts
│   │   ├── baseline/           # Baseline-specific evaluation
│   │   └── attention/          # Attention-specific evaluation
│   ├── utils/                  # Utility functions
│   └── shell_scripts/          # Automation scripts
├── output/
│   ├── models/                 # Trained models
│   └── visualizations/         # Results plots
├── notebooks/                  # Jupyter notebooks
├── Dockerfile                  # Docker environment
└── docker-compose.yaml        # Container orchestration
```

## Requirements

### Docker Environment (Recommended)
- Docker and Docker Compose
- ROS Noetic base image with:
  - TensorFlow 2.10.0
  - OpenCV 4.7.0
  - ROS bag tools
  - Python scientific stack

### Local Installation (Alternative)
```bash
# ROS Noetic
sudo apt update
sudo apt install ros-noetic-desktop-full

# Python packages
pip install tensorflow==2.10.0 opencv-python==4.7.0.72 \
           numpy pandas matplotlib scikit-learn tqdm \
           pyquaternion h5py scipy tensorboard
```

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd underwater-visual-odometry

# Start Docker environment
docker-compose up -d

# Verify container is running
docker ps
```

### 2. Extract Images from ROS Bags

**Single bag extraction:**
```bash
docker exec underwater_vo bash -c "
cd /app && python3 scripts/extract_rosbag_data.py \
  --bag data/raw/ariel_2023-12-21-14-24-42_0.bag \
  --output data/processed/bag_0"
```

**Extract all 5 bags automatically:**
```bash
# Using new structure
bash extract_all_bags_new.sh

# Or using quick commands interface
python3 scripts/quick_commands.py extract --all
```

**Expected output:**
```
data/processed/
├── bag_0/
│   ├── images/           # 1,096 images
│   ├── image_timestamps.csv
│   └── extracted_poses.csv
├── bag_1/ ... bag_4/     # Similar structure
```

### 3. Train the Model

**Baseline CNN training:**
```bash
# Using new structure
docker exec underwater_vo bash -c "
cd /app && python3 scripts/training/baseline/train_baseline.py \
  --data_dir data/raw \
  --output_dir output/models \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001"

# Or using quick commands
python3 scripts/quick_commands.py train --type baseline --epochs 50
```

**Training with attention mechanism:**
```bash
# Using new structure
docker exec underwater_vo bash -c "
cd /app && python3 scripts/training/attention/train_attention_window.py \
  --window_size 5 \
  --epochs 50"

# Or using quick commands
python3 scripts/quick_commands.py train --type attention --epochs 50
```

**Monitor training:**
```bash
# View logs
docker exec underwater_vo bash -c "cd /app && tensorboard --logdir output/models/*/logs"

# Check training progress
docker exec underwater_vo bash -c "cd /app && ls -la output/models/"
```

### 4. Evaluate the Model

**Quick trajectory evaluation (subsampled):**
```bash
# Using new structure
docker exec underwater_vo bash -c "
cd /app && python3 scripts/evaluation/common/simple_trajectory_plot.py \
  --model_dir output/models/model_20250708_023343 \
  --subsample 20"

# Or using quick commands
python3 scripts/quick_commands.py eval --model output/models/model_20250708_023343 --type quick
```

**Full sequential evaluation:**
```bash
# Using new structure
docker exec underwater_vo bash -c "
cd /app && python3 scripts/evaluation/common/evaluate_sequential_bags.py \
  --model_dir output/models/model_20250708_023343 \
  --data_dir data/processed"

# Or using quick commands
python3 scripts/quick_commands.py eval --model output/models/model_20250708_023343 --type sequential
```

**Comprehensive model evaluation:**
```bash
# Using new structure
docker exec underwater_vo bash -c "
cd /app && python3 scripts/evaluation/common/evaluate_model.py \
  --model_dir output/models/model_20250708_023343 \
  --data_dir data/raw"

# Or using quick commands
python3 scripts/quick_commands.py eval --model output/models/model_20250708_023343 --type comprehensive
```

## Detailed Workflows

### Data Extraction Workflow

1. **Inspect ROS bag contents:**
```bash
docker exec underwater_vo bash -c "
cd /app && python3 scripts/extract_rosbag_data.py \
  --bag data/raw/ariel_2023-12-21-14-24-42_0.bag \
  --output /tmp/test_extraction"
```

2. **Available topics in bags:**
   - **Images**: `/alphasense_driver_ros/cam0-4` (5 cameras available)
   - **Poses**: `/mavros/local_position/odom` (ground truth)
   - **IMU**: `/alphasense_driver_ros/imu`
   - **Qualisys**: `/qualisys/ariel/odom` (external tracking)

3. **Extraction parameters:**
   - `--image_topic`: Specify camera (default: auto-detect cam0)
   - `--pose_topic`: Specify pose source (default: mavros odom)
   - `--max_images`: Limit number of images

### Training Workflow

1. **Model architectures available:**
   - **Baseline CNN**: Simple encoder-decoder
   - **Attention Window**: Multi-frame attention mechanism

2. **Training configuration:**
```json
{
  "model_type": "baseline",
  "image_height": 224,
  "image_width": 224,
  "batch_size": 32,
  "epochs": 50,
  "learning_rate": 0.001,
  "optimizer": "adam",
  "loss_type": "huber",
  "val_ratio": 0.2
}
```

3. **Training outputs:**
   - `best_model.h5`: Best validation model
   - `final_model.h5`: Final epoch model
   - `history.json`: Training metrics
   - `config.json`: Model configuration
   - `logs/`: TensorBoard logs

4. **Monitor training:**
```bash
# Real-time monitoring
docker exec underwater_vo bash -c "cd /app && watch -n 5 'ls -la output/models/*/'"

# TensorBoard visualization
docker exec -p 6006:6006 underwater_vo bash -c "
cd /app && tensorboard --logdir output/models --host 0.0.0.0"
# Access at http://localhost:6006
```

### Evaluation Workflow

1. **Trajectory evaluation types:**
   - **Simple**: Fast subsampled trajectory comparison
   - **Sequential**: Full bags 0-3 sequential processing
   - **Comprehensive**: Multiple sequences with error analysis

2. **Evaluation metrics:**
   - **ATE**: Absolute Trajectory Error
   - **RPE**: Relative Pose Error (translation/rotation)
   - **Position Error**: Euclidean distance error
   - **Drift Analysis**: Error accumulation over time

3. **Output visualizations:**
   - **Trajectory plots**: 2D/3D path comparisons
   - **Error analysis**: Error over time
   - **Performance summary**: Statistical analysis

## Key Scripts Reference

### Data Processing
- `extract_rosbag_data.py`: Extract images/poses from ROS bags
- `prepare_underwater_data.py`: Data preprocessing utilities
- `data_loader.py`: Dataset loading for training

### Model Training
- `train_baseline.py`: Train baseline CNN model
- `train_attention_window.py`: Train attention-based model
- `models/baseline_cnn.py`: CNN architecture definitions
- `models/losses.py`: Custom loss functions

### Evaluation & Analysis
- `evaluate_model.py`: Comprehensive model evaluation
- `evaluate_sequential_bags.py`: Sequential trajectory evaluation
- `simple_trajectory_plot.py`: Quick trajectory visualization
- `coordinate_transforms.py`: Pose transformation utilities
- `visualization.py`: Plotting utilities

## Configuration Files

### Docker Configuration
```yaml
# docker-compose.yaml
services:
  underwater-vo:
    build: .
    volumes:
      - ./scripts:/app/scripts
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - PYTHONPATH=/app:/app/scripts
```

### Model Configuration
```json
{
  "data_dir": "/app/data/raw",
  "output_dir": "/app/output/models",
  "model_type": "baseline",
  "image_height": 224,
  "image_width": 224,
  "batch_size": 32,
  "epochs": 50,
  "learning_rate": 0.001,
  "loss_type": "huber",
  "patience": 10
}
```

## Results & Performance

### Latest Model Performance
- **Model**: `model_20250708_023343`
- **Architecture**: Baseline CNN
- **Training**: 50 epochs, Huber loss
- **Validation**: 20% split

### Sequential Evaluation Results
- **Test data**: Bags 0-3 (4,382 poses)
- **Subsampling**: 1:20 (219 frame pairs)
- **Mean position error**: 4.974 m
- **Std position error**: 2.029 m
- **Max position error**: 9.310 m

### Generated Outputs
- **Trajectory plots**: `output/sequential_results/`
- **Training curves**: `output/models/*/logs/`
- **Error analysis**: `output/models/*/evaluation/`

## Troubleshooting

### Common Issues

1. **Docker permission errors:**
```bash
sudo chmod -R 777 ./data ./output
```

2. **ROS bag extraction fails:**
```bash
# Check bag file integrity
docker exec underwater_vo bash -c "cd /app && rosbag info data/raw/ariel_*.bag"
```

3. **CUDA/GPU issues:**
```bash
# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

4. **Memory issues during training:**
```bash
# Reduce batch size
python3 scripts/train_baseline.py --batch_size 16
```

### Debug Commands

```bash
# Check Docker container status
docker logs underwater_vo

# Interactive container access
docker exec -it underwater_vo bash

# Check Python environment
docker exec underwater_vo python3 -c "import tensorflow as tf; print(tf.__version__)"

# Monitor system resources
docker stats underwater_vo
```

## Development

### Adding New Models
1. Create model architecture in `scripts/models/`
2. Add training script following `train_baseline.py` pattern
3. Update `create_model()` function with new model type
4. Test with small dataset first

### Custom Loss Functions
1. Add loss function to `scripts/models/losses.py`
2. Update training script to use new loss
3. Configure loss weights for translation/rotation components

### New Evaluation Metrics
1. Add metric computation to `coordinate_transforms.py`
2. Update evaluation scripts to compute new metrics
3. Add visualization to `visualization.py`

## Citations & References

- TensorFlow Visual Odometry implementations
- ROS bag processing utilities
- Underwater computer vision datasets
- SE(3) pose representation and transformations

## License & Contact

[Add your license and contact information here]

---

## New Organized Structure

The scripts folder has been restructured for better organization and scalability:

### Quick Commands Interface
```bash
# List available models
python3 scripts/quick_commands.py list

# Extract all data
python3 scripts/quick_commands.py extract --all

# Train baseline model
python3 scripts/quick_commands.py train --type baseline --epochs 50

# Quick evaluation
python3 scripts/quick_commands.py eval --model output/models/model_X --type quick
```

### Direct Script Access
```bash
# Data processing
python3 scripts/data_processing/extract_rosbag_data.py [args]

# Training (by model type)
python3 scripts/training/baseline/train_baseline.py [args]
python3 scripts/training/attention/train_attention_window.py [args]

# Evaluation (common tools)
python3 scripts/evaluation/common/simple_trajectory_plot.py [args]
python3 scripts/evaluation/common/evaluate_model.py [args]
```

### Adding New Models
1. **Model architecture**: `scripts/models/new_model_type/`
2. **Training script**: `scripts/training/new_model_type/`
3. **Evaluation (if needed)**: `scripts/evaluation/new_model_type/`

See `scripts/STRUCTURE.md` for detailed documentation.

## Quick Command Reference

```bash
# Complete workflow with new structure
docker-compose up -d
bash extract_all_bags_new.sh
python3 scripts/quick_commands.py train --type baseline
python3 scripts/quick_commands.py eval --model output/models/model_X --type quick

# View results
python3 scripts/quick_commands.py list
ls output/sequential_results/
```