# Scripts Folder Structure

This document describes the reorganized structure of the scripts folder for better organization and scalability when adding new models.

## Folder Organization

```
scripts/
├── data_processing/           # Data extraction and preprocessing
│   ├── __init__.py
│   ├── extract_rosbag_data.py     # Extract images/poses from ROS bags
│   ├── data_loader.py             # Dataset loading utilities
│   ├── prepare_underwater_data.py # Data preprocessing
│   └── process_tum_trajectory.py  # TUM format processing
│
├── models/                    # Model architectures
│   ├── __init__.py
│   ├── losses.py                  # Custom loss functions
│   ├── baseline/                  # Baseline CNN models
│   │   ├── __init__.py
│   │   └── baseline_cnn.py        # Baseline CNN architecture
│   └── attention/                 # Attention-based models
│       ├── __init__.py
│       └── [future attention models]
│
├── training/                  # Training scripts per model type
│   ├── __init__.py
│   ├── baseline/                  # Baseline model training
│   │   ├── __init__.py
│   │   └── train_baseline.py      # Train baseline CNN
│   └── attention/                 # Attention model training
│       ├── __init__.py
│       └── train_attention_window.py # Train attention model
│
├── evaluation/                # Model evaluation and analysis
│   ├── __init__.py
│   ├── common/                    # Shared evaluation tools
│   │   ├── __init__.py
│   │   ├── evaluate_model.py          # Comprehensive evaluation
│   │   ├── evaluate_sequential_bags.py # Sequential trajectory eval
│   │   ├── simple_trajectory_plot.py  # Quick trajectory plotting
│   │   ├── quick_evaluate.py          # Fast evaluation
│   │   ├── debug_trajectory.py        # Trajectory debugging
│   │   ├── visualize_*.py             # Visualization scripts
│   │   └── [other evaluation tools]
│   ├── baseline/                  # Baseline-specific evaluation
│   │   └── [future baseline-specific evaluations]
│   └── attention/                 # Attention-specific evaluation
│       └── [future attention-specific evaluations]
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── coordinate_transforms.py   # Pose transformations
│   └── visualization.py           # Plotting utilities
│
├── shell_scripts/             # Shell/batch scripts
│   ├── check_training_status.sh
│   ├── run_first_experiment.sh
│   ├── see_results.sh
│   └── [other automation scripts]
│
└── __init__.py                # Main scripts module
```

## Usage Patterns

### Adding a New Model Type (e.g., "transformer")

1. **Create model architecture:**
   ```
   models/transformer/
   ├── __init__.py
   └── transformer_vo.py
   ```

2. **Create training script:**
   ```
   training/transformer/
   ├── __init__.py
   └── train_transformer.py
   ```

3. **Add evaluation (if model-specific):**
   ```
   evaluation/transformer/
   ├── __init__.py
   └── transformer_specific_eval.py
   ```

### Import Patterns

From any script, use these import patterns:

```python
# Data processing
from data_processing.data_loader import UnderwaterVODataset
from data_processing.extract_rosbag_data import RosbagExtractor

# Models
from models.baseline.baseline_cnn import create_model
from models.losses import HuberPoseLoss

# Utils
from utils.coordinate_transforms import integrate_trajectory
from utils.visualization import plot_trajectory_3d

# Evaluation tools
from evaluation.common.evaluate_model import ModelEvaluator
```

## Key Scripts by Category

### Data Processing
- **`extract_rosbag_data.py`**: Extract images and poses from ROS bags
- **`data_loader.py`**: Dataset class for training/evaluation
- **`prepare_underwater_data.py`**: Data preprocessing utilities

### Model Training
- **`training/baseline/train_baseline.py`**: Train baseline CNN model
- **`training/attention/train_attention_window.py`**: Train attention-based model

### Model Evaluation
- **`evaluation/common/simple_trajectory_plot.py`**: Quick trajectory visualization
- **`evaluation/common/evaluate_model.py`**: Comprehensive model evaluation  
- **`evaluation/common/evaluate_sequential_bags.py`**: Sequential bag evaluation

### Utilities
- **`utils/coordinate_transforms.py`**: SE(3) transformations and trajectory integration
- **`utils/visualization.py`**: Plotting functions for trajectories and errors

## Running Scripts with New Structure

### From Docker Container Root (/app):

```bash
# Data extraction
python3 scripts/data_processing/extract_rosbag_data.py --bag data/raw/bag.bag

# Train baseline model
python3 scripts/training/baseline/train_baseline.py --epochs 50

# Train attention model  
python3 scripts/training/attention/train_attention_window.py --window_size 5

# Evaluate model
python3 scripts/evaluation/common/simple_trajectory_plot.py --model_dir output/models/model_X

# Comprehensive evaluation
python3 scripts/evaluation/common/evaluate_model.py --model_dir output/models/model_X
```

## Benefits of This Structure

1. **Scalability**: Easy to add new model types without cluttering
2. **Organization**: Clear separation of concerns (data/training/evaluation/utils)
3. **Modularity**: Each model type has its own training/evaluation space
4. **Maintainability**: Related functionality is grouped together
5. **Import Clarity**: Clear import paths that show dependencies

## Migration Notes

- All import paths have been updated to use the new structure
- Shell scripts moved to `shell_scripts/` folder
- Common evaluation tools remain in `evaluation/common/`
- Model-specific evaluation can be added to `evaluation/{model_type}/`
- All folders have `__init__.py` for proper Python module structure