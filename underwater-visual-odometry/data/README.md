# Underwater Visual Odometry Dataset

## Overview

This dataset contains underwater visual odometry data collected from the "Ariel" underwater vehicle system on December 21, 2023. The dataset includes synchronized multi-camera imagery, ground truth trajectory data, and processed sequential frame sequences suitable for visual odometry research and deep learning applications.

## Dataset Summary

- **Total Duration**: 273.99 seconds (~4.6 minutes)
- **Total Frames**: 5,481 synchronized frame sequences
- **Total Images**: 27,399 individual camera images
- **Camera Array**: 5 cameras (cam0-cam4) with near-perfect coverage
- **Data Collection Date**: December 21, 2023, 14:24-14:28 UTC
- **Platform**: Ariel underwater vehicle
- **Ground Truth**: Qualisys motion capture system

## Data Structure

```
data/
├── raw/                           # Original ROS bag files and ground truth
│   ├── ariel_2023-12-21-14-24-42_0.bag
│   ├── ariel_2023-12-21-14-25-37_1.bag
│   ├── ariel_2023-12-21-14-26-32_2.bag
│   ├── ariel_2023-12-21-14-27-27_3.bag
│   ├── ariel_2023-12-21-14-28-22_4.bag
│   └── qualisys_ariel_odom_traj_8_id6.tum
├── sequential_frames/             # Processed synchronized frame sequences
│   ├── data/
│   │   ├── frame_sequence.csv     # Frame metadata and synchronization
│   │   └── sequence_metadata.json # Dataset statistics and configuration
│   └── images/                    # Extracted and organized images
│       ├── cam0/                  # Camera 0 images (5,477 frames, 99.9% coverage)
│       ├── cam1/                  # Camera 1 images (5,479 frames, 100.0% coverage)
│       ├── cam2/                  # Camera 2 images (5,481 frames, 100.0% coverage)
│       ├── cam3/                  # Camera 3 images (5,480 frames, 100.0% coverage)
│       └── cam4/                  # Camera 4 images (5,477 frames, 99.9% coverage)
├── processed/                     # Additional processed data
│   ├── quick_test/               # Small test dataset (10 frames)
│   └── test_extraction/          # Test extraction results
└── test_images/                  # Sample images from cam0 (340 images)
```

## Data Formats

### 1. Raw Data
- **ROS Bags**: 5 bag files containing synchronized camera data and navigation information
- **Ground Truth**: TUM format trajectory file with pose information (timestamp, tx, ty, tz, qx, qy, qz, qw)

### 2. Sequential Frame Data
- **frame_sequence.csv**: Contains synchronized frame metadata with columns:
  - `global_frame_id`: Unique frame identifier across all bags
  - `bag_name`: Source bag file name
  - `bag_index`: Bag file index (0-4)
  - `bag_frame_id`: Frame index within specific bag
  - `timestamp`: Global timestamp
  - `num_cameras`: Number of cameras with data for this frame
  - `cam[0-4]_path`: Image file paths for each camera
  - `cam[0-4]_timestamp`: Individual camera timestamps

### 3. Image Data
- **Format**: JPEG images
- **Naming Convention**: `frame_XXXXXX_camX.jpg`
- **Organization**: Separated by camera (cam0-cam4)
- **Resolution**: Standard underwater camera resolution
- **Total Images**: 27,399 across all cameras

### 4. Ground Truth Trajectory
- **Format**: TUM trajectory format
- **Fields**: timestamp, tx, ty, tz, qx, qy, qz, qw
- **Coordinate System**: Standard robotics convention
- **Frequency**: High-frequency pose estimates from Qualisys system

## Camera Configuration

The dataset includes a 5-camera array with the following coverage statistics:

| Camera | Total Frames | Coverage | Notes |
|--------|-------------|----------|-------|
| cam0   | 5,477       | 99.9%    | Minor frame losses |
| cam1   | 5,479       | 100.0%   | Complete coverage |
| cam2   | 5,481       | 100.0%   | Complete coverage |
| cam3   | 5,480       | 100.0%   | Complete coverage |
| cam4   | 5,477       | 99.9%    | Minor frame losses |

## Temporal Information

- **Start Time**: 1703165082.986978 (Unix timestamp)
- **End Time**: 1703165356.9790084 (Unix timestamp)
- **Duration**: 273.99 seconds
- **Average Frame Rate**: ~20 Hz across all cameras
- **Synchronization**: Sub-millisecond precision between cameras

## Data Collection Details

### Recording Sessions (Bags)
1. **Bag 0**: ariel_2023-12-21-14-24-42_0 (1,096 frames)
2. **Bag 1**: ariel_2023-12-21-14-25-37_1 (1,096 frames)
3. **Bag 2**: ariel_2023-12-21-14-26-32_2 (1,097 frames)
4. **Bag 3**: ariel_2023-12-21-14-27-27_3 (1,096 frames)
5. **Bag 4**: ariel_2023-12-21-14-28-22_4 (1,096 frames)

### Platform Specifications
- **Vehicle**: Ariel underwater robot
- **Environment**: Underwater
- **Motion Capture**: Qualisys system for ground truth
- **Sensors**: Multi-camera stereo array

## Usage Guidelines

### For Visual Odometry Research
1. Use `frame_sequence.csv` to load synchronized multi-camera sequences
2. Ground truth poses available in `qualisys_ariel_odom_traj_8_id6.tum`
3. Images organized by camera for easy stereo/multi-view processing

### For Deep Learning
1. Sequential frame data provides temporal relationships
2. High frame rate suitable for motion estimation
3. Multi-camera setup enables various stereo configurations

### For Testing
- Use `processed/quick_test/` for rapid algorithm validation
- Contains 10-frame subset with pose deltas pre-calculated

## Data Quality

- **Synchronization**: High precision temporal alignment across cameras
- **Coverage**: Near-perfect frame coverage (>99.9% for all cameras)
- **Ground Truth**: High-accuracy motion capture reference
- **Image Quality**: Standard underwater imaging conditions

## File Size Information

- **Raw Bags**: ~5 GB total
- **Extracted Images**: ~27,399 JPEG files
- **Metadata**: CSV and JSON files for easy parsing
- **Total Dataset**: Approximately 6-8 GB

## Citation

If you use this dataset in your research, please cite the associated work and acknowledge the data collection from the Ariel underwater vehicle platform.

## License

Please refer to the project documentation for licensing terms and usage restrictions.

---

*Dataset processed and organized for underwater visual odometry research. For questions or issues, please refer to the project documentation.*