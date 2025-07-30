#!/bin/bash

# Build and run the underwater visual odometry project with data preparation

echo "==================================="
echo "UNDERWATER VISUAL ODOMETRY PIPELINE"
echo "==================================="

# Build Docker image
echo -e "\n1. Building Docker image..."
docker-compose build

# Start container
echo -e "\n2. Starting container..."
docker-compose up -d

# Wait for container to be ready
sleep 2

# Prepare data from ROS bag and TUM file
echo -e "\n3. Preparing data from ROS bag and TUM trajectory..."
docker-compose exec underwater-vo bash -c "source /opt/ros/noetic/setup.bash && python3 /app/scripts/prepare_underwater_data.py --image_topic /alphasense_driver_ros/cam0 --pose_topic /qualisys/ariel/odom --skip_extraction"

# Check if data preparation was successful
if [ $? -eq 0 ]; then
    echo -e "\n✓ Data preparation successful!"
    
    # Run tests
    echo -e "\n4. Running system tests..."
    echo "   - Testing data loader..."
    docker-compose exec underwater-vo python3 /app/scripts/data_loader.py
    
    echo "   - Testing coordinate transformations..."
    docker-compose exec underwater-vo python3 /app/scripts/coordinate_transforms.py
    
    echo "   - Testing model creation..."
    docker-compose exec underwater-vo python3 /app/scripts/models/baseline_cnn.py
    
    # Start training
    echo -e "\n5. Starting baseline training..."
    docker-compose exec underwater-vo python3 /app/scripts/train_baseline.py \
        --data_dir /app/data/raw \
        --output_dir /app/output/models \
        --model_type baseline \
        --batch_size 32 \
        --epochs 50 \
        --learning_rate 0.001 \
        --loss_type huber
    
    echo -e "\n✓ Pipeline complete!"
    echo -e "\nTo monitor training with TensorBoard:"
    echo "docker-compose exec underwater-vo tensorboard --logdir=/app/output/models --host=0.0.0.0"
    echo "Then open http://localhost:6006"
    
    echo -e "\nTo view the dataset statistics:"
    echo "Check /app/data/raw/dataset_statistics.png"
    
else
    echo -e "\n✗ Data preparation failed!"
    echo "Check the logs above for errors."
    echo -e "\nPossible issues:"
    echo "1. ROS bag might have different topic names"
    echo "2. Time synchronization issues between bag and TUM file"
    echo -e "\nTo debug, run the data preparation with specific topics:"
    echo "docker-compose exec underwater-vo python3 /app/scripts/extract_rosbag_data.py --bag /app/data/raw/ariel_2023-12-21-14-26-32_2.bag"
fi