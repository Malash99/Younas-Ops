#!/bin/bash

# Build and run the underwater visual odometry project

echo "Building Docker image..."
docker-compose build

echo -e "\nStarting container..."
docker-compose up -d

echo -e "\nRunning initial data exploration..."
docker-compose exec underwater-vo python3 /app/scripts/data_loader.py

echo -e "\nTesting coordinate transformations..."
docker-compose exec underwater-vo python3 /app/scripts/coordinate_transforms.py

echo -e "\nTesting model creation..."
docker-compose exec underwater-vo python3 /app/scripts/models/baseline_cnn.py

echo -e "\nTesting loss functions..."
docker-compose exec underwater-vo python3 /app/scripts/models/losses.py

echo -e "\nStarting baseline training with default parameters..."
docker-compose exec underwater-vo python3 /app/scripts/train_baseline.py \
    --data_dir /app/data/raw \
    --output_dir /app/output/models \
    --model_type baseline \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --loss_type huber

echo -e "\nTraining complete! Check /app/output/models for results."
echo "To monitor training with TensorBoard, run:"
echo "docker-compose exec underwater-vo tensorboard --logdir=/app/output/models --host=0.0.0.0"