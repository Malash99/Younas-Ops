#!/bin/bash

# Script to run the full data extraction

echo "Starting data extraction from ROS bags..."

# First, test on one bag
echo "Testing on single bag..."
python3 scripts/extract_all_data.py --input_dir data/raw --output_dir data/processed/single_bag_test --max_bags 1

# If successful, run on all bags
if [ $? -eq 0 ]; then
    echo "Single bag test successful! Processing all bags..."
    python3 scripts/extract_all_data.py --input_dir data/raw --output_dir data/processed/full_dataset
else
    echo "Single bag test failed. Check the script."
    exit 1
fi

echo "Extraction complete!"