#!/bin/bash

echo "Starting image extraction from all ROS bags..."

# Create main output directory
mkdir -p data/images

# Extract from all bags
echo "Extracting images from all 5 bag files..."
python3 scripts/extract_images_only.py --input_dir data/raw --output_dir data/images

echo "Image extraction complete!"

# Show statistics
echo ""
echo "=== EXTRACTION SUMMARY ==="
for cam in cam0 cam1 cam2 cam3 cam4; do
    count=$(find data/images/$cam -name "*.jpg" 2>/dev/null | wc -l)
    echo "$cam: $count images"
done

total=$(find data/images -name "*.jpg" 2>/dev/null | wc -l)
echo "Total: $total images"

# Show disk usage
echo ""
echo "Disk usage:"
du -sh data/images