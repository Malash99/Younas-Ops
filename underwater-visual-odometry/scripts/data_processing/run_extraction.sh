#!/bin/bash

# Professional Data Extraction Pipeline
# Underwater Visual Odometry Dataset Generation
#
# This script runs the complete data extraction pipeline to process
# ROS bag files and generate a training-ready visual odometry dataset.
#
# Usage:
#   ./scripts/data_processing/run_extraction.sh
#   
# Or with custom parameters:
#   ./scripts/data_processing/run_extraction.sh --bag_dir custom/path --output_dir custom/output

set -e  # Exit on any error

echo "=============================================================================="
echo "UNDERWATER VISUAL ODOMETRY - DATA EXTRACTION PIPELINE"
echo "=============================================================================="

# Default parameters
BAG_DIR="data/raw"
OUTPUT_DIR="data/processed/visual_odometry_dataset" 
CSV_NAME="visual_odometry_dataset.csv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bag_dir)
            BAG_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --csv_name)
            CSV_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --bag_dir DIR     Directory containing ROS bag files (default: data/raw)"
            echo "  --output_dir DIR  Output directory for dataset (default: data/processed/visual_odometry_dataset)"
            echo "  --csv_name NAME   Name of output CSV file (default: visual_odometry_dataset.csv)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Bag directory: $BAG_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  CSV filename: $CSV_NAME"
echo ""

# Check if bag directory exists
if [ ! -d "$BAG_DIR" ]; then
    echo "❌ Error: Bag directory '$BAG_DIR' does not exist"
    exit 1
fi

# Check for bag files
BAG_COUNT=$(find "$BAG_DIR" -name "*.bag" | wc -l)
if [ $BAG_COUNT -eq 0 ]; then
    echo "❌ Error: No .bag files found in '$BAG_DIR'"
    exit 1
fi

echo "📁 Found $BAG_COUNT bag files in $BAG_DIR"

# Check Python dependencies
echo "🔍 Checking Python dependencies..."
python3 -c "import rosbag, cv2, pandas, numpy, scipy, tf" 2>/dev/null || {
    echo "❌ Error: Missing required Python packages"
    echo "Please install: rosbag, opencv-python, pandas, numpy, scipy, tf"
    echo "Run: pip install rosbag opencv-python pandas numpy scipy tf"
    exit 1
}
echo "✅ All dependencies available"

# Create output directory
echo "📂 Creating output directory structure..."
mkdir -p "$OUTPUT_DIR"

# Run the extraction
echo ""
echo "🚀 Starting data extraction pipeline..."
echo "This may take several minutes depending on bag file sizes..."

python3 scripts/data_processing/extract_visual_odometry_dataset.py \
    --bag_dir "$BAG_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --csv_name "$CSV_NAME"

# Check if extraction was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 EXTRACTION COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📊 Generated files:"
    echo "  - Dataset CSV: $OUTPUT_DIR/$CSV_NAME"
    echo "  - Statistics: $OUTPUT_DIR/dataset_statistics.json"
    echo "  - Images: $OUTPUT_DIR/images/"
    echo ""
    echo "📈 Dataset summary:"
    if [ -f "$OUTPUT_DIR/dataset_statistics.json" ]; then
        python3 -c "
import json
with open('$OUTPUT_DIR/dataset_statistics.json', 'r') as f:
    stats = json.load(f)
print(f\"  - Total frames: {stats.get('total_frames', 'N/A')}\")
print(f\"  - Frames with ground truth: {stats.get('frames_with_ground_truth', 'N/A')}\")
print(f\"  - Unique bags: {stats.get('unique_bags', 'N/A')}\")
"
    fi
    echo ""
    echo "🔬 Next steps:"
    echo "  1. Verify dataset quality: head $OUTPUT_DIR/$CSV_NAME"
    echo "  2. Check images: ls $OUTPUT_DIR/images/cam0/ | head"
    echo "  3. Start training your visual odometry model!"
    echo ""
else
    echo "❌ Extraction failed. Check error messages above."
    exit 1
fi