#!/bin/bash

# Script to visualize results after training

echo "=================================="
echo "VISUALIZING YOUR MODEL RESULTS"
echo "=================================="

# Find the latest model
LATEST_MODEL=$(find /app/output/models -name "model_*" -type d | sort -r | head -n 1)

if [ -z "$LATEST_MODEL" ]; then
    echo "No trained models found!"
    echo "Please run training first."
    exit 1
fi

echo "Found latest model: $LATEST_MODEL"

# Option 1: Full evaluation with plots
echo -e "\n1. Running full evaluation..."
docker-compose exec underwater-vo python3 /app/scripts/evaluate_latest_model.py

# Show where results are saved
echo -e "\n=================================="
echo "RESULTS LOCATION"
echo "=================================="
echo "Your results are saved in:"
echo "$LATEST_MODEL/evaluation/"
echo ""
echo "Files generated:"
echo "  📊 performance_summary.png - Overall model performance"
echo "  📈 training_history.png - Training curves"
echo "  🗺️ trajectory_3d_*.png - 3D trajectory comparisons"
echo "  📍 trajectory_xy_*.png - 2D trajectory views"
echo "  📉 errors_*.png - Error plots over time"
echo "  🖼️ sample_predictions/ - Sample predictions on images"
echo ""
echo "To view these files:"
echo "1. From your host machine, navigate to:"
echo "   $(pwd)/output/models/$(basename $LATEST_MODEL)/evaluation/"
echo ""
echo "2. Or copy them out of the container:"
echo "   docker cp underwater_vo:$LATEST_MODEL/evaluation/ ./results/"

# Option 2: Quick interactive visualization (optional)
echo -e "\n=================================="
echo "INTERACTIVE VISUALIZATION"
echo "=================================="
echo "To see live predictions on image sequences, run:"
echo "docker-compose exec underwater-vo python3 /app/scripts/quick_visualize.py --model $LATEST_MODEL/best_model.h5"