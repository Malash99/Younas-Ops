#!/bin/bash

# Script to visualize results after training - Windows compatible version

echo "=================================="
echo "VISUALIZING YOUR MODEL RESULTS"
echo "=================================="

# Check if docker-compose is running
if ! docker-compose ps | grep -q "underwater_vo.*Up"; then
    echo "Starting Docker container..."
    docker-compose up -d
    sleep 3
fi

# Run evaluation inside the container
echo -e "\nRunning evaluation inside Docker container..."
docker-compose exec underwater-vo python3 /app/scripts/evaluate_latest_model.py

# Get the latest model directory from inside container
LATEST_MODEL=$(docker-compose exec underwater-vo bash -c "find /app/output/models -name 'model_*' -type d | sort -r | head -n 1" | tr -d '\r')

if [ -z "$LATEST_MODEL" ]; then
    echo "No trained models found!"
    echo ""
    echo "To check if you have any models:"
    echo "  docker-compose exec underwater-vo ls -la /app/output/models/"
    echo ""
    echo "To train a model:"
    echo "  docker-compose exec underwater-vo python3 /app/scripts/train_baseline.py"
    exit 1
fi

echo -e "\n=================================="
echo "COPYING RESULTS TO YOUR COMPUTER"
echo "=================================="

# Create local results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_RESULTS_DIR="./results_${TIMESTAMP}"
mkdir -p "$LOCAL_RESULTS_DIR"

# Copy evaluation results from container to local machine
echo "Copying results from Docker container..."
docker cp "underwater_vo:${LATEST_MODEL}/evaluation" "$LOCAL_RESULTS_DIR/" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "\n✅ SUCCESS! Results copied to: $LOCAL_RESULTS_DIR/evaluation/"
    echo ""
    echo "📊 Files you can view:"
    ls -la "$LOCAL_RESULTS_DIR/evaluation/"*.png 2>/dev/null | awk '{print "  - " $NF}'
    echo ""
    echo "To view the results:"
    echo "  1. Open Windows Explorer"
    echo "  2. Navigate to: $(pwd)/$LOCAL_RESULTS_DIR/evaluation/"
    echo "  3. Double-click on any .png file to view"
    
    # Try to open the results folder in Windows Explorer
    if command -v explorer.exe &> /dev/null; then
        echo -e "\nOpening results folder..."
        explorer.exe "$(cygpath -w "$LOCAL_RESULTS_DIR/evaluation")" 2>/dev/null || true
    fi
else
    echo "⚠️  Warning: Could not find evaluation results."
    echo ""
    echo "The model might still be training or evaluation hasn't been run yet."
    echo ""
    echo "To check training status:"
    echo "  docker-compose logs -f underwater-vo"
fi