#!/bin/bash

echo "=================================="
echo "CHECKING TRAINING STATUS"
echo "=================================="

# Make sure container is running
if ! docker-compose ps | grep -q "underwater_vo.*Up"; then
    echo "Starting Docker container..."
    docker-compose up -d
    sleep 3
fi

echo -e "\n1. Checking for trained models..."
echo "-----------------------------------"
docker-compose exec underwater-vo bash -c "ls -la /app/output/models/" || echo "No models directory found"

echo -e "\n2. Checking data preparation..."
echo "-----------------------------------"
docker-compose exec underwater-vo bash -c "ls -la /app/data/raw/*.csv 2>/dev/null | head -5" || echo "No processed data found"
docker-compose exec underwater-vo bash -c "ls /app/data/raw/images/*.png 2>/dev/null | wc -l | xargs echo 'Number of extracted images:'" || echo "No images found"

echo -e "\n3. Checking latest logs..."
echo "-----------------------------------"
docker-compose logs --tail=20 underwater-vo

echo -e "\n=================================="
echo "WHAT TO DO NEXT:"
echo "=================================="

# Check if data exists
if docker-compose exec underwater-vo test -f /app/data/raw/ground_truth.csv; then
    echo "✅ Data is prepared"
    
    # Check if models exist
    if docker-compose exec underwater-vo bash -c "ls /app/output/models/model_*/best_model.h5 2>/dev/null" > /dev/null; then
        echo "✅ Models are trained"
        echo ""
        echo "To see results, run:"
        echo "  ./see_results_windows.sh"
    else
        echo "❌ No trained models found"
        echo ""
        echo "To train a model, run:"
        echo "  docker-compose exec underwater-vo python3 /app/scripts/train_baseline.py"
    fi
else
    echo "❌ Data not prepared"
    echo ""
    echo "To prepare data, run:"
    echo "  docker-compose exec underwater-vo python3 /app/scripts/prepare_underwater_data.py"
fi