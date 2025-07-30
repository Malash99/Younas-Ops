#!/bin/bash

# Extract all 5 ROS bags using Docker
echo "Starting extraction of all 5 ROS bags..."

# Array of bag files
bags=(
    "ariel_2023-12-21-14-24-42_0.bag"
    "ariel_2023-12-21-14-25-37_1.bag"
    "ariel_2023-12-21-14-26-32_2.bag"
    "ariel_2023-12-21-14-27-27_3.bag"
    "ariel_2023-12-21-14-28-22_4.bag"
)

# Extract each bag
for i in "${!bags[@]}"; do
    bag="${bags[$i]}"
    echo "============================================================"
    echo "Processing bag $i: $bag"
    echo "============================================================"
    
    docker exec underwater_vo bash -c "source /opt/ros/noetic/setup.bash && cd /app && python3 scripts/extract_rosbag_data.py --bag data/raw/$bag --output data/processed/bag_$i"
    
    if [ $? -eq 0 ]; then
        echo "Successfully extracted bag $i"
    else
        echo "Error extracting bag $i"
    fi
    echo ""
done

echo "============================================================"
echo "Extraction complete! Check data/processed/ for results"
echo "============================================================"