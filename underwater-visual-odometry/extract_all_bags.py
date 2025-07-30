#!/usr/bin/env python3
"""
Extract images and poses from all 5 ROS bag files.
"""

import os
import subprocess
import sys

def extract_all_bags():
    """Extract data from all 5 ROS bag files."""
    
    # Base paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    # Create processed directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # List of bag files
    bag_files = [
        "ariel_2023-12-21-14-24-42_0.bag",
        "ariel_2023-12-21-14-25-37_1.bag", 
        "ariel_2023-12-21-14-26-32_2.bag",
        "ariel_2023-12-21-14-27-27_3.bag",
        "ariel_2023-12-21-14-28-22_4.bag"
    ]
    
    for i, bag_file in enumerate(bag_files):
        print(f"\n{'='*60}")
        print(f"Processing bag {i}: {bag_file}")
        print(f"{'='*60}")
        
        # Paths
        bag_path = os.path.join(raw_dir, bag_file)
        output_dir = os.path.join(processed_dir, f"bag_{i}")
        
        # Check if bag exists
        if not os.path.exists(bag_path):
            print(f"Warning: Bag file not found: {bag_path}")
            continue
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run extraction
        cmd = [
            sys.executable, 
            os.path.join(base_dir, "scripts", "extract_rosbag_data.py"),
            "--bag", bag_path,
            "--output", output_dir
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Successfully extracted bag {i}")
            if result.stdout:
                print("Output:", result.stdout)
                
        except subprocess.CalledProcessError as e:
            print(f"Error extracting bag {i}: {e}")
            if e.stdout:
                print("Stdout:", e.stdout)
            if e.stderr:
                print("Stderr:", e.stderr)
                
        except Exception as e:
            print(f"Unexpected error processing bag {i}: {e}")
    
    print(f"\n{'='*60}")
    print("Extraction complete! Check data/processed/ for results")
    print(f"{'='*60}")

if __name__ == "__main__":
    extract_all_bags()