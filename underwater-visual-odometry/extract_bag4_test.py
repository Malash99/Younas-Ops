#!/usr/bin/env python3
"""
Extract ROS bag 4 for testing the corrected model on unseen data.
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from data_processing.extract_rosbag_data import extract_rosbag_data


def extract_bag4_for_testing():
    """Extract bag 4 for testing purposes."""
    print("="*80)
    print("EXTRACTING ROS BAG 4 FOR UNSEEN DATA TESTING")
    print("="*80)
    
    # Configuration
    bag_path = "./data/raw/ariel_2023-12-21-14-28-22_4.bag"
    output_dir = "./data/bag4_test"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    if not os.path.exists(bag_path):
        print(f"Error: ROS bag not found at {bag_path}")
        return
    
    print(f"Extracting from: {bag_path}")
    print(f"Output directory: {output_dir}")
    
    # Extract data
    try:
        print("Starting extraction...")
        extract_rosbag_data(
            bag_path=bag_path,
            output_dir=output_dir,
            image_topic="/alphasense/cam0/image",
            pose_topic="/qualisys/ariel/odom",
            max_images=None  # Extract all images
        )
        
        print("Extraction completed successfully!")
        
        # Check extracted data
        if os.path.exists(os.path.join(output_dir, 'ground_truth.csv')):
            gt_df = pd.read_csv(os.path.join(output_dir, 'ground_truth.csv'))
            print(f"Extracted {len(gt_df)} ground truth poses")
            
            # Count images
            image_files = [f for f in os.listdir(os.path.join(output_dir, 'images')) 
                          if f.endswith('.png')]
            print(f"Extracted {len(image_files)} images")
            
            if len(gt_df) > 0:
                print(f"Time duration: {gt_df['timestamp'].max() - gt_df['timestamp'].min():.2f} seconds")
                print("Data ready for testing!")
            else:
                print("Warning: No ground truth poses extracted")
        else:
            print("Warning: ground_truth.csv not found")
            
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False
    
    return True


if __name__ == "__main__":
    extract_bag4_for_testing()