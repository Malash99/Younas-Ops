#!/usr/bin/env python3
"""
Simple test without emojis for Windows
"""

import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path


def main():
    print("UW-TransVO Data Loading Tests")
    print("=" * 50)
    
    # Test 1: Load training data
    training_csv = 'data/processed/training_dataset/training_data.csv'
    print(f"Loading training data from: {training_csv}")
    
    if not os.path.exists(training_csv):
        print(f"ERROR: Training data not found: {training_csv}")
        return
    
    df = pd.read_csv(training_csv)
    print(f"SUCCESS: Loaded {len(df)} training samples")
    
    # Test 2: Check first few images
    print("\nTesting image loading...")
    
    for i in range(min(3, len(df))):
        sample = df.iloc[i]
        print(f"\nSample {i}:")
        
        for cam_id in range(4):
            cam_col = f'cam{cam_id}_path'
            if pd.notna(sample[cam_col]):
                img_path = sample[cam_col]
                full_path = Path(img_path)  # Path is already relative to project root
                
                if full_path.exists():
                    img = cv2.imread(str(full_path))
                    if img is not None:
                        print(f"  cam{cam_id}: OK ({img.shape})")
                    else:
                        print(f"  cam{cam_id}: FAILED to load")
                else:
                    print(f"  cam{cam_id}: File not found")
            else:
                print(f"  cam{cam_id}: No path")
    
    # Test 3: Check pose data
    print("\nPose statistics:")
    pose_cols = ['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw']
    stats = df[pose_cols].describe()
    print(stats)
    
    # Test 4: Camera coverage
    print("\nCamera coverage:")
    for cam_id in range(5):
        cam_col = f'cam{cam_id}_path'
        if cam_col in df.columns:
            available = df[cam_col].notna().sum()
            coverage = (available / len(df)) * 100
            print(f"  cam{cam_id}: {available}/{len(df)} ({coverage:.1f}%)")
    
    # Test 5: Create splits
    print("\nCreating train/val/test splits...")
    bags = df['bag_name'].unique()
    bags = sorted(bags)
    print(f"Found bags: {bags}")
    
    # Simple split: bags 0-2 = train, bag 3 = val, bag 4 = test
    train_bags = bags[:3]
    val_bags = bags[3:4] if len(bags) > 3 else []
    test_bags = bags[4:] if len(bags) > 4 else []
    
    print(f"Train bags: {train_bags}")
    print(f"Val bags: {val_bags}")
    print(f"Test bags: {test_bags}")
    
    # Add split column
    df['split'] = 'train'
    if val_bags:
        df.loc[df['bag_name'].isin(val_bags), 'split'] = 'val'
    if test_bags:
        df.loc[df['bag_name'].isin(test_bags), 'split'] = 'test'
    
    # Save updated CSV
    df.to_csv(training_csv, index=False)
    
    split_counts = df['split'].value_counts()
    print("Split distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} samples")
    
    print("\n" + "=" * 50)
    print("SUCCESS: Data is ready for training!")
    print("Next: Install PyTorch and run training")


if __name__ == '__main__':
    main()