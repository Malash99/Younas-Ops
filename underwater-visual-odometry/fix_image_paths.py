#!/usr/bin/env python3
"""
Fix image paths in training data CSV
"""

import pandas as pd
import os
from pathlib import Path


def fix_paths():
    """Fix image paths to be relative to project root"""
    
    training_csv = 'data/processed/training_dataset/training_data.csv'
    df = pd.read_csv(training_csv)
    
    print(f"Loaded {len(df)} samples")
    
    # Fix paths for each camera
    for cam_id in range(5):
        cam_col = f'cam{cam_id}_path'
        if cam_col in df.columns:
            # Replace the path prefix
            df[cam_col] = df[cam_col].str.replace(
                '../sequential_frames/', 
                'data/sequential_frames/', 
                regex=False
            )
    
    # Save fixed CSV
    df.to_csv(training_csv, index=False)
    print(f"Fixed paths in {training_csv}")
    
    # Test a few samples
    print("\nTesting fixed paths:")
    for i in range(3):
        sample = df.iloc[i]
        cam0_path = sample['cam0_path']
        if pd.notna(cam0_path):
            full_path = Path(cam0_path)
            exists = full_path.exists()
            print(f"Sample {i}: {cam0_path} -> {'EXISTS' if exists else 'NOT FOUND'}")


if __name__ == '__main__':
    fix_paths()