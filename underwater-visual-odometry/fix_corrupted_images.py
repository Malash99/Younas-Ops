#!/usr/bin/env python3
"""
Fix Corrupted Images in Dataset

This script identifies and handles corrupted images that are causing training issues.
"""

import os
import pandas as pd
from PIL import Image
from pathlib import Path
import shutil

def check_and_fix_corrupted_images(data_root, csv_path, backup=True):
    """
    Check for corrupted images and remove them from dataset.
    
    Args:
        data_root: Root directory containing images
        csv_path: Path to dataset CSV
        backup: Whether to backup corrupted files
    """
    print("Checking for corrupted images...")
    
    # Load dataset CSV
    df = pd.read_csv(csv_path)
    
    corrupted_files = []
    corrupted_rows = []
    
    # Create backup directory if needed
    backup_dir = None
    if backup:
        backup_dir = Path(data_root) / "corrupted_backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"Backup directory: {backup_dir}")
    
    # Check each image in the dataset
    for idx, row in df.iterrows():
        image_path = Path(data_root) / row['cam0_path']
        
        if not image_path.exists():
            print(f"Missing file: {image_path}")
            corrupted_rows.append(idx)
            continue
            
        try:
            # Try to open and verify the image
            with Image.open(image_path) as img:
                img.verify()  # This will raise an exception if corrupted
                
            # Try to load the image data
            with Image.open(image_path) as img:
                img.load()  # This will catch more corruption types
                
        except Exception as e:
            print(f"Corrupted: {image_path} - {e}")
            corrupted_files.append(str(image_path))
            corrupted_rows.append(idx)
            
            # Move to backup if requested
            if backup and backup_dir:
                backup_path = backup_dir / image_path.name
                try:
                    shutil.move(str(image_path), str(backup_path))
                    print(f"  -> Moved to backup: {backup_path}")
                except Exception as move_error:
                    print(f"  -> Could not move file: {move_error}")
    
    print(f"\nFound {len(corrupted_files)} corrupted images")
    print(f"Affected dataset rows: {len(corrupted_rows)}")
    
    if corrupted_rows:
        # Remove corrupted entries from CSV
        df_clean = df.drop(corrupted_rows).reset_index(drop=True)
        
        # Save cleaned CSV
        clean_csv_path = csv_path.replace('.csv', '_clean.csv')
        df_clean.to_csv(clean_csv_path, index=False)
        
        print(f"\nCleaned dataset saved: {clean_csv_path}")
        print(f"Original rows: {len(df)}")
        print(f"Clean rows: {len(df_clean)}")
        print(f"Removed rows: {len(corrupted_rows)}")
        
        return clean_csv_path, corrupted_files
    else:
        print("No corrupted images found!")
        return csv_path, []

def main():
    data_root = "data/processed/visual_odometry_dataset"
    csv_path = "data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv"
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    if not os.path.exists(data_root):
        print(f"Data root not found: {data_root}")
        return
    
    clean_csv, corrupted_files = check_and_fix_corrupted_images(data_root, csv_path, backup=True)
    
    print(f"\n{'='*60}")
    print("CORRUPTION FIX COMPLETE")
    print(f"{'='*60}")
    print(f"Clean CSV: {clean_csv}")
    print(f"Use this command to train with clean data:")
    print(f"python train_tsformer.py --csv_path {clean_csv} --sequence_length 8 --num_epochs 50 --batch_size 4")

if __name__ == "__main__":
    main()