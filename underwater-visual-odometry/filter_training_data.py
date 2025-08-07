#!/usr/bin/env python3
"""
Filter problematic samples from training data
Remove outliers that cause NaN gradients
"""

import pandas as pd
import numpy as np

def filter_problematic_samples():
    """Remove samples that cause training instability"""
    
    # Load original data
    df = pd.read_csv('data/processed/training_dataset/training_data.csv')
    print(f"Original dataset: {len(df)} samples")
    
    # Define pose columns
    pose_cols = ['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw']
    
    # Identify problematic samples
    problematic_samples = set()
    
    # Method 1: Remove extreme outliers (>4 sigma)
    for col in pose_cols:
        threshold = df[col].std() * 4  # 4 sigma threshold
        outliers = df[abs(df[col]) > threshold].index
        problematic_samples.update(outliers)
        print(f"{col}: {len(outliers)} outliers (>{threshold:.6f})")
    
    # Method 2: Remove sudden jumps (top 1%)
    for col in pose_cols:
        diff = df[col].diff().abs()
        jump_threshold = diff.quantile(0.99)  # Top 1% of changes
        jumps = df[diff > jump_threshold].index
        problematic_samples.update(jumps)
        print(f"{col}: {len(jumps)} sudden jumps")
    
    # Method 3: Remove samples with very large accumulated errors
    # These can cause gradient explosion in long sequences
    accumulated_drift = np.sqrt(
        df['delta_x'].cumsum()**2 + 
        df['delta_y'].cumsum()**2 + 
        df['delta_z'].cumsum()**2
    )
    
    # Find samples where accumulated drift suddenly increases
    drift_diff = accumulated_drift.diff()
    large_drift_jumps = df[drift_diff > drift_diff.quantile(0.995)].index
    problematic_samples.update(large_drift_jumps)
    print(f"Large drift jumps: {len(large_drift_jumps)}")
    
    print(f"\nTotal problematic samples: {len(problematic_samples)}")
    print(f"Sample IDs: {sorted(list(problematic_samples))[:20]}...")  # Show first 20
    
    # Create clean dataset
    clean_df = df.drop(index=problematic_samples).reset_index(drop=True)
    clean_df['sample_id'] = range(len(clean_df))  # Reset sample IDs
    
    print(f"\nFiltered dataset: {len(clean_df)} samples")
    print(f"Removed: {len(df) - len(clean_df)} samples ({100*(len(df)-len(clean_df))/len(df):.1f}%)")
    
    # Save filtered dataset
    clean_df.to_csv('data/processed/training_dataset/training_data_filtered.csv', index=False)
    print(f"Saved: training_data_filtered.csv")
    
    # Verify the cleaned data
    print(f"\nCleaned data statistics:")
    for col in pose_cols:
        print(f"  {col}: [{clean_df[col].min():.6f}, {clean_df[col].max():.6f}] (std: {clean_df[col].std():.6f})")
    
    return len(problematic_samples)

if __name__ == '__main__':
    removed_count = filter_problematic_samples()
    print(f"\n✅ Data filtering complete! Removed {removed_count} problematic samples.")
    print("Now use 'training_data_filtered.csv' for stable training.")