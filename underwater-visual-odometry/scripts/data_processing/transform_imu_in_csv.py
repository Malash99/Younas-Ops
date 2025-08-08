#!/usr/bin/env python3
"""
Transform IMU data in existing CSV from old transformation to Kalibr cam_0 calibration.

This script reads the existing visual_odometry_dataset.csv, applies the correct
Kalibr transformation to the IMU columns, and saves the corrected dataset.

Author: Underwater Visual Odometry Research Team
Date: January 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def get_kalibr_cam0_transformation():
    """
    Get the Kalibr calibration transformation matrix for cam_0.
    
    From ReaqrVIO Kalibr calibration:
    - qCM (quaternion): [-0.5000, 0.5024, -0.5002, -0.4974]  
    - MrMC (translation): [0.0482, -0.0097, -0.0506]
    
    Returns:
        4x4 transformation matrix from IMU to camera frame
    """
    # Quaternion to rotation matrix conversion
    qx, qy, qz, qw = -0.5000, 0.5024, -0.5002, -0.4974
    
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    # Translation vector  
    t = np.array([0.0482, -0.0097, -0.0506])
    
    # Build 4x4 transformation matrix
    T_camera_imu = np.eye(4)
    T_camera_imu[:3, :3] = R
    T_camera_imu[:3, 3] = t
    
    return T_camera_imu

def get_old_transformation():
    """
    Get the old "standard_rov" transformation matrix.
    
    Returns:
        4x4 transformation matrix from old method
    """
    return np.array([
        [0, 1, 0, 0],  # Camera X = IMU Y (right)
        [0, 0, 1, 0],  # Camera Y = IMU Z (down)  
        [1, 0, 0, 0],  # Camera Z = IMU X (forward)
        [0, 0, 0, 1]
    ])

def transform_imu_data(df):
    """
    Transform IMU data from old transformation to Kalibr calibration.
    
    Args:
        df: DataFrame with IMU columns (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
        
    Returns:
        DataFrame with corrected IMU data
    """
    print("Transforming IMU data from old transformation to Kalibr cam_0 calibration...")
    
    # Get transformation matrices
    T_new = get_kalibr_cam0_transformation()  # Kalibr calibration
    T_old = get_old_transformation()          # Old transformation
    
    # Extract rotation matrices (3x3)
    R_new = T_new[:3, :3]
    R_old = T_old[:3, :3]
    
    # Inverse of old transformation to get back to IMU frame
    R_old_inv = R_old.T  # For rotation matrices, inverse = transpose
    
    # Combined transformation: IMU -> old_camera -> IMU -> new_camera
    R_combined = R_new @ R_old_inv
    
    print(f"Old transformation (standard_rov):")
    print(R_old)
    print(f"\nNew transformation (Kalibr cam_0):")
    print(R_new)
    print(f"\nCombined transformation matrix:")
    print(R_combined)
    
    # Create a copy of the dataframe
    df_corrected = df.copy()
    
    # Transform IMU data for each row
    imu_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Check if IMU columns exist
    missing_columns = [col for col in imu_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing IMU columns in dataset: {missing_columns}")
    
    print(f"\nTransforming {len(df)} rows of IMU data...")
    
    for idx, row in df.iterrows():
        # Get original IMU data (in old camera frame)
        accel_old = np.array([row['accel_x'], row['accel_y'], row['accel_z']])
        gyro_old = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])
        
        # Apply combined transformation to get new camera frame
        accel_new = R_combined @ accel_old
        gyro_new = R_combined @ gyro_old
        
        # Update the dataframe
        df_corrected.loc[idx, 'accel_x'] = accel_new[0]
        df_corrected.loc[idx, 'accel_y'] = accel_new[1] 
        df_corrected.loc[idx, 'accel_z'] = accel_new[2]
        df_corrected.loc[idx, 'gyro_x'] = gyro_new[0]
        df_corrected.loc[idx, 'gyro_y'] = gyro_new[1]
        df_corrected.loc[idx, 'gyro_z'] = gyro_new[2]
    
    return df_corrected

def main():
    """Main function to transform IMU data in CSV."""
    parser = argparse.ArgumentParser(description="Transform IMU data in CSV to use Kalibr calibration")
    parser.add_argument("--input_csv", type=str, 
                       default="data/processed/visual_odometry_dataset/visual_odometry_dataset.csv",
                       help="Input CSV file path")
    parser.add_argument("--output_csv", type=str,
                       default="data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr.csv", 
                       help="Output CSV file path")
    
    args = parser.parse_args()
    
    print("="*80)
    print("IMU TRANSFORMATION - CSV UPDATE")
    print("="*80)
    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    
    # Load the existing CSV
    print(f"\nLoading existing dataset...")
    if not Path(args.input_csv).exists():
        raise FileNotFoundError(f"Input CSV file not found: {args.input_csv}")
        
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows from dataset")
    
    # Show sample of old IMU data
    print(f"\nSample of OLD IMU data (first 3 rows):")
    imu_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    print(df[imu_cols].head(3).to_string())
    
    # Transform the IMU data
    df_corrected = transform_imu_data(df)
    
    # Show sample of new IMU data
    print(f"\nSample of NEW IMU data (first 3 rows):")
    print(df_corrected[imu_cols].head(3).to_string())
    
    # Show difference magnitude
    print(f"\nMagnitude of changes (first 3 rows):")
    for i in range(min(3, len(df))):
        old_accel = np.array([df.iloc[i]['accel_x'], df.iloc[i]['accel_y'], df.iloc[i]['accel_z']])
        new_accel = np.array([df_corrected.iloc[i]['accel_x'], df_corrected.iloc[i]['accel_y'], df_corrected.iloc[i]['accel_z']])
        diff_accel = np.linalg.norm(new_accel - old_accel)
        
        old_gyro = np.array([df.iloc[i]['gyro_x'], df.iloc[i]['gyro_y'], df.iloc[i]['gyro_z']])
        new_gyro = np.array([df_corrected.iloc[i]['gyro_x'], df_corrected.iloc[i]['gyro_y'], df_corrected.iloc[i]['gyro_z']])
        diff_gyro = np.linalg.norm(new_gyro - old_gyro)
        
        print(f"Row {i}: Accel diff = {diff_accel:.6f}, Gyro diff = {diff_gyro:.6f}")
    
    # Save the corrected dataset
    print(f"\nSaving corrected dataset to {args.output_csv}...")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_corrected.to_csv(args.output_csv, index=False, float_format='%.6f')
    
    print(f"\nTransformation complete!")
    print(f"Original dataset: {args.input_csv}")
    print(f"Corrected dataset: {args.output_csv}")
    print(f"IMU data now uses proper Kalibr cam_0 calibration.")

if __name__ == "__main__":
    main()