#!/usr/bin/env python3
"""
Find and evaluate the latest trained model.
"""

import os
import glob
import subprocess
from datetime import datetime


def find_latest_model(models_dir='/app/output/models'):
    """Find the most recently created model directory."""
    # Look for model directories with timestamp pattern
    model_dirs = glob.glob(os.path.join(models_dir, 'model_*'))
    
    if not model_dirs:
        print(f"No model directories found in {models_dir}")
        return None
    
    # Sort by modification time
    latest_model = max(model_dirs, key=os.path.getmtime)
    
    # Check if it contains model files
    required_files = ['best_model.h5', 'final_model.h5']
    has_model = any(os.path.exists(os.path.join(latest_model, f)) for f in required_files)
    
    if not has_model:
        print(f"No model weights found in {latest_model}")
        return None
    
    return latest_model


def main():
    print("="*60)
    print("EVALUATING LATEST TRAINED MODEL")
    print("="*60)
    
    # Find latest model
    latest_model = find_latest_model()
    
    if latest_model is None:
        print("\nNo trained models found!")
        print("Please train a model first using:")
        print("  python3 /app/scripts/train_baseline.py")
        return
    
    print(f"\nFound latest model: {latest_model}")
    
    # Check what files are in the model directory
    print("\nModel directory contents:")
    for file in os.listdir(latest_model):
        file_path = os.path.join(latest_model, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  - {file} ({size:.2f} MB)")
    
    # Run evaluation
    print("\nStarting evaluation...")
    cmd = f"python3 /app/scripts/evaluate_model.py --model_dir {latest_model}"
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {latest_model}/evaluation/")
        print("\nYou can view the results:")
        print(f"  - Trajectories: {latest_model}/evaluation/trajectory_*.png")
        print(f"  - Performance: {latest_model}/evaluation/performance_summary.png")
        print(f"  - Training curves: {latest_model}/evaluation/training_history.png")
        print(f"  - Sample predictions: {latest_model}/evaluation/sample_predictions/")
    else:
        print("\nEvaluation failed! Check the error messages above.")


if __name__ == "__main__":
    main()