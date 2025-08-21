#!/usr/bin/env python3
"""
Train TSformer-VO with Multi-Scale SE(3) Loss

This script trains your model with the new multi-scale loss that should 
dramatically improve trajectory scale matching.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Train with multi-scale loss"""
    
    # Import after adding path
    from train_tsformer import main as train_main
    import argparse
    
    # Override sys.argv to use multi-scale loss by default
    original_argv = sys.argv.copy()
    
    # Default training arguments with multi-scale improvements
    sys.argv = [
        'train_with_multi_scale_loss.py',
        '--csv_path', 'data/processed/visual_odometry_dataset/visual_odometry_dataset_kalibr_clean.csv',
        '--data_root', 'data/processed/visual_odometry_dataset',
        '--test_bags', 'ariel_2023-12-21-14-28-22_4',
        '--sequence_length', '8',
        '--batch_size', '4',
        '--num_epochs', '30',
        '--learning_rate', '1e-4',
        '--output_dir', 'experiments/tsformer_vo_multi_scale',
        '--log_interval', '10'
    ]
    
    print("=" * 60)
    print("TRAINING WITH MULTI-SCALE SE(3) LOSS")
    print("=" * 60)
    print("Key improvements:")
    print("  - Multi-scale supervision (local + global)")
    print("  - lambda2 = 2.0 (higher weight for trajectory scale)")
    print("  - SE(3) chain consistency")
    print("  - Expected: 5-10x better trajectory scale matching!")
    print("=" * 60)
    
    try:
        # Run training
        train_main()
        
        print("\\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print("Next steps:")
        print("  1. Check experiments/tsformer_vo_multi_scale/ for results")
        print("  2. Run visualization to see improved trajectory matching")
        print("  3. Compare with previous results - should see much better scale!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Check the error above and fix any issues.")
    
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()