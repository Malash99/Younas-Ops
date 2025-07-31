#!/usr/bin/env python3
"""
Test script to run data extraction on a single bag file
"""

import os
import sys
sys.path.append('/app/scripts')

from extract_training_data import DataExtractor

def test_single_bag():
    """Test extraction on the first bag file"""
    
    # Set up paths
    bag_path = "data/raw/ariel_2023-12-21-14-24-42_0.bag"
    output_dir = "data/processed/test_extraction"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = DataExtractor(output_dir)
    
    # Process single bag
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    print(f"Testing extraction on: {bag_name}")
    
    try:
        extractor.extract_bag_data(bag_path, bag_name)
        extractor.save_csv()
        print("Test extraction completed successfully!")
        
        # Print some statistics
        print(f"Total samples extracted: {len(extractor.data_records)}")
        if extractor.data_records:
            sample = extractor.data_records[0]
            print("\nFirst sample keys:")
            for key in sorted(sample.keys()):
                print(f"  {key}: {sample[key]}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_single_bag()