#!/usr/bin/env python3
"""
Test image extraction on single bag file
"""

import os
import sys
sys.path.append('/app/scripts')

from extract_images_only import extract_images_from_bag

def test_single_bag():
    """Test image extraction on first bag"""
    
    bag_path = "data/raw/ariel_2023-12-21-14-24-42_0.bag"
    output_dir = "data/test_images"
    
    print("Testing image extraction on single bag...")
    print(f"Bag: {bag_path}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract images
    try:
        counts = extract_images_from_bag(bag_path, output_dir)
        
        total = sum(counts.values())
        print(f"\nTest successful! Extracted {total} images total")
        
        # Check if images were actually saved
        for cam_id in range(5):
            cam_dir = os.path.join(output_dir, f'cam{cam_id}')
            if os.path.exists(cam_dir):
                actual_files = len([f for f in os.listdir(cam_dir) if f.endswith('.jpg')])
                print(f"cam{cam_id}: {counts[f'cam{cam_id}']} extracted, {actual_files} files on disk")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_single_bag()