#!/usr/bin/env python3
"""
Simple script to extract all images from all bags
"""

import subprocess
import sys

def main():
    print("Starting image extraction from all ROS bags...")
    
    try:
        # Run the extraction
        result = subprocess.run([
            'python3', 'scripts/extract_images_only.py',
            '--input_dir', 'data/raw',
            '--output_dir', 'data/images'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Extraction completed successfully!")
            print(result.stdout)
        else:
            print("Extraction failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"Error running extraction: {e}")

if __name__ == '__main__':
    main()