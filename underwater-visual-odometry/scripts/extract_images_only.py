#!/usr/bin/env python3
"""
Extract only images from all 5 cameras from ROS bags
Fast extraction for training purposes
"""

import rosbag
import cv2
import os
from cv_bridge import CvBridge
from tqdm import tqdm
import argparse
import glob

def extract_images_from_bag(bag_path, output_dir):
    """Extract all camera images from a single bag"""
    
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    print(f"Processing {bag_name}...")
    
    # Create camera directories
    for cam_id in range(5):
        cam_dir = os.path.join(output_dir, f'cam{cam_id}')
        os.makedirs(cam_dir, exist_ok=True)
    
    bridge = CvBridge()
    image_counters = {f'cam{i}': 0 for i in range(5)}
    
    # Read bag and extract images
    with rosbag.Bag(bag_path, 'r') as bag:
        # Get total camera messages for progress bar
        camera_topics = [f'/alphasense_driver_ros/cam{i}' for i in range(5)]
        total_msgs = bag.get_message_count(camera_topics)
        
        with tqdm(total=total_msgs, desc=f"Extracting images from {bag_name}") as pbar:
            for topic, msg, t in bag.read_messages(topics=camera_topics):
                
                # Get camera ID from topic
                cam_id = int(topic.split('cam')[1])
                
                try:
                    # Convert ROS image to OpenCV
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    
                    # Create filename with timestamp and counter
                    timestamp = t.to_sec()
                    filename = f"{bag_name}_cam{cam_id}_{image_counters[f'cam{cam_id}']:06d}_{timestamp:.6f}.jpg"
                    
                    # Save image
                    image_path = os.path.join(output_dir, f'cam{cam_id}', filename)
                    cv2.imwrite(image_path, cv_image)
                    
                    image_counters[f'cam{cam_id}'] += 1
                    
                except Exception as e:
                    print(f"Error processing {topic} at {timestamp}: {e}")
                
                pbar.update(1)
    
    # Print statistics
    total_images = sum(image_counters.values())
    print(f"Extracted {total_images} images from {bag_name}:")
    for cam_id in range(5):
        count = image_counters[f'cam{cam_id}']
        print(f"  cam{cam_id}: {count} images")
    
    return image_counters

def main():
    parser = argparse.ArgumentParser(description='Extract images from ROS bags')
    parser.add_argument('--input_dir', default='data/raw', help='Directory with bag files')
    parser.add_argument('--output_dir', default='data/images', help='Output directory for images')
    parser.add_argument('--max_bags', type=int, help='Maximum number of bags to process')
    
    args = parser.parse_args()
    
    # Find all bag files
    bag_files = glob.glob(os.path.join(args.input_dir, "*.bag"))
    if not bag_files:
        print(f"No bag files found in {args.input_dir}")
        return
    
    bag_files.sort()
    if args.max_bags:
        bag_files = bag_files[:args.max_bags]
    
    print(f"Found {len(bag_files)} bag files to process")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each bag file
    total_counts = {f'cam{i}': 0 for i in range(5)}
    
    for bag_path in bag_files:
        bag_counts = extract_images_from_bag(bag_path, args.output_dir)
        
        # Add to total counts
        for cam_id in range(5):
            total_counts[f'cam{cam_id}'] += bag_counts[f'cam{cam_id}']
    
    # Print final statistics
    total_images = sum(total_counts.values())
    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Total images extracted: {total_images}")
    print(f"Images per camera:")
    for cam_id in range(5):
        count = total_counts[f'cam{cam_id}']
        print(f"  cam{cam_id}: {count} images")
    
    print(f"\nImages saved to: {args.output_dir}")

if __name__ == '__main__':
    main()