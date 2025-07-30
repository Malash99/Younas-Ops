#!/usr/bin/env python3
"""
Extract images from all 5 bags and create a CSV with frame sequences
"""

import rosbag
import cv2
import numpy as np
import pandas as pd
import os
from cv_bridge import CvBridge
from tqdm import tqdm
import glob
import json

class SequentialImageExtractor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.bridge = CvBridge()
        
        # Create directories
        self.images_dir = os.path.join(output_dir, 'images')
        self.data_dir = os.path.join(output_dir, 'data')
        
        for cam_id in range(5):
            os.makedirs(os.path.join(self.images_dir, f'cam{cam_id}'), exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Global frame counter and sequence tracking
        self.global_frame_id = 0
        self.frame_records = []
        
    def extract_bag_images(self, bag_path, bag_name, bag_index):
        """Extract images from a single bag and record sequence info"""
        
        print(f"Processing bag {bag_index + 1}/5: {bag_name}")
        
        # Collect all camera messages
        camera_messages = []
        
        with rosbag.Bag(bag_path, 'r') as bag:
            # Read only camera topics
            camera_topics = [f'/alphasense_driver_ros/cam{i}' for i in range(5)]
            total_msgs = bag.get_message_count(camera_topics)
            
            with tqdm(total=total_msgs, desc=f"Reading {bag_name}") as pbar:
                for topic, msg, t in bag.read_messages(topics=camera_topics):
                    cam_id = int(topic.split('cam')[1])
                    if cam_id < 5:
                        camera_messages.append({
                            'timestamp': t.to_sec(),
                            'cam_id': cam_id,
                            'msg': msg,
                            'ros_time': t
                        })
                    pbar.update(1)
        
        # Sort all messages by timestamp to maintain temporal order
        camera_messages.sort(key=lambda x: x['timestamp'])
        
        print(f"Found {len(camera_messages)} camera messages in {bag_name}")
        
        # Group messages by timestamp (since cameras are synchronized)
        # Messages within 10ms are considered the same timestamp
        frame_groups = []
        current_group = []
        last_timestamp = None
        
        for msg_data in camera_messages:
            timestamp = msg_data['timestamp']
            
            if last_timestamp is None or abs(timestamp - last_timestamp) < 0.01:  # 10ms tolerance
                current_group.append(msg_data)
                last_timestamp = timestamp
            else:
                if current_group:
                    frame_groups.append(current_group)
                current_group = [msg_data]
                last_timestamp = timestamp
        
        # Add the last group
        if current_group:
            frame_groups.append(current_group)
        
        print(f"Grouped into {len(frame_groups)} synchronized frames")
        
        # Process each frame group
        for group_idx, frame_group in enumerate(tqdm(frame_groups, desc=f"Processing frames from {bag_name}")):
            
            # Use the average timestamp for this frame
            avg_timestamp = np.mean([msg['timestamp'] for msg in frame_group])
            
            # Create frame record
            frame_record = {
                'global_frame_id': self.global_frame_id,
                'bag_name': bag_name,
                'bag_index': bag_index,
                'bag_frame_id': group_idx,
                'timestamp': avg_timestamp,
                'num_cameras': len(frame_group)
            }
            
            # Save images for each camera in this frame
            cameras_saved = []
            for msg_data in frame_group:
                cam_id = msg_data['cam_id']
                
                try:
                    # Convert to OpenCV image
                    cv_image = self.bridge.imgmsg_to_cv2(msg_data['msg'], "bgr8")
                    
                    # Create filename
                    filename = f"frame_{self.global_frame_id:06d}_cam{cam_id}.jpg"
                    image_path = os.path.join(self.images_dir, f'cam{cam_id}', filename)
                    
                    # Save image
                    cv2.imwrite(image_path, cv_image)
                    
                    # Record in frame data
                    frame_record[f'cam{cam_id}_path'] = f'images/cam{cam_id}/{filename}'
                    frame_record[f'cam{cam_id}_timestamp'] = msg_data['timestamp']
                    cameras_saved.append(cam_id)
                    
                except Exception as e:
                    print(f"Error saving cam{cam_id} for frame {self.global_frame_id}: {e}")
                    frame_record[f'cam{cam_id}_path'] = None
                    frame_record[f'cam{cam_id}_timestamp'] = None
            
            # Only keep frames that have at least 3 cameras
            if len(cameras_saved) >= 3:
                # Fill missing cameras with None
                for cam_id in range(5):
                    if cam_id not in cameras_saved:
                        frame_record[f'cam{cam_id}_path'] = None
                        frame_record[f'cam{cam_id}_timestamp'] = None
                
                self.frame_records.append(frame_record)
                self.global_frame_id += 1
            
        print(f"Extracted {self.global_frame_id} frames from {bag_name}")
    
    def process_all_bags(self, input_dir):
        """Process all bag files in sequence"""
        
        # Find all bag files
        bag_files = glob.glob(os.path.join(input_dir, "*.bag"))
        bag_files.sort()  # Important: process in chronological order
        
        if not bag_files:
            print(f"No bag files found in {input_dir}")
            return
        
        print(f"Found {len(bag_files)} bag files to process")
        
        # Process each bag in order
        for bag_index, bag_path in enumerate(bag_files):
            bag_name = os.path.splitext(os.path.basename(bag_path))[0]
            self.extract_bag_images(bag_path, bag_name, bag_index)
    
    def save_frame_sequence_csv(self):
        """Save the frame sequence CSV"""
        
        if not self.frame_records:
            print("No frames to save!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.frame_records)
        
        # Sort by global frame ID to ensure proper sequence
        df = df.sort_values('global_frame_id').reset_index(drop=True)
        
        # Save CSV
        csv_path = os.path.join(self.data_dir, 'frame_sequence.csv')
        df.to_csv(csv_path, index=False)
        
        # Create metadata
        metadata = {
            'total_frames': len(df),
            'total_bags': len(df['bag_name'].unique()),
            'bags_processed': sorted(df['bag_name'].unique().tolist()),
            'cameras': 5,
            'time_span': {
                'start_timestamp': float(df['timestamp'].min()),
                'end_timestamp': float(df['timestamp'].max()),
                'duration_seconds': float(df['timestamp'].max() - df['timestamp'].min())
            },
            'frames_per_bag': df.groupby('bag_name').size().to_dict(),
            'description': 'Sequential frame data from all bags with synchronized camera images'
        }
        
        metadata_path = os.path.join(self.data_dir, 'sequence_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print statistics
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"Total frames extracted: {len(df)}")
        print(f"Total bags processed: {len(df['bag_name'].unique())}")
        print(f"Time span: {metadata['time_span']['duration_seconds']:.1f} seconds")
        print(f"CSV saved to: {csv_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        # Print per-bag statistics
        print(f"\nFrames per bag:")
        for bag_name, count in metadata['frames_per_bag'].items():
            print(f"  {bag_name}: {count} frames")
        
        # Print camera coverage statistics
        camera_coverage = {}
        for cam_id in range(5):
            non_null_count = df[f'cam{cam_id}_path'].notna().sum()
            coverage_pct = (non_null_count / len(df)) * 100
            camera_coverage[f'cam{cam_id}'] = {
                'frames_with_images': int(non_null_count),
                'coverage_percentage': round(coverage_pct, 1)
            }
            print(f"  cam{cam_id}: {non_null_count}/{len(df)} frames ({coverage_pct:.1f}%)")
        
        # Add camera coverage to metadata
        metadata['camera_coverage'] = camera_coverage
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return csv_path, metadata_path

def main():
    input_dir = "data/raw"
    output_dir = "data/sequential_frames"
    
    print("Starting sequential image extraction from all ROS bags...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = SequentialImageExtractor(output_dir)
    
    # Process all bags
    extractor.process_all_bags(input_dir)
    
    # Save CSV and metadata
    extractor.save_frame_sequence_csv()
    
    print("Sequential extraction complete!")

if __name__ == '__main__':
    main()