#!/usr/bin/env python3
"""
Quick test script - processes only first 100 frames
"""

import rosbag
import cv2
import numpy as np
import pandas as pd
import os
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def quick_test():
    """Quick test with limited frames"""
    
    bag_path = "data/raw/ariel_2023-12-21-14-24-42_0.bag"
    output_dir = "data/processed/quick_test"
    os.makedirs(output_dir, exist_ok=True)
    
    bridge = CvBridge()
    camera_data = []
    pose_data = []
    
    print("Reading bag (first 1000 messages only)...")
    
    with rosbag.Bag(bag_path, 'r') as bag:
        count = 0
        for topic, msg, t in bag.read_messages():
            if count >= 1000:  # Limit for quick test
                break
            
            timestamp = t.to_sec()
            
            # Only process cam0 and poses for quick test
            if topic == '/alphasense_driver_ros/cam0':
                try:
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    camera_data.append({
                        'timestamp': timestamp,
                        'image': cv_image
                    })
                except Exception as e:
                    print(f"Error: {e}")
            
            elif topic == '/qualisys/ariel/pose':
                pose = msg.pose
                pose_data.append({
                    'timestamp': timestamp,
                    'x': pose.position.x,
                    'y': pose.position.y,
                    'z': pose.position.z,
                    'qx': pose.orientation.x,
                    'qy': pose.orientation.y,
                    'qz': pose.orientation.z,
                    'qw': pose.orientation.w
                })
            
            count += 1
    
    print(f"Found {len(camera_data)} camera frames and {len(pose_data)} poses")
    
    # Sort by timestamp
    camera_data.sort(key=lambda x: x['timestamp'])
    pose_data.sort(key=lambda x: x['timestamp'])
    
    # Process only first 10 frames for test
    records = []
    for i in range(min(10, len(camera_data) - 1)):
        current_frame = camera_data[i + 1]
        previous_frame = camera_data[i]
        
        # Find closest poses
        current_pose = None
        previous_pose = None
        
        current_time = current_frame['timestamp']
        previous_time = previous_frame['timestamp']
        
        # Simple closest match
        for pose in pose_data:
            if abs(pose['timestamp'] - current_time) < 0.1:  # 100ms tolerance
                current_pose = pose
                break
        
        for pose in pose_data:
            if abs(pose['timestamp'] - previous_time) < 0.1:
                previous_pose = pose
                break
        
        if current_pose and previous_pose:
            # Compute simple position delta
            delta_x = current_pose['x'] - previous_pose['x']
            delta_y = current_pose['y'] - previous_pose['y']
            delta_z = current_pose['z'] - previous_pose['z']
            
            # Save image
            image_filename = f"test_frame_{i:03d}.jpg"
            image_path = os.path.join(output_dir, image_filename)
            cv2.imwrite(image_path, current_frame['image'])
            
            records.append({
                'frame_id': i,
                'timestamp': current_time,
                'dt': current_time - previous_time,
                'delta_x': delta_x,
                'delta_y': delta_y,
                'delta_z': delta_z,
                'image_path': image_filename
            })
    
    # Save test CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'test_data.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Quick test complete! Saved {len(records)} samples to {csv_path}")
    if records:
        print("\nSample record:")
        for key, value in records[0].items():
            print(f"  {key}: {value}")

if __name__ == '__main__':
    quick_test()