#!/usr/bin/env python3
"""
Production script to extract training data from all ROS bags
Optimized version with better performance
"""

import rosbag
import cv2
import numpy as np
import pandas as pd
import os
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import argparse
import json
import glob

class OptimizedDataExtractor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.bridge = CvBridge()
        
        # Create output directories
        self.images_dir = os.path.join(output_dir, 'images')
        self.data_dir = os.path.join(output_dir, 'data')
        
        for cam_id in range(5):
            os.makedirs(os.path.join(self.images_dir, f'cam{cam_id}'), exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Data storage
        self.data_records = []
        self.sample_counter = 0
        
    def process_bag(self, bag_path, bag_name):
        """Process a single bag file"""
        print(f"Processing {bag_name}...")
        
        # Data collectors
        camera_msgs = {f'cam{i}': [] for i in range(5)}
        pose_msgs = []
        imu_msgs = []
        pressure_msgs = []
        
        # Read all messages first
        with rosbag.Bag(bag_path, 'r') as bag:
            total_msgs = bag.get_message_count([
                '/alphasense_driver_ros/cam0', '/alphasense_driver_ros/cam1', 
                '/alphasense_driver_ros/cam2', '/alphasense_driver_ros/cam3', 
                '/alphasense_driver_ros/cam4', '/qualisys/ariel/pose',
                '/alphasense_driver_ros/imu', '/mavros/imu/static_pressure'
            ])
            
            with tqdm(total=total_msgs, desc=f"Reading {bag_name}") as pbar:
                for topic, msg, t in bag.read_messages(topics=[
                    '/alphasense_driver_ros/cam0', '/alphasense_driver_ros/cam1', 
                    '/alphasense_driver_ros/cam2', '/alphasense_driver_ros/cam3', 
                    '/alphasense_driver_ros/cam4', '/qualisys/ariel/pose',
                    '/alphasense_driver_ros/imu', '/mavros/imu/static_pressure'
                ]):
                    timestamp = t.to_sec()
                    
                    if topic.startswith('/alphasense_driver_ros/cam'):
                        cam_id = int(topic[-1])
                        if cam_id < 5:
                            camera_msgs[f'cam{cam_id}'].append((timestamp, msg))
                    
                    elif topic == '/qualisys/ariel/pose':
                        pose_msgs.append((timestamp, msg))
                    
                    elif topic == '/alphasense_driver_ros/imu':
                        imu_msgs.append((timestamp, msg))
                    
                    elif topic == '/mavros/imu/static_pressure':
                        pressure_msgs.append((timestamp, msg))
                    
                    pbar.update(1)
        
        # Sort by timestamp
        for cam_id in range(5):
            camera_msgs[f'cam{cam_id}'].sort(key=lambda x: x[0])
        pose_msgs.sort(key=lambda x: x[0])
        imu_msgs.sort(key=lambda x: x[0])
        pressure_msgs.sort(key=lambda x: x[0])
        
        # Create training samples
        self.create_samples(camera_msgs, pose_msgs, imu_msgs, pressure_msgs, bag_name)
    
    def find_closest_msg(self, target_time, msg_list, tolerance=0.1):
        """Find closest message by timestamp using binary search"""
        if not msg_list:
            return None
        
        # Binary search for closest timestamp
        left, right = 0, len(msg_list) - 1
        closest_msg = None
        min_diff = float('inf')
        
        while left <= right:
            mid = (left + right) // 2
            timestamp = msg_list[mid][0]
            diff = abs(timestamp - target_time)
            
            if diff < min_diff and diff < tolerance:
                min_diff = diff
                closest_msg = msg_list[mid]
            
            if timestamp < target_time:
                left = mid + 1
            else:
                right = mid - 1
        
        return closest_msg
    
    def compute_relative_motion(self, pose_prev, pose_curr):
        """Compute relative motion between poses"""
        # Extract positions and orientations
        pos_prev = np.array([pose_prev.position.x, pose_prev.position.y, pose_prev.position.z])
        pos_curr = np.array([pose_curr.position.x, pose_curr.position.y, pose_curr.position.z])
        
        quat_prev = np.array([pose_prev.orientation.x, pose_prev.orientation.y, 
                             pose_prev.orientation.z, pose_prev.orientation.w])
        quat_curr = np.array([pose_curr.orientation.x, pose_curr.orientation.y, 
                             pose_curr.orientation.z, pose_curr.orientation.w])
        
        # Global position change
        global_delta = pos_curr - pos_prev
        
        # Transform to robot frame
        R_prev = R.from_quat(quat_prev).as_matrix()
        robot_delta = R_prev.T @ global_delta
        
        # Relative rotation
        R_curr = R.from_quat(quat_curr).as_matrix()
        R_rel = R_prev.T @ R_curr
        euler_rel = R.from_matrix(R_rel).as_euler('xyz', degrees=False)
        
        return {
            'delta_x': robot_delta[0],
            'delta_y': robot_delta[1],
            'delta_z': robot_delta[2],
            'delta_roll': euler_rel[0],
            'delta_pitch': euler_rel[1],
            'delta_yaw': euler_rel[2]
        }
    
    def create_samples(self, camera_msgs, pose_msgs, imu_msgs, pressure_msgs, bag_name):
        """Create training samples from synchronized data"""
        
        # Use cam0 as reference timing
        cam0_msgs = camera_msgs['cam0']
        
        print(f"Creating samples from {len(cam0_msgs)} cam0 frames...")
        
        for i in tqdm(range(1, len(cam0_msgs)), desc="Processing frames"):
            curr_time, curr_cam0_msg = cam0_msgs[i]
            prev_time, prev_cam0_msg = cam0_msgs[i-1]
            
            # Find corresponding poses
            curr_pose_msg = self.find_closest_msg(curr_time, pose_msgs)
            prev_pose_msg = self.find_closest_msg(prev_time, pose_msgs)
            
            if not curr_pose_msg or not prev_pose_msg:
                continue
            
            # Compute relative motion
            relative_motion = self.compute_relative_motion(
                prev_pose_msg[1].pose, curr_pose_msg[1].pose
            )
            
            # Find sensor data
            imu_msg = self.find_closest_msg(curr_time, imu_msgs)
            pressure_msg = self.find_closest_msg(curr_time, pressure_msgs)
            
            # Save images and create record
            sample_record = {
                'sample_id': self.sample_counter,
                'bag_name': bag_name,
                'timestamp': curr_time,
                'dt': curr_time - prev_time,
                **relative_motion,
                'pose_x': curr_pose_msg[1].pose.position.x,
                'pose_y': curr_pose_msg[1].pose.position.y,
                'pose_z': curr_pose_msg[1].pose.position.z,
            }
            
            # Add IMU data
            if imu_msg:
                imu = imu_msg[1]
                sample_record.update({
                    'imu_accel_x': imu.linear_acceleration.x,
                    'imu_accel_y': imu.linear_acceleration.y,
                    'imu_accel_z': imu.linear_acceleration.z,
                    'imu_gyro_x': imu.angular_velocity.x,
                    'imu_gyro_y': imu.angular_velocity.y,
                    'imu_gyro_z': imu.angular_velocity.z,
                })
            
            # Add pressure data
            if pressure_msg:
                sample_record['pressure'] = pressure_msg[1].fluid_pressure
                sample_record['pressure_variance'] = pressure_msg[1].variance
            
            # Save camera images
            for cam_id in range(5):
                cam_msg = self.find_closest_msg(curr_time, camera_msgs[f'cam{cam_id}'])
                if cam_msg:
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(cam_msg[1], "bgr8")
                        img_filename = f"{bag_name}_{self.sample_counter:06d}_cam{cam_id}.jpg"
                        img_path = os.path.join(self.images_dir, f'cam{cam_id}', img_filename)
                        cv2.imwrite(img_path, cv_image)
                        sample_record[f'cam{cam_id}_path'] = f'images/cam{cam_id}/{img_filename}'
                    except Exception as e:
                        print(f"Error saving cam{cam_id} image: {e}")
                        sample_record[f'cam{cam_id}_path'] = None
            
            self.data_records.append(sample_record)
            self.sample_counter += 1
    
    def save_data(self):
        """Save extracted data to CSV and metadata"""
        if not self.data_records:
            print("No data to save!")
            return
        
        # Save CSV
        df = pd.DataFrame(self.data_records)
        csv_path = os.path.join(self.data_dir, 'training_data.csv')
        df.to_csv(csv_path, index=False)
        
        # Save metadata
        metadata = {
            'total_samples': len(self.data_records),
            'cameras': 5,
            'coordinate_frame': 'robot_relative_motion',
            'output_variables': ['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw'],
            'input_sources': ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'imu', 'pressure'],
            'description': 'Visual odometry training data with relative motion ground truth'
        }
        
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nExtraction complete!")
        print(f"Total samples: {len(self.data_records)}")
        print(f"CSV saved to: {csv_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        # Print sample statistics
        if self.data_records:
            df_stats = df[['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw']].describe()
            print(f"\nMotion statistics:")
            print(df_stats)

def main():
    parser = argparse.ArgumentParser(description='Extract training data from ROS bags')
    parser.add_argument('--input_dir', default='data/raw', help='Directory with bag files')
    parser.add_argument('--output_dir', default='data/processed/full_dataset', help='Output directory')
    parser.add_argument('--max_bags', type=int, help='Maximum number of bags to process')
    
    args = parser.parse_args()
    
    # Find bag files
    bag_files = glob.glob(os.path.join(args.input_dir, "*.bag"))
    if not bag_files:
        print(f"No bag files found in {args.input_dir}")
        return
    
    bag_files.sort()
    if args.max_bags:
        bag_files = bag_files[:args.max_bags]
    
    print(f"Found {len(bag_files)} bag files to process")
    
    # Create extractor
    os.makedirs(args.output_dir, exist_ok=True)
    extractor = OptimizedDataExtractor(args.output_dir)
    
    # Process each bag
    for bag_path in bag_files:
        bag_name = os.path.splitext(os.path.basename(bag_path))[0]
        extractor.process_bag(bag_path, bag_name)
    
    # Save final dataset
    extractor.save_data()

if __name__ == '__main__':
    main()