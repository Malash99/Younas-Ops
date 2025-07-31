#!/usr/bin/env python3
"""
ROS Bag Data Extraction Script for Visual Odometry Training

This script extracts:
- Images from 5 cameras (cam0-cam4)
- Ground truth poses (transformed to relative motion)
- IMU data (acceleration, angular velocity)
- Barometer/pressure data

The ground truth is transformed from global coordinates to relative motion
between consecutive frames for ML training.
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
from collections import defaultdict
import json

class DataExtractor:
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
        self.image_counter = 0
        
    def extract_bag_data(self, bag_path, bag_name):
        """Extract data from a single ROS bag file"""
        print(f"Processing bag: {bag_name}")
        
        # Storage for synchronized data
        camera_data = {f'cam{i}': [] for i in range(5)}
        pose_data = []
        imu_data = []
        pressure_data = []
        
        # Read bag and collect data
        with rosbag.Bag(bag_path, 'r') as bag:
            print("Reading bag contents...")
            
            # Get bag info for progress bar
            total_messages = bag.get_message_count()
            
            with tqdm(total=total_messages, desc="Reading messages") as pbar:
                for topic, msg, t in bag.read_messages():
                    timestamp = t.to_sec()
                    
                    # Extract camera images
                    if topic.startswith('/alphasense_driver_ros/cam'):
                        cam_id = int(topic.split('cam')[1])
                        if cam_id < 5:  # Only process cam0-cam4
                            try:
                                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                                camera_data[f'cam{cam_id}'].append({
                                    'timestamp': timestamp,
                                    'image': cv_image,
                                    'msg_timestamp': msg.header.stamp.to_sec()
                                })
                            except Exception as e:
                                print(f"Error processing {topic}: {e}")
                    
                    # Extract ground truth poses (Qualisys)
                    elif topic == '/qualisys/ariel/pose':
                        pose = msg.pose
                        pose_data.append({
                            'timestamp': timestamp,
                            'msg_timestamp': msg.header.stamp.to_sec(),
                            'x': pose.position.x,
                            'y': pose.position.y,
                            'z': pose.position.z,
                            'qx': pose.orientation.x,
                            'qy': pose.orientation.y,
                            'qz': pose.orientation.z,
                            'qw': pose.orientation.w
                        })
                    
                    # Extract IMU data (Alphasense IMU - higher frequency)
                    elif topic == '/alphasense_driver_ros/imu':
                        imu_data.append({
                            'timestamp': timestamp,
                            'msg_timestamp': msg.header.stamp.to_sec(),
                            'accel_x': msg.linear_acceleration.x,
                            'accel_y': msg.linear_acceleration.y,
                            'accel_z': msg.linear_acceleration.z,
                            'gyro_x': msg.angular_velocity.x,
                            'gyro_y': msg.angular_velocity.y,
                            'gyro_z': msg.angular_velocity.z
                        })
                    
                    # Extract pressure data
                    elif topic == '/mavros/imu/static_pressure':
                        pressure_data.append({
                            'timestamp': timestamp,
                            'msg_timestamp': msg.header.stamp.to_sec(),
                            'pressure': msg.fluid_pressure,
                            'variance': msg.variance
                        })
                    
                    pbar.update(1)
        
        # Sort all data by timestamp
        for cam_id in range(5):
            camera_data[f'cam{cam_id}'].sort(key=lambda x: x['timestamp'])
        pose_data.sort(key=lambda x: x['timestamp'])
        imu_data.sort(key=lambda x: x['timestamp'])
        pressure_data.sort(key=lambda x: x['timestamp'])
        
        # Synchronize and create training samples
        self.create_training_samples(camera_data, pose_data, imu_data, pressure_data, bag_name)
    
    def find_closest_data(self, target_time, data_list, time_key='timestamp', max_diff=0.1):
        """Find the closest data point in time"""
        if not data_list:
            return None
        
        min_diff = float('inf')
        closest_data = None
        
        for data in data_list:
            diff = abs(data[time_key] - target_time)
            if diff < min_diff and diff < max_diff:
                min_diff = diff
                closest_data = data
        
        return closest_data
    
    def compute_relative_motion(self, pose1, pose2):
        """
        Compute relative motion between two poses.
        
        Args:
            pose1, pose2: Dict with keys 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'
        
        Returns:
            Dict with delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw
        """
        # Position differences in global frame
        global_delta = np.array([
            pose2['x'] - pose1['x'],
            pose2['y'] - pose1['y'],
            pose2['z'] - pose1['z']
        ])
        
        # Convert quaternions to rotation matrices
        q1 = np.array([pose1['qx'], pose1['qy'], pose1['qz'], pose1['qw']])
        q2 = np.array([pose2['qx'], pose2['qy'], pose2['qz'], pose2['qw']])
        
        R1 = R.from_quat(q1).as_matrix()
        R2 = R.from_quat(q2).as_matrix()
        
        # Transform global position change to robot frame (at pose1)
        robot_delta = R1.T @ global_delta
        
        # Compute relative rotation
        R_rel = R1.T @ R2
        relative_rotation = R.from_matrix(R_rel)
        
        # Convert to Euler angles (roll, pitch, yaw)
        euler_angles = relative_rotation.as_euler('xyz', degrees=False)
        
        return {
            'delta_x': robot_delta[0],
            'delta_y': robot_delta[1], 
            'delta_z': robot_delta[2],
            'delta_roll': euler_angles[0],
            'delta_pitch': euler_angles[1],
            'delta_yaw': euler_angles[2]
        }
    
    def create_training_samples(self, camera_data, pose_data, imu_data, pressure_data, bag_name):
        """Create synchronized training samples"""
        print("Creating training samples...")
        
        # Use cam0 as the reference for timing (assuming all cameras are synchronized)
        ref_camera = camera_data['cam0']
        
        for i in tqdm(range(len(ref_camera)), desc="Processing frames"):
            current_frame = ref_camera[i]
            current_time = current_frame['timestamp']
            
            # Skip if this is the first frame (no previous frame for relative motion)
            if i == 0:
                continue
            
            previous_frame = ref_camera[i-1]
            previous_time = previous_frame['timestamp']
            
            # Find corresponding poses
            current_pose = self.find_closest_data(current_time, pose_data)
            previous_pose = self.find_closest_data(previous_time, pose_data)
            
            if current_pose is None or previous_pose is None:
                continue
            
            # Compute relative motion
            relative_motion = self.compute_relative_motion(previous_pose, current_pose)
            
            # Find corresponding sensor data
            imu_current = self.find_closest_data(current_time, imu_data)
            pressure_current = self.find_closest_data(current_time, pressure_data)
            
            # Save images for all cameras
            image_paths = {}
            for cam_id in range(5):
                cam_frame = self.find_closest_data(current_time, camera_data[f'cam{cam_id}'])
                if cam_frame is not None:
                    image_filename = f"{bag_name}_frame_{self.image_counter:06d}_cam{cam_id}.jpg"
                    image_path = os.path.join(self.images_dir, f'cam{cam_id}', image_filename)
                    cv2.imwrite(image_path, cam_frame['image'])
                    image_paths[f'cam{cam_id}_path'] = os.path.join('images', f'cam{cam_id}', image_filename)
                else:
                    image_paths[f'cam{cam_id}_path'] = None
            
            # Create data record
            record = {
                'sample_id': self.image_counter,
                'bag_name': bag_name,
                'timestamp': current_time,
                'dt': current_time - previous_time,
                
                # Ground truth relative motion (target outputs)
                'delta_x': relative_motion['delta_x'],
                'delta_y': relative_motion['delta_y'],
                'delta_z': relative_motion['delta_z'],
                'delta_roll': relative_motion['delta_roll'],
                'delta_pitch': relative_motion['delta_pitch'],
                'delta_yaw': relative_motion['delta_yaw'],
                
                # Absolute poses (for reference)
                'pose_x': current_pose['x'],
                'pose_y': current_pose['y'],
                'pose_z': current_pose['z'],
                'pose_qx': current_pose['qx'],
                'pose_qy': current_pose['qy'],
                'pose_qz': current_pose['qz'],
                'pose_qw': current_pose['qw'],
                
                # IMU data
                'imu_accel_x': imu_current['accel_x'] if imu_current else None,
                'imu_accel_y': imu_current['accel_y'] if imu_current else None,
                'imu_accel_z': imu_current['accel_z'] if imu_current else None,
                'imu_gyro_x': imu_current['gyro_x'] if imu_current else None,
                'imu_gyro_y': imu_current['gyro_y'] if imu_current else None,
                'imu_gyro_z': imu_current['gyro_z'] if imu_current else None,
                
                # Pressure data
                'pressure': pressure_current['pressure'] if pressure_current else None,
                'pressure_variance': pressure_current['variance'] if pressure_current else None,
                
                # Image paths
                **image_paths
            }
            
            self.data_records.append(record)
            self.image_counter += 1
    
    def save_csv(self):
        """Save all extracted data to CSV"""
        df = pd.DataFrame(self.data_records)
        csv_path = os.path.join(self.data_dir, 'training_data.csv')
        df.to_csv(csv_path, index=False)
        
        # Save metadata
        metadata = {
            'total_samples': len(self.data_records),
            'cameras': 5,
            'coordinate_frame': 'robot_frame_relative_motion',
            'description': 'Relative motion between consecutive frames in robot coordinate frame'
        }
        
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(self.data_records)} training samples to {csv_path}")
        print(f"Saved metadata to {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract training data from ROS bags')
    parser.add_argument('--input_dir', default='data/raw', help='Directory containing bag files')
    parser.add_argument('--output_dir', default='data/processed', help='Output directory')
    parser.add_argument('--bag_pattern', default='*.bag', help='Pattern for bag files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = DataExtractor(args.output_dir)
    
    # Find all bag files
    import glob
    bag_files = glob.glob(os.path.join(args.input_dir, args.bag_pattern))
    
    if not bag_files:
        print(f"No bag files found in {args.input_dir}")
        return
    
    print(f"Found {len(bag_files)} bag files")
    
    # Process each bag file
    for bag_path in sorted(bag_files):
        bag_name = os.path.splitext(os.path.basename(bag_path))[0]
        extractor.extract_bag_data(bag_path, bag_name)
    
    # Save final CSV
    extractor.save_csv()
    
    print("Data extraction complete!")

if __name__ == '__main__':
    main()