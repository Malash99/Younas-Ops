#!/usr/bin/env python3
"""
Visual Odometry Dataset Extraction Pipeline

This script extracts synchronized camera frames, ground truth poses (transformed to camera frame),
IMU data, barometer readings, and thrust controls from ROS bag files to create a comprehensive
visual odometry training dataset.

Features:
- Extracts images from all 5 Alphasense cameras
- Transforms Qualisys ground truth from world frame to camera frame
- Includes IMU data (accelerometer, gyroscope)
- Includes barometer/pressure sensor readings
- Includes thrust/motor control commands
- Generates professional CSV dataset with proper synchronization
- Creates organized directory structure for images and metadata

Author: Underwater Visual Odometry Research Team
Date: January 2025
Version: 1.0
"""

import os
import sys
import cv2
import rosbag
import rospy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
try:
    import tf.transformations as tf_trans
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: tf.transformations not available. Using scipy for rotations.")

from scipy.spatial.transform import Rotation as R

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False
    print("Warning: cv_bridge not available. Will use alternative image conversion.")
import argparse
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class VisualOdometryDataExtractor:
    """
    Professional-grade data extraction pipeline for underwater visual odometry.
    
    This class handles:
    1. Synchronized extraction of multi-camera images
    2. Ground truth pose transformation to camera coordinate frame
    3. IMU data synchronization and interpolation
    4. Barometer/pressure sensor data extraction
    5. Thrust/motor control data extraction
    6. Professional dataset organization and CSV generation
    """
    
    def __init__(self, output_dir: str = "data/processed/visual_odometry_dataset"):
        """
        Initialize the data extractor.
        
        Args:
            output_dir: Directory to save extracted data
        """
        self.output_dir = Path(output_dir)
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()
        else:
            self.bridge = None
        
        # Create organized directory structure
        self.setup_directory_structure()
        
        # Setup IMU-camera coordinate frame transformation
        self.setup_imu_camera_calibration("kalibr_cam0")  # Use Kalibr calibration for cam_0
        
        # Topic configurations
        self.camera_topics = [
            "/alphasense_driver_ros/cam0",
            "/alphasense_driver_ros/cam1", 
            "/alphasense_driver_ros/cam2",
            "/alphasense_driver_ros/cam3",
            "/alphasense_driver_ros/cam4"
        ]
        
        self.pose_topics = [
            "/qualisys/ariel/pose",    # Primary ground truth
            "/qualisys/ariel/odom"     # Alternative with covariance
        ]
        
        self.imu_topics = [
            "/alphasense_driver_ros/imu",  # Primary IMU
            "/mavros/imu/data"             # Secondary IMU
        ]
        
        self.pressure_topics = [
            "/mavros/imu/static_pressure"
        ]
        
        self.thrust_topics = [
            "/mavros/rc/out"  # Motor/thrust commands
        ]
        
        # Data storage
        self.extracted_data = []
        self.stats = {
            'total_frames': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'bags_processed': 0
        }
    
    def setup_directory_structure(self):
        """Create professional directory structure for dataset."""
        directories = [
            self.output_dir,
            self.output_dir / "images" / "cam0",
            self.output_dir / "images" / "cam1", 
            self.output_dir / "images" / "cam2",
            self.output_dir / "images" / "cam3",
            self.output_dir / "images" / "cam4",
            self.output_dir / "metadata",
            self.output_dir / "ground_truth",
            self.output_dir / "sensor_data"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Created directory structure in: {self.output_dir}")
    
    def transform_pose_world_to_camera(self, world_pose: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
        """
        Transform pose from world coordinate frame to camera coordinate frame.
        
        This is crucial for visual odometry training - the model predicts motion
        in the camera's local coordinate system, not the world coordinate system.
        
        Args:
            world_pose: [x, y, z, qx, qy, qz, qw] in world frame
            camera_pose: Current camera pose [x, y, z, qx, qy, qz, qw] in world frame
            
        Returns:
            camera_frame_pose: [dx, dy, dz, droll, dpitch, dyaw] in camera frame
        """
        if TF_AVAILABLE:
            # Use tf.transformations (preferred method)
            world_pos = world_pose[:3]
            world_quat = world_pose[3:7]  # [qx, qy, qz, qw]
            
            cam_pos = camera_pose[:3]
            cam_quat = camera_pose[3:7]
            
            # Create transformation matrices
            T_world_target = tf_trans.concatenate_matrices(
                tf_trans.translation_matrix(world_pos),
                tf_trans.quaternion_matrix(world_quat)
            )
            
            T_world_camera = tf_trans.concatenate_matrices(
                tf_trans.translation_matrix(cam_pos),
                tf_trans.quaternion_matrix(cam_quat)
            )
            
            # Transform to camera frame
            T_camera_world = tf_trans.inverse_matrix(T_world_camera)
            T_camera_target = np.dot(T_camera_world, T_world_target)
            
            # Extract relative position and orientation
            relative_pos = tf_trans.translation_from_matrix(T_camera_target)
            relative_quat = tf_trans.quaternion_from_matrix(T_camera_target)
            relative_euler = tf_trans.euler_from_quaternion(relative_quat)
            
            return np.concatenate([relative_pos, relative_euler])
        
        else:
            # Fallback using scipy (alternative method)
            world_pos = world_pose[:3]
            world_quat = world_pose[3:7]  # [qx, qy, qz, qw] -> [x, y, z, w] for scipy
            world_quat_scipy = [world_quat[0], world_quat[1], world_quat[2], world_quat[3]]
            
            cam_pos = camera_pose[:3] 
            cam_quat = camera_pose[3:7]
            cam_quat_scipy = [cam_quat[0], cam_quat[1], cam_quat[2], cam_quat[3]]
            
            # Create rotation objects
            world_rot = R.from_quat(world_quat_scipy)
            cam_rot = R.from_quat(cam_quat_scipy)
            
            # Compute relative transformation
            rel_pos = world_pos - cam_pos
            rel_pos_camera = cam_rot.inv().apply(rel_pos)
            
            rel_rot = cam_rot.inv() * world_rot
            rel_euler = rel_rot.as_euler('xyz')  # roll, pitch, yaw
            
            return np.concatenate([rel_pos_camera, rel_euler])
    
    def transform_imu_to_camera_frame(self, imu_data: np.ndarray, current_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform IMU data from IMU coordinate frame to camera coordinate frame.
        
        This is crucial for multi-sensor fusion - IMU and camera must be in the same coordinate system.
        
        Args:
            imu_data: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z] in IMU frame
            current_pose: Current camera pose [x, y, z, qx, qy, qz, qw] (optional, for gravity compensation)
            
        Returns:
            IMU data transformed to camera coordinate frame
        """
        # Extract accelerometer and gyroscope data
        accel_imu = imu_data[:3]  # [ax, ay, az] in IMU frame
        gyro_imu = imu_data[3:]   # [gx, gy, gz] in IMU frame
        
        # Define IMU-to-Camera transformation (this depends on your specific setup)
        # For typical underwater ROV setups, common transformations are:
        
        # OPTION 1: IMU and camera are aligned (no transformation needed)
        # This assumes IMU is mounted with same orientation as camera
        if self.imu_camera_aligned:
            return imu_data
        
        # OPTION 2: Standard IMU-to-Camera transformation
        # Assuming IMU frame: X=forward, Y=right, Z=down
        # Camera frame: X=right, Y=down, Z=forward
        # Transformation: [imu_y, imu_z, imu_x]
        accel_camera = np.array([accel_imu[1], accel_imu[2], accel_imu[0]])  # Y->X, Z->Y, X->Z
        gyro_camera = np.array([gyro_imu[1], gyro_imu[2], gyro_imu[0]])     # Y->X, Z->Y, X->Z
        
        # OPTION 3: Use rotation matrix if you know the exact IMU mounting orientation
        # This is the most accurate approach if you have calibration data
        if hasattr(self, 'T_camera_imu') and self.T_camera_imu is not None:
            # Apply 3x3 rotation matrix transformation
            accel_camera = self.T_camera_imu[:3, :3] @ accel_imu
            gyro_camera = self.T_camera_imu[:3, :3] @ gyro_imu
        
        # OPTION 4: Gravity compensation using current pose (advanced)
        if current_pose is not None and len(current_pose) >= 7:
            # Remove gravity component from accelerometer using current orientation
            # This gives you linear acceleration in camera frame
            quat = current_pose[3:7]  # [qx, qy, qz, qw]
            
            if TF_AVAILABLE:
                # Use tf transformations
                gravity_world = np.array([0, 0, 9.81])  # Gravity in world frame
                rotation_matrix = tf_trans.quaternion_matrix(quat)[:3, :3]
                gravity_camera = rotation_matrix.T @ gravity_world
                accel_camera = accel_camera - gravity_camera  # Remove gravity
            else:
                # Use scipy fallback
                gravity_world = np.array([0, 0, 9.81])
                rot = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
                gravity_camera = rot.inv().apply(gravity_world)
                accel_camera = accel_camera - gravity_camera
        
        return np.concatenate([accel_camera, gyro_camera])
    
    def setup_imu_camera_calibration(self, calibration_method: str = "auto_detect"):
        """
        Setup IMU-to-camera transformation based on your specific hardware setup.
        
        Args:
            calibration_method: "auto_detect", "manual", "aligned", "kalibr_cam0", "kalibr_cam1"
        """
        if calibration_method == "aligned":
            # IMU and camera are perfectly aligned
            self.imu_camera_aligned = True
            self.T_camera_imu = np.eye(4)
            
        elif calibration_method == "kalibr_cam0":
            # Kalibr calibrated transformation for cam_0
            # From ReaqrVIO Kalibr calibration: 
            # qCM (quaternion): [-0.5000, 0.5024, -0.5002, -0.4974]
            # MrMC (translation): [0.0482, -0.0097, -0.0506]
            self.imu_camera_aligned = False
            
            # Convert quaternion to rotation matrix
            qx, qy, qz, qw = -0.5000, 0.5024, -0.5002, -0.4974
            
            # Quaternion to rotation matrix conversion
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
            ])
            
            # Translation vector
            t = np.array([0.0482, -0.0097, -0.0506])
            
            # Build 4x4 transformation matrix
            self.T_camera_imu = np.eye(4)
            self.T_camera_imu[:3, :3] = R
            self.T_camera_imu[:3, 3] = t
            
            print(f"Using Kalibr calibration for cam_0")
            print(f"  Translation: {t}")
            print(f"  Rotation quaternion: [{qx}, {qy}, {qz}, {qw}]")
            
        elif calibration_method == "kalibr_cam1":
            # Kalibr calibrated transformation for cam_1
            # qCM: [0.5012, 0.5012, 0.4991, -0.4985]
            # MrMC: [0.0617, 0.0098, -0.0507]
            self.imu_camera_aligned = False
            
            qx, qy, qz, qw = 0.5012, 0.5012, 0.4991, -0.4985
            
            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
            ])
            
            t = np.array([0.0617, 0.0098, -0.0507])
            
            self.T_camera_imu = np.eye(4)
            self.T_camera_imu[:3, :3] = R
            self.T_camera_imu[:3, 3] = t
            
            print(f"Using Kalibr calibration for cam_1")
            print(f"  Translation: {t}")
            print(f"  Rotation quaternion: [{qx}, {qy}, {qz}, {qw}]")
            
        elif calibration_method == "standard_rov":
            # Standard ROV mounting: IMU forward-right-down, Camera right-down-forward
            self.imu_camera_aligned = False
            self.T_camera_imu = np.array([
                [0, 1, 0, 0],  # Camera X = IMU Y (right)
                [0, 0, 1, 0],  # Camera Y = IMU Z (down)  
                [1, 0, 0, 0],  # Camera Z = IMU X (forward)
                [0, 0, 0, 1]
            ])
            
        elif calibration_method == "manual":
            # You can manually set this based on your specific setup
            print("Manual IMU-camera calibration needed. Please set self.T_camera_imu manually.")
            self.imu_camera_aligned = False
            self.T_camera_imu = None
            
        else:  # auto_detect
            # Use Kalibr cam_0 calibration as default
            print("Using Kalibr cam_0 calibration as default.")
            self.setup_imu_camera_calibration("kalibr_cam0")
    
    def interpolate_sensor_data(self, target_timestamp: float, sensor_data: List[Tuple], 
                              max_time_diff: float = 0.1) -> Optional[np.ndarray]:
        """
        Interpolate sensor data to match camera timestamp.
        
        Args:
            target_timestamp: Camera frame timestamp
            sensor_data: List of (timestamp, data) tuples
            max_time_diff: Maximum time difference for valid interpolation
            
        Returns:
            Interpolated sensor data or None if no valid data
        """
        if len(sensor_data) < 2:
            return None
        
        # Find nearest timestamps
        timestamps = np.array([item[0] for item in sensor_data])
        idx = np.searchsorted(timestamps, target_timestamp)
        
        if idx == 0 or idx >= len(timestamps):
            # Use nearest neighbor if at boundaries
            nearest_idx = np.argmin(np.abs(timestamps - target_timestamp))
            if abs(timestamps[nearest_idx] - target_timestamp) <= max_time_diff:
                return sensor_data[nearest_idx][1]
            return None
        
        # Linear interpolation
        t1, data1 = sensor_data[idx-1]
        t2, data2 = sensor_data[idx]
        
        if abs(t2 - t1) > max_time_diff * 2:  # Gap too large
            return None
        
        # Interpolation weight
        alpha = (target_timestamp - t1) / (t2 - t1)
        interpolated = data1 + alpha * (data2 - data1)
        
        return interpolated
    
    def extract_images_from_message(self, msg, camera_id: int, timestamp: float, 
                                  frame_id: str) -> Dict[str, str]:
        """
        Extract and save image from ROS message.
        
        Args:
            msg: ROS Image message
            camera_id: Camera ID (0-4)
            timestamp: Message timestamp
            frame_id: Unique frame identifier
            
        Returns:
            Dictionary with image paths
        """
        try:
            if CV_BRIDGE_AVAILABLE and self.bridge:
                # Convert ROS image to OpenCV image using cv_bridge
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            else:
                # Fallback: manual conversion from ROS Image message
                import struct
                # Basic conversion for common formats
                if msg.encoding == "bgr8":
                    cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                elif msg.encoding == "rgb8":
                    cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                elif msg.encoding == "mono8":
                    cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                else:
                    print(f"Unsupported encoding: {msg.encoding}")
                    return {f'cam{camera_id}_path': None}
            
            # Create filename
            filename = f"{frame_id}_cam{camera_id}_{timestamp:.6f}.png"
            filepath = self.output_dir / "images" / f"cam{camera_id}" / filename
            
            # Save image
            cv2.imwrite(str(filepath), cv_image)
            
            # Return relative path for CSV
            relative_path = f"images/cam{camera_id}/{filename}"
            
            return {
                f'cam{camera_id}_path': relative_path,
                f'cam{camera_id}_width': cv_image.shape[1],
                f'cam{camera_id}_height': cv_image.shape[0]
            }
            
        except Exception as e:
            print(f"Error extracting image from cam{camera_id}: {e}")
            return {f'cam{camera_id}_path': None}
    
    def process_bag_file(self, bag_path: str, bag_name: str) -> List[Dict]:
        """
        Process a single bag file and extract all synchronized data.
        
        Args:
            bag_path: Path to the ROS bag file
            bag_name: Name identifier for the bag
            
        Returns:
            List of extracted data records
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING BAG: {bag_name}")
        print(f"{'='*60}")
        
        bag_data = []
        
        # Storage for different sensor data streams
        camera_messages = {i: [] for i in range(5)}
        pose_messages = []
        imu_messages = []
        pressure_messages = []
        thrust_messages = []
        
        try:
            with rosbag.Bag(bag_path, 'r') as bag:
                # Get bag info
                bag_info = bag.get_type_and_topic_info()
                total_messages = sum([topic_info.message_count for topic_info in bag_info.topics.values()])
                
                print(f"Reading {total_messages} messages from bag...")
                
                # Read all messages and organize by type
                progress_bar = tqdm(bag.read_messages(), total=total_messages, desc="Reading bag")
                
                for topic, msg, t in progress_bar:
                    timestamp = t.to_sec()
                    
                    # Camera messages
                    for i, cam_topic in enumerate(self.camera_topics):
                        if topic == cam_topic:
                            camera_messages[i].append((timestamp, msg))
                            break
                    
                    # Pose messages (ground truth)
                    if topic in self.pose_topics:
                        if hasattr(msg, 'pose'):
                            pose = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose
                        else:
                            pose = msg
                        
                        pos = [pose.position.x, pose.position.y, pose.position.z]
                        quat = [pose.orientation.x, pose.orientation.y, 
                               pose.orientation.z, pose.orientation.w]
                        pose_data = np.array(pos + quat)
                        pose_messages.append((timestamp, pose_data))
                    
                    # IMU messages
                    if topic in self.imu_topics:
                        accel = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                        gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                        imu_data = np.array(accel + gyro)
                        imu_messages.append((timestamp, imu_data))
                    
                    # Pressure messages
                    if topic in self.pressure_topics:
                        pressure_data = np.array([msg.fluid_pressure])
                        pressure_messages.append((timestamp, pressure_data))
                    
                    # Thrust messages
                    if topic in self.thrust_topics:
                        if hasattr(msg, 'channels'):
                            thrust_data = np.array(msg.channels)
                            thrust_messages.append((timestamp, thrust_data))
                
                progress_bar.close()
                
                # Sort all messages by timestamp
                for cam_id in camera_messages:
                    camera_messages[cam_id].sort(key=lambda x: x[0])
                pose_messages.sort(key=lambda x: x[0])
                imu_messages.sort(key=lambda x: x[0])
                pressure_messages.sort(key=lambda x: x[0])
                thrust_messages.sort(key=lambda x: x[0])
                
                print(f"Organized messages:")
                print(f"  Camera frames: {[len(camera_messages[i]) for i in range(5)]}")
                print(f"  Pose messages: {len(pose_messages)}")
                print(f"  IMU messages: {len(imu_messages)}")
                print(f"  Pressure messages: {len(pressure_messages)}")
                print(f"  Thrust messages: {len(thrust_messages)}")
                
                # Process synchronized frames
                print(f"Processing synchronized frames...")
                
                # Use cam0 as the reference camera for timing
                reference_camera = 0
                cam0_messages = camera_messages[reference_camera]
                
                for frame_idx, (ref_timestamp, ref_msg) in enumerate(tqdm(cam0_messages, desc="Processing frames")):
                    
                    try:
                        frame_id = f"{bag_name}_{frame_idx:06d}"
                        
                        # Initialize frame data
                        frame_data = {
                            'frame_id': frame_id,
                            'bag_name': bag_name,
                            'timestamp': ref_timestamp,
                            'frame_index': frame_idx
                        }
                        
                        # Extract images from all cameras (find closest timestamp)
                        all_images_extracted = True
                        for cam_id in range(5):
                            if cam_id == reference_camera:
                                # Use reference message
                                image_data = self.extract_images_from_message(
                                    ref_msg, cam_id, ref_timestamp, frame_id
                                )
                            else:
                                # Find closest timestamp in other cameras
                                cam_messages = camera_messages[cam_id]
                                if not cam_messages:
                                    all_images_extracted = False
                                    break
                                
                                timestamps = [msg[0] for msg in cam_messages]
                                closest_idx = np.argmin(np.abs(np.array(timestamps) - ref_timestamp))
                                
                                if abs(timestamps[closest_idx] - ref_timestamp) > 0.1:  # 100ms threshold
                                    all_images_extracted = False
                                    break
                                
                                closest_msg = cam_messages[closest_idx][1]
                                image_data = self.extract_images_from_message(
                                    closest_msg, cam_id, ref_timestamp, frame_id
                                )
                            
                            frame_data.update(image_data)
                        
                        if not all_images_extracted:
                            continue
                        
                        # Get ground truth pose (interpolated)
                        gt_pose = self.interpolate_sensor_data(ref_timestamp, pose_messages)
                        if gt_pose is not None:
                            # Store world frame pose
                            frame_data.update({
                                'world_x': gt_pose[0], 'world_y': gt_pose[1], 'world_z': gt_pose[2],
                                'world_qx': gt_pose[3], 'world_qy': gt_pose[4], 
                                'world_qz': gt_pose[5], 'world_qw': gt_pose[6]
                            })
                            
                            # Transform to camera frame for delta calculation (will be done later)
                            frame_data['has_ground_truth'] = True
                        else:
                            frame_data['has_ground_truth'] = False
                        
                        # Get IMU data (interpolated and transformed to camera frame)
                        imu_data = self.interpolate_sensor_data(ref_timestamp, imu_messages)
                        if imu_data is not None:
                            # Transform IMU data to camera coordinate frame
                            imu_camera_frame = self.transform_imu_to_camera_frame(imu_data, gt_pose)
                            frame_data.update({
                                'accel_x': imu_camera_frame[0], 'accel_y': imu_camera_frame[1], 'accel_z': imu_camera_frame[2],
                                'gyro_x': imu_camera_frame[3], 'gyro_y': imu_camera_frame[4], 'gyro_z': imu_camera_frame[5],
                                # Also store raw IMU data for reference
                                'accel_x_raw': imu_data[0], 'accel_y_raw': imu_data[1], 'accel_z_raw': imu_data[2],
                                'gyro_x_raw': imu_data[3], 'gyro_y_raw': imu_data[4], 'gyro_z_raw': imu_data[5]
                            })
                        
                        # Get pressure data (interpolated)
                        pressure_data = self.interpolate_sensor_data(ref_timestamp, pressure_messages)
                        if pressure_data is not None:
                            frame_data['pressure'] = pressure_data[0]
                        
                        # Get thrust data (interpolated)
                        thrust_data = self.interpolate_sensor_data(ref_timestamp, thrust_messages)
                        if thrust_data is not None:
                            for i, thrust_val in enumerate(thrust_data):
                                frame_data[f'thrust_ch{i}'] = thrust_val
                        
                        bag_data.append(frame_data)
                        self.stats['successful_extractions'] += 1
                        
                    except Exception as e:
                        print(f"Error processing frame {frame_idx}: {e}")
                        self.stats['failed_extractions'] += 1
                        continue
                
                print(f"Extracted {len(bag_data)} synchronized frames from {bag_name}")
                self.stats['bags_processed'] += 1
                
        except Exception as e:
            print(f"Error processing bag {bag_name}: {e}")
        
        return bag_data
    
    def compute_camera_frame_deltas(self, data: List[Dict]) -> List[Dict]:
        """
        Compute delta poses in camera coordinate frame.
        
        This transforms the ground truth poses from world frame to camera frame
        and computes frame-to-frame deltas for visual odometry training.
        
        Args:
            data: List of extracted frame data
            
        Returns:
            Data with added camera-frame delta poses
        """
        print(f"\n{'='*60}")
        print("COMPUTING CAMERA FRAME DELTAS")
        print(f"{'='*60}")
        
        enhanced_data = []
        
        for i, frame in enumerate(tqdm(data, desc="Computing deltas")):
            enhanced_frame = frame.copy()
            
            if i == 0 or not frame['has_ground_truth']:
                # First frame or no ground truth - set deltas to zero
                enhanced_frame.update({
                    'delta_x': 0.0, 'delta_y': 0.0, 'delta_z': 0.0,
                    'delta_roll': 0.0, 'delta_pitch': 0.0, 'delta_yaw': 0.0
                })
                enhanced_data.append(enhanced_frame)
                continue
            
            prev_frame = data[i-1]
            if not prev_frame['has_ground_truth']:
                # Previous frame has no ground truth
                enhanced_frame.update({
                    'delta_x': 0.0, 'delta_y': 0.0, 'delta_z': 0.0,
                    'delta_roll': 0.0, 'delta_pitch': 0.0, 'delta_yaw': 0.0
                })
                enhanced_data.append(enhanced_frame)
                continue
            
            try:
                # Current and previous poses in world frame
                curr_pose = np.array([
                    frame['world_x'], frame['world_y'], frame['world_z'],
                    frame['world_qx'], frame['world_qy'], frame['world_qz'], frame['world_qw']
                ])
                
                prev_pose = np.array([
                    prev_frame['world_x'], prev_frame['world_y'], prev_frame['world_z'],
                    prev_frame['world_qx'], prev_frame['world_qy'], prev_frame['world_qz'], prev_frame['world_qw']
                ])
                
                # Transform current pose to previous camera's coordinate frame
                delta_camera_frame = self.transform_pose_world_to_camera(curr_pose, prev_pose)
                
                # Store camera frame deltas
                enhanced_frame.update({
                    'delta_x': delta_camera_frame[0],
                    'delta_y': delta_camera_frame[1], 
                    'delta_z': delta_camera_frame[2],
                    'delta_roll': delta_camera_frame[3],
                    'delta_pitch': delta_camera_frame[4],
                    'delta_yaw': delta_camera_frame[5]
                })
                
            except Exception as e:
                print(f"Error computing delta for frame {i}: {e}")
                enhanced_frame.update({
                    'delta_x': 0.0, 'delta_y': 0.0, 'delta_z': 0.0,
                    'delta_roll': 0.0, 'delta_pitch': 0.0, 'delta_yaw': 0.0
                })
            
            enhanced_data.append(enhanced_frame)
        
        print(f"Computed camera frame deltas for {len(enhanced_data)} frames")
        return enhanced_data
    
    def save_dataset(self, data: List[Dict], output_filename: str = "visual_odometry_dataset.csv"):
        """
        Save the complete dataset to CSV with professional formatting.
        
        Args:
            data: Complete extracted and processed data
            output_filename: Name of output CSV file
        """
        print(f"\n{'='*60}")
        print("SAVING DATASET")
        print(f"{'='*60}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Reorder columns for better organization
        column_order = [
            # Frame identification
            'frame_id', 'bag_name', 'timestamp', 'frame_index',
            
            # Camera paths
            'cam0_path', 'cam1_path', 'cam2_path', 'cam3_path', 'cam4_path',
            
            # Camera dimensions (if available)
            'cam0_width', 'cam0_height',
            
            # Ground truth deltas (camera frame) - PRIMARY TARGET
            'delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw',
            
            # World frame poses (reference)
            'world_x', 'world_y', 'world_z', 'world_qx', 'world_qy', 'world_qz', 'world_qw',
            
            # IMU data
            'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
            
            # Pressure/depth
            'pressure',
            
            # Control inputs
        ] + [col for col in df.columns if col.startswith('thrust_ch')] + [
            
            # Metadata
            'has_ground_truth'
        ]
        
        # Reorder DataFrame columns
        available_columns = [col for col in column_order if col in df.columns]
        other_columns = [col for col in df.columns if col not in column_order]
        final_columns = available_columns + other_columns
        
        df = df[final_columns]
        
        # Save CSV
        csv_path = self.output_dir / output_filename
        df.to_csv(csv_path, index=False, float_format='%.6f')
        
        # Generate statistics
        stats = {
            'total_frames': len(df),
            'frames_with_ground_truth': df['has_ground_truth'].sum(),
            'unique_bags': df['bag_name'].nunique(),
            'cameras_per_frame': 5,
            'delta_statistics': {
                'delta_x': {'mean': df['delta_x'].mean(), 'std': df['delta_x'].std(), 'min': df['delta_x'].min(), 'max': df['delta_x'].max()},
                'delta_y': {'mean': df['delta_y'].mean(), 'std': df['delta_y'].std(), 'min': df['delta_y'].min(), 'max': df['delta_y'].max()},
                'delta_z': {'mean': df['delta_z'].mean(), 'std': df['delta_z'].std(), 'min': df['delta_z'].min(), 'max': df['delta_z'].max()},
            },
            'extraction_date': datetime.now().isoformat()
        }
        
        # Save statistics
        stats_path = self.output_dir / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"Dataset saved to: {csv_path}")
        print(f"Statistics saved to: {stats_path}")
        print(f"Dataset summary:")
        print(f"  - Total frames: {stats['total_frames']}")
        print(f"  - Frames with ground truth: {stats['frames_with_ground_truth']}")
        print(f"  - Unique bags: {stats['unique_bags']}")
        print(f"  - Delta X range: {stats['delta_statistics']['delta_x']['min']:.4f} to {stats['delta_statistics']['delta_x']['max']:.4f}")
        
        return csv_path, stats_path
    
    def process_all_bags(self, bag_directory: str) -> Tuple[str, str]:
        """
        Process all bag files in the specified directory.
        
        Args:
            bag_directory: Directory containing ROS bag files
            
        Returns:
            Tuple of (csv_path, stats_path)
        """
        bag_dir = Path(bag_directory)
        bag_files = list(bag_dir.glob("*.bag"))
        
        if not bag_files:
            raise ValueError(f"No bag files found in {bag_directory}")
        
        print(f"Found {len(bag_files)} bag files to process")
        
        all_data = []
        
        for bag_file in sorted(bag_files):
            bag_name = bag_file.stem  # Filename without extension
            bag_data = self.process_bag_file(str(bag_file), bag_name)
            all_data.extend(bag_data)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total extracted frames: {len(all_data)}")
        
        # Compute camera frame deltas
        enhanced_data = self.compute_camera_frame_deltas(all_data)
        
        # Save dataset
        csv_path, stats_path = self.save_dataset(enhanced_data)
        
        return csv_path, stats_path

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Extract visual odometry dataset from ROS bags")
    parser.add_argument("--bag_dir", type=str, default="data/raw", 
                       help="Directory containing ROS bag files")
    parser.add_argument("--output_dir", type=str, default="data/processed/visual_odometry_dataset",
                       help="Output directory for extracted dataset")
    parser.add_argument("--csv_name", type=str, default="visual_odometry_dataset.csv",
                       help="Name of output CSV file")
    
    args = parser.parse_args()
    
    print("="*80)
    print("UNDERWATER VISUAL ODOMETRY - DATASET EXTRACTION PIPELINE")
    print("="*80)
    print(f"Bag directory: {args.bag_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"CSV filename: {args.csv_name}")
    
    # Initialize extractor
    extractor = VisualOdometryDataExtractor(args.output_dir)
    
    # Process all bags
    try:
        csv_path, stats_path = extractor.process_all_bags(args.bag_dir)
        
        print(f"\nEXTRACTION COMPLETE!")
        print(f"Dataset CSV: {csv_path}")
        print(f"Statistics: {stats_path}")
        print(f"Images stored in: {extractor.output_dir / 'images'}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise

if __name__ == "__main__":
    main()