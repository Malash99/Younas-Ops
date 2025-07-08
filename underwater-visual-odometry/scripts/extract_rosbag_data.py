#!/usr/bin/env python3
"""
Extract images and timestamps from ROS bag file for underwater visual odometry.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd
from cv_bridge import CvBridge
from tqdm import tqdm

# ROS imports
import rospy
import rosbag
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class RosbagExtractor:
    """Extract data from ROS bag files."""
    
    def __init__(self, bag_path: str, output_dir: str):
        """
        Initialize extractor.
        
        Args:
            bag_path: Path to ROS bag file
            output_dir: Directory to save extracted data
        """
        self.bag_path = bag_path
        self.output_dir = output_dir
        self.bridge = CvBridge()
        
        # Create output directories
        self.images_dir = os.path.join(output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Data storage
        self.image_data = []
        self.pose_data = []
        
    def extract_topics_info(self):
        """Print information about topics in the bag."""
        print(f"\nAnalyzing bag file: {self.bag_path}")
        
        with rosbag.Bag(self.bag_path, 'r') as bag:
            info = bag.get_type_and_topic_info()
            
            print("\nAvailable topics:")
            for topic, topic_info in info.topics.items():
                print(f"  {topic}:")
                print(f"    Type: {topic_info.msg_type}")
                print(f"    Count: {topic_info.message_count}")
                print(f"    Frequency: {topic_info.frequency:.2f} Hz" if topic_info.frequency else "    Frequency: N/A")
            
            # Get bag duration
            start_time = bag.get_start_time()
            end_time = bag.get_end_time()
            duration = end_time - start_time
            print(f"\nBag duration: {duration:.2f} seconds")
            
        return info.topics
    
    def find_image_topic(self, topics):
        """Find the most likely image topic."""
        image_types = ['sensor_msgs/Image', 'sensor_msgs/CompressedImage']
        image_keywords = ['camera', 'image', 'rgb', 'color']
        
        image_topics = []
        for topic, info in topics.items():
            if info.msg_type in image_types:
                image_topics.append(topic)
            else:
                # Check if topic name suggests it's an image
                for keyword in image_keywords:
                    if keyword in topic.lower():
                        image_topics.append(topic)
                        break
        
        if not image_topics:
            print("Warning: No image topics found!")
            return None
        
        if len(image_topics) == 1:
            return image_topics[0]
        
        # If multiple topics, let user choose
        print("\nMultiple image topics found:")
        for i, topic in enumerate(image_topics):
            print(f"  {i}: {topic}")
        
        # For now, just use the first one
        return image_topics[0]
    
    def find_pose_topic(self, topics):
        """Find the most likely pose/odometry topic."""
        pose_types = ['geometry_msgs/PoseStamped', 'nav_msgs/Odometry', 
                      'geometry_msgs/PoseWithCovarianceStamped']
        pose_keywords = ['pose', 'odom', 'odometry', 'position']
        
        pose_topics = []
        for topic, info in topics.items():
            if info.msg_type in pose_types:
                pose_topics.append(topic)
            else:
                # Check if topic name suggests it's pose data
                for keyword in pose_keywords:
                    if keyword in topic.lower():
                        pose_topics.append(topic)
                        break
        
        if not pose_topics:
            print("Warning: No pose topics found!")
            return None
        
        if len(pose_topics) == 1:
            return pose_topics[0]
        
        # If multiple topics, prefer 'odom' topics
        for topic in pose_topics:
            if 'odom' in topic.lower():
                return topic
        
        return pose_topics[0]
    
    def extract_images(self, image_topic: str, max_images: int = None):
        """Extract images from the bag."""
        print(f"\nExtracting images from topic: {image_topic}")
        
        count = 0
        with rosbag.Bag(self.bag_path, 'r') as bag:
            total_msgs = bag.get_message_count(image_topic)
            
            for topic, msg, t in tqdm(bag.read_messages(topics=[image_topic]), 
                                     total=total_msgs, 
                                     desc="Extracting images"):
                
                timestamp = t.to_sec()
                
                try:
                    # Convert to OpenCV image
                    if msg._type == 'sensor_msgs/CompressedImage':
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    else:  # sensor_msgs/Image
                        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    
                    # Save image
                    filename = f"frame_{count:06d}.png"
                    filepath = os.path.join(self.images_dir, filename)
                    cv2.imwrite(filepath, cv_image)
                    
                    # Store metadata
                    self.image_data.append({
                        'timestamp': timestamp,
                        'filename': filename,
                        'frame_id': count
                    })
                    
                    count += 1
                    
                    if max_images and count >= max_images:
                        break
                        
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue
        
        print(f"Extracted {count} images")
        
        # Save image metadata
        if self.image_data:
            df = pd.DataFrame(self.image_data)
            df.to_csv(os.path.join(self.output_dir, 'image_timestamps.csv'), index=False)
            print(f"Saved image timestamps to image_timestamps.csv")
    
    def extract_poses(self, pose_topic: str):
        """Extract pose data from the bag."""
        print(f"\nExtracting poses from topic: {pose_topic}")
        
        with rosbag.Bag(self.bag_path, 'r') as bag:
            # First, determine the message type
            info = bag.get_type_and_topic_info()
            msg_type = info.topics[pose_topic].msg_type
            
            total_msgs = bag.get_message_count(pose_topic)
            
            for topic, msg, t in tqdm(bag.read_messages(topics=[pose_topic]), 
                                     total=total_msgs,
                                     desc="Extracting poses"):
                
                timestamp = t.to_sec()
                
                try:
                    # Extract pose based on message type
                    if msg_type == 'nav_msgs/Odometry':
                        pose = msg.pose.pose
                        position = pose.position
                        orientation = pose.orientation
                    elif msg_type == 'geometry_msgs/PoseStamped':
                        pose = msg.pose
                        position = pose.position
                        orientation = pose.orientation
                    elif msg_type == 'geometry_msgs/PoseWithCovarianceStamped':
                        pose = msg.pose.pose
                        position = pose.position
                        orientation = pose.orientation
                    else:
                        print(f"Unknown pose message type: {msg_type}")
                        continue
                    
                    # Store pose data
                    self.pose_data.append({
                        'timestamp': timestamp,
                        'x': position.x,
                        'y': position.y,
                        'z': position.z,
                        'qx': orientation.x,
                        'qy': orientation.y,
                        'qz': orientation.z,
                        'qw': orientation.w
                    })
                    
                except Exception as e:
                    print(f"Error processing pose: {e}")
                    continue
        
        print(f"Extracted {len(self.pose_data)} poses")
        
        # Save pose data
        if self.pose_data:
            df = pd.DataFrame(self.pose_data)
            df.to_csv(os.path.join(self.output_dir, 'extracted_poses.csv'), index=False)
            print(f"Saved poses to extracted_poses.csv")
    
    def extract_all(self, image_topic: str = None, pose_topic: str = None, max_images: int = None):
        """Extract all data from the bag."""
        # Get topics info
        topics = self.extract_topics_info()
        
        # Find topics if not specified
        if image_topic is None:
            image_topic = self.find_image_topic(topics)
            if image_topic:
                print(f"Auto-detected image topic: {image_topic}")
        
        if pose_topic is None:
            pose_topic = self.find_pose_topic(topics)
            if pose_topic:
                print(f"Auto-detected pose topic: {pose_topic}")
        
        # Extract data
        if image_topic and image_topic in topics:
            self.extract_images(image_topic, max_images)
        else:
            print(f"Warning: Image topic '{image_topic}' not found in bag")
        
        if pose_topic and pose_topic in topics:
            self.extract_poses(pose_topic)
        else:
            print(f"Warning: Pose topic '{pose_topic}' not found in bag")
        
        print("\nExtraction complete!")


def main():
    parser = argparse.ArgumentParser(description='Extract data from ROS bag file')
    parser.add_argument('--bag', type=str, default='/app/data/raw/ariel_2023-12-21-14-26-32_2.bag',
                        help='Path to ROS bag file')
    parser.add_argument('--output', type=str, default='/app/data/raw',
                        help='Output directory')
    parser.add_argument('--image_topic', type=str, default=None,
                        help='Image topic name (auto-detect if not specified)')
    parser.add_argument('--pose_topic', type=str, default=None,
                        help='Pose topic name (auto-detect if not specified)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to extract')
    
    args = parser.parse_args()
    
    # Initialize ROS node (required for cv_bridge)
    rospy.init_node('rosbag_extractor', anonymous=True)
    
    # Extract data
    extractor = RosbagExtractor(args.bag, args.output)
    extractor.extract_all(args.image_topic, args.pose_topic, args.max_images)


if __name__ == "__main__":
    main()