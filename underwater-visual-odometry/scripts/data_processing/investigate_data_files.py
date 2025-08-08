#!/usr/bin/env python3
"""
Data Files Investigation Script

This script analyzes all .bag files and .tum files in the data/raw directory
to understand what data we have available for training.

Author: Claude Code Assistant
Date: 2025-01-08
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    import rosbag
    import rospy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: rosbag/rospy not available. Will analyze what we can without ROS.")

class DataInvestigator:
    def __init__(self, data_dir=None):
        if data_dir is None:
            self.data_dir = project_root / "data" / "raw"
        else:
            self.data_dir = Path(data_dir)
        
        self.results = {
            'investigation_date': datetime.now().isoformat(),
            'data_directory': str(self.data_dir),
            'bag_files': [],
            'tum_files': [],
            'summary': {}
        }
    
    def investigate_bag_file(self, bag_path):
        """Investigate a single .bag file"""
        print(f"\n{'='*60}")
        print(f"INVESTIGATING BAG FILE: {bag_path.name}")
        print(f"{'='*60}")
        
        bag_info = {
            'filename': bag_path.name,
            'full_path': str(bag_path),
            'file_size_mb': round(bag_path.stat().st_size / (1024*1024), 2),
            'exists': bag_path.exists(),
            'topics': [],
            'duration_seconds': 0,
            'message_counts': {},
            'start_time': None,
            'end_time': None
        }
        
        print(f"File size: {bag_info['file_size_mb']} MB")
        print(f"File exists: {bag_info['exists']}")
        
        if not bag_path.exists():
            print(f"ERROR: File does not exist!")
            return bag_info
        
        if not ROS_AVAILABLE:
            print("Cannot analyze bag contents without rosbag library")
            print("Install with: pip install bagpy rosbag rospkg")
            return bag_info
        
        try:
            # Open bag file
            with rosbag.Bag(str(bag_path), 'r') as bag:
                # Get bag info
                bag_info['start_time'] = bag.get_start_time()
                bag_info['end_time'] = bag.get_end_time() 
                bag_info['duration_seconds'] = bag.get_end_time() - bag.get_start_time()
                
                print(f"Duration: {bag_info['duration_seconds']:.2f} seconds")
                print(f"Start time: {datetime.fromtimestamp(bag_info['start_time'])}")
                print(f"End time: {datetime.fromtimestamp(bag_info['end_time'])}")
                
                # Get topic info
                topics_info = bag.get_type_and_topic_info()
                
                print(f"\nTopics found ({len(topics_info.topics)}):")
                print(f"{'Topic':<50} {'Type':<30} {'Messages':<10}")
                print("-" * 90)
                
                for topic_name, topic_info in topics_info.topics.items():
                    msg_count = topic_info.message_count
                    msg_type = topic_info.msg_type
                    
                    bag_info['topics'].append({
                        'name': topic_name,
                        'type': msg_type,
                        'message_count': msg_count
                    })
                    
                    bag_info['message_counts'][topic_name] = msg_count
                    
                    print(f"{topic_name:<50} {msg_type:<30} {msg_count:<10}")
                
                # Analyze camera topics specifically
                camera_topics = [t for t in bag_info['topics'] if 'camera' in t['name'] or 'image' in t['name']]
                if camera_topics:
                    print(f"\nCamera/Image Topics Found:")
                    for topic in camera_topics:
                        fps = topic['message_count'] / bag_info['duration_seconds'] if bag_info['duration_seconds'] > 0 else 0
                        print(f"  {topic['name']}: {topic['message_count']} messages, ~{fps:.1f} fps")
                
                # Analyze odometry/pose topics
                odom_topics = [t for t in bag_info['topics'] if any(keyword in t['name'].lower() for keyword in ['odom', 'pose', 'nav', 'tf'])]
                if odom_topics:
                    print(f"\nOdometry/Pose Topics Found:")
                    for topic in odom_topics:
                        print(f"  {topic['name']}: {topic['message_count']} messages, type: {topic['type']}")
                
        except Exception as e:
            print(f"ERROR analyzing bag file: {e}")
            bag_info['error'] = str(e)
        
        return bag_info
    
    def investigate_tum_file(self, tum_path):
        """Investigate a single .tum file"""
        print(f"\n{'='*60}")
        print(f"INVESTIGATING TUM FILE: {tum_path.name}")  
        print(f"{'='*60}")
        
        tum_info = {
            'filename': tum_path.name,
            'full_path': str(tum_path),
            'file_size_kb': round(tum_path.stat().st_size / 1024, 2),
            'exists': tum_path.exists(),
            'num_poses': 0,
            'duration_seconds': 0,
            'trajectory_length_meters': 0,
            'start_time': None,
            'end_time': None,
            'position_range': {},
            'sample_poses': []
        }
        
        print(f"File size: {tum_info['file_size_kb']} KB")
        print(f"File exists: {tum_info['exists']}")
        
        if not tum_path.exists():
            print(f"ERROR: File does not exist!")
            return tum_info
        
        try:
            # Read TUM file (format: timestamp tx ty tz qx qy qz qw)
            with open(tum_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
            
            if not lines:
                print("TUM file is empty or contains only comments")
                return tum_info
            
            # Parse poses
            poses = []
            for line in lines[:10]:  # Show first 10 lines as samples
                parts = line.split()
                if len(parts) >= 8:  # timestamp + 7 pose values
                    timestamp = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    poses.append([timestamp, tx, ty, tz, qx, qy, qz, qw])
            
            # Parse all poses for statistics
            all_poses = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 8:
                    timestamp = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    all_poses.append([timestamp, tx, ty, tz])
            
            if all_poses:
                all_poses = np.array(all_poses)
                timestamps = all_poses[:, 0]
                positions = all_poses[:, 1:4]
                
                tum_info['num_poses'] = len(all_poses)
                tum_info['start_time'] = timestamps[0]
                tum_info['end_time'] = timestamps[-1]
                tum_info['duration_seconds'] = timestamps[-1] - timestamps[0]
                
                # Calculate trajectory length
                if len(positions) > 1:
                    diffs = np.diff(positions, axis=0)
                    distances = np.linalg.norm(diffs, axis=1)
                    tum_info['trajectory_length_meters'] = np.sum(distances)
                
                # Position ranges
                tum_info['position_range'] = {
                    'x_min': float(positions[:, 0].min()),
                    'x_max': float(positions[:, 0].max()),
                    'y_min': float(positions[:, 1].min()),
                    'y_max': float(positions[:, 1].max()),
                    'z_min': float(positions[:, 2].min()),
                    'z_max': float(positions[:, 2].max())
                }
                
                # Store sample poses
                tum_info['sample_poses'] = poses[:5]  # First 5 poses
                
                print(f"Number of poses: {tum_info['num_poses']}")
                print(f"Duration: {tum_info['duration_seconds']:.2f} seconds")
                print(f"Trajectory length: {tum_info['trajectory_length_meters']:.2f} meters")
                print(f"Average speed: {tum_info['trajectory_length_meters']/tum_info['duration_seconds']:.3f} m/s")
                
                print(f"\nPosition ranges:")
                print(f"  X: {tum_info['position_range']['x_min']:.3f} to {tum_info['position_range']['x_max']:.3f} meters")
                print(f"  Y: {tum_info['position_range']['y_min']:.3f} to {tum_info['position_range']['y_max']:.3f} meters") 
                print(f"  Z: {tum_info['position_range']['z_min']:.3f} to {tum_info['position_range']['z_max']:.3f} meters")
                
                print(f"\nSample poses (first 5):")
                print(f"{'Timestamp':<15} {'X':<10} {'Y':<10} {'Z':<10}")
                print("-" * 50)
                for pose in poses[:5]:
                    print(f"{pose[0]:<15.3f} {pose[1]:<10.3f} {pose[2]:<10.3f} {pose[3]:<10.3f}")
                
        except Exception as e:
            print(f"ERROR analyzing TUM file: {e}")
            tum_info['error'] = str(e)
        
        return tum_info
    
    def run_investigation(self):
        """Run complete data investigation"""
        print("="*80)
        print("UNDERWATER VISUAL ODOMETRY - DATA FILES INVESTIGATION")
        print("="*80)
        print(f"Investigating data directory: {self.data_dir}")
        print(f"ROS tools available: {ROS_AVAILABLE}")
        
        # Find all .bag files
        bag_files = list(self.data_dir.glob("*.bag"))
        print(f"\nFound {len(bag_files)} .bag files")
        
        # Find all .tum files  
        tum_files = list(self.data_dir.glob("*.tum"))
        print(f"Found {len(tum_files)} .tum files")
        
        # Investigate each bag file
        for bag_file in sorted(bag_files):
            bag_info = self.investigate_bag_file(bag_file)
            self.results['bag_files'].append(bag_info)
        
        # Investigate each tum file
        for tum_file in sorted(tum_files):
            tum_info = self.investigate_tum_file(tum_file)
            self.results['tum_files'].append(tum_info)
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def generate_summary(self):
        """Generate investigation summary"""
        print(f"\n{'='*80}")
        print("INVESTIGATION SUMMARY")
        print("="*80)
        
        summary = {
            'total_bag_files': len(self.results['bag_files']),
            'total_tum_files': len(self.results['tum_files']),
            'total_data_size_mb': 0,
            'bag_duration_total': 0,
            'tum_poses_total': 0,
            'camera_info': {},
            'recommendations': []
        }
        
        # Analyze bag files
        if self.results['bag_files']:
            print(f"\nBAG FILES SUMMARY:")
            for i, bag in enumerate(self.results['bag_files']):
                print(f"  {i+1}. {bag['filename']}")
                print(f"     Size: {bag['file_size_mb']} MB")
                if 'duration_seconds' in bag:
                    print(f"     Duration: {bag['duration_seconds']:.1f}s")
                    summary['bag_duration_total'] += bag['duration_seconds']
                if 'topics' in bag:
                    print(f"     Topics: {len(bag['topics'])}")
                summary['total_data_size_mb'] += bag['file_size_mb']
                print()
        
        # Analyze TUM files
        if self.results['tum_files']:
            print(f"TUM FILES SUMMARY:")
            for i, tum in enumerate(self.results['tum_files']):
                print(f"  {i+1}. {tum['filename']}")
                print(f"     Size: {tum['file_size_kb']} KB")
                if 'num_poses' in tum:
                    print(f"     Poses: {tum['num_poses']}")
                    print(f"     Duration: {tum.get('duration_seconds', 0):.1f}s")
                    print(f"     Trajectory: {tum.get('trajectory_length_meters', 0):.2f}m")
                    summary['tum_poses_total'] += tum['num_poses']
                print()
        
        # Generate recommendations
        recommendations = []
        
        if summary['total_bag_files'] == 0:
            recommendations.append("❌ No bag files found - need ROS bag data for visual odometry")
        elif summary['total_bag_files'] < 3:
            recommendations.append("⚠️ Only few bag files - consider recording more sequences for robust training")
        else:
            recommendations.append("✅ Good number of bag files for training")
        
        if summary['total_tum_files'] == 0:
            recommendations.append("❌ No ground truth (.tum) files found - need reference trajectories")
        else:
            recommendations.append("✅ Ground truth trajectory available")
        
        if summary['total_data_size_mb'] > 1000:
            recommendations.append("✅ Substantial dataset size for deep learning")
        elif summary['total_data_size_mb'] > 100:
            recommendations.append("⚠️ Moderate dataset size - may need data augmentation")
        else:
            recommendations.append("❌ Small dataset - definitely need more data or strong augmentation")
        
        summary['recommendations'] = recommendations
        
        print("RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")
        
        print(f"\nOVERALL DATA STATUS:")
        print(f"  Total bag files: {summary['total_bag_files']}")
        print(f"  Total TUM files: {summary['total_tum_files']}")
        print(f"  Total data size: {summary['total_data_size_mb']:.1f} MB")
        print(f"  Total bag duration: {summary['bag_duration_total']:.1f} seconds")
        print(f"  Total ground truth poses: {summary['tum_poses_total']}")
        
        self.results['summary'] = summary
    
    def save_results(self):
        """Save investigation results to JSON file"""
        output_file = self.data_dir.parent / "data_investigation_report.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n📊 Investigation report saved to: {output_file}")
        except Exception as e:
            print(f"Error saving report: {e}")
        
        return output_file

def main():
    """Main function"""
    print("Starting data files investigation...")
    
    # Initialize investigator
    investigator = DataInvestigator()
    
    # Run investigation
    results = investigator.run_investigation()
    
    print(f"\n🎯 Investigation complete!")
    print(f"Check the generated JSON report for detailed results.")
    
    return results

if __name__ == "__main__":
    main()