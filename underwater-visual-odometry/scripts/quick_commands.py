#!/usr/bin/env python3
"""
Quick command interface for the reorganized underwater visual odometry project.
"""

import os
import sys
import subprocess
import argparse

def run_docker_command(command):
    """Run command in Docker container."""
    full_command = f'docker exec underwater_vo bash -c "cd /app && {command}"'
    print(f"Running: {command}")
    return subprocess.run(full_command, shell=True)

def extract_data(bag_id=None, all_bags=False):
    """Extract data from ROS bags."""
    if all_bags:
        print("Extracting all 5 bags...")
        for i in range(5):
            bag_name = f"ariel_2023-12-21-14-{24+i//2}-{42+(i%2)*55}_{i}.bag"
            cmd = f"python3 scripts/data_processing/extract_rosbag_data.py --bag data/raw/{bag_name} --output data/processed/bag_{i}"
            run_docker_command(cmd)
    elif bag_id is not None:
        bag_names = [
            "ariel_2023-12-21-14-24-42_0.bag",
            "ariel_2023-12-21-14-25-37_1.bag", 
            "ariel_2023-12-21-14-26-32_2.bag",
            "ariel_2023-12-21-14-27-27_3.bag",
            "ariel_2023-12-21-14-28-22_4.bag"
        ]
        if 0 <= bag_id < len(bag_names):
            cmd = f"python3 scripts/data_processing/extract_rosbag_data.py --bag data/raw/{bag_names[bag_id]} --output data/processed/bag_{bag_id}"
            run_docker_command(cmd)
        else:
            print(f"Invalid bag ID: {bag_id}. Use 0-4.")

def train_model(model_type="baseline", epochs=50, batch_size=32):
    """Train a model."""
    if model_type == "baseline":
        cmd = f"python3 scripts/training/baseline/train_baseline.py --epochs {epochs} --batch_size {batch_size}"
    elif model_type == "attention":
        cmd = f"python3 scripts/training/attention/train_attention_window.py --epochs {epochs} --window_size 5"
    else:
        print(f"Unknown model type: {model_type}")
        return
    
    run_docker_command(cmd)

def evaluate_model(model_dir, eval_type="quick"):
    """Evaluate a trained model."""
    if eval_type == "quick":
        cmd = f"python3 scripts/evaluation/common/simple_trajectory_plot.py --model_dir {model_dir} --subsample 20"
    elif eval_type == "sequential":
        cmd = f"python3 scripts/evaluation/common/evaluate_sequential_bags.py --model_dir {model_dir}"
    elif eval_type == "comprehensive":
        cmd = f"python3 scripts/evaluation/common/evaluate_model.py --model_dir {model_dir}"
    else:
        print(f"Unknown evaluation type: {eval_type}")
        return
    
    run_docker_command(cmd)

def list_models():
    """List available trained models."""
    cmd = "ls -la output/models/"
    run_docker_command(cmd)

def main():
    parser = argparse.ArgumentParser(description='Quick commands for underwater VO project')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract data from ROS bags')
    extract_parser.add_argument('--bag', type=int, help='Extract specific bag (0-4)')
    extract_parser.add_argument('--all', action='store_true', help='Extract all bags')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--type', choices=['baseline', 'attention'], default='baseline')
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch_size', type=int, default=32)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a model')
    eval_parser.add_argument('--model', required=True, help='Model directory (e.g. output/models/model_X)')
    eval_parser.add_argument('--type', choices=['quick', 'sequential', 'comprehensive'], default='quick')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extract_data(args.bag, args.all)
    elif args.command == 'train':
        train_model(args.type, args.epochs, args.batch_size)
    elif args.command == 'eval':
        evaluate_model(args.model, args.type)
    elif args.command == 'list':
        list_models()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()