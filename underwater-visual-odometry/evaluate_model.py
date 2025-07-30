#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for UW-TransVO
Evaluates trained underwater visual odometry transformer model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import sys
from pathlib import Path
import time
from datetime import datetime
import argparse

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.datasets import UnderwaterVODataset
from training.losses import PoseLoss

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, checkpoint_path, config, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.config = config
        
        # Load model
        self.model = self._load_model()
        
        # Evaluation metrics storage
        self.results = {
            'translation_errors': [],
            'rotation_errors': [],
            'pose_predictions': [],
            'ground_truth_poses': [],
            'per_sample_metrics': [],
            'summary_metrics': {},
            'evaluation_time': None
        }
        
        print(f"Model Evaluator initialized")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self):
        """Load trained model from checkpoint"""
        print(f"Loading model from: {self.checkpoint_path}")
        
        # Create model
        model = UWTransVO(**self.config['model']).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        print("Model loaded successfully")
        return model
    
    def evaluate_dataset(self, dataset, batch_size=4):
        """Evaluate model on given dataset"""
        print(f"\\nEvaluating on dataset with {len(dataset)} samples...")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
        
        criterion = PoseLoss(translation_weight=1.0, rotation_weight=10.0)
        
        start_time = time.time()
        total_samples = 0
        batch_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                images = batch['images'].to(self.device)
                camera_ids = batch['camera_ids'].to(self.device)
                camera_mask = batch['camera_mask'].to(self.device)
                pose_target = batch['pose_target'].to(self.device)
                
                # Forward pass
                predictions = self.model(
                    images=images,
                    camera_ids=camera_ids,
                    camera_mask=camera_mask
                )
                
                # Calculate losses
                loss_dict = criterion(predictions['pose'], pose_target)
                batch_losses.append(loss_dict['total_loss'].item())
                
                # Store predictions and targets for detailed analysis
                pred_poses = predictions['pose'].cpu().numpy()
                true_poses = pose_target.cpu().numpy()
                
                # Calculate per-sample metrics
                for i in range(pred_poses.shape[0]):
                    pred_pose = pred_poses[i]
                    true_pose = true_poses[i]
                    
                    # Translation error (Euclidean distance)
                    trans_error = np.linalg.norm(pred_pose[:3] - true_pose[:3])
                    
                    # Rotation error (angular difference)
                    rot_error = np.linalg.norm(pred_pose[3:] - true_pose[3:])
                    
                    self.results['translation_errors'].append(trans_error)
                    self.results['rotation_errors'].append(rot_error)
                    self.results['pose_predictions'].append(pred_pose)
                    self.results['ground_truth_poses'].append(true_pose)
                    
                    self.results['per_sample_metrics'].append({
                        'sample_idx': total_samples,
                        'translation_error': trans_error,
                        'rotation_error': rot_error,
                        'total_loss': loss_dict['total_loss'].item(),
                        'translation_loss': loss_dict.get('translation_loss', 0),
                        'rotation_loss': loss_dict.get('rotation_loss', 0)
                    })
                    
                    total_samples += 1
                
                if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                    print(f"  Batch {batch_idx+1}/{len(dataloader)} - "
                          f"Loss: {loss_dict['total_loss'].item():.6f}")
        
        evaluation_time = time.time() - start_time
        self.results['evaluation_time'] = evaluation_time
        
        print(f"Evaluation completed in {evaluation_time:.2f} seconds")
        return self._calculate_summary_metrics()
    
    def _calculate_summary_metrics(self):
        """Calculate summary statistics"""
        trans_errors = np.array(self.results['translation_errors'])
        rot_errors = np.array(self.results['rotation_errors'])
        
        summary = {
            'total_samples': len(trans_errors),
            'translation_metrics': {
                'mean_error': float(np.mean(trans_errors)),
                'median_error': float(np.median(trans_errors)),
                'std_error': float(np.std(trans_errors)),
                'min_error': float(np.min(trans_errors)),
                'max_error': float(np.max(trans_errors)),
                'rmse': float(np.sqrt(np.mean(trans_errors**2)))
            },
            'rotation_metrics': {
                'mean_error': float(np.mean(rot_errors)),
                'median_error': float(np.median(rot_errors)),
                'std_error': float(np.std(rot_errors)),
                'min_error': float(np.min(rot_errors)),
                'max_error': float(np.max(rot_errors)),
                'rmse': float(np.sqrt(np.mean(rot_errors**2)))
            },
            'combined_metrics': {
                'mean_total_error': float(np.mean(trans_errors + rot_errors)),
                'samples_under_threshold': {
                    'trans_0.1': int(np.sum(trans_errors < 0.1)),
                    'trans_0.05': int(np.sum(trans_errors < 0.05)),
                    'rot_0.1': int(np.sum(rot_errors < 0.1)),
                    'rot_0.05': int(np.sum(rot_errors < 0.05))
                }
            }
        }
        
        self.results['summary_metrics'] = summary
        return summary
    
    def generate_visualizations(self, output_dir='evaluation_results'):
        """Generate evaluation visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\\nGenerating visualizations in {output_dir}...")
        
        # Set up matplotlib
        plt.style.use('dark_background')
        fig_size = (15, 10)
        
        # 1. Error Distribution Plots
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle('UW-TransVO Model Evaluation Results', fontsize=16, color='white')
        
        # Translation errors histogram
        axes[0,0].hist(self.results['translation_errors'], bins=50, color='cyan', alpha=0.7)
        axes[0,0].set_title('Translation Error Distribution', color='white')
        axes[0,0].set_xlabel('Translation Error (m)', color='white')
        axes[0,0].set_ylabel('Frequency', color='white')
        axes[0,0].grid(True, alpha=0.3)
        
        # Rotation errors histogram
        axes[0,1].hist(self.results['rotation_errors'], bins=50, color='orange', alpha=0.7)
        axes[0,1].set_title('Rotation Error Distribution', color='white')
        axes[0,1].set_xlabel('Rotation Error (rad)', color='white')
        axes[0,1].set_ylabel('Frequency', color='white')
        axes[0,1].grid(True, alpha=0.3)
        
        # Translation error over samples
        sample_indices = range(len(self.results['translation_errors']))
        axes[1,0].plot(sample_indices, self.results['translation_errors'], 
                      color='cyan', alpha=0.7, linewidth=1)
        axes[1,0].set_title('Translation Error Over Samples', color='white')
        axes[1,0].set_xlabel('Sample Index', color='white')
        axes[1,0].set_ylabel('Translation Error (m)', color='white')
        axes[1,0].grid(True, alpha=0.3)
        
        # Rotation error over samples
        axes[1,1].plot(sample_indices, self.results['rotation_errors'], 
                      color='orange', alpha=0.7, linewidth=1)
        axes[1,1].set_title('Rotation Error Over Samples', color='white')
        axes[1,1].set_xlabel('Sample Index', color='white')
        axes[1,1].set_ylabel('Rotation Error (rad)', color='white')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='black')
        plt.close()
        
        # 2. Prediction vs Ground Truth Scatter Plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Predictions vs Ground Truth', fontsize=16, color='white')
        
        predictions = np.array(self.results['pose_predictions'])
        ground_truth = np.array(self.results['ground_truth_poses'])
        
        pose_labels = ['X (m)', 'Y (m)', 'Z (m)', 'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
        
        for i in range(6):
            row = i // 3
            col = i % 3
            
            axes[row, col].scatter(ground_truth[:, i], predictions[:, i], 
                                 c=colors[i], alpha=0.6, s=10)
            
            # Perfect prediction line
            min_val = min(ground_truth[:, i].min(), predictions[:, i].min())
            max_val = max(ground_truth[:, i].max(), predictions[:, i].max())
            axes[row, col].plot([min_val, max_val], [min_val, max_val], 
                              'white', linestyle='--', alpha=0.8)
            
            axes[row, col].set_title(f'{pose_labels[i]}', color='white')
            axes[row, col].set_xlabel('Ground Truth', color='white')
            axes[row, col].set_ylabel('Predicted', color='white')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'predictions_vs_truth.png', dpi=300, bbox_inches='tight',
                   facecolor='black')
        plt.close()
        
        # 3. Performance Summary Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        metrics = self.results['summary_metrics']
        trans_metrics = [
            metrics['translation_metrics']['mean_error'],
            metrics['translation_metrics']['median_error'],
            metrics['translation_metrics']['rmse']
        ]
        rot_metrics = [
            metrics['rotation_metrics']['mean_error'],
            metrics['rotation_metrics']['median_error'],
            metrics['rotation_metrics']['rmse']
        ]
        
        x = np.arange(3)
        width = 0.35
        
        ax.bar(x - width/2, trans_metrics, width, label='Translation', color='cyan', alpha=0.7)
        ax.bar(x + width/2, rot_metrics, width, label='Rotation', color='orange', alpha=0.7)
        
        ax.set_title('Model Performance Summary', color='white', fontsize=14)
        ax.set_xlabel('Metrics', color='white')
        ax.set_ylabel('Error', color='white')
        ax.set_xticks(x)
        ax.set_xticklabels(['Mean', 'Median', 'RMSE'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight',
                   facecolor='black')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def save_results(self, output_dir='evaluation_results'):
        """Save evaluation results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save summary metrics as JSON
        results_file = output_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'evaluation_info': {
                    'model_checkpoint': str(self.checkpoint_path),
                    'evaluation_date': datetime.now().isoformat(),
                    'evaluation_time_seconds': self.results['evaluation_time'],
                    'device': str(self.device)
                },
                'summary_metrics': self.results['summary_metrics']
            }, f, indent=2)
        
        # Save detailed per-sample results as CSV
        df = pd.DataFrame(self.results['per_sample_metrics'])
        df.to_csv(output_dir / 'per_sample_results.csv', index=False)
        
        # Save predictions and ground truth
        pred_df = pd.DataFrame(
            self.results['pose_predictions'],
            columns=['pred_x', 'pred_y', 'pred_z', 'pred_roll', 'pred_pitch', 'pred_yaw']
        )
        gt_df = pd.DataFrame(
            self.results['ground_truth_poses'],
            columns=['gt_x', 'gt_y', 'gt_z', 'gt_roll', 'gt_pitch', 'gt_yaw']
        )
        combined_df = pd.concat([pred_df, gt_df], axis=1)
        combined_df.to_csv(output_dir / 'predictions_and_ground_truth.csv', index=False)
        
        print(f"\\nResults saved to {output_dir}")
        print(f"  - Summary: {results_file}")
        print(f"  - Per-sample: {output_dir / 'per_sample_results.csv'}")
        print(f"  - Predictions: {output_dir / 'predictions_and_ground_truth.csv'}")
    
    def print_summary(self):
        """Print evaluation summary"""
        metrics = self.results['summary_metrics']
        
        print("\\n" + "="*80)
        print("UW-TRANSVO MODEL EVALUATION SUMMARY")
        print("="*80)
        
        print(f"Total Samples Evaluated: {metrics['total_samples']}")
        print(f"Evaluation Time: {self.results['evaluation_time']:.2f} seconds")
        print(f"Samples per Second: {metrics['total_samples'] / self.results['evaluation_time']:.1f}")
        
        print("\\nTRANSLATION PERFORMANCE:")
        print("-" * 40)
        trans = metrics['translation_metrics']
        print(f"  Mean Error:   {trans['mean_error']:.6f} m")
        print(f"  Median Error: {trans['median_error']:.6f} m")
        print(f"  RMSE:         {trans['rmse']:.6f} m")
        print(f"  Std Dev:      {trans['std_error']:.6f} m")
        print(f"  Min Error:    {trans['min_error']:.6f} m")
        print(f"  Max Error:    {trans['max_error']:.6f} m")
        
        print("\\nROTATION PERFORMANCE:")
        print("-" * 40)
        rot = metrics['rotation_metrics']
        print(f"  Mean Error:   {rot['mean_error']:.6f} rad ({rot['mean_error']*180/np.pi:.2f}°)")
        print(f"  Median Error: {rot['median_error']:.6f} rad ({rot['median_error']*180/np.pi:.2f}°)")
        print(f"  RMSE:         {rot['rmse']:.6f} rad ({rot['rmse']*180/np.pi:.2f}°)")
        print(f"  Std Dev:      {rot['std_error']:.6f} rad ({rot['std_error']*180/np.pi:.2f}°)")
        print(f"  Min Error:    {rot['min_error']:.6f} rad ({rot['min_error']*180/np.pi:.2f}°)")
        print(f"  Max Error:    {rot['max_error']:.6f} rad ({rot['max_error']*180/np.pi:.2f}°)")
        
        print("\\nACCURACY THRESHOLDS:")
        print("-" * 40)
        thresh = metrics['combined_metrics']['samples_under_threshold']
        total = metrics['total_samples']
        print(f"  Translation < 0.05m: {thresh['trans_0.05']}/{total} ({thresh['trans_0.05']/total*100:.1f}%)")
        print(f"  Translation < 0.10m: {thresh['trans_0.1']}/{total} ({thresh['trans_0.1']/total*100:.1f}%)")
        print(f"  Rotation < 0.05rad:  {thresh['rot_0.05']}/{total} ({thresh['rot_0.05']/total*100:.1f}%)")
        print(f"  Rotation < 0.10rad:  {thresh['rot_0.1']}/{total} ({thresh['rot_0.1']/total*100:.1f}%)")
        
        print("\\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate UW-TransVO model')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/test_4cam_seq2_vision/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       default='checkpoints/test_4cam_seq2_vision/config.json',
                       help='Path to model config')
    parser.add_argument('--data_csv', type=str,
                       default='data/processed/training_dataset/training_data.csv',
                       help='Path to evaluation data')
    parser.add_argument('--output_dir', type=str,
                       default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to evaluate (None for all)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("UW-TransVO Model Evaluation")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Data: {args.data_csv}")
    print(f"Output: {args.output_dir}")
    
    # Create evaluator
    evaluator = ModelEvaluator(args.checkpoint, config)
    
    # Create evaluation dataset
    eval_dataset = UnderwaterVODataset(
        data_csv=args.data_csv,
        data_root='.',
        camera_ids=config['data']['camera_ids'],
        sequence_length=config['data']['sequence_length'],
        img_size=config['data']['img_size'],
        use_imu=config['data']['use_imu'],
        use_pressure=config['data']['use_pressure'],
        augmentation=False,  # No augmentation for evaluation
        split='val',  # Use validation split
        max_samples=args.max_samples
    )
    
    print(f"Evaluation dataset: {len(eval_dataset)} samples")
    
    # Run evaluation
    summary_metrics = evaluator.evaluate_dataset(eval_dataset, args.batch_size)
    
    # Generate results
    evaluator.print_summary()
    evaluator.generate_visualizations(args.output_dir)
    evaluator.save_results(args.output_dir)
    
    print(f"\\nEvaluation complete! Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()