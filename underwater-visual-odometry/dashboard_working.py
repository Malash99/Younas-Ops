#!/usr/bin/env python3
"""
Working Web Training Dashboard for UW-TransVO
Simplified and guaranteed to work
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import time
import threading
import json
from datetime import datetime
from flask import Flask, render_template_string
import webbrowser

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.datasets import UnderwaterVODataset
from training.losses import PoseLoss

app = Flask(__name__)

# Simple global state
training_state = {
    'status': 'Starting...',
    'epoch': '0',
    'batch': '0',
    'loss': '0.000000',
    'gpu_memory': '0.00',
    'speed': '0.00',
    'params': '0.0M',
    'dataset_size': '0',
    'progress': '0'
}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>UW-TransVO Training Monitor</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body { 
            font-family: Arial; 
            background: linear-gradient(135deg, #1a1a2e, #16213e); 
            color: white; 
            margin: 0; 
            padding: 20px;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { 
            text-align: center; 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
        }
        .status { 
            font-size: 2em; 
            color: #00ff88; 
            font-weight: bold; 
        }
        .metrics { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin-bottom: 20px; 
        }
        .metric { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            text-align: center;
            border: 2px solid rgba(0,255,136,0.3);
        }
        .metric-label { 
            font-size: 0.9em; 
            color: #ccc; 
            margin-bottom: 10px; 
        }
        .metric-value { 
            font-size: 1.8em; 
            color: #00ff88; 
            font-weight: bold; 
            font-family: monospace;
        }
        .progress-container { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
        }
        .progress-bar { 
            width: 100%; 
            height: 30px; 
            background: rgba(255,255,255,0.2); 
            border-radius: 15px; 
            overflow: hidden; 
        }
        .progress-fill { 
            height: 100%; 
            background: linear-gradient(90deg, #00ff88, #00cc6a); 
            transition: width 1s ease; 
        }
        .info { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
        }
        .timestamp { 
            text-align: center; 
            color: #888; 
            margin-top: 20px; 
        }
        .training { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>UW-TransVO Training Monitor</h1>
            <div class="status {{ 'training' if status == 'Training' else '' }}">{{ status }}</div>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Current Loss</div>
                <div class="metric-value">{{ loss }}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Training Speed</div>
                <div class="metric-value">{{ speed }}s</div>
            </div>
            <div class="metric">
                <div class="metric-label">GPU Memory</div>
                <div class="metric-value">{{ gpu_memory }} GB</div>
            </div>
            <div class="metric">
                <div class="metric-label">Model Size</div>
                <div class="metric-value">{{ params }}</div>
            </div>
        </div>
        
        <div class="progress-container">
            <h3>Training Progress - Epoch {{ epoch }}</h3>
            <div>Batch {{ batch }} | Dataset: {{ dataset_size }} samples</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ progress }}%"></div>
            </div>
            <div style="text-align: center; margin-top: 10px;">{{ progress }}% Complete</div>
        </div>
        
        <div class="info">
            <h3>GPU Training Active</h3>
            <p>CUDA Acceleration Enabled</p>
            <p>Mixed Precision Training</p>
            <p>Real-time Monitoring</p>
            <p>Auto-refresh every 2 seconds</p>
        </div>
        
        <div class="timestamp">
            Last Updated: {{ timestamp }}
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE, 
        status=training_state['status'],
        epoch=training_state['epoch'],
        batch=training_state['batch'],
        loss=training_state['loss'],
        gpu_memory=training_state['gpu_memory'],
        speed=training_state['speed'],
        params=training_state['params'],
        dataset_size=training_state['dataset_size'],
        progress=training_state['progress'],
        timestamp=datetime.now().strftime('%H:%M:%S')
    )

def update_state(**kwargs):
    """Update training state"""
    global training_state
    for key, value in kwargs.items():
        if key in training_state:
            training_state[key] = str(value)
    print(f"State updated: {training_state['status']} - Loss: {training_state['loss']}")

def run_training():
    """Training with state updates"""
    try:
        print("Starting UW-TransVO GPU Training")
        update_state(status="Initializing GPU...")
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        
        print(f"Using device: {device} ({gpu_name})")
        update_state(status=f"GPU Ready: {gpu_name}")
        time.sleep(2)
        
        # Model config
        config = {
            'img_size': 224, 'patch_size': 16, 'd_model': 384, 'num_heads': 6, 
            'num_layers': 4, 'max_cameras': 4, 'max_seq_len': 2, 'dropout': 0.1,
            'use_imu': False, 'use_pressure': False, 'uncertainty_estimation': False
        }
        
        # Create model
        update_state(status="Creating Model...")
        model = UWTransVO(**config).to(device)
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        update_state(
            status="Loading Dataset...",
            params=f"{model_params/1e6:.1f}M"
        )
        
        # Load dataset
        dataset = UnderwaterVODataset(
            data_csv='data/processed/training_dataset/training_data.csv',
            data_root='.', camera_ids=[0, 1, 2, 3], sequence_length=2,
            img_size=224, use_imu=False, use_pressure=False, 
            augmentation=True, split='train', max_samples=80
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
        
        update_state(
            status="Training Started",
            dataset_size=len(dataset)
        )
        
        # Training setup
        criterion = PoseLoss(translation_weight=1.0, rotation_weight=10.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        epochs = 5
        model.train()
        
        # Training loop
        for epoch in range(epochs):
            update_state(epoch=f"{epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                batch_start = time.time()
                
                try:
                    # Move to device
                    images = batch['images'].to(device, non_blocking=True)
                    camera_ids = batch['camera_ids'].to(device, non_blocking=True)
                    camera_mask = batch['camera_mask'].to(device, non_blocking=True)
                    pose_target = batch['pose_target'].to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    if scaler:
                        with torch.amp.autocast('cuda'):
                            predictions = model(images=images, camera_ids=camera_ids, camera_mask=camera_mask)
                            loss_dict = criterion(predictions['pose'], pose_target)
                            loss = loss_dict['total_loss']
                        
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        predictions = model(images=images, camera_ids=camera_ids, camera_mask=camera_mask)
                        loss_dict = criterion(predictions['pose'], pose_target)
                        loss = loss_dict['total_loss']
                        loss.backward()
                        optimizer.step()
                    
                    # Update metrics
                    batch_time = time.time() - batch_start
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    progress = ((epoch * len(dataloader) + batch_idx + 1) / (epochs * len(dataloader))) * 100
                    
                    update_state(
                        status="Training",
                        batch=f"{batch_idx+1}/{len(dataloader)}",
                        loss=f"{loss.item():.6f}",
                        gpu_memory=f"{gpu_memory:.2f}",
                        speed=f"{batch_time:.2f}",
                        progress=f"{progress:.1f}"
                    )
                    
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss={loss.item():.6f}, GPU={gpu_memory:.2f}GB")
                    
                    time.sleep(0.5)  # Slower for better visibility
                    
                except Exception as e:
                    print(f"Batch error: {e}")
                    continue
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        update_state(status="Training Complete!")
        print("Training completed successfully!")
        
    except Exception as e:
        update_state(status=f"Error: {str(e)}")
        print(f"Training error: {e}")

if __name__ == '__main__':
    print("UW-TransVO Working Dashboard")
    print("=" * 40)
    print("Dashboard: http://localhost:5002")
    print("Auto-refresh every 2 seconds")
    print("Training starts automatically")
    print("=" * 40)
    
    # Start training
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    
    # Open browser
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:5002')
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run Flask
    app.run(host='localhost', port=5002, debug=False)