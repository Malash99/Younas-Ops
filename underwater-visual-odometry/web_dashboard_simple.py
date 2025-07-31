#!/usr/bin/env python3
"""
Web Training Dashboard for UW-TransVO - Simple Version
Real-time monitoring via localhost in Chrome browser
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import time
import json
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import webbrowser

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.datasets import UnderwaterVODataset
from training.losses import PoseLoss

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'underwater-visual-odometry'
socketio = SocketIO(app, cors_allowed_origins="*")

# Training state
training_data = {
    'status': 'idle',
    'current_epoch': 0,
    'current_batch': 0,
    'total_epochs': 0,
    'total_batches': 0,
    'current_loss': 0,
    'losses': [],
    'gpu_memory': 0,
    'model_params': 0,
    'dataset_size': 0,
    'batch_time': 0,
    'start_time': None,
}

def update_dashboard(data):
    """Send update to all connected clients"""
    training_data.update(data)
    socketio.emit('training_update', training_data)

# Simple HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>UW-TransVO Training Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.4/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: white; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: #00ff88; margin-bottom: 30px; }
        .status-bar { background: #16213e; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: #16213e; padding: 20px; border-radius: 10px; text-align: center; }
        .metric-value { font-size: 2em; color: #00ff88; font-weight: bold; }
        .progress-section { background: #16213e; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .progress-bar { width: 100%; height: 30px; background: #333; border-radius: 15px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #00ff88, #00cc6a); transition: width 0.3s; }
        .chart-container { background: #16213e; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .log-container { background: #0f0f23; padding: 20px; border-radius: 10px; height: 300px; overflow-y: auto; font-family: monospace; }
        .log-entry { margin: 5px 0; color: #00ff88; }
        .training { animation: pulse 2s infinite; } 
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>UW-TransVO Training Dashboard</h1>
            <p>Real-time Underwater Visual Odometry Training Monitor</p>
        </div>
        
        <div class="status-bar">
            <h2>Status: <span id="status">Initializing...</span></h2>
            <p>GPU: <span id="gpu-info">Checking...</span> | Time: <span id="current-time">--:--:--</span></p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>Current Loss</h3>
                <div class="metric-value" id="current-loss">--</div>
            </div>
            <div class="metric-card">
                <h3>Training Speed</h3>
                <div class="metric-value" id="batch-time">-- s/batch</div>
            </div>
            <div class="metric-card">
                <h3>GPU Memory</h3>
                <div class="metric-value" id="gpu-memory">-- GB</div>
            </div>
            <div class="metric-card">
                <h3>Model Parameters</h3>
                <div class="metric-value" id="model-params">-- M</div>
            </div>
        </div>
        
        <div class="progress-section">
            <h3>Training Progress</h3>
            <div>Epoch <span id="current-epoch">0</span>/<span id="total-epochs">0</span></div>
            <div class="progress-bar">
                <div class="progress-fill" id="epoch-progress" style="width: 0%"></div>
            </div>
            <div>Batch <span id="current-batch">0</span>/<span id="total-batches">0</span></div>
            <div class="progress-bar">
                <div class="progress-fill" id="batch-progress" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Loss Over Time</h3>
            <canvas id="lossChart" width="800" height="300"></canvas>
        </div>
        
        <div class="log-container">
            <div id="training-log"></div>
        </div>
    </div>

    <script>
        const socket = io();
        
        // Initialize loss chart
        const ctx = document.getElementById('lossChart').getContext('2d');
        const lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0,255,136,0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { labels: { color: 'white' } } },
                scales: {
                    x: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                    y: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                }
            }
        });
        
        function addLog(message) {
            const log = document.getElementById('training-log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
        }
        
        socket.on('training_update', function(data) {
            // Update status
            const status = document.getElementById('status');
            status.textContent = data.status || 'Unknown';
            if (data.status === 'training') {
                status.classList.add('training');
            } else {
                status.classList.remove('training');
            }
            
            // Update metrics
            if (data.current_loss !== undefined) {
                document.getElementById('current-loss').textContent = data.current_loss.toFixed(6);
            }
            if (data.batch_time !== undefined) {
                document.getElementById('batch-time').textContent = data.batch_time.toFixed(2) + ' s/batch';
            }
            if (data.gpu_memory !== undefined) {
                document.getElementById('gpu-memory').textContent = data.gpu_memory.toFixed(2) + ' GB';
            }
            if (data.model_params) {
                document.getElementById('model-params').textContent = (data.model_params / 1e6).toFixed(1) + ' M';
            }
            if (data.gpu_info) {
                document.getElementById('gpu-info').textContent = data.gpu_info.gpu_name || 'Unknown';
            }
            
            // Update progress
            if (data.current_epoch !== undefined) {
                document.getElementById('current-epoch').textContent = data.current_epoch;
            }
            if (data.total_epochs !== undefined) {
                document.getElementById('total-epochs').textContent = data.total_epochs;
            }
            if (data.current_batch !== undefined) {
                document.getElementById('current-batch').textContent = data.current_batch;
            }
            if (data.total_batches !== undefined) {
                document.getElementById('total-batches').textContent = data.total_batches;
            }
            
            // Update progress bars
            if (data.current_epoch && data.total_epochs) {
                const epochPercent = (data.current_epoch / data.total_epochs) * 100;
                document.getElementById('epoch-progress').style.width = epochPercent + '%';
            }
            if (data.current_batch && data.total_batches) {
                const batchPercent = (data.current_batch / data.total_batches) * 100;
                document.getElementById('batch-progress').style.width = batchPercent + '%';
            }
            
            // Update chart
            if (data.losses && data.losses.length > 0) {
                const recent = data.losses.slice(-50);
                lossChart.data.labels = recent.map((_, i) => i + 1);
                lossChart.data.datasets[0].data = recent.map(l => l.total_loss || l);
                lossChart.update('none');
            }
            
            // Update time
            document.getElementById('current-time').textContent = new Date().toLocaleTimeString();
        });
        
        // Initial setup
        addLog('Dashboard initialized. Waiting for training to start...');
        setInterval(() => {
            document.getElementById('current-time').textContent = new Date().toLocaleTimeString();
        }, 1000);
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE)

@socketio.on('connect')
def handle_connect():
    emit('training_update', training_data)
    print(f"Client connected. Status: {training_data['status']}")

def run_training():
    """Main training function"""
    print("Starting UW-TransVO GPU Training")
    print("=" * 40)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_info = {
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    
    update_dashboard({
        'status': 'initializing',
        'gpu_info': gpu_info,
        'start_time': datetime.now().strftime('%H:%M:%S')
    })
    
    # Model config optimized for GTX 1050
    config = {
        'img_size': 224,
        'patch_size': 16,
        'd_model': 384,
        'num_heads': 6,
        'num_layers': 4,
        'max_cameras': 4,
        'max_seq_len': 2,
        'dropout': 0.1,
        'use_imu': False,
        'use_pressure': False,
        'uncertainty_estimation': False
    }
    
    # Create model
    update_dashboard({'status': 'creating_model'})
    model = UWTransVO(**config).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    update_dashboard({'model_params': model_params})
    
    # Load dataset
    update_dashboard({'status': 'loading_data'})
    dataset = UnderwaterVODataset(
        data_csv='data/processed/training_dataset/training_data.csv',
        data_root='.',
        camera_ids=[0, 1, 2, 3],
        sequence_length=2,
        img_size=224,
        use_imu=False,
        use_pressure=False,
        augmentation=True,
        split='train',
        max_samples=150  # Reasonable for demo
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    
    # Training setup
    criterion = PoseLoss(translation_weight=1.0, rotation_weight=10.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    epochs = 8
    
    update_dashboard({
        'status': 'training',
        'dataset_size': len(dataset),
        'total_epochs': epochs,
        'total_batches': len(dataloader)
    })
    
    # Training loop
    model.train()
    all_losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.time()
            
            try:
                # Move to device
                images = batch['images'].to(device, non_blocking=True)
                camera_ids = batch['camera_ids'].to(device, non_blocking=True)
                camera_mask = batch['camera_mask'].to(device, non_blocking=True)
                pose_target = batch['pose_target'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if scaler:
                    with torch.cuda.amp.autocast():
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Metrics
                batch_time = time.time() - batch_start
                epoch_losses.append(loss.item())
                all_losses.append({'total_loss': loss.item()})
                
                gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                
                # Update dashboard
                update_dashboard({
                    'current_epoch': epoch + 1,
                    'current_batch': batch_idx + 1,
                    'current_loss': loss.item(),
                    'batch_time': batch_time,
                    'gpu_memory': gpu_memory,
                    'losses': all_losses[-20:]  # Keep last 20 points
                })
                
                time.sleep(0.1)  # Small delay for UI updates
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    update_dashboard({
                        'status': 'error',
                        'error_message': f"GPU OUT OF MEMORY: {e}"
                    })
                    return
                continue
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Epoch complete
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1}/{epochs} complete. Average loss: {avg_loss:.6f}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Training complete
    update_dashboard({
        'status': 'completed',
        'completion_time': datetime.now().strftime('%H:%M:%S')
    })
    
    print("Training completed successfully!")

if __name__ == '__main__':
    print("UW-TransVO Web Training Dashboard")
    print("=" * 40)
    print("Starting web server...")
    print("Dashboard: http://localhost:5000")
    print("Training will begin automatically")
    print("=" * 40)
    
    # Start training thread
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    
    # Open browser automatically
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:5000')
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run Flask app
    try:
        socketio.run(app, host='localhost', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nTraining dashboard stopped by user")