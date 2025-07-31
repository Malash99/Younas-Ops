#!/usr/bin/env python3
"""
Web-based Training Dashboard for UW-TransVO
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
import queue
from datetime import datetime
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import webbrowser
import os

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.datasets import UnderwaterVODataset
from training.losses import PoseLoss

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'underwater-visual-odometry-2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for training state
training_data = {
    'status': 'idle',
    'current_epoch': 0,
    'current_batch': 0,
    'total_epochs': 0,
    'total_batches': 0,
    'losses': [],
    'gpu_memory': [],
    'timestamps': [],
    'model_params': 0,
    'dataset_size': 0,
    'training_speed': [],
    'start_time': None,
    'last_update': None
}

training_queue = queue.Queue()

def update_dashboard(data):
    """Send update to all connected clients"""
    training_data.update(data)
    training_data['last_update'] = datetime.now().strftime('%H:%M:%S')
    socketio.emit('training_update', training_data)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """API endpoint for training status"""
    return jsonify(training_data)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('training_update', training_data)
    print(f"Client connected. Current status: {training_data['status']}")

def run_training():
    """Training function that runs in separate thread"""
    print("Starting UW-TransVO Training with Web Dashboard")
    print("=" * 60)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_info = {
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'gpu_memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else 'N/A'
    }
    
    update_dashboard({
        'status': 'initializing',
        'gpu_info': gpu_info,
        'start_time': datetime.now().strftime('%H:%M:%S')
    })
    
    # Model configuration
    config = {
        'img_size': 224,
        'patch_size': 16,
        'd_model': 384,  # Optimized for GTX 1050
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
    
    update_dashboard({
        'model_params': model_params,
        'config': config
    })
    
    # Create dataset
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
        max_samples=200  # Reasonable size for demo
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    # Setup training components
    criterion = PoseLoss(translation_weight=1.0, rotation_weight=10.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    epochs = 10
    total_batches = len(dataloader)
    
    update_dashboard({
        'status': 'training',
        'dataset_size': len(dataset),
        'total_epochs': epochs,
        'total_batches': total_batches
    })
    
    # Training loop
    model.train()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        update_dashboard({
            'current_epoch': epoch + 1,
            'current_batch': 0
        })
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            
            try:
                # Move to device
                images = batch['images'].to(device, non_blocking=True)
                camera_ids = batch['camera_ids'].to(device, non_blocking=True)
                camera_mask = batch['camera_mask'].to(device, non_blocking=True)
                pose_target = batch['pose_target'].to(device, non_blocking=True)
                
                # Forward pass
                optimizer.zero_grad()
                
                if scaler and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        predictions = model(
                            images=images,
                            camera_ids=camera_ids,
                            camera_mask=camera_mask
                        )
                        loss_dict = criterion(predictions['pose'], pose_target)
                        loss = loss_dict['total_loss']
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    predictions = model(
                        images=images,
                        camera_ids=camera_ids,
                        camera_mask=camera_mask
                    )
                    loss_dict = criterion(predictions['pose'], pose_target)
                    loss = loss_dict['total_loss']
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Calculate metrics
                batch_time = time.time() - batch_start_time
                epoch_losses.append(loss.item())
                
                # GPU memory usage
                gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                
                # Update dashboard
                update_data = {
                    'current_batch': batch_idx + 1,
                    'current_loss': loss.item(),
                    'translation_loss': loss_dict.get('translation_loss', 0),
                    'rotation_loss': loss_dict.get('rotation_loss', 0),
                    'batch_time': batch_time,
                    'gpu_memory_used': gpu_memory
                }
                
                # Add to history every few batches
                if batch_idx % 5 == 0:
                    training_data['losses'].append({
                        'epoch': epoch + 1,
                        'batch': batch_idx + 1,
                        'total_loss': loss.item(),
                        'translation_loss': loss_dict.get('translation_loss', 0),
                        'rotation_loss': loss_dict.get('rotation_loss', 0)
                    })
                    training_data['gpu_memory'].append(gpu_memory)
                    training_data['training_speed'].append(batch_time)  
                    training_data['timestamps'].append(datetime.now().strftime('%H:%M:%S'))
                
                update_dashboard(update_data)
                
                # Small delay to make updates visible
                time.sleep(0.1)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    error_msg = f"GPU OUT OF MEMORY: {e}"
                    update_dashboard({
                        'status': 'error',
                        'error_message': error_msg
                    })
                    torch.cuda.empty_cache()
                    return
                else:
                    print(f"Runtime error: {e}")
                    continue
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        
        epoch_summary = {
            'epoch_complete': epoch + 1,
            'epoch_loss': avg_epoch_loss,
            'epoch_time': epoch_time,
            'epochs_remaining': epochs - (epoch + 1)
        }
        
        update_dashboard(epoch_summary)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        time.sleep(1)  # Brief pause between epochs
    
    # Training complete
    total_time = time.time() - time.mktime(datetime.strptime(training_data['start_time'], '%H:%M:%S').timetuple())
    
    update_dashboard({
        'status': 'completed',
        'total_training_time': total_time,
        'final_loss': training_data['losses'][-1]['total_loss'] if training_data['losses'] else 0,
        'completion_time': datetime.now().strftime('%H:%M:%S')
    })
    
    print("Training completed successfully!")

if __name__ == '__main__':
    # Create templates directory and HTML template
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # HTML template for dashboard
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UW-TransVO Training Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.4/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .status-bar {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            color: #fff;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #00ff88;
        }
        .progress-container {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #00cc6a);
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        .log-container {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 10px;
            height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            backdrop-filter: blur(10px);
        }
        .log-entry {
            margin: 5px 0;
            padding: 5px;
            border-left: 3px solid #00ff88;
            padding-left: 10px;
        }
        .status-idle { color: #ffd700; }
        .status-training { color: #00ff88; }
        .status-completed { color: #00ccff; }
        .status-error { color: #ff4444; }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .training { animation: pulse 2s infinite; }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🌊 UW-TransVO Training Dashboard</h1>
            <p>Real-time Underwater Visual Odometry Training Monitor</p>
        </div>
        
        <div class="status-bar">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>Status:</strong> 
                    <span id="status" class="status-idle">Initializing...</span>
                </div>
                <div>
                    <strong>GPU:</strong> <span id="gpu-info">Checking...</span>
                </div>
                <div>
                    <strong>Last Update:</strong> <span id="last-update">--:--:--</span>
                </div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>📈 Current Loss</h3>
                <div class="metric-value" id="current-loss">--</div>
                <small>Translation: <span id="trans-loss">--</span> | Rotation: <span id="rot-loss">--</span></small>
            </div>
            <div class="metric-card">
                <h3>⚡ Training Speed</h3>
                <div class="metric-value" id="batch-time">-- s/batch</div>
                <small>GPU Memory: <span id="gpu-memory">-- GB</span></small>
            </div>
            <div class="metric-card">
                <h3>🎯 Model Info</h3>
                <div class="metric-value" id="model-params">-- M</div>
                <small>Dataset: <span id="dataset-size">--</span> samples</small>
            </div>
            <div class="metric-card">
                <h3>⏱️ Progress</h3>
                <div class="metric-value" id="epoch-progress">Epoch --/--</div>
                <small>Batch: <span id="batch-progress">--/--</span></small>
            </div>
        </div>
        
        <div class="progress-container">
            <h3>Training Progress</h3>
            <div>Epoch Progress:</div>
            <div class="progress-bar">
                <div class="progress-fill" id="epoch-progress-bar" style="width: 0%"></div>
            </div>
            <div>Batch Progress:</div>
            <div class="progress-bar">
                <div class="progress-fill" id="batch-progress-bar" style="width: 0%"></div>
            </div>
        </div>
        
        <div class="charts-container">
            <div class="chart-card">
                <h3>Loss Over Time</h3>
                <canvas id="lossChart" width="400" height="200"></canvas>
            </div>
            <div class="chart-card">
                <h3>GPU Memory Usage</h3>
                <canvas id="memoryChart" width="400" height="200"></canvas>
            </div>
        </div>
        
        <div class="log-container">
            <h3>Training Log</h3>
            <div id="training-log"></div>
        </div>
    </div>

    <script>
        const socket = io();
        
        // Initialize charts
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        const memoryCtx = document.getElementById('memoryChart').getContext('2d');
        
        const lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Total Loss',
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
        
        const memoryChart = new Chart(memoryCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'GPU Memory (GB)',
                    data: [],
                    borderColor: '#00ccff',
                    backgroundColor: 'rgba(0,204,255,0.1)',
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
        
        function addLogEntry(message) {
            const logDiv = document.getElementById('training-log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
            logDiv.appendChild(entry);
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        socket.on('training_update', function(data) {
            // Update status
            const statusElement = document.getElementById('status');
            statusElement.textContent = data.status;
            statusElement.className = `status-${data.status}`;
            
            if (data.status === 'training') {
                statusElement.classList.add('training');
            } else {
                statusElement.classList.remove('training');
            }
            
            // Update metrics
            if (data.current_loss !== undefined) {
                document.getElementById('current-loss').textContent = data.current_loss.toFixed(6);
            }
            if (data.translation_loss !== undefined) {
                document.getElementById('trans-loss').textContent = data.translation_loss.toFixed(6);
            }
            if (data.rotation_loss !== undefined) {
                document.getElementById('rot-loss').textContent = data.rotation_loss.toFixed(6);
            }
            if (data.batch_time !== undefined) {
                document.getElementById('batch-time').textContent = data.batch_time.toFixed(2) + ' s/batch';
            }
            if (data.gpu_memory_used !== undefined) {
                document.getElementById('gpu-memory').textContent = data.gpu_memory_used.toFixed(2) + ' GB';
            }
            if (data.model_params) {
                document.getElementById('model-params').textContent = (data.model_params / 1e6).toFixed(1) + ' M';
            }
            if (data.dataset_size) {
                document.getElementById('dataset-size').textContent = data.dataset_size;
            }
            if (data.gpu_info) {
                document.getElementById('gpu-info').textContent = data.gpu_info.gpu_name;
            }
            if (data.last_update) {
                document.getElementById('last-update').textContent = data.last_update;
            }
            
            // Update progress
            if (data.current_epoch && data.total_epochs) {
                const epochProgress = (data.current_epoch / data.total_epochs) * 100;
                document.getElementById('epoch-progress-bar').style.width = epochProgress + '%';
                document.getElementById('epoch-progress').textContent = `Epoch ${data.current_epoch}/${data.total_epochs}`;
            }
            if (data.current_batch && data.total_batches) {
                const batchProgress = (data.current_batch / data.total_batches) * 100;
                document.getElementById('batch-progress-bar').style.width = batchProgress + '%';
                document.getElementById('batch-progress').textContent = `${data.current_batch}/${data.total_batches}`;
            }
            
            // Update charts
            if (data.losses && data.losses.length > 0) {
                const recentLosses = data.losses.slice(-20); // Show last 20 points
                lossChart.data.labels = recentLosses.map((_, i) => i + 1);
                lossChart.data.datasets[0].data = recentLosses.map(l => l.total_loss);
                lossChart.update('none');
            }
            
            if (data.gpu_memory && data.gpu_memory.length > 0) {
                const recentMemory = data.gpu_memory.slice(-20);
                memoryChart.data.labels = recentMemory.map((_, i) => i + 1);
                memoryChart.data.datasets[0].data = recentMemory;
                memoryChart.update('none');
            }
            
            // Add log entries for significant events
            if (data.status === 'training' && data.current_batch === 1) {
                addLogEntry(`Starting epoch ${data.current_epoch}/${data.total_epochs}`);
            }
            if (data.epoch_complete) {
                addLogEntry(`Epoch ${data.epoch_complete} completed. Loss: ${data.epoch_loss.toFixed(6)}`);
            }
            if (data.status === 'completed') {
                addLogEntry(`Training completed! Final loss: ${data.final_loss.toFixed(6)}`);
            }
            if (data.error_message) {
                addLogEntry(`ERROR: ${data.error_message}`);
            }
        });
        
        // Initial log entry
        addLogEntry('Dashboard initialized. Waiting for training to start...');
    </script>
</body>
</html>
    '''
    
    with open(templates_dir / 'dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("🌊 UW-TransVO Web Training Dashboard")
    print("=" * 50)
    print("🚀 Starting web server...")
    print("📊 Dashboard will open in your browser at: http://localhost:5000")
    print("⚡ GPU training will begin automatically")
    print("=" * 50)
    
    # Start training in background thread
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    
    # Open browser
    def open_browser():
        time.sleep(2)  # Wait for server to start
        webbrowser.open('http://localhost:5000')
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)