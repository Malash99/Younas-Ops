#!/usr/bin/env python3
"""
Fixed Web Training Dashboard for UW-TransVO
Ensures proper data transmission to browser
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import time
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import webbrowser
import json

sys.path.append(str(Path(__file__).parent))

from models.transformer import UWTransVO
from data.datasets import UnderwaterVODataset
from training.losses import PoseLoss

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'underwater-visual-odometry'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Global training state - initialize with default values
training_data = {
    'status': 'initializing',
    'current_epoch': 0,
    'current_batch': 0,
    'total_epochs': 8,
    'total_batches': 0,
    'current_loss': 0.0,
    'translation_loss': 0.0,
    'rotation_loss': 0.0,
    'losses': [],
    'gpu_memory': 0.0,
    'model_params': 0,
    'dataset_size': 0,
    'batch_time': 0.0,
    'gpu_info': {'gpu_name': 'Loading...'},
    'start_time': datetime.now().strftime('%H:%M:%S'),
    'last_update': datetime.now().strftime('%H:%M:%S')
}

def broadcast_update(data):
    """Broadcast update to all connected clients"""
    global training_data
    training_data.update(data)
    training_data['last_update'] = datetime.now().strftime('%H:%M:%S')
    
    print(f"Broadcasting update: {data.get('status', 'unknown')} - Loss: {data.get('current_loss', 'N/A')}")
    socketio.emit('training_update', training_data)
    time.sleep(0.01)  # Small delay to ensure transmission

# HTML Template with better error handling
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>UW-TransVO Training Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.4/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            color: white; 
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            text-align: center; 
            color: #00ff88; 
            margin-bottom: 30px;
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 15px;
        }
        .status-bar { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px;
            border: 1px solid rgba(0,255,136,0.3);
        }
        .connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .connected { background: #00ff88; color: black; }
        .disconnected { background: #ff4444; color: white; }
        .metrics { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 20px; 
        }
        .metric-card { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            text-align: center;
            border: 1px solid rgba(0,255,136,0.2);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            border-color: rgba(0,255,136,0.5);
            transform: translateY(-2px);
        }
        .metric-label { font-size: 0.9em; color: #ccc; margin-bottom: 5px; }
        .metric-value { 
            font-size: 2em; 
            color: #00ff88; 
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }
        .metric-sub { font-size: 0.8em; color: #aaa; margin-top: 5px; }
        .progress-section { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px;
            border: 1px solid rgba(0,255,136,0.2);
        }
        .progress-bar { 
            width: 100%; 
            height: 25px; 
            background: rgba(255,255,255,0.2); 
            border-radius: 12px; 
            overflow: hidden; 
            margin: 10px 0;
            position: relative;
        }
        .progress-fill { 
            height: 100%; 
            background: linear-gradient(90deg, #00ff88, #00cc6a); 
            transition: width 0.5s ease;
            border-radius: 12px;
        }
        .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        }
        .chart-container { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px;
            border: 1px solid rgba(0,255,136,0.2);
        }
        .log-container { 
            background: rgba(0,0,0,0.4); 
            padding: 20px; 
            border-radius: 10px; 
            height: 300px; 
            overflow-y: auto; 
            font-family: 'Courier New', monospace;
            border: 1px solid rgba(0,255,136,0.2);
        }
        .log-entry { 
            margin: 3px 0; 
            color: #00ff88;
            font-size: 0.9em;
        }
        .log-entry.error { color: #ff4444; }
        .log-entry.warning { color: #ffaa00; }
        .training { animation: pulse 2s infinite; } 
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .status-training { color: #00ff88; }
        .status-completed { color: #00ccff; }
        .status-error { color: #ff4444; }
        .status-initializing { color: #ffaa00; }
    </style>
</head>
<body>
    <div class="connection-status" id="connection-status">Connecting...</div>
    
    <div class="container">
        <div class="header">
            <h1>🌊 UW-TransVO Training Dashboard</h1>
            <p>Real-time Underwater Visual Odometry Training Monitor</p>
            <div>Last Update: <span id="last-update">Connecting...</span></div>
        </div>
        
        <div class="status-bar">
            <h2>Status: <span id="status" class="status-initializing">Connecting...</span></h2>
            <p>
                GPU: <span id="gpu-info">Checking...</span> | 
                Time: <span id="current-time">--:--:--</span> |
                Training Started: <span id="start-time">--:--:--</span>
            </p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Current Loss</div>
                <div class="metric-value" id="current-loss">--</div>
                <div class="metric-sub">
                    Trans: <span id="trans-loss">--</span> | 
                    Rot: <span id="rot-loss">--</span>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Training Speed</div>
                <div class="metric-value" id="batch-time">-- s</div>
                <div class="metric-sub">per batch</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Memory</div>
                <div class="metric-value" id="gpu-memory">-- GB</div>
                <div class="metric-sub">VRAM usage</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Model Size</div>
                <div class="metric-value" id="model-params">-- M</div>
                <div class="metric-sub">parameters</div>
            </div>
        </div>
        
        <div class="progress-section">
            <h3>Training Progress</h3>
            <div>
                <strong>Epoch Progress:</strong> 
                <span id="current-epoch">0</span>/<span id="total-epochs">0</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="epoch-progress" style="width: 0%"></div>
                <div class="progress-text" id="epoch-progress-text">0%</div>
            </div>
            <div>
                <strong>Batch Progress:</strong> 
                <span id="current-batch">0</span>/<span id="total-batches">0</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="batch-progress" style="width: 0%"></div>
                <div class="progress-text" id="batch-progress-text">0%</div>
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
        let socket;
        let lossChart;
        let isConnected = false;
        
        function initializeChart() {
            const ctx = document.getElementById('lossChart').getContext('2d');
            lossChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0,255,136,0.1)',
                        tension: 0.4,
                        pointRadius: 2
                    }]
                },
                options: {
                    responsive: true,
                    animation: false,
                    plugins: { 
                        legend: { labels: { color: 'white' } } 
                    },
                    scales: {
                        x: { 
                            ticks: { color: 'white' }, 
                            grid: { color: 'rgba(255,255,255,0.1)' } 
                        },
                        y: { 
                            ticks: { color: 'white' }, 
                            grid: { color: 'rgba(255,255,255,0.1)' } 
                        }
                    }
                }
            });
        }
        
        function updateConnectionStatus(connected) {
            const statusEl = document.getElementById('connection-status');
            isConnected = connected;
            
            if (connected) {
                statusEl.textContent = 'Connected';
                statusEl.className = 'connection-status connected';
            } else {
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'connection-status disconnected';
            }
        }
        
        function addLog(message, type = 'info') {
            const log = document.getElementById('training-log');
            const entry = document.createElement('div');
            entry.className = `log-entry ${type}`;
            entry.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
            
            // Keep only last 50 entries
            while (log.children.length > 50) {
                log.removeChild(log.firstChild);
            }
        }
        
        function updateUI(data) {
            try {
                // Update status
                if (data.status) {
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = data.status;
                    statusEl.className = `status-${data.status}`;
                    
                    if (data.status === 'training') {
                        statusEl.classList.add('training');
                    }
                }
                
                // Update metrics with proper formatting
                if (data.current_loss !== undefined && data.current_loss !== null) {
                    document.getElementById('current-loss').textContent = 
                        typeof data.current_loss === 'number' ? data.current_loss.toFixed(6) : data.current_loss;
                }
                
                if (data.translation_loss !== undefined) {
                    document.getElementById('trans-loss').textContent = 
                        typeof data.translation_loss === 'number' ? data.translation_loss.toFixed(6) : data.translation_loss;
                }
                
                if (data.rotation_loss !== undefined) {
                    document.getElementById('rot-loss').textContent = 
                        typeof data.rotation_loss === 'number' ? data.rotation_loss.toFixed(6) : data.rotation_loss;
                }
                
                if (data.batch_time !== undefined) {
                    document.getElementById('batch-time').textContent = 
                        typeof data.batch_time === 'number' ? data.batch_time.toFixed(2) : data.batch_time;
                }
                
                if (data.gpu_memory !== undefined) {
                    document.getElementById('gpu-memory').textContent = 
                        typeof data.gpu_memory === 'number' ? data.gpu_memory.toFixed(2) : data.gpu_memory;
                }
                
                if (data.model_params) {
                    document.getElementById('model-params').textContent = 
                        (data.model_params / 1e6).toFixed(1);
                }
                
                if (data.gpu_info && data.gpu_info.gpu_name) {
                    document.getElementById('gpu-info').textContent = data.gpu_info.gpu_name;
                }
                
                if (data.start_time) {
                    document.getElementById('start-time').textContent = data.start_time;
                }
                
                if (data.last_update) {
                    document.getElementById('last-update').textContent = data.last_update;
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
                    document.getElementById('epoch-progress-text').textContent = epochPercent.toFixed(1) + '%';
                }
                
                if (data.current_batch && data.total_batches) {
                    const batchPercent = (data.current_batch / data.total_batches) * 100;
                    document.getElementById('batch-progress').style.width = batchPercent + '%';
                    document.getElementById('batch-progress-text').textContent = batchPercent.toFixed(1) + '%';
                }
                
                // Update chart
                if (data.losses && data.losses.length > 0 && lossChart) {
                    const recent = data.losses.slice(-30); // Last 30 points
                    lossChart.data.labels = recent.map((_, i) => i + 1);
                    lossChart.data.datasets[0].data = recent.map(l => 
                        typeof l === 'object' ? l.total_loss : l
                    );
                    lossChart.update('none');
                }
                
            } catch (error) {
                console.error('Error updating UI:', error);
                addLog(`UI Update Error: ${error.message}`, 'error');
            }
        }
        
        function connectSocket() {
            socket = io();
            
            socket.on('connect', function() {
                updateConnectionStatus(true);
                addLog('Connected to training dashboard');
            });
            
            socket.on('disconnect', function() {
                updateConnectionStatus(false);
                addLog('Disconnected from server', 'warning');
            });
            
            socket.on('training_update', function(data) {
                console.log('Received training update:', data);
                updateUI(data);
                
                // Log significant events
                if (data.status === 'training' && data.current_batch === 1) {
                    addLog(`Starting epoch ${data.current_epoch}/${data.total_epochs}`);
                }
                if (data.status === 'completed') {
                    addLog('Training completed successfully!');
                }
                if (data.status === 'error') {
                    addLog(`Error: ${data.error_message || 'Unknown error'}`, 'error');
                }
            });
            
            socket.on('connect_error', function(error) {
                updateConnectionStatus(false);
                addLog(`Connection error: ${error}`, 'error');
            });
        }
        
        // Initialize everything
        document.addEventListener('DOMContentLoaded', function() {
            initializeChart();
            connectSocket();
            
            // Update current time every second
            setInterval(() => {
                document.getElementById('current-time').textContent = new Date().toLocaleTimeString();
            }, 1000);
            
            addLog('Dashboard initialized');
        });
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def get_status():
    return jsonify(training_data)

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {training_data['status']}")
    emit('training_update', training_data)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def run_training():
    """Main training function with proper data broadcasting"""
    print("Starting UW-TransVO GPU Training")
    print("=" * 40)
    
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_info = {
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
        
        print(f"Using device: {device}")
        print(f"GPU: {gpu_info['gpu_name']}")
        
        broadcast_update({
            'status': 'initializing',
            'gpu_info': gpu_info
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
        print("Creating model...")
        broadcast_update({'status': 'creating_model'})
        
        model = UWTransVO(**config).to(device)
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model parameters: {model_params:,}")
        
        broadcast_update({
            'status': 'loading_data',
            'model_params': model_params
        })
        
        # Load dataset
        print("Loading dataset...")
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
            max_samples=100  # Smaller for demo
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Batches per epoch: {len(dataloader)}")
        
        # Training setup
        criterion = PoseLoss(translation_weight=1.0, rotation_weight=10.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        epochs = 5  # Shorter for demo
        
        broadcast_update({
            'status': 'training',
            'dataset_size': len(dataset),
            'total_epochs': epochs,
            'total_batches': len(dataloader)
        })
        
        print("Starting training loop...")
        
        # Training loop
        model.train()
        all_losses = []
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
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
                    if scaler and torch.cuda.is_available():
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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    # Calculate metrics
                    batch_time = time.time() - batch_start
                    current_loss = loss.item()
                    translation_loss = loss_dict.get('translation_loss', 0)
                    rotation_loss = loss_dict.get('rotation_loss', 0)
                    
                    epoch_losses.append(current_loss)
                    all_losses.append(current_loss)
                    
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    
                    # Print progress
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}: "
                          f"Loss: {current_loss:.6f}, Time: {batch_time:.2f}s, GPU: {gpu_memory:.2f}GB")
                    
                    # Broadcast update to dashboard
                    update_data = {
                        'status': 'training',
                        'current_epoch': epoch + 1,
                        'current_batch': batch_idx + 1,
                        'current_loss': current_loss,
                        'translation_loss': translation_loss,
                        'rotation_loss': rotation_loss,
                        'batch_time': batch_time,
                        'gpu_memory': gpu_memory,
                        'losses': all_losses[-20:]  # Keep last 20 points for chart
                    }
                    
                    broadcast_update(update_data)
                    
                    # Small delay for UI updates
                    time.sleep(0.2)
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    print(f"Runtime Error: {error_msg}")
                    
                    if "out of memory" in error_msg.lower():
                        broadcast_update({
                            'status': 'error',
                            'error_message': f"GPU OUT OF MEMORY: {error_msg}"
                        })
                        return
                    else:
                        continue
                        
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            # Epoch summary
            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.6f}")
                
                broadcast_update({
                    'epoch_complete': epoch + 1,
                    'epoch_loss': avg_loss
                })
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Training complete
        final_loss = all_losses[-1] if all_losses else 0
        print(f"\nTraining completed! Final loss: {final_loss:.6f}")
        
        broadcast_update({
            'status': 'completed',
            'final_loss': final_loss,
            'completion_time': datetime.now().strftime('%H:%M:%S')
        })
        
    except Exception as e:
        print(f"Training error: {e}")
        broadcast_update({
            'status': 'error',
            'error_message': str(e)
        })

if __name__ == '__main__':
    print("UW-TransVO Fixed Web Training Dashboard")
    print("=" * 50)
    print("Starting web server...")
    print("Dashboard: http://localhost:5001")
    print("Training will begin automatically")
    print("=" * 50)
    
    # Start training thread
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    
    # Open browser automatically
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open('http://localhost:5001')
        except:
            print("Could not open browser automatically. Please open http://localhost:5001")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run Flask app
    try:
        socketio.run(app, host='localhost', port=5001, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nTraining dashboard stopped by user")