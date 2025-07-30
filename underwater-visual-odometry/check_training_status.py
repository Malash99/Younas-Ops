#!/usr/bin/env python3
"""
Check training status and open browser if needed
"""

import webbrowser
import time
import requests
import json

def check_dashboard():
    """Check if dashboard is running and show status"""
    try:
        response = requests.get('http://localhost:5000/api/status', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("=" * 60)
            print("🌊 UW-TransVO Training Dashboard Status")
            print("=" * 60)
            print(f"Status: {data.get('status', 'Unknown')}")
            print(f"Current Epoch: {data.get('current_epoch', 0)}/{data.get('total_epochs', 0)}")
            print(f"Current Batch: {data.get('current_batch', 0)}/{data.get('total_batches', 0)}")
            print(f"Current Loss: {data.get('current_loss', 0):.6f}")
            print(f"GPU Memory: {data.get('gpu_memory', 0):.2f} GB")
            print(f"Model Parameters: {data.get('model_params', 0) / 1e6:.1f}M")
            print(f"Dataset Size: {data.get('dataset_size', 0)} samples")
            print("=" * 60)
            print("🌐 Dashboard URL: http://localhost:5000")
            print("📊 View real-time charts and progress in your browser!")
            print("=" * 60)
            return True
        else:
            print("Dashboard not responding properly")
            return False
    except requests.exceptions.RequestException:
        print("Dashboard not running or not accessible")
        return False

def main():
    print("Checking UW-TransVO Training Dashboard...")
    
    if check_dashboard():
        print("\n✅ Dashboard is running successfully!")
        print("\n🚀 Opening dashboard in browser...")
        webbrowser.open('http://localhost:5000')
        print("\n💡 You can:")
        print("   - Watch real-time training progress")
        print("   - Monitor GPU memory usage")
        print("   - View loss curves")
        print("   - Track training metrics")
        print("\n🔄 Refresh the page to see latest updates")
    else:
        print("\n❌ Dashboard not running.")
        print("To start the dashboard, run:")
        print("   python web_dashboard_simple.py")

if __name__ == '__main__':
    main()