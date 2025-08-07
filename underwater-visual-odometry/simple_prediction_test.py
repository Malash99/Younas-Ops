"""
Simple Prediction Test for Current Model

Tests the existing working model to see trajectory direction and scale
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_synthetic_test():
    """Create a simple synthetic test to see model behavior"""
    print("=" * 60)
    print("SIMPLE PREDICTION TEST - CURRENT ARCHITECTURE")
    print("=" * 60)
    
    # Check if we have the working model
    model_path = 'final_fixed_model.pth'
    if not os.path.exists(model_path):
        print("ERROR: final_fixed_model.pth not found")
        print("This was the working model from your previous successful training")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load the working model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
        
        # Let's simulate some inputs to test the model's behavior
        # We'll create simple synthetic data to see prediction patterns
        
        print("\nTesting synthetic inputs...")
        
        # Create a batch of images (5 cameras, 1 batch, 10 frames, 3 channels, 192x192)
        batch_size = 1
        seq_len = 10
        num_cameras = 5
        img_size = 192
        
        # Random images
        images = torch.randn(num_cameras, batch_size, seq_len, 3, img_size, img_size).to(device)
        
        # Test different scenarios
        print("\nTest 1: Random images - What does model predict?")
        
        # We'll analyze what the existing model predicts for synthetic data
        # This will tell us about the model's current behavior patterns
        
        predictions = []
        
        for test_num in range(3):
            # Generate slightly different random images
            test_images = torch.randn(num_cameras, batch_size, seq_len, 3, img_size, img_size).to(device) * 0.1
            
            # We need to load the model architecture first
            # For now, let's analyze the saved checkpoint structure
            print(f"\nAnalyzing checkpoint structure...")
            print(f"Keys in checkpoint: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                model_keys = list(checkpoint['model_state_dict'].keys())
                print(f"Model layers: {len(model_keys)} total")
                print(f"Sample keys: {model_keys[:5]}")
                
                # Look for pose prediction layers
                pose_layers = [k for k in model_keys if 'pose' in k.lower() or 'head' in k.lower()]
                print(f"Pose-related layers: {pose_layers[:10]}")
        
        print("\nTo properly test the model, we need to:")
        print("1. Load the exact same architecture used in training")
        print("2. Feed it real image data") 
        print("3. Compare predictions with ground truth")
        print("\nLet's use the test script that worked before...")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

def analyze_previous_test_results():
    """Analyze the previous test results if available"""
    print("\n" + "=" * 60)
    print("ANALYZING PREVIOUS TEST RESULTS")
    print("=" * 60)
    
    # Check for previous trajectory plot
    plot_path = 'final_model_trajectory_test.png'
    if os.path.exists(plot_path):
        print(f"Found previous trajectory plot: {plot_path}")
        print("This shows the model's current behavior:")
        print("- If trajectories are curved: GOOD (no more straight lines)")
        print("- If predictions are smaller than GT: Scale problem")  
        print("- If predictions go opposite direction: Direction problem")
    else:
        print("No previous trajectory plot found")
    
    # Run the existing test that we know works
    print(f"\nLet's run the existing test that worked...")
    
    # Import and run the test that we know works
    try:
        os.system('python test_final_model.py > current_test_output.txt 2>&1')
        
        # Read the output
        if os.path.exists('current_test_output.txt'):
            with open('current_test_output.txt', 'r') as f:
                output = f.read()
            
            print("Current model test results:")
            print("-" * 40)
            
            # Extract key information
            lines = output.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['prediction', 'ground truth', 'error', 'trajectory', 'curved', 'direction', 'magnitude']):
                    print(line)
            
            print("-" * 40)
            
        else:
            print("Could not capture test output")
            
    except Exception as e:
        print(f"Error running test: {e}")

def create_direction_analysis_plot():
    """Create a simple plot showing what we expect vs what model might predict"""
    print("\n" + "=" * 60) 
    print("CREATING DIRECTION ANALYSIS PLOT")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Expected forward motion
    ax = axes[0]
    t = np.linspace(0, 1, 10)
    x_forward = t * 0.02  # 20mm forward motion per frame
    y_forward = np.sin(t * 2) * 0.005  # Small lateral movement
    
    ax.plot(x_forward, y_forward, 'b-', linewidth=3, label='Expected Forward')
    ax.scatter(x_forward[0], y_forward[0], color='green', s=100, label='Start')
    ax.scatter(x_forward[-1], y_forward[-1], color='red', s=100, label='End')
    ax.set_title('Expected: Forward Motion\n(Positive X direction)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Previous problem (backward motion)
    ax = axes[1]
    x_backward = -t * 0.004  # 4mm backward motion per frame (old problem)
    y_backward = np.sin(t * 2) * 0.002
    
    ax.plot(x_backward, y_backward, 'r--', linewidth=3, label='Old Problem: Backward')
    ax.scatter(x_backward[0], y_backward[0], color='green', s=100, label='Start')
    ax.scatter(x_backward[-1], y_backward[-1], color='red', s=100, label='End') 
    ax.set_title('Previous Problem: Backward Motion\n(Negative X, 5x too small)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 3: Hoped for fix
    ax = axes[2]
    x_fixed = t * 0.018  # Correctly scaled forward motion
    y_fixed = np.sin(t * 2) * 0.004
    
    ax.plot(x_fixed, y_fixed, 'g-', linewidth=3, label='Target: Fixed Motion')
    ax.scatter(x_fixed[0], y_fixed[0], color='green', s=100, label='Start')
    ax.scatter(x_fixed[-1], y_fixed[-1], color='red', s=100, label='End')
    ax.set_title('Target: Fixed Motion\n(Forward direction, correct scale)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('direction_analysis_expected.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created expected behavior plot: direction_analysis_expected.png")
    print("\nNext steps:")
    print("1. Run test_final_model.py to see current model behavior")
    print("2. Compare with our target behavior")
    print("3. If needed, continue training the scale/direction fixed model")

def main():
    """Main function"""
    create_synthetic_test()
    analyze_previous_test_results()
    create_direction_analysis_plot()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print("From your PROJECT_PROGRESS_SUMMARY.md, we know:")
    print("- CURRENT ISSUE: Predictions 5.3x too small, wrong direction")
    print("- SOLUTION IMPLEMENTED: Scale factors + direction consistency")
    print("- STATUS: Core fixes are working in training")
    print("")
    print("To see current model predictions:")
    print("1. python test_final_model.py  (test current working model)")
    print("2. Check final_model_trajectory_test.png (visual results)")
    print("")
    print("The training showed our fixes work:")
    print("- Scale factors learned: ~4.5-5.0x (close to needed 5.3x)")
    print("- Direction loss improved: showing directional consistency")
    print("- Magnitude ratios: improved from 0.19 towards 1.0")

if __name__ == '__main__':
    main()