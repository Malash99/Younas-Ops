#!/usr/bin/env python3
"""
Validation script to check reference frame consistency between 
ground truth computation and trajectory integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

# Import both old and new implementations
from utils.coordinate_transforms import (
    compute_relative_transformation as old_compute_relative,
    integrate_trajectory as old_integrate,
    transform_pose_to_global as old_transform
)

from utils.coordinate_transforms_fixed import (
    compute_relative_pose_correct as new_compute_relative,
    integrate_trajectory_correct as new_integrate,
    validate_reference_frames
)


def create_test_trajectory():
    """Create a known test trajectory for validation."""
    # Simple trajectory: forward motion with some rotation
    poses = []
    
    # Start at origin
    poses.append({
        'position': np.array([0., 0., 0.]),
        'quaternion': np.array([1., 0., 0., 0.])  # [w, x, y, z]
    })
    
    # Move forward 1m
    poses.append({
        'position': np.array([1., 0., 0.]),
        'quaternion': np.array([1., 0., 0., 0.])
    })
    
    # Turn 90° and move forward 1m more
    q_90 = Quaternion(axis=[0, 0, 1], angle=np.pi/2)
    poses.append({
        'position': np.array([1., 1., 0.]),
        'quaternion': q_90.elements
    })
    
    # Turn another 45° and move diagonally
    q_135 = Quaternion(axis=[0, 0, 1], angle=3*np.pi/4)
    poses.append({
        'position': np.array([1. - np.sqrt(2)/2, 1. + np.sqrt(2)/2, 0.]),
        'quaternion': q_135.elements
    })
    
    return poses


def test_relative_pose_computation():
    """Test relative pose computation consistency."""
    print("="*80)
    print("TESTING RELATIVE POSE COMPUTATION")
    print("="*80)
    
    poses = create_test_trajectory()
    
    for i in range(len(poses) - 1):
        pose1 = poses[i]
        pose2 = poses[i + 1]
        
        print(f"\\nFrame {i} -> {i+1}:")
        print(f"  Pose1: pos={pose1['position']}, quat={pose1['quaternion']}")
        print(f"  Pose2: pos={pose2['position']}, quat={pose2['quaternion']}")
        
        # Compute with both methods
        try:
            old_rel = old_compute_relative(pose1, pose2)
            print(f"  Old method: {old_rel}")
        except Exception as e:
            print(f"  Old method failed: {e}")
            old_rel = None
        
        try:
            new_rel = new_compute_relative(pose1, pose2)
            print(f"  New method: {new_rel}")
        except Exception as e:
            print(f"  New method failed: {e}")
            new_rel = None
        
        if old_rel is not None and new_rel is not None:
            diff = np.linalg.norm(old_rel - new_rel)
            print(f"  Difference magnitude: {diff:.6f}")
            if diff > 0.01:
                print(f"  WARNING: Large difference between methods!")


def test_trajectory_integration():
    """Test trajectory integration consistency."""
    print("="*80)
    print("TESTING TRAJECTORY INTEGRATION")
    print("="*80)
    
    poses = create_test_trajectory()
    
    # Compute relative poses with new method
    relative_poses = []
    for i in range(len(poses) - 1):
        rel_pose = new_compute_relative(poses[i], poses[i + 1])
        relative_poses.append(rel_pose)
    
    relative_poses = np.array(relative_poses)
    print(f"\\nComputed {len(relative_poses)} relative poses")
    print(f"Relative poses shape: {relative_poses.shape}")
    
    # Test integration with both methods
    initial_pose = poses[0]
    
    # Old method integration
    try:
        old_trajectory = old_integrate(initial_pose, relative_poses)
        print(f"\\nOld integration result shape: {old_trajectory.shape}")
        print(f"Old trajectory (positions only):")
        if len(old_trajectory.shape) == 2:
            for i, pos in enumerate(old_trajectory):
                if len(pos) >= 3:
                    print(f"  Frame {i}: {pos[:3]}")
                else:
                    print(f"  Frame {i}: {pos}")
    except Exception as e:
        print(f"\\nOld integration failed: {e}")
        old_trajectory = None
    
    # New method integration
    try:
        new_trajectory = new_integrate(initial_pose, relative_poses)
        print(f"\\nNew integration result (list length): {len(new_trajectory)}")
        print(f"New trajectory (positions only):")
        for i, pose_dict in enumerate(new_trajectory):
            print(f"  Frame {i}: {pose_dict['position']}")
    except Exception as e:
        print(f"\\nNew integration failed: {e}")
        new_trajectory = None
    
    # Compare with ground truth
    print(f"\\nGround truth positions:")
    for i, pose in enumerate(poses):
        print(f"  Frame {i}: {pose['position']}")
    
    # Compute errors
    if new_trajectory is not None:
        print(f"\\nNew method errors:")
        for i, (gt_pose, pred_pose) in enumerate(zip(poses, new_trajectory)):
            error = np.linalg.norm(gt_pose['position'] - pred_pose['position'])
            print(f"  Frame {i}: {error:.6f} m")


def test_round_trip_consistency():
    """Test that relative pose computation and integration are inverses."""
    print("="*80)
    print("TESTING ROUND-TRIP CONSISTENCY")
    print("="*80)
    
    poses = create_test_trajectory()
    
    # For each consecutive pair
    for i in range(len(poses) - 1):
        pose1 = poses[i]
        pose2 = poses[i + 1]
        
        print(f"\\nTesting pair {i} -> {i+1}:")
        
        # Compute relative pose
        rel_pose = new_compute_relative(pose1, pose2)
        print(f"  Relative pose: {rel_pose}")
        
        # Integrate to reconstruct pose2
        reconstructed_trajectory = new_integrate(pose1, [rel_pose])
        reconstructed_pose2 = reconstructed_trajectory[-1]
        
        # Compare
        pos_error = np.linalg.norm(pose2['position'] - reconstructed_pose2['position'])
        
        # Compare quaternions
        q_gt = Quaternion(pose2['quaternion'])
        q_recon = Quaternion(reconstructed_pose2['quaternion'])
        rot_error = Quaternion.absolute_distance(q_gt, q_recon)
        
        print(f"  Original pose2: pos={pose2['position']}, quat={pose2['quaternion']}")
        print(f"  Reconstructed:  pos={reconstructed_pose2['position']}, quat={reconstructed_pose2['quaternion']}")
        print(f"  Position error: {pos_error:.8f} m")
        print(f"  Rotation error: {rot_error:.8f} rad ({np.degrees(rot_error):.6f}°)")
        
        if pos_error > 1e-6:
            print(f"  WARNING: Position error too large!")
        if rot_error > 1e-6:
            print(f"  WARNING: Rotation error too large!")


def visualize_trajectory_comparison():
    """Visualize trajectory differences."""
    print("="*80)
    print("CREATING TRAJECTORY VISUALIZATION")
    print("="*80)
    
    poses = create_test_trajectory()
    
    # Compute relative poses
    relative_poses = []
    for i in range(len(poses) - 1):
        rel_pose = new_compute_relative(poses[i], poses[i + 1])
        relative_poses.append(rel_pose)
    
    relative_poses = np.array(relative_poses)
    
    # Integrate trajectory
    initial_pose = poses[0]
    integrated_trajectory = new_integrate(initial_pose, relative_poses)
    
    # Extract positions
    gt_positions = np.array([pose['position'] for pose in poses])
    integrated_positions = np.array([pose['position'] for pose in integrated_trajectory])
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], 'bo-', label='Ground Truth', markersize=8)
    plt.plot(integrated_positions[:, 0], integrated_positions[:, 1], 'rx-', label='Integrated', markersize=8)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('XY Trajectory Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.subplot(2, 2, 2)
    errors = np.linalg.norm(gt_positions - integrated_positions, axis=1)
    plt.plot(errors, 'g-', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Position Error (m)')
    plt.title('Position Error Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(gt_positions[:, 0], label='GT X', linewidth=2)
    plt.plot(integrated_positions[:, 0], '--', label='Integrated X', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('X Position (m)')
    plt.title('X Position Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(gt_positions[:, 1], label='GT Y', linewidth=2)
    plt.plot(integrated_positions[:, 1], '--', label='Integrated Y', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'validation')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'reference_frame_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation plot saved to: {output_dir}/reference_frame_validation.png")
    print(f"Maximum position error: {np.max(errors):.8f} m")
    print(f"Mean position error: {np.mean(errors):.8f} m")


def main():
    """Run all validation tests."""
    print("REFERENCE FRAME VALIDATION SUITE")
    print("="*80)
    
    # Create validation output directory
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'validation'), exist_ok=True)
    
    # Run tests
    validate_reference_frames()
    test_relative_pose_computation() 
    test_trajectory_integration()
    test_round_trip_consistency()
    visualize_trajectory_comparison()
    
    print("\\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\\nKey Findings:")
    print("1. Check console output above for any WARNING messages")
    print("2. Round-trip consistency errors should be < 1e-6")
    print("3. Integration should exactly reconstruct ground truth")
    print("4. Check validation plot for visual confirmation")
    print("\\nIf all tests pass, the fixed implementation is ready to use!")


if __name__ == "__main__":
    main()