"""
CORRECTED coordinate transformation utilities for underwater visual odometry.
Ensures consistent reference frame handling between ground truth and predictions.
"""

import numpy as np
from pyquaternion import Quaternion
from typing import Tuple, Union, Dict


def compute_relative_pose_correct(pose1: Dict, pose2: Dict) -> np.ndarray:
    """
    Compute relative pose from pose1 to pose2 in pose1's local frame.
    This is what the model should predict and what ground truth should provide.
    
    Args:
        pose1, pose2: Pose dictionaries with 'position' and 'quaternion'
        
    Returns:
        6D relative pose [dx, dy, dz, d_rx, d_ry, d_rz] in pose1's local frame
    """
    # Positions
    p1 = np.array(pose1['position'])
    p2 = np.array(pose2['position'])
    
    # Quaternions (ensure [w, x, y, z] format)
    if len(pose1['quaternion']) == 4:
        q1 = Quaternion(pose1['quaternion'])  # Assumes [w, x, y, z]
        q2 = Quaternion(pose2['quaternion'])
    else:
        raise ValueError("Quaternion must have 4 elements [w, x, y, z]")
    
    # Relative rotation: q_rel = q1^-1 * q2
    q_rel = q1.inverse * q2
    
    # Relative translation in pose1's local frame
    t_rel = q1.inverse.rotate(p2 - p1)
    
    # Convert relative rotation to rotation vector (axis * angle)
    if q_rel.angle > 1e-6:  # Avoid division by zero
        axis = q_rel.axis
        angle = q_rel.angle
        rot_vec = axis * angle
    else:
        rot_vec = np.array([0., 0., 0.])
    
    return np.concatenate([t_rel, rot_vec]).astype(np.float32)


def integrate_trajectory_correct(initial_pose: Dict, relative_poses: np.ndarray) -> list:
    """
    Correctly integrate relative poses to global trajectory.
    
    Args:
        initial_pose: Starting pose {'position': [x,y,z], 'quaternion': [w,x,y,z]}
        relative_poses: Array of relative poses (N, 6) in local frames
        
    Returns:
        List of pose dictionaries representing global trajectory
    """
    trajectory = [initial_pose.copy()]
    current_pose = initial_pose.copy()
    
    for rel_pose in relative_poses:
        # Extract relative translation and rotation vector
        rel_trans = rel_pose[:3]
        rel_rot_vec = rel_pose[3:]
        
        # Current global rotation
        q_current = Quaternion(current_pose['quaternion'])
        
        # Transform relative translation to global frame
        global_trans_delta = q_current.rotate(rel_trans)
        new_position = current_pose['position'] + global_trans_delta
        
        # Compose rotations: q_new = q_current * q_rel
        if np.linalg.norm(rel_rot_vec) > 1e-6:
            angle = np.linalg.norm(rel_rot_vec)
            axis = rel_rot_vec / angle
            q_rel = Quaternion(axis=axis, angle=angle)
        else:
            q_rel = Quaternion()  # Identity rotation
        
        q_new = q_current * q_rel
        
        # Update current pose
        current_pose = {
            'position': new_position,
            'quaternion': q_new.elements  # [w, x, y, z]
        }
        
        trajectory.append(current_pose.copy())
    
    return trajectory


def extract_positions_from_trajectory(trajectory: list) -> np.ndarray:
    """Extract positions from trajectory for plotting."""
    return np.array([pose['position'] for pose in trajectory])


def compute_trajectory_errors_correct(pred_trajectory: list, gt_trajectory: list) -> Dict:
    """
    Compute trajectory errors correctly.
    
    Args:
        pred_trajectory, gt_trajectory: Lists of pose dictionaries
        
    Returns:
        Dictionary with error metrics
    """
    # Extract positions
    pred_positions = extract_positions_from_trajectory(pred_trajectory)
    gt_positions = extract_positions_from_trajectory(gt_trajectory)
    
    # Ensure same length
    min_len = min(len(pred_positions), len(gt_positions))
    pred_positions = pred_positions[:min_len]
    gt_positions = gt_positions[:min_len]
    
    # Absolute Trajectory Error (ATE)
    position_errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
    ate_rmse = np.sqrt(np.mean(position_errors ** 2))
    ate_mean = np.mean(position_errors)
    ate_std = np.std(position_errors)
    
    # Relative Pose Error (RPE) - consecutive frames
    rpe_trans_errors = []
    rpe_rot_errors = []
    
    for i in range(min_len - 1):
        # Compute relative poses for consecutive frames
        pred_rel = compute_relative_pose_correct(pred_trajectory[i], pred_trajectory[i+1])
        gt_rel = compute_relative_pose_correct(gt_trajectory[i], gt_trajectory[i+1])
        
        # Translation error
        trans_error = np.linalg.norm(pred_rel[:3] - gt_rel[:3])
        rpe_trans_errors.append(trans_error)
        
        # Rotation error (angle between rotation vectors)
        pred_rot = pred_rel[3:]
        gt_rot = gt_rel[3:]
        
        pred_angle = np.linalg.norm(pred_rot)
        gt_angle = np.linalg.norm(gt_rot)
        
        if pred_angle > 1e-6 and gt_angle > 1e-6:
            pred_axis = pred_rot / pred_angle
            gt_axis = gt_rot / gt_angle
            
            # Angle between rotation axes
            cos_angle = np.clip(np.dot(pred_axis, gt_axis), -1, 1)
            axis_error = np.arccos(np.abs(cos_angle))
            
            # Total rotation error
            rot_error = np.abs(pred_angle - gt_angle) + axis_error
        else:
            rot_error = np.abs(pred_angle - gt_angle)
        
        rpe_rot_errors.append(rot_error)
    
    return {
        'ate_rmse': ate_rmse,
        'ate_mean': ate_mean,
        'ate_std': ate_std,
        'rpe_trans_mean': np.mean(rpe_trans_errors),
        'rpe_trans_std': np.std(rpe_trans_errors),
        'rpe_rot_mean': np.mean(rpe_rot_errors),
        'rpe_rot_std': np.std(rpe_rot_errors),
        'position_errors': position_errors
    }


def validate_reference_frames():
    """
    Validate that the reference frame transformations are working correctly.
    """
    print("="*60)
    print("VALIDATING REFERENCE FRAME TRANSFORMATIONS")
    print("="*60)
    
    # Test case 1: Simple forward motion
    pose1 = {
        'position': np.array([0., 0., 0.]),
        'quaternion': np.array([1., 0., 0., 0.])  # No rotation
    }
    
    pose2 = {
        'position': np.array([1., 0., 0.]),  # Move 1m forward
        'quaternion': np.array([1., 0., 0., 0.])  # No rotation
    }
    
    # Compute relative pose
    rel_pose = compute_relative_pose_correct(pose1, pose2)
    print(f"Test 1 - Forward motion:")
    print(f"  Relative pose: {rel_pose}")
    print(f"  Expected: [1, 0, 0, 0, 0, 0]")
    print(f"  Match: {np.allclose(rel_pose, [1, 0, 0, 0, 0, 0])}")
    
    # Test trajectory integration
    trajectory = integrate_trajectory_correct(pose1, [rel_pose])
    final_pos = trajectory[-1]['position']  
    print(f"  Integrated position: {final_pos}")
    print(f"  Expected: [1, 0, 0]")
    print(f"  Match: {np.allclose(final_pos, [1, 0, 0])}")
    
    # Test case 2: Rotation
    pose3 = {
        'position': np.array([0., 0., 0.]),
        'quaternion': Quaternion(axis=[0, 0, 1], angle=np.pi/2).elements  # 90° yaw
    }
    
    rel_pose_rot = compute_relative_pose_correct(pose1, pose3)
    print(f"\\nTest 2 - Pure rotation:")
    print(f"  Relative pose: {rel_pose_rot}")
    print(f"  Rotation magnitude: {np.linalg.norm(rel_pose_rot[3:]):.4f}")
    print(f"  Expected rotation: {np.pi/2:.4f}")
    
    # Test case 3: Combined motion
    pose4 = {
        'position': np.array([1., 1., 0.]),
        'quaternion': Quaternion(axis=[0, 0, 1], angle=np.pi/4).elements  # 45° yaw
    }
    
    rel_pose_combined = compute_relative_pose_correct(pose1, pose4)
    print(f"\\nTest 3 - Combined motion:")
    print(f"  Relative pose: {rel_pose_combined}")
    
    # Integrate and check
    trajectory_combined = integrate_trajectory_correct(pose1, [rel_pose_combined])
    final_pos_combined = trajectory_combined[-1]['position']
    final_quat_combined = trajectory_combined[-1]['quaternion']
    
    print(f"  Integrated position: {final_pos_combined}")
    print(f"  Expected position: [1, 1, 0]")
    print(f"  Position match: {np.allclose(final_pos_combined, [1, 1, 0], atol=1e-6)}")
    
    final_angle = Quaternion(final_quat_combined).angle
    print(f"  Integrated rotation angle: {final_angle:.4f}")
    print(f"  Expected angle: {np.pi/4:.4f}")
    print(f"  Rotation match: {np.allclose(final_angle, np.pi/4, atol=1e-6)}")
    
    print("="*60)
    return True


if __name__ == "__main__":
    validate_reference_frames()