"""
Coordinate transformation utilities for underwater visual odometry.
Handles conversions between different reference frames and pose representations.
"""

import numpy as np
from pyquaternion import Quaternion
from typing import Tuple, Union


def quaternion_to_rotation_matrix(q: Union[Quaternion, np.ndarray]) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion object or array [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    if isinstance(q, np.ndarray):
        q = Quaternion(q[0], q[1], q[2], q[3])
    
    return q.rotation_matrix


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Array of [roll, pitch, yaw] in radians
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # roll
        y = np.arctan2(-R[2, 0], sy)      # pitch
        z = np.arctan2(R[1, 0], R[0, 0])  # yaw
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.array([x, y, z])


def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        euler: Array of [roll, pitch, yaw] in radians
        
    Returns:
        3x3 rotation matrix
    """
    roll, pitch, yaw = euler
    
    # Rotation matrix from Euler angles
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    return R_z @ R_y @ R_x


def transform_pose_to_global(pose_robot: np.ndarray, current_global_pose: dict) -> dict:
    """
    Transform pose from robot frame to global frame.
    
    Args:
        pose_robot: 6D pose in robot frame [dx, dy, dz, droll, dpitch, dyaw]
        current_global_pose: Current global pose {'position': [x,y,z], 'quaternion': [w,x,y,z]}
        
    Returns:
        New global pose dictionary
    """
    # Extract translation and rotation
    dt_robot = pose_robot[:3]
    deuler_robot = pose_robot[3:]
    
    # Current global rotation
    q_global = Quaternion(current_global_pose['quaternion'])
    R_global = q_global.rotation_matrix
    
    # Transform translation to global frame
    dt_global = R_global @ dt_robot
    new_position = current_global_pose['position'] + dt_global
    
    # Transform rotation
    dR_robot = euler_to_rotation_matrix(deuler_robot)
    new_R_global = R_global @ dR_robot
    
    # Convert back to quaternion
    new_q_global = Quaternion(matrix=new_R_global)
    
    return {
        'position': new_position,
        'quaternion': np.array([new_q_global.w, new_q_global.x, new_q_global.y, new_q_global.z])
    }


def compute_relative_transformation(pose1: dict, pose2: dict) -> np.ndarray:
    """
    Compute relative transformation from pose1 to pose2.
    
    Args:
        pose1, pose2: Pose dictionaries with 'position' and 'quaternion'
        
    Returns:
        6D relative pose [dx, dy, dz, droll, dpitch, dyaw]
    """
    # Positions
    p1 = np.array(pose1['position'])
    p2 = np.array(pose2['position'])
    
    # Quaternions
    q1 = Quaternion(pose1['quaternion'])
    q2 = Quaternion(pose2['quaternion'])
    
    # Relative rotation
    q_rel = q1.inverse * q2
    
    # Relative translation in frame 1
    t_rel = q1.inverse.rotate(p2 - p1)
    
    # Convert to Euler angles
    euler_rel = rotation_matrix_to_euler(q_rel.rotation_matrix)
    
    return np.concatenate([t_rel, euler_rel])


def integrate_trajectory(initial_pose: dict, relative_poses: np.ndarray) -> np.ndarray:
    """
    Integrate relative poses to get global trajectory.
    
    Args:
        initial_pose: Starting pose dictionary
        relative_poses: Array of relative poses, shape (N, 6)
        
    Returns:
        Global trajectory array, shape (N+1, 7) with [x, y, z, qw, qx, qy, qz]
    """
    trajectory = []
    current_pose = initial_pose.copy()
    
    # Add initial pose
    trajectory.append(np.concatenate([
        current_pose['position'],
        current_pose['quaternion']
    ]))
    
    # Integrate relative poses
    for rel_pose in relative_poses:
        current_pose = transform_pose_to_global(rel_pose, current_pose)
        trajectory.append(np.concatenate([
            current_pose['position'],
            current_pose['quaternion']
        ]))
    
    return np.array(trajectory)


def compute_trajectory_error(pred_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> dict:
    """
    Compute trajectory errors (ATE and RPE).
    
    Args:
        pred_trajectory: Predicted trajectory, shape (N, 7)
        gt_trajectory: Ground truth trajectory, shape (N, 7)
        
    Returns:
        Dictionary with error metrics
    """
    # Absolute Trajectory Error (ATE)
    position_errors = np.linalg.norm(pred_trajectory[:, :3] - gt_trajectory[:, :3], axis=1)
    ate = np.mean(position_errors)
    
    # Relative Pose Error (RPE)
    rpe_trans = []
    rpe_rot = []
    
    for i in range(len(pred_trajectory) - 1):
        # Compute relative transformations
        pred_rel = compute_relative_transformation(
            {'position': pred_trajectory[i, :3], 'quaternion': pred_trajectory[i, 3:]},
            {'position': pred_trajectory[i+1, :3], 'quaternion': pred_trajectory[i+1, 3:]}
        )
        
        gt_rel = compute_relative_transformation(
            {'position': gt_trajectory[i, :3], 'quaternion': gt_trajectory[i, 3:]},
            {'position': gt_trajectory[i+1, :3], 'quaternion': gt_trajectory[i+1, 3:]}
        )
        
        # Translation error
        rpe_trans.append(np.linalg.norm(pred_rel[:3] - gt_rel[:3]))
        
        # Rotation error (angle difference)
        pred_q = Quaternion(axis=[0, 0, 1], angle=pred_rel[5])  # Using yaw for simplicity
        gt_q = Quaternion(axis=[0, 0, 1], angle=gt_rel[5])
        angle_diff = Quaternion.absolute_distance(pred_q, gt_q)
        rpe_rot.append(angle_diff)
    
    return {
        'ate': ate,
        'ate_std': np.std(position_errors),
        'rpe_trans': np.mean(rpe_trans),
        'rpe_trans_std': np.std(rpe_trans),
        'rpe_rot': np.mean(rpe_rot),
        'rpe_rot_std': np.std(rpe_rot)
    }


if __name__ == "__main__":
    # Test transformations
    print("Testing coordinate transformations...")
    
    # Test quaternion to rotation matrix
    q = Quaternion(axis=[0, 0, 1], angle=np.pi/4)
    R = quaternion_to_rotation_matrix(q)
    print(f"Rotation matrix for 45° yaw:\n{R}")
    
    # Test Euler conversion
    euler = rotation_matrix_to_euler(R)
    print(f"Euler angles: {np.degrees(euler)} degrees")
    
    # Test relative transformation
    pose1 = {
        'position': np.array([0, 0, 0]),
        'quaternion': np.array([1, 0, 0, 0])
    }
    pose2 = {
        'position': np.array([1, 0, 0]),
        'quaternion': Quaternion(axis=[0, 0, 1], angle=np.pi/6).elements
    }
    
    rel_pose = compute_relative_transformation(pose1, pose2)
    print(f"Relative pose: {rel_pose}")
    print(f"Translation: {rel_pose[:3]}, Rotation (deg): {np.degrees(rel_pose[3:])}")