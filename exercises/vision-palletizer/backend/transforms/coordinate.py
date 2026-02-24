"""
Coordinate Transformations
=========================

Transform coordinates between camera frame and robot base frame.

Camera mounting specifications (from README):
    Position (X, Y, Z): 500mm, 300mm, 800mm
    Orientation (Roll, Pitch, Yaw): 15°, -10°, 45°
    Rotation convention: Intrinsic rotations Z → Y → X
    Optical axis: Camera Z points toward the scene
"""

import numpy as np

# Camera mounting parameters
CAMERA_POSITION_MM = np.array([500.0, 300.0, 800.0])
CAMERA_ROLL_DEG = 15.0
CAMERA_PITCH_DEG = -10.0
CAMERA_YAW_DEG = 45.0


def build_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from Roll-Pitch-Yaw (Euler) angles.

    Uses intrinsic rotation order Z → Y → X:
        R = Rz(yaw) · Ry(pitch) · Rx(roll)

    Args:
        roll: Rotation about X-axis in radians
        pitch: Rotation about Y-axis in radians
        yaw: Rotation about Z-axis in radians

    Returns:
        3x3 rotation matrix
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Rz(yaw)
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,    0, 1],
    ])

    # Ry(pitch)
    Ry = np.array([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp],
    ])

    # Rx(roll)
    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr],
    ])

    return Rz @ Ry @ Rx


def build_homogeneous_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Build a 4x4 homogeneous transformation matrix.

    Args:
        rotation: 3x3 rotation matrix
        translation: 3x1 or (3,) translation vector

    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation).flatten()
    return T


def _get_camera_to_robot_transform() -> np.ndarray:
    """Build the 4x4 camera-to-robot homogeneous transform from mounting specs."""
    roll = np.deg2rad(CAMERA_ROLL_DEG)
    pitch = np.deg2rad(CAMERA_PITCH_DEG)
    yaw = np.deg2rad(CAMERA_YAW_DEG)

    R = build_rotation_matrix(roll, pitch, yaw)
    return build_homogeneous_transform(R, CAMERA_POSITION_MM)


# Pre-compute the transform and its inverse for efficiency
T_camera_to_robot = _get_camera_to_robot_transform()
T_robot_to_camera = np.linalg.inv(T_camera_to_robot)


def camera_to_robot(point_camera: np.ndarray) -> np.ndarray:
    """
    Transform a point from camera frame to robot base frame.

    Args:
        point_camera: [x, y, z] coordinates in camera frame (mm)

    Returns:
        [x, y, z] coordinates in robot base frame (mm)
    """
    point_camera = np.asarray(point_camera, dtype=float)
    # Convert to homogeneous coordinates [x, y, z, 1]
    point_h = np.append(point_camera, 1.0)
    # Apply transform
    result_h = T_camera_to_robot @ point_h
    return result_h[:3]


def robot_to_camera(point_robot: np.ndarray) -> np.ndarray:
    """
    Transform a point from robot base frame to camera frame.

    Args:
        point_robot: [x, y, z] coordinates in robot base frame (mm)

    Returns:
        [x, y, z] coordinates in camera frame (mm)
    """
    point_robot = np.asarray(point_robot, dtype=float)
    # Convert to homogeneous coordinates [x, y, z, 1]
    point_h = np.append(point_robot, 1.0)
    # Apply inverse transform
    result_h = T_robot_to_camera @ point_h
    return result_h[:3]
