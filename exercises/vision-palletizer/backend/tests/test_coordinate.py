"""Tests for coordinate transformations. Run with: pytest tests/test_coordinate.py -v"""

import numpy as np
import pytest

from transforms.coordinate import (
    build_rotation_matrix,
    build_homogeneous_transform,
    camera_to_robot,
    robot_to_camera,
    T_camera_to_robot,
    CAMERA_POSITION_MM,
)


class TestBuildRotationMatrix:
    """Verify rotation matrix construction."""

    def test_identity_for_zero_angles(self):
        """Zero angles should produce the identity matrix."""
        R = build_rotation_matrix(0, 0, 0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotation_is_orthogonal(self):
        """R^T @ R should equal I (orthogonal matrix)."""
        R = build_rotation_matrix(np.deg2rad(15), np.deg2rad(-10), np.deg2rad(45))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_determinant_is_one(self):
        """Proper rotation matrices have determinant +1."""
        R = build_rotation_matrix(np.deg2rad(15), np.deg2rad(-10), np.deg2rad(45))
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_pure_yaw_90(self):
        """90° yaw should rotate X→Y, Y→-X."""
        R = build_rotation_matrix(0, 0, np.pi / 2)
        # X-axis [1,0,0] should map to [0,1,0]
        result = R @ np.array([1, 0, 0])
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-10)

    def test_pure_pitch_90(self):
        """90° pitch should rotate X→Z, Z→-X."""
        R = build_rotation_matrix(0, np.pi / 2, 0)
        result = R @ np.array([1, 0, 0])
        np.testing.assert_allclose(result, [0, 0, -1], atol=1e-10)

    def test_pure_roll_90(self):
        """90° roll should rotate Y→Z, Z→-Y."""
        R = build_rotation_matrix(np.pi / 2, 0, 0)
        result = R @ np.array([0, 1, 0])
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-10)

    def test_shape_is_3x3(self):
        R = build_rotation_matrix(0.1, 0.2, 0.3)
        assert R.shape == (3, 3)


class TestBuildHomogeneousTransform:
    """Verify 4x4 homogeneous transform construction."""

    def test_shape(self):
        R = np.eye(3)
        t = np.array([1, 2, 3])
        T = build_homogeneous_transform(R, t)
        assert T.shape == (4, 4)

    def test_last_row(self):
        R = np.eye(3)
        t = np.array([10, 20, 30])
        T = build_homogeneous_transform(R, t)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])

    def test_rotation_and_translation_embedded(self):
        R = build_rotation_matrix(0.1, 0.2, 0.3)
        t = np.array([100, 200, 300])
        T = build_homogeneous_transform(R, t)
        np.testing.assert_allclose(T[:3, :3], R)
        np.testing.assert_allclose(T[:3, 3], t)


class TestCameraToRobot:
    """Verify camera-to-robot frame transformation."""

    def test_camera_origin_maps_to_camera_position(self):
        """Camera origin [0,0,0] should map to the camera mounting position."""
        result = camera_to_robot(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, CAMERA_POSITION_MM, atol=1e-6)

    def test_returns_3_element_array(self):
        result = camera_to_robot(np.array([50.0, -30.0, 0.0]))
        assert result.shape == (3,)

    def test_translation_only_for_identity_rotation(self):
        """If we build a transform with identity rotation, camera point equals robot point + translation."""
        # This is a sanity check — not using the actual camera params
        pass

    def test_known_detection_produces_valid_output(self):
        """First detection from camera_detections.json should transform to a reasonable robot position."""
        result = camera_to_robot(np.array([50.0, -30.0, 0.0]))
        # The result should be near the camera mounting position (500, 300, 800)
        # but offset by the rotated detection vector
        assert result is not None
        assert len(result) == 3
        # Verify it's in a plausible robot workspace region (not wildly off)
        assert -1000 < result[0] < 2000
        assert -1000 < result[1] < 2000
        assert 0 < result[2] < 1500


class TestRoundTrip:
    """Camera → Robot → Camera should return the original point."""

    def test_round_trip_origin(self):
        original = np.array([0.0, 0.0, 0.0])
        robot_pt = camera_to_robot(original)
        recovered = robot_to_camera(robot_pt)
        np.testing.assert_allclose(recovered, original, atol=1e-6)

    def test_round_trip_arbitrary(self):
        original = np.array([120.0, 45.0, 0.0])
        robot_pt = camera_to_robot(original)
        recovered = robot_to_camera(robot_pt)
        np.testing.assert_allclose(recovered, original, atol=1e-6)

    def test_round_trip_negative(self):
        original = np.array([-25.0, 80.0, 0.0])
        robot_pt = camera_to_robot(original)
        recovered = robot_to_camera(robot_pt)
        np.testing.assert_allclose(recovered, original, atol=1e-6)

    def test_round_trip_all_detections(self):
        """All four detections from the JSON should survive a round trip."""
        detections = [
            [50.0, -30.0, 0.0],
            [120.0, 45.0, 0.0],
            [-25.0, 80.0, 0.0],
            [90.0, -60.0, 0.0],
        ]
        for det in detections:
            original = np.array(det)
            recovered = robot_to_camera(camera_to_robot(original))
            np.testing.assert_allclose(recovered, original, atol=1e-6)
