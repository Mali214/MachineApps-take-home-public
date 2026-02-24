"""Tests for MotionController. Run with: pytest tests/test_motion.py -v

These tests use mock mode (no real robot connection needed).
"""

import math
import pytest

from robot.connection import RobotConnection
from robot.motion import MotionController


@pytest.fixture
def mock_controller():
    """Create a MotionController backed by a mock connection."""
    conn = RobotConnection(host="mock-host")
    # In mock mode (ur_rtde not installed or not connected)
    conn._mock_mode = True
    conn._connected = True
    return MotionController(conn)


class TestMoveToHome:
    def test_home_returns_true(self, mock_controller):
        assert mock_controller.move_to_home() is True


class TestPickSequence:
    def test_pick_returns_true(self, mock_controller):
        result = mock_controller.move_to_pick([0.4, -0.2, 0.1])
        assert result is True

    def test_pick_closes_gripper(self, mock_controller):
        mock_controller.move_to_pick([0.4, -0.2, 0.1])
        assert mock_controller._gripper_closed is True

    def test_pick_with_custom_orientation(self, mock_controller):
        result = mock_controller.move_to_pick(
            [0.4, -0.2, 0.1],
            orientation=[0.0, math.pi, 0.1],
        )
        assert result is True


class TestPlaceSequence:
    def test_place_returns_true(self, mock_controller):
        result = mock_controller.move_to_place([0.5, -0.3, 0.15])
        assert result is True

    def test_place_opens_gripper(self, mock_controller):
        mock_controller._gripper_closed = True
        mock_controller.move_to_place([0.5, -0.3, 0.15])
        assert mock_controller._gripper_closed is False

    def test_place_with_custom_orientation(self, mock_controller):
        result = mock_controller.move_to_place(
            [0.5, -0.3, 0.15],
            orientation=[0.0, math.pi, 0.0],
        )
        assert result is True


class TestApproachHeight:
    def test_approach_offset_is_at_least_50mm(self):
        assert MotionController.APPROACH_HEIGHT_OFFSET >= 0.050


class TestDefaultOrientation:
    def test_default_orientation_points_down(self, mock_controller):
        orient = mock_controller.get_default_orientation()
        assert len(orient) == 3
        # Should be [0, pi, 0] for tool-down
        assert abs(orient[1] - math.pi) < 1e-6
