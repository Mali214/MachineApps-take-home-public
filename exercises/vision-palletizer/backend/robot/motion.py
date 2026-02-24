"""
Motion Controller
================

Implements robot motion commands for pick and place operations.
"""

import time
import math
from typing import Optional
import numpy as np

from .connection import RobotConnection


class MotionController:
    """
    Controls robot motion for palletizing operations.

    Coordinates:
    - All positions are in meters
    - All orientations are in radians (axis-angle representation for UR)

    Motion strategy:
    - Large moves use getInverseKinematics + moveJ (seeded with HOME_JOINTS to reduce flips)
    - Short moves (approach ↔ target, ~100mm) use moveL
    """

    # Safety parameters
    APPROACH_HEIGHT_OFFSET = 0.100  # 100mm above pick/place position
    DEFAULT_VELOCITY = 0.5         # m/s  (linear moves)
    DEFAULT_ACCELERATION = 0.5      # m/s²
    JOINT_VELOCITY = 1.05           # rad/s (joint moves)
    JOINT_ACCELERATION = 1.4        # rad/s²
    GRIPPER_DWELL = 0.5             # seconds to simulate gripper actuation

    # UR5e workspace limits (metres)
    MAX_REACH_M = 0.850             # UR5e max reach radius
    MIN_REACH_M = 0.150             # inner dead-zone — arm can't reach directly under base
    MIN_Z_M = -0.01                 # slightly below base plane
    MAX_Z_M = 1.200                 # well above base

    # ── Home / reference joint configuration ────────────────────────────
    #
    # Safe home posture — tool pointing down.
    #   J1 =    0°   base facing forward
    #   J2 =  -90°   shoulder horizontal
    #   J3 =  -90°   elbow pointing down/forward
    #   J4 =  -90°   wrist1 compensates
    #   J5 =   90°   wrist2 — TCP pointing straight down
    #   J6 =    0°   no extra rotation
    HOME_JOINTS = [
        0.0,
        math.radians(-90),    # -1.5708
        math.radians(-90),    # -1.5708
        math.radians(-90),    # -1.5708
        math.radians( 90),    #  1.5708
        0.0,
    ]

    def __init__(self, connection: RobotConnection):
        self.connection = connection
        self._gripper_closed = False

    # ── workspace validation ─────────────────────────────────────────────

    def validate_position(self, position: list[float]) -> bool:
        """
        Check that a Cartesian position [x, y, z] (metres) is within the
        UR5e workspace.  Raises ValueError if out of range.
        """
        x, y, z = position[0], position[1], position[2]
        reach = math.sqrt(x * x + y * y)
        if reach < self.MIN_REACH_M:
            raise ValueError(
                f"Position ({x:.3f}, {y:.3f}) is inside dead-zone "
                f"({reach:.3f} m < {self.MIN_REACH_M} m)"
            )
        if reach > self.MAX_REACH_M:
            raise ValueError(
                f"Position ({x:.3f}, {y:.3f}) exceeds max reach "
                f"({reach:.3f} m > {self.MAX_REACH_M} m)"
            )
        if z < self.MIN_Z_M or z > self.MAX_Z_M:
            raise ValueError(
                f"Z={z:.3f} m is outside workspace [{self.MIN_Z_M}, {self.MAX_Z_M}] m"
            )
        return True

    # ── high-level motions ──────────────────────────────────────────────

    def move_to_home(self) -> bool:
        """Move robot to a safe home position."""
        print("[MOTION] Moving to home position ...")
        return self._move_joint(self.HOME_JOINTS)

    def move_to_pick(
        self,
        position: list[float],
        orientation: Optional[list[float]] = None,
    ) -> bool:
        """
        Pick sequence:
          1. Joint-move to approach (IK, elbow-up)
          2. Linear descend
          3. Close gripper
          4. Linear retract
        """
        if orientation is None:
            orientation = self.get_default_orientation()

        x, y, z = position
        approach_z = z + self.APPROACH_HEIGHT_OFFSET

        approach_pose = [x, y, approach_z] + orientation
        pick_pose     = [x, y, z]          + orientation

        self.validate_position([x, y, z])
        self.validate_position([x, y, approach_z])

        print(f"[MOTION] Pick: approach z={approach_z:.3f}")
        if not self._move_to_pose_joint(approach_pose):
            return False

        print(f"[MOTION] Pick: descend to z={z:.3f}")
        if not self._move_linear(pick_pose):
            return False

        print("[MOTION] Pick: closing gripper")
        if not self.close_gripper():
            return False

        print("[MOTION] Pick: retracting")
        if not self._move_linear(approach_pose):
            return False

        print("[MOTION] Pick complete")
        return True

    def move_to_place(
        self,
        position: list[float],
        orientation: Optional[list[float]] = None,
    ) -> bool:
        """
        Place sequence:
          1. Joint-move to approach (IK, elbow-up)
          2. Linear descend
          3. Open gripper
          4. Linear retract
        """
        if orientation is None:
            orientation = self.get_default_orientation()

        x, y, z = position
        approach_z = z + self.APPROACH_HEIGHT_OFFSET

        approach_pose = [x, y, approach_z] + orientation
        place_pose    = [x, y, z]          + orientation

        self.validate_position([x, y, z])
        self.validate_position([x, y, approach_z])

        print(f"[MOTION] Place: approach z={approach_z:.3f}")
        if not self._move_to_pose_joint(approach_pose):
            return False

        print(f"[MOTION] Place: descend to z={z:.3f}")
        if not self._move_linear(place_pose):
            return False

        print("[MOTION] Place: opening gripper")
        if not self.open_gripper():
            return False

        print("[MOTION] Place: retracting")
        if not self._move_linear(approach_pose):
            return False

        print("[MOTION] Place complete")
        return True

    # ── gripper ─────────────────────────────────────────────────────────

    def open_gripper(self) -> bool:
        """Simulate opening the gripper (no physical I/O in URSim)."""
        self._gripper_closed = False
        time.sleep(self.GRIPPER_DWELL)
        print("[GRIPPER] Opened")
        return True

    def close_gripper(self) -> bool:
        """Simulate closing the gripper (no physical I/O in URSim)."""
        self._gripper_closed = True
        time.sleep(self.GRIPPER_DWELL)
        print("[GRIPPER] Closed")
        return True

    # ── low-level moves ─────────────────────────────────────────────────

    def _move_linear(
        self,
        pose: list[float],
        velocity: float = DEFAULT_VELOCITY,
        acceleration: float = DEFAULT_ACCELERATION,
    ) -> bool:
        """Linear (Cartesian) move — use only for short distances."""
        if self.connection.is_mock_mode():
            print(f"[MOCK] moveL to {[round(v, 4) for v in pose[:3]]}")
            return True

        ctrl = self.connection.control
        if ctrl is None:
            raise RuntimeError("Robot not connected")

        ctrl.moveL(pose, velocity, acceleration)
        return True

    def _move_to_pose_joint(
        self,
        pose: list[float],
        velocity: float = JOINT_VELOCITY,
        acceleration: float = JOINT_ACCELERATION,
    ) -> bool:
        """
        Move to a Cartesian pose via joint-space.

        Uses getInverseKinematics with the current joint positions as
        q_near so the IK solver picks a solution close to where the arm
        already is (preserving the elbow-up configuration).
        Then executes moveJ to those joints.
        """
        if self.connection.is_mock_mode():
            print(f"[MOCK] moveJ→pose {[round(v, 4) for v in pose[:3]]}")
            return True

        ctrl = self.connection.control
        recv = self.connection.receive
        if ctrl is None or recv is None:
            raise RuntimeError("Robot not connected")

        # Seed IK with HOME_JOINTS to keep solutions consistent and avoid flips.
        q_near = list(self.HOME_JOINTS)          # plain Python floats
        q_target = ctrl.getInverseKinematics(pose, q_near)

        print(f"  [IK] q_near  = {[round(q, 2) for q in q_near]}")
        print(f"  [IK] q_target= {[round(q, 2) for q in q_target]}")

        ctrl.moveJ(q_target, velocity, acceleration)
        return True

    def _move_joint(
        self,
        joints: list[float],
        velocity: float = JOINT_VELOCITY,
        acceleration: float = JOINT_ACCELERATION,
    ) -> bool:
        """Joint-space move to explicit joint angles."""
        if self.connection.is_mock_mode():
            print(f"[MOCK] moveJ to {[round(j, 4) for j in joints]}")
            return True

        ctrl = self.connection.control
        if ctrl is None:
            raise RuntimeError("Robot not connected")

        ctrl.moveJ(joints, velocity, acceleration)
        return True

    def get_default_orientation(self) -> list[float]:
        """
        Default tool orientation for picking (pointing straight down).

        Returns [rx, ry, rz] in axis-angle representation.
        """
        return [0.0, np.pi, 0.0]

    @staticmethod
    def compose_orientation_with_yaw(
        base_rotvec: list[float],
        yaw_rad: float,
    ) -> list[float]:
        """
        Compose a base orientation (rotation vector) with a Z-axis yaw.

        UR robots express TCP orientation as a *rotation vector* (axis-angle).
        Simply adding ``yaw`` to one element is incorrect.  This method:

        1. Converts the base rotation vector to a rotation matrix (Rodrigues).
        2. Builds ``Rz(yaw)`` in the robot base frame.
        3. Composes ``R_final = Rz(yaw) @ R_base``.
        4. Converts back to a rotation vector.

        Args:
            base_rotvec: [rx, ry, rz] rotation vector (axis-angle, radians).
            yaw_rad: Yaw angle to apply about the robot Z-axis (radians).

        Returns:
            New [rx, ry, rz] rotation vector with yaw composed in.
        """
        if abs(yaw_rad) < 1e-10:
            return list(base_rotvec)

        # --- Rodrigues: rotation vector → rotation matrix -----------------
        rv = np.asarray(base_rotvec, dtype=float)
        angle = float(np.linalg.norm(rv))
        if angle < 1e-10:
            R_base = np.eye(3)
        else:
            k = rv / angle
            K = np.array([
                [    0, -k[2],  k[1]],
                [ k[2],     0, -k[0]],
                [-k[1],  k[0],     0],
            ])
            R_base = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # --- Rz(yaw) -----------------------------------------------------
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw = np.array([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [ 0,   0, 1],
        ])

        # --- compose ------------------------------------------------------
        R_final = R_yaw @ R_base

        # --- rotation matrix → rotation vector (inverse Rodrigues) --------
        angle_f = np.arccos(np.clip((np.trace(R_final) - 1.0) / 2.0, -1.0, 1.0))
        if angle_f < 1e-10:
            return [0.0, 0.0, 0.0]

        axis_f = np.array([
            R_final[2, 1] - R_final[1, 2],
            R_final[0, 2] - R_final[2, 0],
            R_final[1, 0] - R_final[0, 1],
        ]) / (2.0 * np.sin(angle_f))

        result = axis_f * angle_f
        return [float(result[0]), float(result[1]), float(result[2])]
