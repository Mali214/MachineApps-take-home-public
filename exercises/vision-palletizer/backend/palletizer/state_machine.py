"""
Palletizer State Machine

Manages the lifecycle of palletizing operations using vention-state-machine.
Documentation: https://docs.vention.io/docs/state-machine
"""

import json
import threading
from enum import Enum, auto
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent

from state_machine.core import StateMachine, BaseTriggers
from state_machine.defs import StateGroup, State, Trigger
from state_machine.decorators import on_enter_state, on_state_change

from transforms.coordinate import camera_to_robot
from palletizer.grid import calculate_place_positions
from robot.motion import MotionController
from robot.connection import RobotConnection


class PalletizerState(Enum):
    """Palletizer operation states."""
    IDLE = auto()
    HOMING = auto()
    PICKING = auto()
    PLACING = auto()
    FAULT = auto()


class Running(StateGroup):
    """Active operation states."""
    homing: State = State()
    picking: State = State()
    placing: State = State()


class States:
    running = Running()


class Triggers:
    """Named events that initiate transitions."""
    finished_homing = Trigger("finished_homing")
    finished_picking = Trigger("finished_picking")
    finished_placing = Trigger("finished_placing")
    cycle_complete = Trigger("cycle_complete")
    stop = Trigger("stop")


TRANSITIONS = [
    Trigger("start").transition("ready", States.running.homing),
    Triggers.finished_homing.transition(States.running.homing, States.running.picking),
    Triggers.finished_picking.transition(States.running.picking, States.running.placing),
    Triggers.finished_placing.transition(States.running.placing, States.running.picking),
    Triggers.cycle_complete.transition(States.running.placing, "ready"),
    Triggers.stop.transition(States.running.homing, "ready"),
    Triggers.stop.transition(States.running.picking, "ready"),
    Triggers.stop.transition(States.running.placing, "ready"),
]


@dataclass
class PalletizerContext:
    """Shared context for state machine operations."""
    rows: int = 2
    cols: int = 2
    box_size_mm: tuple[float, float, float] = (100.0, 100.0, 50.0)
    pallet_origin_mm: tuple[float, float, float] = (400.0, -200.0, 150.0)
    spacing_mm: float = 10.0
    current_box_index: int = 0
    total_boxes: int = 0
    place_positions: list[tuple[float, float, float]] = field(default_factory=list)
    error_message: str = ""
    # Height of the pick surface in mm (robot base frame).
    # Camera gives us X/Y; the actual pick Z is this known table height.
    # Default 100mm above robot base — a safe reachable height for URSim.
    pick_surface_height_mm: float = 100.0
    # Detections in camera frame loaded from JSON or supplied via API
    detections: list[dict] = field(default_factory=list)
    # Transformed pick positions in robot frame (meters)
    pick_positions_m: list[list[float]] = field(default_factory=list)
    # Yaw angles from detections (radians)
    pick_yaws_rad: list[float] = field(default_factory=list)


class PalletizerStateMachine(StateMachine):
    """
    State machine for palletizing operations.
    
    Usage:
        machine = PalletizerStateMachine()
        machine.trigger('start')  # Transitions to HOMING
    """
    
    def __init__(self, robot_connection: Optional[RobotConnection] = None):
        super().__init__(
            states=States,
            transitions=TRANSITIONS,
            enable_last_state_recovery=False,
        )
        self.context = PalletizerContext()
        # Motion controller (may be None if no robot is connected)
        self._motion: Optional[MotionController] = None
        if robot_connection is not None:
            self._motion = MotionController(robot_connection)
        # Lock that is held for the duration of a running cycle.
        # Prevents two threads from calling begin() concurrently.
        self._run_lock = threading.Lock()

    def set_robot_connection(self, connection: RobotConnection) -> None:
        """Attach or replace the robot connection (and motion controller)."""
        self._motion = MotionController(connection)
    
    @property
    def current_state(self) -> PalletizerState:
        """Get current state. Note: library uses format 'Running_homing' not 'running.homing'."""
        state_str = self.state
        mapping = {
            "ready": PalletizerState.IDLE,
            "fault": PalletizerState.FAULT,
            "Running_homing": PalletizerState.HOMING,
            "Running_picking": PalletizerState.PICKING,
            "Running_placing": PalletizerState.PLACING,
        }
        return mapping.get(state_str, PalletizerState.IDLE)
    
    @property
    def progress(self) -> dict:
        """Get current progress: state, current_box, total_boxes, error."""
        return {
            "state": self.current_state.name,
            "current_box": self.context.current_box_index,
            "total_boxes": self.context.total_boxes,
            "error": self.context.error_message if self.context.error_message else None,
        }
    
    def configure(
        self,
        rows: int,
        cols: int,
        box_size_mm: tuple[float, float, float],
        pallet_origin_mm: tuple[float, float, float],
        spacing_mm: float = 10.0,
    ) -> bool:
        """Configure palletizing parameters. Only valid in IDLE state."""
        if self.current_state != PalletizerState.IDLE:
            return False
        
        self.context.rows = rows
        self.context.cols = cols
        self.context.box_size_mm = box_size_mm
        self.context.pallet_origin_mm = pallet_origin_mm
        self.context.spacing_mm = spacing_mm
        self.context.total_boxes = rows * cols
        self.context.current_box_index = 0
        # Pre-compute grid place positions
        self.context.place_positions = calculate_place_positions(
            rows, cols, box_size_mm, pallet_origin_mm, spacing_mm,
        )
        return True

    def load_detections(self, path: str | None = None) -> int:
        """
        Load detections from JSON file and transform to robot frame.

        Args:
            path: Optional path to the JSON file.  Defaults to
                  ``<project>/data/camera_detections.json`` resolved
                  relative to this source file so it works regardless
                  of the process working directory.

        Returns:
            Number of detections loaded.
        """
        if path is None:
            path = str(_THIS_DIR.parent / "data" / "camera_detections.json")

        with open(path, "r") as f:
            data = json.load(f)

        self.context.detections = data.get("detections", [])
        self._transform_detections()
        return len(self.context.detections)

    def add_detection(self, x_mm: float, y_mm: float, z_mm: float, yaw_deg: float = 0.0) -> None:
        """Add a single detection (camera frame) and transform it."""
        self.context.detections.append({
            "x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm, "yaw_deg": yaw_deg,
        })
        self._transform_detections()

    def _transform_detections(self) -> None:
        """Transform all detections from camera frame to robot frame (meters).

        The full 3D transform is always applied.  However, when a detection
        reports z_mm == 0 (planar / 2D detection without depth), the
        transformed Z would land at camera-mounting height (~800 mm) which
        is unreachable.  In that case we fall back to the configurable
        ``pick_surface_height_mm`` so the robot targets the known table
        surface.  When z_mm is non-zero (a real 3D measurement), the
        fully-transformed Z is used as-is.
        """
        self.context.pick_positions_m = []
        self.context.pick_yaws_rad = []
        fallback_z_m = self.context.pick_surface_height_mm / 1000.0

        for det in self.context.detections:
            cam_pt = np.array([det["x_mm"], det["y_mm"], det["z_mm"]])
            robot_pt_mm = camera_to_robot(cam_pt)

            # Use the full 3D transformed Z when the detection has depth;
            # fall back to the known pick-surface height for planar (z==0)
            # detections where the camera provides only X/Y.
            if abs(det["z_mm"]) > 1e-6:
                z_m = robot_pt_mm[2] / 1000.0
            else:
                z_m = fallback_z_m

            robot_pt_m = [robot_pt_mm[0] / 1000.0,
                          robot_pt_mm[1] / 1000.0,
                          z_m]
            self.context.pick_positions_m.append(robot_pt_m)
            self.context.pick_yaws_rad.append(np.deg2rad(det.get("yaw_deg", 0.0)))

            print(f"[TRANSFORM] cam({det['x_mm']:.1f}, {det['y_mm']:.1f}, {det['z_mm']:.1f}) "
                  f"→ robot({robot_pt_m[0]:.4f}, {robot_pt_m[1]:.4f}, {robot_pt_m[2]:.4f}) m")

    def begin(self) -> bool:
        """Start the palletizing sequence."""
        if self.current_state != PalletizerState.IDLE:
            return False

        # Ensure we have detections and place positions ready
        if not self.context.pick_positions_m:
            self.load_detections()

        if not self.context.place_positions:
            self.configure(
                self.context.rows,
                self.context.cols,
                self.context.box_size_mm,
                self.context.pallet_origin_mm,
                self.context.spacing_mm,
            )

        self.context.total_boxes = min(
            len(self.context.pick_positions_m),
            len(self.context.place_positions),
        )
        self.context.current_box_index = 0

        try:
            self.trigger("start")
            return True
        except Exception:
            return False
    
    # =========================================================================
    # Stop / Reset policy
    # =========================================================================
    #
    # stop()  — graceful halt from any running state → IDLE.
    #           The background thread finishes its current move, sees the
    #           state-guard check, then exits cleanly.  No fault is raised.
    #           The machine is immediately ready for a new /start.
    #
    # reset() — recover from FAULT only.  A fault means something unexpected
    #           happened (bad position, motion failure, etc.) and the operator
    #           must explicitly acknowledge it before restarting.
    # =========================================================================

    def begin_async(self) -> bool:
        """
        Start the palletizing cycle in a daemon background thread.

        Uses a non-blocking lock acquisition so that a second call while the
        cycle is already running returns False immediately (no double threads).
        The lock is released inside a ``finally`` block so it is always freed
        even if ``begin()`` raises.

        Returns:
            True  – cycle thread was started successfully.
            False – a cycle is already running (lock is held).
        """
        acquired = self._run_lock.acquire(blocking=False)
        if not acquired:
            return False

        def _run() -> None:
            try:
                self.begin()
            finally:
                self._run_lock.release()

        thread = threading.Thread(target=_run, daemon=True, name="palletizer-cycle")
        thread.start()
        return True

    def stop(self) -> bool:
        """Stop the palletizing sequence and return to IDLE."""
        if self.current_state == PalletizerState.IDLE:
            return True
        try:
            self.trigger("stop")
            return True
        except Exception:
            return False
    
    def reset(self) -> bool:
        """Reset from FAULT state to IDLE and move robot to home."""
        try:
            self.trigger(BaseTriggers.RESET.value)
            self.context.error_message = ""
            # Move robot to safe home position on recovery
            if self._motion:
                try:
                    self._motion.move_to_home()
                except Exception as exc:
                    print(f"[WARN] Homing after reset failed: {exc}")
            return True
        except Exception:
            return False
    
    def fault(self, message: str) -> bool:
        """Transition to FAULT state with an error message."""
        self.context.error_message = message
        try:
            self.trigger(BaseTriggers.TO_FAULT.value)
            return True
        except Exception:
            return False

    # =========================================================================
    # State Entry Callbacks – actual business logic
    # =========================================================================
    
    @on_enter_state(States.running.homing)
    def on_enter_homing(self, _):
        """Move robot to home position, then advance to PICKING."""
        print("[STATE] Entering HOMING")
        try:
            if self._motion:
                self._motion.move_to_home()

            # Guard: stop() may have transitioned us to IDLE while the
            # blocking home move was executing.  Only advance if still HOMING.
            if self.current_state != PalletizerState.HOMING:
                print("[STATE] Homing interrupted by stop – not firing finished_homing")
                return

            self.trigger("finished_homing")
        except Exception as exc:
            self.fault(f"Homing failed: {exc}")
    
    @on_enter_state(States.running.picking)
    def on_enter_picking(self, _):
        """Pick the next box using the transformed camera detection."""
        print(f"[STATE] Entering PICKING (box {self.context.current_box_index})")
        try:
            idx = self.context.current_box_index
            if idx >= len(self.context.pick_positions_m):
                self.fault("No more pick positions available")
                return

            pick_pos = self.context.pick_positions_m[idx]
            yaw = self.context.pick_yaws_rad[idx]

            # Build orientation: default tool-down + yaw from vision
            orientation = None
            if self._motion:
                base_orient = self._motion.get_default_orientation()
                # Properly compose yaw with the rotation vector
                orientation = self._motion.compose_orientation_with_yaw(
                    base_orient, yaw,
                )

            if self._motion:
                success = self._motion.move_to_pick(pick_pos, orientation)
                if not success:
                    self.fault(f"Pick motion failed at box {idx}")
                    return

            # Guard: stop() may have transitioned us to IDLE while the
            # blocking motion was executing.  Only advance if still PICKING.
            if self.current_state != PalletizerState.PICKING:
                print("[STATE] Picking interrupted by stop – not firing finished_picking")
                return

            self.trigger("finished_picking")
        except Exception as exc:
            self.fault(f"Picking failed: {exc}")
    
    @on_enter_state(States.running.placing)
    def on_enter_placing(self, _):
        """Place the box at the next grid position and advance."""
        idx = self.context.current_box_index
        print(f"[STATE] Entering PLACING (box {idx})")
        try:
            if idx >= len(self.context.place_positions):
                self.fault("No more place positions available")
                return

            place_mm = self.context.place_positions[idx]
            # Convert mm → metres
            place_m = [place_mm[0] / 1000.0, place_mm[1] / 1000.0, place_mm[2] / 1000.0]

            if self._motion:
                success = self._motion.move_to_place(place_m)
                if not success:
                    self.fault(f"Place motion failed at box {idx}")
                    return

            # Guard: stop() may have transitioned us to IDLE while the
            # blocking motion was executing.  Only advance if still PLACING.
            if self.current_state != PalletizerState.PLACING:
                print("[STATE] Placing interrupted by stop – not firing finished_placing")
                return

            self.context.current_box_index += 1

            if self.context.current_box_index >= self.context.total_boxes:
                print("[STATE] All boxes placed – cycle complete")
                self.trigger("cycle_complete")
            else:
                self.trigger("finished_placing")
        except Exception as exc:
            self.fault(f"Placing failed: {exc}")
    
    @on_state_change
    def on_any_state_change(self, old_state: str, new_state: str, trigger: str):
        """Called on every state transition. Useful for logging."""
        print(f"[TRANSITION] {old_state} --({trigger})--> {new_state}")
