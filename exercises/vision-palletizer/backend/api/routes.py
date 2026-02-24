"""
Palletizer API Routes
====================

FastAPI routes for palletizer control.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

import numpy as np

from transforms.coordinate import camera_to_robot
from palletizer.state_machine import PalletizerStateMachine, PalletizerState

router = APIRouter()

# ============================================================================
# Singleton state machine – created once, wired to the robot on first use
# ============================================================================
_machine: Optional[PalletizerStateMachine] = None


def get_machine() -> PalletizerStateMachine:
    """Return or lazily create the singleton PalletizerStateMachine."""
    global _machine
    if _machine is None:
        # Import here to avoid circular imports at module level
        from main import get_robot_connection
        conn = get_robot_connection()
        _machine = PalletizerStateMachine(robot_connection=conn)
    return _machine


# ============================================================================
# Request/Response Models
# ============================================================================

class PalletConfig(BaseModel):
    """Configuration for palletizing operation."""
    
    rows: int = Field(..., ge=1, le=10, description="Number of rows in the grid")
    cols: int = Field(..., ge=1, le=10, description="Number of columns in the grid")
    box_width_mm: float = Field(..., gt=0, description="Box width in mm (X direction)")
    box_depth_mm: float = Field(..., gt=0, description="Box depth in mm (Y direction)")
    box_height_mm: float = Field(..., gt=0, description="Box height in mm (Z direction)")
    pallet_origin_x_mm: float = Field(..., description="Pallet origin X in mm")
    pallet_origin_y_mm: float = Field(..., description="Pallet origin Y in mm")
    pallet_origin_z_mm: float = Field(..., description="Pallet origin Z in mm")
    spacing_mm: float = Field(10.0, ge=0, description="Gap between boxes in mm")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rows": 2,
                "cols": 2,
                "box_width_mm": 100.0,
                "box_depth_mm": 100.0,
                "box_height_mm": 50.0,
                "pallet_origin_x_mm": 400.0,
                "pallet_origin_y_mm": -200.0,
                "pallet_origin_z_mm": 100.0,
                "spacing_mm": 10.0,
            }
        }


class VisionDetection(BaseModel):
    """Simulated vision detection of a box."""
    
    x_mm: float = Field(..., description="Box X position in camera frame (mm)")
    y_mm: float = Field(..., description="Box Y position in camera frame (mm)")
    z_mm: float = Field(..., description="Box Z position in camera frame (mm)")
    yaw_deg: Optional[float] = Field(0.0, description="Box rotation about Z (degrees)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "x_mm": 50.0,
                "y_mm": -30.0,
                "z_mm": 0.0,
                "yaw_deg": 15.0,
            }
        }


class StatusResponse(BaseModel):
    """Palletizer status response."""
    
    state: str = Field(..., description="Current state machine state")
    current_box: int = Field(..., description="Current box index (0-based)")
    total_boxes: int = Field(..., description="Total boxes to palletize")
    error: Optional[str] = Field(None, description="Error message if in FAULT state")


class ConfigResponse(BaseModel):
    """Configuration response."""
    
    success: bool
    message: str
    grid_size: Optional[str] = None


class CommandResponse(BaseModel):
    """Generic command response."""
    
    success: bool
    message: str


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/configure", response_model=ConfigResponse)
async def configure_palletizer(config: PalletConfig):
    """
    Configure the palletizing operation.
    
    Sets up the grid dimensions, box size, and pallet origin.
    Can only be called when the palletizer is in IDLE state.
    """
    machine = get_machine()

    if machine.current_state != PalletizerState.IDLE:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot configure while in {machine.current_state.name} state. Stop first.",
        )

    box_size = (config.box_width_mm, config.box_depth_mm, config.box_height_mm)
    pallet_origin = (config.pallet_origin_x_mm, config.pallet_origin_y_mm, config.pallet_origin_z_mm)

    success = machine.configure(
        rows=config.rows,
        cols=config.cols,
        box_size_mm=box_size,
        pallet_origin_mm=pallet_origin,
        spacing_mm=config.spacing_mm,
    )

    if not success:
        raise HTTPException(status_code=400, detail="Configuration failed")

    return ConfigResponse(
        success=True,
        message=f"Configured {config.rows}x{config.cols} grid ({config.rows * config.cols} boxes)",
        grid_size=f"{config.rows}x{config.cols}",
    )


@router.post("/start", response_model=CommandResponse)
async def start_palletizer():
    """
    Start the palletizing sequence.

    Returns immediately. The pick-and-place cycle runs in a background thread
    so that /stop and /reset remain responsive during execution.
    Poll /status to track progress.
    """
    machine = get_machine()

    if machine.current_state != PalletizerState.IDLE:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot start while in {machine.current_state.name} state",
        )

    # Validate preconditions here (before the thread starts) so we can return
    # a meaningful error code instead of silently failing in the background.
    if not machine.context.pick_positions_m:
        # begin() would auto-load, but we load eagerly so errors surface now.
        machine.load_detections()

    if not machine.context.place_positions:
        machine.configure(
            machine.context.rows,
            machine.context.cols,
            machine.context.box_size_mm,
            machine.context.pallet_origin_mm,
            machine.context.spacing_mm,
        )

    # Delegate thread creation + lock management to begin_async().
    # Returns False (without starting a thread) if a cycle is already running.
    if not machine.begin_async():
        raise HTTPException(status_code=409, detail="Palletizer is already running")

    return CommandResponse(success=True, message="Palletizing started (running in background)")


@router.post("/stop", response_model=CommandResponse)
async def stop_palletizer():
    """
    Stop the palletizing sequence.
    
    Gracefully stops the operation and returns to IDLE state.
    """
    machine = get_machine()
    success = machine.stop()

    if not success:
        raise HTTPException(status_code=500, detail="Failed to stop palletizer")

    return CommandResponse(success=True, message="Palletizer stopped")


@router.post("/reset", response_model=CommandResponse)
async def reset_palletizer():
    """
    Reset from FAULT state.
    
    Clears the fault and returns to IDLE state.
    """
    machine = get_machine()

    if machine.current_state != PalletizerState.FAULT:
        raise HTTPException(status_code=409, detail="Not in FAULT state – nothing to reset")

    success = machine.reset()
    if not success:
        raise HTTPException(status_code=500, detail="Reset failed")

    return CommandResponse(success=True, message="Palletizer reset to IDLE")


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get current palletizer status.
    
    Returns the current state, progress, and any error messages.
    """
    machine = get_machine()
    prog = machine.progress
    return StatusResponse(**prog)


@router.post("/vision/detect", response_model=CommandResponse)
async def simulate_vision_detection(detection: VisionDetection):
    """
    Simulate a vision detection event.
    
    In a real system, this would come from the vision system.
    For this exercise, use this endpoint to simulate box detections.
    
    The coordinates are in the camera frame and must be transformed
    to the robot frame before use.
    """
    machine = get_machine()
    machine.add_detection(
        x_mm=detection.x_mm,
        y_mm=detection.y_mm,
        z_mm=detection.z_mm,
        yaw_deg=detection.yaw_deg or 0.0,
    )
    return CommandResponse(
        success=True,
        message=f"Detection added. Total detections: {len(machine.context.detections)}",
    )


# ============================================================================
# Helper/Debug Endpoints
# ============================================================================

@router.get("/debug/positions")
async def get_calculated_positions():
    """
    Debug endpoint: Get all calculated place positions.
    
    Useful for verifying grid calculations without running the full sequence.
    """
    machine = get_machine()
    return {
        "place_positions_mm": machine.context.place_positions,
        "pick_positions_m": machine.context.pick_positions_m,
        "total_place": len(machine.context.place_positions),
        "total_pick": len(machine.context.pick_positions_m),
    }


@router.post("/debug/transform")
async def test_transform(detection: VisionDetection):
    """
    Debug endpoint: Test coordinate transformation.
    
    Transforms the input coordinates and returns both camera and robot frame values.
    Useful for verifying transformation math.
    """
    cam_pt = np.array([detection.x_mm, detection.y_mm, detection.z_mm])
    robot_pt = camera_to_robot(cam_pt)
    return {
        "camera_frame_mm": {"x": cam_pt[0], "y": cam_pt[1], "z": cam_pt[2]},
        "robot_frame_mm": {"x": float(robot_pt[0]), "y": float(robot_pt[1]), "z": float(robot_pt[2])},
        "robot_frame_m": {
            "x": float(robot_pt[0] / 1000),
            "y": float(robot_pt[1] / 1000),
            "z": float(robot_pt[2] / 1000),
        },
    }
