"""Tests for REST API routes. Run with: pytest tests/test_routes.py -v"""

import pytest
from fastapi.testclient import TestClient

# Override the machine singleton BEFORE importing the app
import api.routes as routes_module
from palletizer.state_machine import PalletizerStateMachine, PalletizerState


@pytest.fixture(autouse=True)
def reset_machine():
    """Give every test a fresh PalletizerStateMachine (no robot connection)."""
    routes_module._machine = PalletizerStateMachine(robot_connection=None)
    yield
    routes_module._machine = None


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


# ── Status ──────────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_idle(self, client):
        resp = client.get("/palletizer/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "IDLE"
        assert data["current_box"] == 0

    def test_status_after_configure(self, client):
        client.post("/palletizer/configure", json={
            "rows": 2, "cols": 3,
            "box_width_mm": 100, "box_depth_mm": 100, "box_height_mm": 50,
            "pallet_origin_x_mm": 400, "pallet_origin_y_mm": -200, "pallet_origin_z_mm": 100,
        })
        resp = client.get("/palletizer/status")
        assert resp.json()["total_boxes"] == 6


# ── Configure ───────────────────────────────────────────────────────────────

class TestConfigure:
    def test_configure_success(self, client):
        resp = client.post("/palletizer/configure", json={
            "rows": 2, "cols": 2,
            "box_width_mm": 100, "box_depth_mm": 100, "box_height_mm": 50,
            "pallet_origin_x_mm": 400, "pallet_origin_y_mm": -200, "pallet_origin_z_mm": 100,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["grid_size"] == "2x2"

    def test_configure_validation_rows(self, client):
        resp = client.post("/palletizer/configure", json={
            "rows": 0, "cols": 2,
            "box_width_mm": 100, "box_depth_mm": 100, "box_height_mm": 50,
            "pallet_origin_x_mm": 400, "pallet_origin_y_mm": -200, "pallet_origin_z_mm": 100,
        })
        assert resp.status_code == 422  # Pydantic validation error


# ── Stop ────────────────────────────────────────────────────────────────────

class TestStop:
    def test_stop_while_idle(self, client):
        resp = client.post("/palletizer/stop")
        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ── Reset ───────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_not_in_fault(self, client):
        resp = client.post("/palletizer/reset")
        assert resp.status_code == 409  # Not in FAULT


# ── Vision Detect ───────────────────────────────────────────────────────────

class TestVisionDetect:
    def test_add_detection(self, client):
        resp = client.post("/palletizer/vision/detect", json={
            "x_mm": 50.0, "y_mm": -30.0, "z_mm": 0.0, "yaw_deg": 15.0,
        })
        assert resp.status_code == 200
        assert "1" in resp.json()["message"]  # "Total detections: 1"

    def test_add_multiple_detections(self, client):
        for _ in range(3):
            client.post("/palletizer/vision/detect", json={
                "x_mm": 10.0, "y_mm": 20.0, "z_mm": 0.0,
            })
        resp = client.post("/palletizer/vision/detect", json={
            "x_mm": 10.0, "y_mm": 20.0, "z_mm": 0.0,
        })
        assert "4" in resp.json()["message"]


# ── Debug Transform ─────────────────────────────────────────────────────────

class TestDebugTransform:
    def test_transform_returns_both_frames(self, client):
        resp = client.post("/palletizer/debug/transform", json={
            "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "camera_frame_mm" in data
        assert "robot_frame_mm" in data
        assert "robot_frame_m" in data

    def test_transform_origin_equals_camera_position(self, client):
        resp = client.post("/palletizer/debug/transform", json={
            "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0,
        })
        robot = resp.json()["robot_frame_mm"]
        assert abs(robot["x"] - 500.0) < 1e-3
        assert abs(robot["y"] - 300.0) < 1e-3
        assert abs(robot["z"] - 800.0) < 1e-3


# ── Debug Positions ─────────────────────────────────────────────────────────

class TestDebugPositions:
    def test_positions_after_configure(self, client):
        client.post("/palletizer/configure", json={
            "rows": 2, "cols": 2,
            "box_width_mm": 100, "box_depth_mm": 100, "box_height_mm": 50,
            "pallet_origin_x_mm": 400, "pallet_origin_y_mm": -200, "pallet_origin_z_mm": 100,
        })
        resp = client.get("/palletizer/debug/positions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_place"] == 4
