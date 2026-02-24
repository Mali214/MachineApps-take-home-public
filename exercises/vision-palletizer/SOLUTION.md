# Solution: Vision Palletizer

## Approach

### Architecture Overview

The solution follows a clean separation of concerns across five modules:

```
transforms/coordinate.py    — Camera ↔ Robot frame math
palletizer/grid.py          — N×M grid place-position calculator
robot/motion.py             — UR5e motion commands (pick/place sequences)
palletizer/state_machine.py — Operation lifecycle (vention-state-machine)
api/routes.py               — REST API (FastAPI)
```

Each module is independently testable, and dependencies are kept minimal (the API lazily wires the state machine to the robot connection).

---

### 1. Coordinate Transformations (35%)

**Approach:**  
Built a standard 4×4 homogeneous transformation pipeline:

1. **Rotation matrix** — `R = Rz(yaw) · Ry(pitch) · Rx(roll)` using intrinsic Z→Y→X convention as specified.
2. **Homogeneous transform** — Combined the 3×3 rotation with the camera mounting translation `[500, 300, 800] mm` into a single 4×4 matrix.
3. **Pre-computed** both `T_camera_to_robot` and its inverse `T_robot_to_camera` at import time for efficiency.
4. Transformations are applied via homogeneous coordinates: append `1` to `[x, y, z]`, multiply by `T`, extract first 3 elements.

**Key decision:** Detections are transformed using a full 3D camera-to-base homogeneous transform. For planar detections where `z_mm=0` (no depth), the pipeline falls back to a configurable pick surface height; when `z_mm` is non-zero, the transformed Z is used directly.

---

### 2. Working Palletizing Sequence (35%)

**Motion strategy:**

- **Large moves** (home → approach, approach → next target): Joint-space via `getInverseKinematics(pose, q_near)` + `moveJ(joints)`. The IK is always seeded with `HOME_JOINTS` to keep the arm in a consistent, non-flipping posture.
- **Short moves** (approach ↔ pick/place, ~100mm vertical): Cartesian `moveL` for precision.
- **Approach height**: 100mm above every pick/place target (exceeds the 50mm minimum).

**Pick sequence:** approach (joint move) → descend (linear) → grip (simulated) → retract (linear)  
**Place sequence:** approach (joint move) → descend (linear) → release (simulated) → retract (linear)

**Gripper:** Simulated with `time.sleep(0.5)` since URSim has no physical I/O. The digital output calls were removed after testing showed `setStandardDigitalOut` is unavailable in the ur_rtde version provided.

**Home position:** `[0°, -90°, -90°, -90°, 90°, 0°]` — a stable home posture with the TCP pointing straight down.

---

### 3. State Machine

Implemented using `vention-state-machine`:

```
IDLE → HOMING → PICKING → PLACING → PICKING → ... → IDLE
         ↓          ↓          ↓
       FAULT      FAULT      FAULT
                    ↓
              (reset → IDLE + home)
```

- `on_enter_homing`: moves to home, then triggers `finished_homing`
- `on_enter_picking`: transforms detection → pick position, adjusts gripper orientation by detected yaw, executes pick sequence
- `on_enter_placing`: converts grid position mm→m, executes place sequence, increments box index
- `stop` trigger: returns to IDLE from any running state (graceful halt — no fault raised)
- `fault()`: transitions to FAULT with error message; `reset()` recovers to IDLE and re-homes the robot


---

### 4. Concurrency Design

Robot motion commands block the calling thread. Running `begin()` directly inside a FastAPI route would block the async event loop for the entire cycle, making `/stop`, `/reset`, and `/status` unresponsive.

**Solution:** `/start` launches the palletizing cycle in a daemon background thread using a non-blocking `threading.Lock`, ensuring the FastAPI event loop remains responsive.  
The lock prevents concurrent cycles.  
Each state callback verifies the current state after blocking motion completes so that a `/stop` mid-move exits cleanly without triggering unintended FAULT transitions.

---

### 5. Grid Calculator

Row-by-row filling: outer loop over rows, inner loop over columns.  
Each position offset = `origin + index × (box_dimension + spacing)`.  
X direction = columns (width), Y direction = rows (depth), Z = constant (pallet surface).

---

### 6. REST API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/palletizer/configure` | POST | Set grid dimensions, box size, pallet origin, spacing |
| `/palletizer/start` | POST | Begin pick-and-place cycle (returns immediately, runs in background) |
| `/palletizer/stop` | POST | Graceful halt → IDLE  |
| `/palletizer/reset` | POST | Recover from FAULT → IDLE (with homing) |
| `/palletizer/status` | GET | Current state, progress, errors |
| `/palletizer/vision/detect` | POST | Add a simulated vision detection |
| `/palletizer/debug/positions` | GET | View all calculated pick/place positions |
| `/palletizer/debug/transform` | POST | Test a single coordinate transformation |

---

## Assumptions

1. **Pick surface height fallback for planar detections** — The full 3D camera-to-robot transform is always applied. When a detection has a non-zero `z_mm` (real depth measurement), the transformed Z is used as-is. When `z_mm == 0` (planar / 2D detection without depth, as in the provided test data), the transformed Z would land at camera mounting height (~800mm), which is unreachable. In that case, Z falls back to a configurable `pick_surface_height_mm` (default 100mm). This respects the spec's "transform to robot frame" requirement while handling the reality of 2D detections.

2. **Tool orientation is straight down** — The default TCP orientation is `[0, π, 0]` (axis-angle rotation vector). When a detected box has a non-zero yaw, the gripper orientation is composed correctly using Rodrigues' formula: the base rotation vector is converted to a rotation matrix, pre-multiplied by `Rz(yaw)`, then converted back to a rotation vector. This avoids the common mistake of naively adding yaw to one element of the rotation vector.

3. **No physical gripper** — URSim doesn't support digital I/O, so gripper actuation is simulated with a 0.5s dwell time. The architecture supports swapping in real gripper commands.

4. **Detections loaded from JSON** — On startup, `camera_detections.json` is loaded automatically if no detections have been added via the API. The 4 detections define the pick positions.

5. **Grid and detections match** — `total_boxes = min(detections, grid_positions)`. If there are fewer detections than grid slots, only the available boxes are placed.

6. **UR5e workspace** — Positions are validated against an outer reach limit (850mm), an inner dead-zone limit (150mm — the arm cannot reach directly under its own base), and Z bounds [-10mm, 1200mm] before motion commands are sent. Out-of-range positions raise a `ValueError` that transitions the machine to FAULT.

7. **Rotation convention** — Intrinsic Z→Y→X as specified in the README. This means `R = Rz(yaw) · Ry(pitch) · Rx(roll)`.

8. **Single-machine concurrency** — One robot, one singleton state machine, one run lock. No concurrent cycles; no distributed coordination needed.

9. **Stop granularity** — Stop takes effect between motion commands, not mid-move, to avoid leaving the robot in an unknown pose.

10. **Units** — Camera/grid calculations use millimetres; robot commands use metres. Conversion occurs at the state machine boundary.

---

## Testing

59 tests across 5 test files:

| File | Tests | Coverage |
|------|-------|----------|
| `test_coordinate.py` | 18 | Rotation matrix properties, homogeneous transforms, camera↔robot round-trips, known input/output pairs |
| `test_grid.py` | 9 | Position count, row-by-row ordering, spacing, single cell, large grid |
| `test_motion.py` | 9 | Mock mode: home, pick sequence, place sequence, gripper state, approach height verification |
| `test_routes.py` | 11 | All API endpoints: configure, start, stop, reset, status, vision detect, debug |
| `test_state_machine.py` | 12 | State transitions, configuration, fault/reset, stop from any state |

Run all tests:
```bash
docker exec palletizer-backend python -m pytest tests/ -v
```

> `httpx` is required by FastAPI's `TestClient` for `test_routes.py`. It is listed in `requirements.txt` and installed automatically when the container is built.


