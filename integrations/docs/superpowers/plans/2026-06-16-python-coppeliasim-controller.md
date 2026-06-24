# Python CoppeliaSim Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python Remote API controller that replaces the scene robot Lua scripts while keeping `zeroBufferBelt.lua` as the only in-scene script.

**Architecture:** Python owns simulation stepping, belt synchronization, robot IK/planning/following, pick/place sequencing, and logging. CoppeliaSim remains the source of truth for scene geometry, physics, collision checks, IK, and OMPL through the ZMQ Remote API.

**Tech Stack:** Python 3.11, `coppeliasim_zmqremoteapi_client`, CoppeliaSim `sim`, `simIK`, `simOMPL`, `pytest`, stdlib `dataclasses`, `argparse`, `logging`.

---

## File Structure

- Create: `python_controller/__init__.py`
  - Package marker.
- Create: `python_controller/remote.py`
  - Remote API connection, simulation start/stop, stepping, object lookup.
- Create: `python_controller/belt.py`
  - Reads `trackedItemsData` and controls `robotPicking`.
- Create: `python_controller/robot.py`
  - Robot handles, joint config helpers, IK setup/solve, collision collection, joint/Cartesian planning and following.
- Create: `python_controller/pick_place.py`
  - Pick/place poses and command sequence.
- Create: `python_controller/main.py`
  - CLI entry point and main loop.
- Create: `tests/test_python_controller_static.py`
  - Static contract tests that can run without CoppeliaSim.
- Create: `tests/test_python_controller_units.py`
  - Unit tests with fake remotes for belt parsing and interpolation helpers.
- Keep: `zeroBufferBelt.lua`
  - In-scene belt script. No changes planned.
- Remove from scene manually after implementation: `external_api.lua`, `rail_ur10_motion_simplify.lua`, `ur10_joint_motion.lua`, `ur10_linear_motion.lua`, `loading_test.lua`.

---

### Task 1: Package Skeleton And Static Contract Tests

**Files:**
- Create: `python_controller/__init__.py`
- Create: `python_controller/remote.py`
- Create: `python_controller/belt.py`
- Create: `python_controller/robot.py`
- Create: `python_controller/pick_place.py`
- Create: `python_controller/main.py`
- Create: `tests/test_python_controller_static.py`

- [ ] **Step 1: Write the failing static test**

```python
# tests/test_python_controller_static.py
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def test_controller_modules_exist_and_use_remote_api_boundaries():
    assert (ROOT / "python_controller" / "__init__.py").exists()
    assert "RemoteAPIClient" in read("python_controller/remote.py")
    assert "client.setStepping(True)" in read("python_controller/remote.py")
    assert "client.step()" in read("python_controller/remote.py")
    assert "trackedItemsData" in read("python_controller/belt.py")
    assert "robotPicking" in read("python_controller/belt.py")
    assert "require(\"simIK\")" in read("python_controller/robot.py")
    assert "require(\"simOMPL\")" in read("python_controller/robot.py")
    assert "def pick(" in read("python_controller/pick_place.py")
    assert "def place(" in read("python_controller/pick_place.py")
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py -q
```

Expected: FAIL because `python_controller` files do not exist.

- [ ] **Step 3: Add minimal package files**

Create `python_controller/__init__.py`:

```python
"""Python-side CoppeliaSim controller for belt-driven pick/place scenes."""
```

Create `python_controller/remote.py`:

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


@dataclass
class RemoteContext:
    client: object
    sim: object
    simIK: object
    simOMPL: object

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self.client.step()


def connect(port: int = 23000) -> RemoteContext:
    client = RemoteAPIClient(port=port)
    sim = client.require("sim")
    simIK = client.require("simIK")
    simOMPL = client.require("simOMPL")
    client.setStepping(True)
    return RemoteContext(client=client, sim=sim, simIK=simIK, simOMPL=simOMPL)


def first_object(sim: object, aliases: Iterable[str]) -> int:
    tried = []
    for alias in aliases:
        tried.append(alias)
        try:
            handle = sim.getObject(alias, {"noError": True})
        except Exception:
            handle = -1
        if handle is not None and handle >= 0:
            return handle
    raise RuntimeError("Could not find object. Tried: " + ", ".join(tried))


def ensure_started(ctx: RemoteContext, scene_path: str | None = None) -> None:
    sim = ctx.sim
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        start = time.time()
        while sim.getSimulationState() != sim.simulation_stopped:
            if time.time() - start > 10:
                raise TimeoutError("Timed out waiting for simulation to stop")
            time.sleep(0.05)
    if scene_path:
        sim.loadScene(scene_path)
    sim.startSimulation()
```

Create `python_controller/belt.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrackedItem:
    handle: int
    size: list[float]
    pose: list[float]
    raw: dict[str, Any]


class BeltMonitor:
    def __init__(self, sim: object):
        self.sim = sim

    def set_robot_picking(self, enabled: bool) -> None:
        self.sim.setInt32Signal("robotPicking", 1 if enabled else 0)

    def read_full_items(self) -> list[TrackedItem]:
        packed = self.sim.getStringSignal("trackedItemsData")
        if not packed:
            return []
        data = self.sim.unpackTable(packed)
        if not data or data.get("full") is not True:
            return []
        items = []
        for item in data.get("items", []):
            handle = item.get("handle")
            if handle is None:
                continue
            items.append(
                TrackedItem(
                    handle=int(handle),
                    size=list(item.get("size", [])),
                    pose=list(item.get("pose", [])),
                    raw=item,
                )
            )
        return items
```

Create `python_controller/robot.py`:

```python
from __future__ import annotations


class RobotController:
    """Initial robot controller shell. Later tasks add motion behavior."""

    def __init__(self, ctx: object):
        self.ctx = ctx
        self.sim = ctx.sim
        self.simIK = ctx.client.require("simIK")
        self.simOMPL = ctx.client.require("simOMPL")
```

Create `python_controller/pick_place.py`:

```python
from __future__ import annotations


class PickPlaceController:
    def __init__(self, robot: object, belt: object):
        self.robot = robot
        self.belt = belt

    def pick(self, handle: int) -> bool:
        raise NotImplementedError("pick is implemented in a later task")

    def place(self, position: list[float]) -> bool:
        raise NotImplementedError("place is implemented in a later task")
```

Create `python_controller/main.py`:

```python
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--place", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py -q
```

Expected: PASS.

---

### Task 2: Belt Data Parsing And Full-Transition Selection

**Files:**
- Modify: `python_controller/belt.py`
- Modify: `tests/test_python_controller_units.py`

- [ ] **Step 1: Write failing unit tests**

```python
# tests/test_python_controller_units.py
from python_controller.belt import BeltMonitor


class FakeSim:
    def __init__(self, packed=None):
        self.packed = packed
        self.signals = {}

    def getStringSignal(self, name):
        assert name == "trackedItemsData"
        return self.packed

    def unpackTable(self, packed):
        return packed

    def setInt32Signal(self, name, value):
        self.signals[name] = value


def test_belt_monitor_returns_items_only_when_full():
    sim = FakeSim({"full": True, "items": [{"handle": 12, "size": [1, 2, 3], "pose": [0, 0, 0, 0, 0, 0, 1]}]})
    items = BeltMonitor(sim).read_full_items()
    assert [item.handle for item in items] == [12]

    sim.packed = {"full": False, "items": [{"handle": 13}]}
    assert BeltMonitor(sim).read_full_items() == []


def test_belt_monitor_writes_robot_picking_signal():
    sim = FakeSim()
    belt = BeltMonitor(sim)
    belt.set_robot_picking(True)
    belt.set_robot_picking(False)
    assert sim.signals["robotPicking"] == 0
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_units.py -q
```

Expected: PASS if Task 1 implementation already matches. If it fails, failure should identify missing belt behavior.

- [ ] **Step 3: Add full-transition helper**

Modify `python_controller/belt.py`:

```python
class BeltMonitor:
    def __init__(self, sim: object):
        self.sim = sim
        self.was_full = False

    def full_transition_items(self) -> list[TrackedItem]:
        items = self.read_full_items()
        is_full = len(items) > 0
        selected = items if is_full and not self.was_full else []
        self.was_full = is_full
        return selected
```

Add test:

```python
def test_full_transition_items_only_returns_on_rising_edge():
    sim = FakeSim({"full": True, "items": [{"handle": 12}]})
    belt = BeltMonitor(sim)
    assert [item.handle for item in belt.full_transition_items()] == [12]
    assert belt.full_transition_items() == []
    sim.packed = {"full": False, "items": []}
    assert belt.full_transition_items() == []
    sim.packed = {"full": True, "items": [{"handle": 13}]}
    assert [item.handle for item in belt.full_transition_items()] == [13]
```

- [ ] **Step 4: Run tests**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_units.py -q
```

Expected: PASS.

---

### Task 3: Robot Handle Discovery And Config Helpers

**Files:**
- Modify: `python_controller/robot.py`
- Modify: `tests/test_python_controller_static.py`

- [ ] **Step 1: Add failing static assertions**

Append to `tests/test_python_controller_static.py`:

```python
def test_robot_controller_discovers_scene_handles_and_configs():
    text = read("python_controller/robot.py")
    assert "class RobotHandles" in text
    assert "def discover_handles(" in text
    assert "def get_config(" in text
    assert "def set_config(" in text
    assert "def set_target_config(" in text
    assert "sim.getObjectsInTree" in text
    assert "sim.sceneobject_joint" in text
```

- [ ] **Step 2: Run static test and verify failure**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py::test_robot_controller_discovers_scene_handles_and_configs -q
```

Expected: FAIL on missing handle/config methods.

- [ ] **Step 3: Implement handle/config helpers**

Replace `python_controller/robot.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

from .remote import first_object


@dataclass
class RobotHandles:
    ik_base: int
    rail_joint: int
    ur10_base: int
    ik_tip: int
    ik_target: int
    pallet: int
    joints: list[int]


class RobotController:
    def __init__(self, ctx: object):
        self.ctx = ctx
        self.sim = ctx.sim
        self.simIK = ctx.client.require("simIK")
        self.simOMPL = ctx.client.require("simOMPL")
        self.handles = self.discover_handles()

    def discover_handles(self) -> RobotHandles:
        sim = self.sim
        ik_base = first_object(sim, ["/mobile_arm"])
        rail_joint = first_object(sim, ["/mobile_arm/railJoint", "/railJoint", ":/railJoint"])
        ur10_base = first_object(sim, ["/mobile_arm/railJoint/UR10", "/mobile_arm/UR10", "/UR10"])
        ik_tip = first_object(sim, ["/mobile_arm/railJoint/UR10/ikTip", "/mobile_arm/UR10/ikTip", "/ikTip", ":/ikTip"])
        ik_target = first_object(sim, ["/ikTarget", ":/ikTarget"])
        pallet = first_object(sim, ["/pallet", ":/pallet"])
        ur10_joints = sim.getObjectsInTree(ur10_base, sim.sceneobject_joint)
        joints = [rail_joint] + list(ur10_joints)
        return RobotHandles(
            ik_base=ik_base,
            rail_joint=rail_joint,
            ur10_base=ur10_base,
            ik_tip=ik_tip,
            ik_target=ik_target,
            pallet=pallet,
            joints=joints,
        )

    def get_config(self) -> list[float]:
        return [self.sim.getJointPosition(joint) for joint in self.handles.joints]

    def set_config(self, config: list[float]) -> None:
        for joint, value in zip(self.handles.joints, config):
            self.sim.setJointPosition(joint, value)

    def set_target_config(self, config: list[float]) -> None:
        for joint, value in zip(self.handles.joints, config):
            self.sim.setJointTargetPosition(joint, value)
```

- [ ] **Step 4: Run static tests**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py -q
```

Expected: PASS.

---

### Task 4: IK Environment And Collision Collection

**Files:**
- Modify: `python_controller/robot.py`
- Modify: `tests/test_python_controller_static.py`

- [ ] **Step 1: Add failing static assertions**

Append:

```python
def test_robot_controller_sets_up_ik_and_collision_collection():
    text = read("python_controller/robot.py")
    assert "def setup_ik(" in text
    assert "simIK.createEnvironment" in text
    assert "simIK.addElementFromScene" in text
    assert "simIK.constraint_pose" in text
    assert "def solve_ik_to_pose(" in text
    assert "simIK.handleGroup" in text
    assert "def update_robot_collection(" in text
    assert "sim.createCollection" in text
    assert "sim.checkCollision" in text
```

- [ ] **Step 2: Run static test and verify failure**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py::test_robot_controller_sets_up_ik_and_collision_collection -q
```

Expected: FAIL.

- [ ] **Step 3: Implement IK and collision helpers**

Add to `RobotController.__init__` after handles:

```python
        self.robot_collection = None
        self.ik_env = None
        self.ik_group = None
        self.sim_to_ik = None
        self.setup_ik()
        self.update_robot_collection()
```

Add methods:

```python
    def setup_ik(self) -> None:
        simIK = self.simIK
        h = self.handles
        self.ik_env = simIK.createEnvironment()
        self.ik_group = simIK.createGroup(self.ik_env)
        simIK.setGroupCalculation(
            self.ik_env,
            self.ik_group,
            simIK.method_damped_least_squares,
            0.3,
            99,
        )
        _, self.sim_to_ik = simIK.addElementFromScene(
            self.ik_env,
            self.ik_group,
            h.ik_base,
            h.ik_tip,
            h.ik_target,
            simIK.constraint_pose,
        )

    def update_robot_collection(self, attached_object: int | None = None) -> None:
        sim = self.sim
        if self.robot_collection is not None:
            sim.destroyCollection(self.robot_collection)
        self.robot_collection = sim.createCollection()
        sim.addItemToCollection(self.robot_collection, sim.handle_tree, self.handles.ik_base, 0)
        if attached_object is not None and attached_object >= 0:
            sim.addItemToCollection(self.robot_collection, sim.handle_single, attached_object, 0)

    def collides(self, configs: list[list[float]], attached_object: int | None = None) -> bool:
        sim = self.sim
        buffered = self.get_config()
        self.update_robot_collection(attached_object=attached_object)
        try:
            for config in configs:
                self.set_config(config)
                if sim.checkCollision(self.robot_collection, sim.handle_all) > 0:
                    return True
                if sim.checkCollision(self.robot_collection, self.robot_collection) > 0:
                    return True
            return False
        finally:
            self.set_config(buffered)

    def solve_ik_to_pose(self, pose: list[float], attached_object: int | None = None) -> list[float] | None:
        simIK = self.simIK
        h = self.handles
        self.sim.setObjectPose(h.ik_target, pose)
        simIK.syncFromSim(self.ik_env, [self.ik_group])
        simIK.setObjectPose(
            self.ik_env,
            self.sim_to_ik[h.ik_target],
            self.sim.getObjectPose(h.ik_target, h.ik_base),
            self.sim_to_ik[h.ik_base],
        )
        result = simIK.handleGroup(self.ik_env, self.ik_group)
        if result != simIK.result_success:
            return None
        config = [
            simIK.getJointPosition(self.ik_env, self.sim_to_ik[joint])
            for joint in h.joints
        ]
        if self.collides([config], attached_object=attached_object):
            return None
        return config
```

- [ ] **Step 4: Run tests**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py -q
```

Expected: PASS.

---

### Task 5: Joint Path Planning And Following

**Files:**
- Modify: `python_controller/robot.py`
- Modify: `tests/test_python_controller_static.py`

- [ ] **Step 1: Add failing static assertions**

Append:

```python
def test_robot_controller_plans_and_follows_joint_paths():
    text = read("python_controller/robot.py")
    assert "def plan_joint_path(" in text
    assert "simOMPL.createTask" in text
    assert "simOMPL.setStateSpaceForJoints" in text
    assert "simOMPL.setCollisionPairs" in text
    assert "simOMPL.solve" in text
    assert "def follow_joint_path(" in text
    assert "sim.generateTimeOptimalTrajectory" in text
    assert "sim.getPathInterpolatedConfig" in text
```

- [ ] **Step 2: Run static test and verify failure**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py::test_robot_controller_plans_and_follows_joint_paths -q
```

Expected: FAIL.

- [ ] **Step 3: Implement planning/following**

Add:

```python
    def plan_joint_path(self, goal_config: list[float], attached_object: int | None = None) -> list[float] | None:
        sim = self.sim
        simOMPL = self.simOMPL
        self.update_robot_collection(attached_object=attached_object)
        task = simOMPL.createTask("python_rail_ur10_path_task")
        try:
            simOMPL.setAlgorithm(task, simOMPL.Algorithm.RRTConnect)
            projection = [1 if i <= 2 else 0 for i in range(len(self.handles.joints))]
            simOMPL.setStateSpaceForJoints(task, self.handles.joints, projection)
            simOMPL.setCollisionPairs(task, [
                self.robot_collection, sim.handle_all,
                self.robot_collection, self.robot_collection,
            ])
            simOMPL.setStartState(task, self.get_config())
            simOMPL.setGoalState(task, goal_config)
            simOMPL.setup(task)
            if simOMPL.solve(task, 10.0) and simOMPL.hasExactSolution(task):
                simOMPL.simplifyPath(task, 10.0)
                return simOMPL.getPath(task)
            return None
        finally:
            simOMPL.destroyTask(task)

    def follow_joint_path(self, path: list[float]) -> bool:
        sim = self.sim
        dof = len(self.handles.joints)
        rail_vel = 0.25
        rail_accel = 0.5
        arm_vel = 3.141592653589793
        arm_accel = 40 * 3.141592653589793 / 180
        min_max_vel = []
        min_max_accel = []
        for i in range(dof):
            vel = rail_vel if i == 0 else arm_vel
            accel = rail_accel if i == 0 else arm_accel
            min_max_vel.extend([-vel, vel])
            min_max_accel.extend([-accel, accel])
        path_lengths = sim.getPathLengths(path, dof)
        path_pts, times = sim.generateTimeOptimalTrajectory(
            path,
            path_lengths,
            min_max_vel,
            min_max_accel,
            1000,
            "not-a-knot",
            5,
        )
        if not path_pts or not times:
            return False
        start_time = sim.getSimulationTime()
        while True:
            elapsed = sim.getSimulationTime() - start_time
            if elapsed >= times[-1]:
                break
            self.set_target_config(sim.getPathInterpolatedConfig(path_pts, times, elapsed))
            self.ctx.step()
        self.set_target_config(sim.getPathInterpolatedConfig(path_pts, times, times[-1]))
        self.ctx.step(10)
        return True
```

- [ ] **Step 4: Run tests**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py -q
```

Expected: PASS.

---

### Task 6: Cartesian Interpolation Helper

**Files:**
- Modify: `python_controller/robot.py`
- Modify: `tests/test_python_controller_units.py`
- Modify: `tests/test_python_controller_static.py`

- [ ] **Step 1: Add failing unit/static tests**

Append to `tests/test_python_controller_units.py`:

```python
from python_controller.robot import interpolate_scalar, interpolate_pose_xyz


def test_interpolate_pose_xyz_keeps_quaternion_and_interpolates_position():
    start = [0, 0, 0, 0, 0, 0, 1]
    target = [1, 2, 3, 0, 0, 0, 1]
    poses = interpolate_pose_xyz(start, target, max_step=1.0)
    assert poses[0] == start
    assert poses[-1] == target
    assert poses[1][:3] == [0.25, 0.5, 0.75]
    assert all(p[3:] == [0, 0, 0, 1] for p in poses)


def test_interpolate_scalar():
    assert interpolate_scalar(2.0, 6.0, 0.25) == 3.0
```

Append to `tests/test_python_controller_static.py`:

```python
def test_robot_controller_has_cartesian_move():
    text = read("python_controller/robot.py")
    assert "def move_cartesian_linear(" in text
    assert "interpolate_pose_xyz" in text
    assert "solve_ik_to_pose" in text
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_units.py::test_interpolate_pose_xyz_keeps_quaternion_and_interpolates_position tests/test_python_controller_static.py::test_robot_controller_has_cartesian_move -q
```

Expected: FAIL because interpolation helpers do not exist.

- [ ] **Step 3: Implement interpolation and Cartesian move**

Add near top of `python_controller/robot.py`:

```python
import math


def interpolate_scalar(start: float, target: float, t: float) -> float:
    return start + (target - start) * t


def interpolate_pose_xyz(start: list[float], target: list[float], max_step: float = 0.03) -> list[list[float]]:
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    dz = target[2] - start[2]
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    steps = max(1, math.ceil(distance / max_step))
    poses = []
    for i in range(steps + 1):
        t = i / steps
        poses.append([
            interpolate_scalar(start[0], target[0], t),
            interpolate_scalar(start[1], target[1], t),
            interpolate_scalar(start[2], target[2], t),
            target[3],
            target[4],
            target[5],
            target[6],
        ])
    return poses
```

Add method:

```python
    def move_cartesian_linear(self, target_pose: list[float], attached_object: int | None = None) -> bool:
        start_pose = self.sim.getObjectPose(self.handles.ik_tip, self.handles.ik_base)
        configs = []
        for pose in interpolate_pose_xyz(start_pose, target_pose, max_step=0.03):
            config = self.solve_ik_to_pose(pose, attached_object=attached_object)
            if config is None:
                return False
            configs.append(config)
        if self.collides(configs, attached_object=attached_object):
            return False
        for config in configs:
            self.set_target_config(config)
            self.ctx.step()
        self.set_target_config(configs[-1])
        self.ctx.step(10)
        return True
```

- [ ] **Step 4: Run tests**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_units.py tests/test_python_controller_static.py -q
```

Expected: PASS.

---

### Task 7: Pick/Place Sequencer

**Files:**
- Modify: `python_controller/pick_place.py`
- Modify: `tests/test_python_controller_static.py`

- [ ] **Step 1: Add failing static assertions**

Append:

```python
def test_pick_place_controller_uses_robot_and_suction_signals():
    text = read("python_controller/pick_place.py")
    assert "def object_size(" in text
    assert "def pick_poses_for_object(" in text
    assert "def place_poses_for_object(" in text
    assert "suctionPadEnabled" in text
    assert "move_to_pose" in text
    assert "position = [0.0, 0.0, 1.0]" in text
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py::test_pick_place_controller_uses_robot_and_suction_signals -q
```

Expected: FAIL.

- [ ] **Step 3: Implement pick/place sequence**

Replace `python_controller/pick_place.py`:

```python
from __future__ import annotations


def object_size(sim: object, handle: int) -> list[float]:
    try:
        bb = sim.getShapeBB(handle)
        if isinstance(bb, list) and len(bb) >= 3:
            return [abs(bb[0]), abs(bb[1]), abs(bb[2])]
    except Exception:
        pass
    mins = [
        sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_x),
        sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_y),
        sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_z),
    ]
    maxs = [
        sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_x),
        sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_y),
        sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_z),
    ]
    return [abs(maxs[i] - mins[i]) for i in range(3)]


class PickPlaceController:
    def __init__(self, robot: object, belt: object):
        self.robot = robot
        self.belt = belt
        self.sim = robot.sim
        self.attached_object: int | None = None
        self.approach = 0.10
        self.contact_offset = 0.005
        self.hold_height = 0.35

    def pick_poses_for_object(self, handle: int) -> tuple[list[float], list[float], list[float]]:
        pose = self.sim.getObjectPose(handle)
        size = object_size(self.sim, handle)
        rotated = self.sim.multiplyPoses(pose, [0, 0, 0, 0, 1, 0, 0])
        top_z = pose[2] + size[2] * 0.5
        pre = [pose[0], pose[1], top_z + self.approach, *rotated[3:]]
        pick = [pose[0], pose[1], top_z + self.contact_offset, *rotated[3:]]
        hold = [pose[0], pose[1], top_z + self.hold_height, *rotated[3:]]
        return pre, pick, hold

    def place_poses_for_object(self, handle: int, position: list[float]) -> tuple[list[float], list[float]]:
        position = [0.0, 0.0, 1.0] if position is None else position
        pallet_pose = self.sim.getObjectPose(self.robot.handles.pallet)
        size = object_size(self.sim, handle)
        center = [
            position[0] + size[0] * 0.5,
            position[1] + size[1] * 0.5,
            position[2] + size[2] * 0.5,
        ]
        orientation = [0, 1, 0, 0]
        place = self.sim.multiplyPoses(pallet_pose, [center[0], center[1], center[2], *orientation])
        pre = self.sim.multiplyPoses(pallet_pose, [center[0], center[1], center[2] + self.approach, *orientation])
        return pre, place

    def set_suction(self, enabled: bool) -> None:
        self.sim.setInt32Signal("suctionPadEnabled", 1 if enabled else 0)

    def move_to_pose(self, pose: list[float], attached_object: int | None = None) -> bool:
        goal = self.robot.solve_ik_to_pose(pose, attached_object=attached_object)
        if goal is None:
            return False
        path = self.robot.plan_joint_path(goal, attached_object=attached_object)
        if path is None:
            return False
        return self.robot.follow_joint_path(path)

    def pick(self, handle: int) -> bool:
        pre, pick, hold = self.pick_poses_for_object(handle)
        if not self.move_to_pose(pre):
            return False
        if not self.robot.move_cartesian_linear(pick):
            return False
        self.set_suction(True)
        self.attached_object = handle
        self.robot.ctx.step(20)
        return self.robot.move_cartesian_linear(hold, attached_object=handle)

    def place(self, position: list[float] | None = None) -> bool:
        if self.attached_object is None:
            return False
        position = [0.0, 0.0, 1.0] if position is None else position
        pre, place = self.place_poses_for_object(self.attached_object, position)
        if not self.move_to_pose(pre, attached_object=self.attached_object):
            return False
        if not self.robot.move_cartesian_linear(place, attached_object=self.attached_object):
            return False
        self.set_suction(False)
        self.attached_object = None
        self.robot.ctx.step(20)
        return self.robot.move_cartesian_linear(pre)
```

- [ ] **Step 4: Run tests**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py -q
```

Expected: PASS.

---

### Task 8: Main Loop CLI

**Files:**
- Modify: `python_controller/main.py`
- Modify: `tests/test_python_controller_static.py`

- [ ] **Step 1: Add failing static assertions**

Append:

```python
def test_main_loop_wires_belt_robot_and_pick_place():
    text = read("python_controller/main.py")
    assert "connect(" in text
    assert "ensure_started(" in text
    assert "BeltMonitor" in text
    assert "RobotController" in text
    assert "PickPlaceController" in text
    assert "full_transition_items" in text
    assert "random.choice" in text
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py::test_main_loop_wires_belt_robot_and_pick_place -q
```

Expected: FAIL.

- [ ] **Step 3: Implement CLI loop**

Replace `python_controller/main.py`:

```python
from __future__ import annotations

import argparse
import logging
import random

from .belt import BeltMonitor
from .pick_place import PickPlaceController
from .remote import connect, ensure_started
from .robot import RobotController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--place", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    ctx = connect(port=args.port)
    ensure_started(ctx, scene_path=args.scene)
    belt = BeltMonitor(ctx.sim)
    robot = RobotController(ctx)
    pick_place = PickPlaceController(robot=robot, belt=belt)

    cycles = 0
    while ctx.sim.getSimulationState() != ctx.sim.simulation_advancing_abouttostop:
        items = belt.full_transition_items()
        if items:
            item = random.choice(items)
            logging.info("selected Cuboid handle=%s", item.handle)
            belt.set_robot_picking(True)
            try:
                ok = pick_place.pick(item.handle) and pick_place.place(args.place)
                logging.info("pick/place handle=%s ok=%s", item.handle, ok)
                cycles += 1
            finally:
                belt.set_robot_picking(False)
        if args.max_cycles and cycles >= args.max_cycles:
            break
        ctx.step()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py tests/test_python_controller_units.py -q
```

Expected: PASS.

---

### Task 9: Smoke Command And Documentation

**Files:**
- Create: `python_controller/README.md`
- Modify: `tests/test_python_controller_static.py`

- [ ] **Step 1: Add README test**

Append:

```python
def test_controller_readme_documents_scene_script_contract():
    text = read("python_controller/README.md")
    assert "Only keep zeroBufferBelt.lua" in text
    assert "trackedItemsData" in text
    assert "robotPicking" in text
    assert "python -m python_controller.main" in text
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py::test_controller_readme_documents_scene_script_contract -q
```

Expected: FAIL because README does not exist.

- [ ] **Step 3: Add README**

Create `python_controller/README.md`:

```markdown
# Python CoppeliaSim Controller

Only keep zeroBufferBelt.lua in the scene. Remove the previous robot motion scripts:

- external_api.lua
- loading_test.lua
- rail_ur10_motion_simplify.lua
- ur10_joint_motion.lua
- ur10_linear_motion.lua

The Python controller reads `trackedItemsData` from `zeroBufferBelt.lua`, waits for the belt to become full, sets `robotPicking=1`, picks one random tracked Cuboid, places it at `(0, 0, 1)`, and sets `robotPicking=0`.

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m python_controller.main --port 23000 --place 0 0 1
```

For a bounded smoke run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m python_controller.main --port 23000 --place 0 0 1 --max-cycles 1
```
```

- [ ] **Step 4: Run tests**

Run:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests/test_python_controller_static.py tests/test_python_controller_units.py -q
```

Expected: PASS.

---

### Task 10: Manual Simulation Verification

**Files:**
- No source changes unless verification exposes a bug.

- [ ] **Step 1: Start CoppeliaSim in true headless or GUI mode**

Use true headless for stability:

```bash
cd /home/gao/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04
./coppeliaSim.sh -H -GzmqRemoteApi.rpcPort=23000
```

Expected: Remote API port `23000` opens.

- [ ] **Step 2: Run one Python controller cycle**

Run from `integrations`:

```bash
/home/gao/anaconda3/envs/packing-toolkit/bin/python -m python_controller.main --port 23000 --place 0 0 1 --max-cycles 1
```

Expected:
- Controller waits until `zeroBufferBelt.lua` publishes full `trackedItemsData`.
- Controller sets `robotPicking=1`.
- Controller picks one tracked Cuboid.
- Controller places it at `(0, 0, 1)`.
- Controller sets `robotPicking=0`.
- Process exits after one successful cycle.

- [ ] **Step 3: If CoppeliaSim crashes**

Record:

```bash
tail -n 200 /path/to/coppeliasim/log
```

Then rerun CoppeliaSim with `-H` and repeat Step 2. If the crash still happens in true headless mode, capture the native stack trace and inspect whether it is in `simIK`, `simOMPL`, or dynamics.

---

## Self-Review

- Spec coverage: The plan covers Python Remote API connection, stepping, belt signal contract, random Cuboid selection, IK, collision checking, collision-free joint planning, Cartesian interpolation, path following, pick/place, CLI, and manual verification.
- No-open-items scan: No deferred markers or open-ended implementation steps remain.
- Type consistency: `RemoteContext`, `BeltMonitor`, `RobotController`, `PickPlaceController`, `TrackedItem`, and method names are defined before use.
- Risk note: Static tests prove structure without CoppeliaSim. Full correctness requires Task 10 against the real scene because `simIK` and `simOMPL` behavior is scene-dependent.
