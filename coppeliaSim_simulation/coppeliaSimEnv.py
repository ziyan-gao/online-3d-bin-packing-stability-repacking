from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class PackingEnv:
    BIN_SIZE = np.array([0.6, 0.6, 0.6])
    BOTTOM_THICKNESS = 0.05
    PHYSICS_ENGINES = {
        "bullet": {"constants": ["physics_bullet"]},
        "bullet_2_78": {"constants": ["physics_bullet_2_78", "physics_bullet278"], "version": 278},
        "bullet_2_83": {"constants": ["physics_bullet_2_83", "physics_bullet283"], "version": 283},
        "bullet2.7": {"constants": ["physics_bullet_2_78", "physics_bullet278"], "version": 278},
        "bullet2.8": {"constants": ["physics_bullet_2_83", "physics_bullet283"], "version": 283},
        "ode": {"constants": ["physics_ode"]},
        "vortex": {"constants": ["physics_vortex"]},
        "newton": {"constants": ["physics_newton"]},
    }

    def __init__(
        self,
        physics_engine=None,
        drop_height=0.005,
        object_mass=0.1,
        port=23000,
        simulation_time_step=None,
        stepping=False,
        scene_path=None,
    ):
        self.client = RemoteAPIClient(port=port)
        self.sim = self.client.require("sim")
        self.stop_simulation(wait=True)
        if scene_path is None:
            scene_path = Path(__file__).with_name("coppeliaSim_bin_packing.ttt")
        self.load_scene(str(scene_path))
        self.scriptHandle = self.sim.getScript(self.sim.scripttype_simulation, "Script")
        self.physics_engine = None
        self.drop_height = drop_height
        self.object_mass = object_mass
        self.simulation_time_step = None
        self.stepping = bool(stepping)
        self.base_top_z = self.BOTTOM_THICKNESS / 2
        if simulation_time_step is not None:
            self.set_simulation_time_step(simulation_time_step)
        if physics_engine is not None:
            self.set_physics_engine(physics_engine)
        if self.stepping:
            self.client.setStepping(True)
        self.sim.startSimulation()
        self.createEnv()

    def _normalize_handles(self, handles):
        if handles is None:
            return []
        if isinstance(handles, (list, tuple)):
            out = []
            for handle in handles:
                out.extend(self._normalize_handles(handle))
            return out
        return [handles]

    def wait_until_stopped(self, timeout=10.0, poll_interval=0.05):
        start = time.time()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            if time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for CoppeliaSim simulation to stop.")
            time.sleep(poll_interval)

    def stop_simulation(self, wait=True):
        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.sim.stopSimulation()
        if wait:
            self.wait_until_stopped()

    def load_scene(self, scene_path):
        try:
            self.sim.loadScene(scene_path)
        except Exception as exc:
            if "simulation is not stopped" not in str(exc):
                raise
            self.stop_simulation(wait=True)
            time.sleep(0.1)
            self.sim.loadScene(scene_path)

    def set_simulation_time_step(self, simulation_time_step):
        simulation_time_step = float(simulation_time_step)
        if simulation_time_step <= 0:
            raise ValueError("simulation_time_step must be positive.")
        if hasattr(self.sim, "setSimulationTimeStep"):
            self.sim.setSimulationTimeStep(simulation_time_step)
        elif hasattr(self.sim, "floatparam_simulation_time_step"):
            self.sim.setFloatParam(self.sim.floatparam_simulation_time_step, simulation_time_step)
        else:
            raise RuntimeError("This CoppeliaSim version does not expose a simulation time-step setter.")
        self.simulation_time_step = self.get_simulation_time_step()

    def get_simulation_time_step(self):
        if hasattr(self.sim, "getSimulationTimeStep"):
            try:
                return float(self.sim.getSimulationTimeStep())
            except Exception:
                pass
        if hasattr(self.sim, "floatparam_simulation_time_step"):
            return float(self.sim.getFloatParam(self.sim.floatparam_simulation_time_step))
        return self.simulation_time_step

    def step(self, n=1):
        if not self.stepping:
            raise RuntimeError("step() requires PackingEnv(..., stepping=True).")
        for _ in range(int(n)):
            self.client.step()

    def run_for_sim_time(self, seconds):
        seconds = float(seconds)
        if seconds <= 0:
            return 0
        if not self.stepping:
            time.sleep(seconds)
            return 0
        dt = self.get_simulation_time_step()
        if dt is None or dt <= 0:
            raise RuntimeError("Cannot step simulation without a positive simulation time step.")
        steps = int(np.ceil(seconds / dt))
        self.step(steps)
        return steps

    def set_physics_engine(self, physics_engine):
        engine_key = str(physics_engine).lower()
        engine_spec = self.PHYSICS_ENGINES.get(engine_key)
        if engine_spec is None:
            raise ValueError("Unknown physics engine: {}".format(physics_engine))
        engine_value = None
        used_version_property = False
        for constant_name in engine_spec["constants"]:
            if hasattr(self.sim, constant_name):
                engine_value = getattr(self.sim, constant_name)
                break
        if engine_value is None and "version" in engine_spec:
            engine_value = self._set_physics_engine_version(engine_spec["version"])
            used_version_property = True
        if engine_value is None:
            raise ValueError("Physics engine is not exposed by this CoppeliaSim version: {}".format(physics_engine))
        if not used_version_property:
            self.sim.setInt32Param(self.sim.intparam_dynamic_engine, engine_value)
        self.physics_engine = engine_key
        self.physics_engine_selection = self.get_physics_engine_selection()

    def _set_physics_engine_version(self, version):
        if not all(hasattr(self.sim, name) for name in (
            "setIntArrayProperty",
            "getIntArrayProperty",
            "handle_scene",
            "physics_bullet",
        )):
            return None
        self.sim.setIntArrayProperty(self.sim.handle_scene, "dynamicsEngine", [self.sim.physics_bullet, version])
        selected = self.sim.getIntArrayProperty(self.sim.handle_scene, "dynamicsEngine")
        if len(selected) >= 2 and selected[1] == version:
            return self.sim.physics_bullet
        raise RuntimeError("Requested Bullet {}, but selected dynamicsEngine={}".format(version, selected))

    def get_physics_engine_selection(self):
        if all(hasattr(self.sim, name) for name in ("getIntArrayProperty", "handle_scene")):
            try:
                return self.sim.getIntArrayProperty(self.sim.handle_scene, "dynamicsEngine")
            except Exception:
                pass
        if hasattr(self.sim, "intparam_dynamic_engine"):
            return [self.sim.getInt32Param(self.sim.intparam_dynamic_engine)]
        return []

    def transform(self, obj, pxyz, drop_height=None):
        drop_height = self.drop_height if drop_height is None else drop_height
        return np.array([
            pxyz[0] + obj[0] / 2 - self.BIN_SIZE[0] / 2,
            pxyz[1] + obj[1] / 2 - self.BIN_SIZE[1] / 2,
            self.base_top_z + pxyz[2] + obj[2] / 2 + drop_height,
        ])

    def createEnv(self):
        self.env_handle = self.sim.callScriptFunction("createEnv", self.scriptHandle, [0, 0], "env")
        print("env created.")
        self.objects = []
        self.object_initial_poses = {}
        self.object_stable_poses = {}

    def add_object(self, obj, pxyz, drop_height=None, mass=None):
        mass = self.object_mass if mass is None else mass
        object_handle = self.sim.callScriptFunction(
            "load_object",
            self.scriptHandle,
            obj.tolist(),
            self.transform(obj, pxyz, drop_height=drop_height).tolist(),
            self.env_handle,
            mass,
        )
        object_handles = self._normalize_handles(object_handle)
        self.objects.extend(object_handles)
        for handle in object_handles:
            self.object_initial_poses[handle] = self.get_object_pose(handle)
        return object_handles

    def get_object_pose(self, handle):
        return np.array(self.sim.getObjectPose(handle, getattr(self.sim, "handle_world", -1)), dtype=np.float64)

    def get_object_velocity(self, handle):
        if hasattr(self.sim, "getObjectVelocity"):
            linear_velocity, angular_velocity = self.sim.getObjectVelocity(handle)
        else:
            linear_velocity, angular_velocity = self.sim.getVelocity(handle)
        return np.array(linear_velocity, dtype=np.float64), np.array(angular_velocity, dtype=np.float64)

    def max_object_speed(self, handles=None):
        object_handles = self._normalize_handles(self.objects if handles is None else handles)
        max_linear_speed = 0.0
        max_angular_speed = 0.0
        for handle in object_handles:
            linear_velocity, angular_velocity = self.get_object_velocity(handle)
            max_linear_speed = max(max_linear_speed, float(np.linalg.norm(linear_velocity)))
            max_angular_speed = max(max_angular_speed, float(np.linalg.norm(angular_velocity)))
        return max_linear_speed, max_angular_speed

    def wait_until_quasi_static(
        self,
        handles=None,
        linear_velocity_threshold=0.01,
        angular_velocity_threshold=0.05,
        stable_duration=0.2,
        max_wait_time=3.0,
        poll_interval=0.05,
    ):
        object_handles = self._normalize_handles(self.objects if handles is None else handles)
        if not object_handles:
            return {
                "settled": True,
                "settle_wait_time": 0.0,
                "max_linear_speed": 0.0,
                "max_angular_speed": 0.0,
            }
        start = time.time()
        below_threshold_since = None
        sim_elapsed = 0.0
        max_linear_speed_seen = 0.0
        max_angular_speed_seen = 0.0
        while True:
            elapsed = sim_elapsed if self.stepping else time.time() - start
            last_linear_speed, last_angular_speed = self.max_object_speed(object_handles)
            max_linear_speed_seen = max(max_linear_speed_seen, last_linear_speed)
            max_angular_speed_seen = max(max_angular_speed_seen, last_angular_speed)
            below_threshold = (
                last_linear_speed <= linear_velocity_threshold
                and last_angular_speed <= angular_velocity_threshold
            )
            if below_threshold:
                if below_threshold_since is None:
                    below_threshold_since = elapsed
                if elapsed - below_threshold_since >= stable_duration:
                    return {
                        "settled": True,
                        "settle_wait_time": elapsed,
                        "max_linear_speed": max_linear_speed_seen,
                        "max_angular_speed": max_angular_speed_seen,
                    }
            else:
                below_threshold_since = None
            if elapsed >= max_wait_time:
                return {
                    "settled": False,
                    "settle_wait_time": elapsed,
                    "max_linear_speed": max_linear_speed_seen,
                    "max_angular_speed": max_angular_speed_seen,
                }
            steps = self.run_for_sim_time(poll_interval)
            if self.stepping:
                sim_elapsed += steps * self.get_simulation_time_step()

    @staticmethod
    def _pose_delta(pose_a, pose_b):
        position_delta = np.linalg.norm(pose_a[:3] - pose_b[:3])
        quat_a = pose_a[3:] / np.linalg.norm(pose_a[3:])
        quat_b = pose_b[3:] / np.linalg.norm(pose_b[3:])
        quat_dot = np.clip(abs(np.dot(quat_a, quat_b)), -1.0, 1.0)
        rotation_delta = 2 * np.arccos(quat_dot)
        return position_delta, rotation_delta

    def check_stability(
        self,
        handles=None,
        settle_time=0.5,
        sample_time=0.1,
        position_tolerance=0.01,
        rotation_tolerance=0.05,
        drift_tolerance=0.002,
    ):
        object_handles = self._normalize_handles(self.objects if handles is None else handles)
        if not object_handles:
            return {
                "stable": True,
                "max_position_delta": 0.0,
                "max_rotation_delta": 0.0,
                "max_settle_drift": 0.0,
                "moved_handles": [],
            }
        self.run_for_sim_time(settle_time)
        settled_poses = {handle: self.get_object_pose(handle) for handle in object_handles}
        self.run_for_sim_time(sample_time)
        final_poses = {handle: self.get_object_pose(handle) for handle in object_handles}
        max_position_delta = 0.0
        max_rotation_delta = 0.0
        max_settle_drift = 0.0
        moved_handles = []
        for handle in object_handles:
            stable_pose = self.object_stable_poses.get(handle)
            if stable_pose is None:
                position_delta = 0.0
                rotation_delta = 0.0
            else:
                position_delta, rotation_delta = self._pose_delta(stable_pose, final_poses[handle])
            settle_drift, _ = self._pose_delta(settled_poses[handle], final_poses[handle])
            max_position_delta = max(max_position_delta, position_delta)
            max_rotation_delta = max(max_rotation_delta, rotation_delta)
            max_settle_drift = max(max_settle_drift, settle_drift)
            if (
                position_delta > position_tolerance
                or rotation_delta > rotation_tolerance
                or settle_drift > drift_tolerance
            ):
                moved_handles.append(handle)
        stable = len(moved_handles) == 0
        if stable:
            self.object_stable_poses.update(final_poses)
        return {
            "stable": stable,
            "max_position_delta": max_position_delta,
            "max_rotation_delta": max_rotation_delta,
            "max_settle_drift": max_settle_drift,
            "moved_handles": moved_handles,
        }

    def reset(self):
        print("env reset.")
        object_handles = self._normalize_handles(self.objects)
        if object_handles:
            self.sim.removeObjects(object_handles)
        self.objects = []
        self.object_initial_poses = {}
        self.object_stable_poses = {}

    def stop(self):
        print("simulation stopped.")
        self.stop_simulation(wait=True)
