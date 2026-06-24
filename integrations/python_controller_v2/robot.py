from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .remote import first_object


def interpolate_scalar(start: float, target: float, t: float) -> float:
    return start + (target - start) * t


def interpolate_config(start: list[float], target: list[float], max_step: float = 5.0 * math.pi / 180.0) -> list[list[float]]:
    if max_step <= 0:
        raise ValueError("max_step must be greater than 0")
    if len(start) != len(target):
        raise ValueError(f"config length mismatch: {len(start)} != {len(target)}")
    max_delta = max((abs(goal - current) for current, goal in zip(start, target)), default=0.0)
    step_count = max(1, math.ceil(max_delta / max_step))
    return [
        [interpolate_scalar(current, goal, index / step_count) for current, goal in zip(start, target)]
        for index in range(step_count + 1)
    ]


def interpolate_pose_xyz(start: list[float], target: list[float], max_step: float = 0.03) -> list[list[float]]:
    if max_step <= 0:
        raise ValueError("max_step must be greater than 0")
    distance = math.dist(start[:3], target[:3])
    step_count = max(1, math.ceil(distance / max_step))
    return [
        [
            interpolate_scalar(start[0], target[0], index / step_count),
            interpolate_scalar(start[1], target[1], index / step_count),
            interpolate_scalar(start[2], target[2], index / step_count),
            *target[3:7],
        ]
        for index in range(step_count + 1)
    ]


def normalize_quaternion(quaternion: list[float]) -> list[float]:
    x, y, z, w = [float(value) for value in quaternion[:4]]
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / length, y / length, z / length, w / length]


def quaternion_to_roll_pitch_yaw(quaternion: list[float]) -> tuple[float, float, float]:
    x, y, z, w = normalize_quaternion(quaternion)

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    if abs(sin_pitch) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sin_pitch)
    else:
        pitch = math.asin(sin_pitch)

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


def quaternion_orientation_error(actual: list[float], target: list[float]) -> float:
    actual = normalize_quaternion(actual)
    target = normalize_quaternion(target)
    dot = sum(a * b for a, b in zip(actual, target))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


def quaternion_z_axis(quaternion: list[float]) -> list[float]:
    x, y, z, w = normalize_quaternion(quaternion)
    return [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]


def vector_angle(actual: list[float], target: list[float]) -> float:
    dot = sum(a * b for a, b in zip(actual, target))
    actual_length = math.sqrt(sum(value * value for value in actual))
    target_length = math.sqrt(sum(value * value for value in target))
    if actual_length <= 0.0 or target_length <= 0.0:
        return math.pi
    dot = max(-1.0, min(1.0, dot / (actual_length * target_length)))
    return math.acos(dot)


@dataclass
class V2Handles:
    mobile_base: int
    rail_joint: int
    arm_base: int
    ik_tip: int
    ik_target: int
    pallet: int
    arm_joints: list[int]


class RobotControllerV2:
    """Small robot controller with separate rail motion and arm-only IK."""

    def __init__(self, ctx: object):
        self.ctx = ctx
        self.sim = ctx.sim
        self.simIK = ctx.client.require("simIK")
        self.simOMPL = ctx.client.require("simOMPL")
        self.last_error: str | None = None
        self.arm_config_hints: dict[str, list[float]] = {}
        self.handles = self.discover_handles()
        self.ik_env: int | None = None
        self.ik_group: int | None = None
        self.ik_element: int | None = None
        self.sim_to_ik: object | None = None
        self.robot_collection: int | None = None
        self.moving_collection: int | None = None
        self.obstacle_collection: int | None = None
        self.setup_arm_ik()
        self.prime_ik_target()

    def discover_handles(self) -> V2Handles:
        sim = self.sim
        mobile_base = first_object(sim, ["/mobile_arm"])
        rail_joint = first_object(sim, ["/mobile_arm/railJoint", ":/mobile_arm/railJoint", "/railJoint", ":/railJoint"])
        arm_base = first_object(sim, ["/mobile_arm/railJoint/UR10", "/mobile_arm/UR10", "/UR10"])
        ik_tip = first_object(sim, ["/mobile_arm/railJoint/UR10/ikTip", "/mobile_arm/UR10/ikTip", "/ikTip", ":/ikTip"])
        ik_target = first_object(sim, ["/ikTarget", ":/ikTarget"])
        pallet = first_object(sim, ["/pallet", ":/pallet"])
        arm_joints = list(sim.getObjectsInTree(arm_base, sim.sceneobject_joint))
        return V2Handles(
            mobile_base=mobile_base,
            rail_joint=rail_joint,
            arm_base=arm_base,
            ik_tip=ik_tip,
            ik_target=ik_target,
            pallet=pallet,
            arm_joints=arm_joints,
        )

    def _clear_error(self) -> None:
        self.last_error = None

    def _set_error(self, reason: str) -> None:
        self.last_error = reason

    def _motion_step_result(self, result: object) -> int:
        if isinstance(result, (list, tuple)):
            if not result:
                return -1
            return int(result[0])
        return int(result)

    def _collision_count(self, result: object) -> int:
        if isinstance(result, (list, tuple)):
            if not result:
                return 0
            return int(result[0])
        return int(result)

    def _remote_success(self, result: object) -> bool:
        if isinstance(result, (list, tuple)):
            if not result:
                return False
            return bool(result[0])
        return bool(result)

    def _path_to_configs(self, path: object) -> list[list[float]]:
        joint_count = len(self.handles.arm_joints)
        if joint_count <= 0 or not isinstance(path, (list, tuple)) or not path:
            return []
        if all(isinstance(config, (list, tuple)) for config in path):
            configs = []
            for config in path:
                if len(config) != joint_count:
                    return []
                configs.append([float(value) for value in config])
            return configs
        if len(path) % joint_count != 0:
            return []
        values = [float(value) for value in path]
        return [values[index : index + joint_count] for index in range(0, len(values), joint_count)]

    def _append_unique_config(
        self,
        configs: list[list[float]],
        config: list[float] | None,
        tolerance: float = 1.0e-4,
    ) -> bool:
        if config is None:
            return False
        self._validate_arm_config(config)
        normalized = [float(value) for value in config]
        for existing in configs:
            if max((abs(a - b) for a, b in zip(existing, normalized)), default=0.0) <= tolerance:
                return False
        configs.append(normalized)
        return True

    def _config_distance(self, config: list[float], reference: list[float]) -> float:
        return math.sqrt(sum((float(value) - float(seed)) ** 2 for value, seed in zip(config, reference)))

    def _normalize_goal_configs(self, goal_config: list[float] | list[list[float]]) -> list[list[float]]:
        if goal_config and all(isinstance(config, (list, tuple)) for config in goal_config):
            goals = [[float(value) for value in config] for config in goal_config]
        else:
            goals = [[float(value) for value in goal_config]]
        if not goals:
            raise ValueError("Expected at least one arm goal config")
        for config in goals:
            self._validate_arm_config(config)
        return goals

    def _ik_handle(self, sim_handle: int) -> int:
        sim_to_ik = self.sim_to_ik
        if sim_to_ik is None:
            raise RuntimeError("arm IK has not been set up")
        try:
            return sim_to_ik[sim_handle]
        except KeyError:
            return sim_to_ik[str(sim_handle)]

    def _ik_constraint_candidates(self) -> list[int]:
        constraints = [self.simIK.constraint_pose]
        position = getattr(self.simIK, "constraint_position", None)
        alpha_beta = getattr(self.simIK, "constraint_alpha_beta", None)
        if position is not None and alpha_beta is not None:
            tool_axis = position | alpha_beta
            if tool_axis not in constraints:
                constraints.append(tool_axis)
        return constraints

    def _set_ik_constraints(self, constraints: int) -> None:
        set_constraints = getattr(self.simIK, "setElementConstraints", None)
        if set_constraints is not None:
            set_constraints(self.ik_env, self.ik_group, self.ik_element, constraints)

    def update_collision_collections(
        self,
        attached_object: int | None = None,
        ignored_objects: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        sim = self.sim
        ignored = [int(handle) for handle in (ignored_objects or []) if handle is not None and handle >= 0]
        previous_robot_collection = self.robot_collection
        previous_moving_collection = self.moving_collection
        previous_obstacle_collection = self.obstacle_collection
        next_robot_collection = sim.createCollection()
        next_moving_collection = sim.createCollection()
        next_obstacle_collection = sim.createCollection()
        created = [next_robot_collection, next_moving_collection, next_obstacle_collection]
        try:
            sim.addItemToCollection(next_robot_collection, sim.handle_tree, self.handles.arm_base, 0)
            sim.addItemToCollection(next_moving_collection, sim.handle_tree, self.handles.arm_base, 0)
            if attached_object is not None and attached_object >= 0:
                sim.addItemToCollection(next_moving_collection, sim.handle_single, attached_object, 0)

            sim.addItemToCollection(next_obstacle_collection, sim.handle_all, -1, 0)
            sim.addItemToCollection(next_obstacle_collection, sim.handle_tree, self.handles.arm_base, 1)
            if attached_object is not None and attached_object >= 0:
                sim.addItemToCollection(next_obstacle_collection, sim.handle_single, attached_object, 1)
            for handle in ignored:
                if handle != attached_object:
                    sim.addItemToCollection(next_obstacle_collection, sim.handle_single, handle, 1)
        except Exception:
            for collection in created:
                sim.destroyCollection(collection)
            raise

        self.robot_collection = next_robot_collection
        self.moving_collection = next_moving_collection
        self.obstacle_collection = next_obstacle_collection
        for collection in (previous_robot_collection, previous_moving_collection, previous_obstacle_collection):
            if collection is not None:
                sim.destroyCollection(collection)

    def _motion_collision_reason(self, include_attached_object: bool = True) -> str | None:
        sim = self.sim
        if include_attached_object:
            if self._collision_count(sim.checkCollision(self.moving_collection, self.obstacle_collection)) > 0:
                return "moving robot or attached object collided with an obstacle"
        elif self._collision_count(sim.checkCollision(self.robot_collection, self.obstacle_collection)) > 0:
            return "robot arm collided with an obstacle"
        if self._collision_count(sim.checkCollision(self.robot_collection, self.robot_collection)) > 0:
            return "robot self-collision"
        return None

    def setup_arm_ik(self) -> None:
        simIK = self.simIK
        h = self.handles
        env = simIK.createEnvironment()
        group = simIK.createGroup(env)
        simIK.setGroupCalculation(env, group, simIK.method_damped_least_squares, 0.3, 99)
        result = simIK.addElementFromScene(
            env,
            group,
            h.arm_base,
            h.ik_tip,
            h.ik_target,
            simIK.constraint_pose,
        )
        element = result[0]
        sim_to_ik = result[1]
        set_precision = getattr(simIK, "setElementPrecision", None)
        if set_precision is not None:
            set_precision(env, group, element, [0.0005, 0.005])
        self.ik_env = env
        self.ik_group = group
        self.ik_element = element
        self.sim_to_ik = sim_to_ik

    def sync_ik_target(self, world_pose: list[float]) -> list[float]:
        h = self.handles
        target_pose = list(world_pose)
        self.sim.setObjectPose(h.ik_target, target_pose)
        self.simIK.setObjectPose(
            self.ik_env,
            self._ik_handle(h.ik_target),
            self.sim.getObjectPose(h.ik_target, h.arm_base),
            self._ik_handle(h.arm_base),
        )
        return target_pose

    def prime_ik_target(self) -> None:
        self.simIK.syncFromSim(self.ik_env, [self.ik_group])
        self.sync_ik_target(self.sim.getObjectPose(self.handles.ik_tip))

    def get_rail_position(self) -> float:
        return float(self.sim.getJointPosition(self.handles.rail_joint))

    def set_rail_position(self, position: float) -> None:
        self.sim.setJointPosition(self.handles.rail_joint, float(position))

    def move_rail_to(self, position: float, max_step: float = 0.02) -> bool:
        start = self.get_rail_position()
        for value in interpolate_config([start], [float(position)], max_step=max_step)[1:]:
            self.set_rail_position(value[0])
            self.ctx.step()
        self.set_rail_position(float(position))
        self._clear_error()
        return True

    def move_rail_by(self, delta: float, max_step: float = 0.02) -> bool:
        return self.move_rail_to(self.get_rail_position() + float(delta), max_step=max_step)

    def get_arm_config(self) -> list[float]:
        return [float(self.sim.getJointPosition(joint)) for joint in self.handles.arm_joints]

    def set_arm_config(self, config: list[float]) -> None:
        self._validate_arm_config(config)
        for joint, value in zip(self.handles.arm_joints, config):
            self.sim.setJointPosition(joint, float(value))

    def remember_arm_config(self, name: str, config: list[float] | None = None) -> list[float]:
        stored = self.get_arm_config() if config is None else [float(value) for value in config]
        self._validate_arm_config(stored)
        self.arm_config_hints[str(name)] = stored
        return list(stored)

    def get_remembered_arm_config(self, name: str) -> list[float] | None:
        config = self.arm_config_hints.get(str(name))
        return None if config is None else list(config)

    def remembered_arm_configs(self) -> list[list[float]]:
        return [list(config) for config in self.arm_config_hints.values()]

    def _validate_arm_config(self, config: list[float]) -> None:
        if len(config) != len(self.handles.arm_joints):
            raise ValueError(f"Expected {len(self.handles.arm_joints)} arm joint values, got {len(config)}")

    def move_arm_to_config(self, config: list[float], max_step: float = 5.0 * math.pi / 180.0) -> bool:
        self._validate_arm_config(config)
        start = self.get_arm_config()
        for waypoint in interpolate_config(start, list(config), max_step=max_step)[1:]:
            self.set_arm_config(waypoint)
            self.ctx.step()
        self.set_arm_config(list(config))
        self._clear_error()
        return True

    def move_arm_by(self, delta_config: list[float], max_step: float = 5.0 * math.pi / 180.0) -> bool:
        if len(delta_config) != len(self.handles.arm_joints):
            raise ValueError(f"Expected {len(self.handles.arm_joints)} arm joint deltas, got {len(delta_config)}")
        target = [current + delta for current, delta in zip(self.get_arm_config(), delta_config)]
        return self.move_arm_to_config(target, max_step=max_step)

    def _pose_constraint_error(self, actual: list[float], target: list[float], constraints: int) -> tuple[float, float, str]:
        position_error = math.dist(actual[:3], target[:3])
        if len(actual) < 7 or len(target) < 7:
            return position_error, 0.0, "orientation"
        alpha_beta = getattr(self.simIK, "constraint_alpha_beta", 0)
        gamma = getattr(self.simIK, "constraint_gamma", 0)
        if constraints & alpha_beta and constraints & gamma:
            return position_error, quaternion_orientation_error(actual[3:7], target[3:7]), "orientation"
        if constraints & alpha_beta:
            return (
                position_error,
                vector_angle(quaternion_z_axis(actual[3:7]), quaternion_z_axis(target[3:7])),
                "tool_axis",
            )
        return position_error, 0.0, "orientation"

    def _config_reaches_pose(self, config: list[float], pose: list[float], constraints: int) -> bool:
        buffered = self.get_arm_config()
        try:
            self.set_arm_config(config)
            tip_pose = self.sim.getObjectPose(self.handles.ik_tip)
        finally:
            self.set_arm_config(buffered)
        position_error, orientation_error, orientation_label = self._pose_constraint_error(tip_pose, pose, constraints)
        if position_error <= 0.005 and orientation_error <= 2.0 * math.pi / 180.0:
            return True
        self._set_error(
            "arm IK candidate misses target "
            f"(position_error={position_error:.6f}, {orientation_label}_error={orientation_error:.6f})"
        )
        return False

    def _tip_reaches_pose(self, pose: list[float], constraints: int) -> bool:
        tip_pose = self.sim.getObjectPose(self.handles.ik_tip)
        position_error, orientation_error, orientation_label = self._pose_constraint_error(tip_pose, pose, constraints)
        if position_error <= 0.005 and orientation_error <= 2.0 * math.pi / 180.0:
            return True
        self._set_error(
            "arm Cartesian motion misses target "
            f"(position_error={position_error:.6f}, {orientation_label}_error={orientation_error:.6f})"
        )
        return False

    def _sample_forward_pose(
        self,
        current_pose: list[float],
        target_pose: list[float],
        rng: random.Random,
        step_size: float = 0.03,
        noise_radius: float = 0.015,
    ) -> list[float]:
        direction = [float(target_pose[index]) - float(current_pose[index]) for index in range(3)]
        distance = math.sqrt(sum(value * value for value in direction))
        if distance <= 0.0:
            return list(target_pose)

        unit = [value / distance for value in direction]
        step = min(step_size, distance)
        nominal = [float(current_pose[index]) + unit[index] * step for index in range(3)]
        if distance <= step_size:
            return [*nominal, *target_pose[3:7]]

        for _ in range(16):
            noise = [rng.uniform(-noise_radius, noise_radius) for _ in range(3)]
            if sum(noise[index] * direction[index] for index in range(3)) < 0.0:
                noise = [-value for value in noise]
            candidate = [nominal[index] + noise[index] for index in range(3)]
            motion = [candidate[index] - float(current_pose[index]) for index in range(3)]
            if sum(motion[index] * direction[index] for index in range(3)) <= 0.0:
                continue
            if math.dist(candidate, target_pose[:3]) >= distance:
                continue
            return [*candidate, *target_pose[3:7]]

        return [*nominal, *target_pose[3:7]]

    def _generate_path_goal_config(self, target_pose: list[float], path_point_count: int = 12) -> list[float] | None:
        generate_path = getattr(self.simIK, "generatePath", None)
        if generate_path is None:
            return None

        constraints = self.simIK.constraint_pose
        self.simIK.syncFromSim(self.ik_env, [self.ik_group])
        self._set_ik_constraints(constraints)
        synced_pose = self.sync_ik_target(target_pose)
        ik_joints = [self._ik_handle(joint) for joint in self.handles.arm_joints]
        path = generate_path(
            self.ik_env,
            self.ik_group,
            ik_joints,
            self._ik_handle(self.handles.ik_tip),
            int(path_point_count),
        )
        configs = self._path_to_configs(path)
        if not configs:
            return None
        config = configs[-1]
        if not self._config_reaches_pose(config, synced_pose, constraints):
            return None
        self._clear_error()
        return config

    def _stochastic_goal_config(
        self,
        target_pose: list[float],
        rng: random.Random,
        max_attempts: int,
    ) -> tuple[list[float] | None, str | None]:
        last_error: str | None = None
        for _attempt in range(max_attempts):
            current_pose = self.sim.getObjectPose(self.handles.ik_tip)
            if math.dist(current_pose[:3], target_pose[:3]) <= 0.005:
                waypoint = list(target_pose)
            else:
                waypoint = self._sample_forward_pose(current_pose, target_pose, rng)
            config = self.solve_arm_ik(waypoint)
            if config is None:
                last_error = self.last_error
                continue
            self.set_arm_config(config)
            if math.dist(waypoint[:3], target_pose[:3]) <= 0.005:
                self._clear_error()
                return config, None

        config = self.solve_arm_ik(target_pose)
        if config is not None:
            self._clear_error()
            return config, None
        return None, self.last_error or last_error

    def solve_arm_ik(self, target_pose: list[float]) -> list[float] | None:
        self._clear_error()
        constraints = self.simIK.constraint_pose
        self.simIK.syncFromSim(self.ik_env, [self.ik_group])
        self._set_ik_constraints(constraints)
        synced_pose = self.sync_ik_target(target_pose)
        result = self.simIK.handleGroup(self.ik_env, self.ik_group)
        result_code = result[0] if isinstance(result, (list, tuple)) else result
        if result_code != self.simIK.result_success:
            self._set_error(f"arm IK did not converge for pose {synced_pose} (result={result_code})")
            return None
        config = [
            float(self.simIK.getJointPosition(self.ik_env, self._ik_handle(joint)))
            for joint in self.handles.arm_joints
        ]
        if not self._config_reaches_pose(config, synced_pose, constraints):
            return None
        self._clear_error()
        return config

    def find_arm_configs_for_pose(
        self,
        target_pose: list[float],
        max_configs: int = 8,
        max_time: float = 10.0,
    ) -> list[list[float]]:
        self._clear_error()
        if max_configs <= 0:
            return []

        buffered_config = self.get_arm_config()
        configs: list[list[float]] = []
        rng = random.Random()
        max_attempts = max(8, int(max(float(max_time), 0.1) * 40.0))
        last_error: str | None = None
        try:
            direct_config = self.solve_arm_ik(target_pose)
            if direct_config is None:
                last_error = self.last_error
            else:
                self._append_unique_config(configs, direct_config)

            for seed_config in self.remembered_arm_configs():
                if len(configs) >= max_configs:
                    break
                self.set_arm_config(seed_config)
                seeded_config = self.solve_arm_ik(target_pose)
                if seeded_config is None:
                    last_error = self.last_error or last_error
                    continue
                self._append_unique_config(configs, seeded_config)

            if len(configs) < max_configs:
                self.set_arm_config(buffered_config)
                path_config = self._generate_path_goal_config(target_pose)
                if path_config is None:
                    last_error = self.last_error or last_error
                else:
                    self._append_unique_config(configs, path_config)

            search_rounds = max(1, max_configs * 2)
            for _round in range(search_rounds):
                if len(configs) >= max_configs:
                    break
                self.set_arm_config(buffered_config)
                config, last_error = self._stochastic_goal_config(target_pose, rng, max_attempts)
                if config is None:
                    break
                self._append_unique_config(configs, config)

            if configs:
                configs.sort(key=lambda config: self._config_distance(config, buffered_config))
                self._clear_error()
                return configs[:max_configs]
        finally:
            self.set_arm_config(buffered_config)

        self._set_error(f"OMPL target IK found no config for pose {target_pose}: {last_error}")
        return []

    def arm_configs_collide(
        self,
        configs: list[list[float]],
        attached_object: int | None = None,
        ignored_objects: list[int] | tuple[int, ...] | None = None,
        allow_final_contact: bool = False,
    ) -> bool:
        buffered_config = self.get_arm_config()
        self.update_collision_collections(attached_object=attached_object, ignored_objects=ignored_objects)
        try:
            final_index = len(configs) - 1
            for index, config in enumerate(configs):
                self.set_arm_config(config)
                include_attached_object = not (allow_final_contact and index == final_index)
                reason = self._motion_collision_reason(include_attached_object=include_attached_object)
                if reason is not None:
                    self._set_error(reason)
                    return True
            self._clear_error()
            return False
        finally:
            self.set_arm_config(buffered_config)

    def plan_arm_joint_path(
        self,
        goal_config: list[float] | list[list[float]],
        attached_object: int | None = None,
        planning_time: float = 3.0,
        path_state_count: int = 100,
    ) -> list[float] | None:
        goal_configs = self._normalize_goal_configs(goal_config)
        self._clear_error()
        self.update_collision_collections(attached_object=attached_object)
        simOMPL = self.simOMPL
        task = simOMPL.createTask("python_v2_arm_path_task")
        try:
            simOMPL.setAlgorithm(task, simOMPL.Algorithm.RRTstar)
            projected_joints = min(3, len(self.handles.arm_joints))
            projection = [1 if index < projected_joints else 0 for index in range(len(self.handles.arm_joints))]
            simOMPL.setStateSpaceForJoints(task, self.handles.arm_joints, projection)
            simOMPL.setCollisionPairs(
                task,
                [
                    self.moving_collection,
                    self.obstacle_collection,
                    self.robot_collection,
                    self.robot_collection,
                ],
            )
            simOMPL.setStartState(task, self.get_arm_config())
            if len(goal_configs) == 1:
                simOMPL.setGoalState(task, goal_configs[0])
            else:
                set_goal_states = getattr(simOMPL, "setGoalStates", None)
                add_goal_state = getattr(simOMPL, "addGoalState", None)
                if set_goal_states is not None:
                    set_goal_states(task, goal_configs)
                else:
                    simOMPL.setGoalState(task, goal_configs[0])
                    if add_goal_state is not None:
                        for config in goal_configs[1:]:
                            add_goal_state(task, config)
            simOMPL.setup(task)
            if not self._remote_success(simOMPL.solve(task, float(planning_time))):
                self._set_error("OMPL could not solve an arm path to the IK goal")
                return None
            has_exact_solution = getattr(simOMPL, "hasExactSolution", None)
            if has_exact_solution is not None and not self._remote_success(has_exact_solution(task)):
                self._set_error("OMPL returned no exact arm path")
                return None
            simOMPL.simplifyPath(task, float(planning_time))
            interpolate_path = getattr(simOMPL, "interpolatePath", None)
            if interpolate_path is not None:
                interpolate_path(task, int(path_state_count))
            path = simOMPL.getPath(task)
            if not path:
                self._set_error("OMPL returned an empty arm path")
                return None
            self._clear_error()
            return path
        finally:
            simOMPL.destroyTask(task)

    def follow_arm_path(self, path: list[float], max_step: float = 5.0 * math.pi / 180.0) -> bool:
        configs = self._path_to_configs(path)
        if not configs:
            self._set_error("cannot follow an empty or malformed arm path")
            return False
        for config in configs:
            if not self.move_arm_to_config(config, max_step=max_step):
                return False
        self._clear_error()
        return True

    def move_arm_ompl_to_config(
        self,
        goal_config: list[float],
        attached_object: int | None = None,
        planning_time: float = 3.0,
        path_state_count: int = 100,
    ) -> bool:
        path = self.plan_arm_joint_path(
            goal_config,
            attached_object=attached_object,
            planning_time=planning_time,
            path_state_count=path_state_count,
        )
        if path is None:
            return False
        return self.follow_arm_path(path)

    def move_arm_ompl_to_pose(
        self,
        target_pose: list[float],
        attached_object: int | None = None,
        ignored_objects: list[int] | tuple[int, ...] | None = None,
        max_configs: int = 8,
        ik_time: float = 10.0,
        planning_time: float = 3.0,
        path_state_count: int = 100,
        allow_final_contact: bool = False,
    ) -> bool:
        self._clear_error()
        configs = self.find_arm_configs_for_pose(target_pose, max_configs=max_configs, max_time=ik_time)
        if not configs:
            detail = self.last_error
            self._set_error(f"OMPL pose move found no arm IK configs: {detail}")
            return False

        planner_errors: list[str] = []
        collision_rejections = 0
        valid_configs: list[list[float]] = []
        for config in configs:
            if self.arm_configs_collide(
                [config],
                attached_object=attached_object,
                ignored_objects=ignored_objects,
                allow_final_contact=allow_final_contact,
            ):
                collision_rejections += 1
                continue

            valid_configs.append(config)

        if valid_configs:
            planner_attached_object = None if allow_final_contact else attached_object
            path = self.plan_arm_joint_path(
                valid_configs,
                attached_object=planner_attached_object,
                planning_time=planning_time,
                path_state_count=path_state_count,
            )
            if path is not None and self.follow_arm_path(path):
                self._clear_error()
                return True
            if self.last_error:
                planner_errors.append(self.last_error)

        details = []
        if collision_rejections:
            details.append(f"{collision_rejections} IK config(s) collided")
        if planner_errors:
            details.append(planner_errors[-1])
        self._set_error("; ".join(details) if details else "OMPL found no arm path to pose")
        return False

    def _validate_linear_motion_collision_free(
        self,
        target_pose: list[float],
        max_step: float = 0.03,
        attached_object: int | None = None,
        ignored_objects: list[int] | tuple[int, ...] | None = None,
        allow_final_contact: bool = False,
    ) -> bool:
        buffered_config = self.get_arm_config()
        start_pose = self.sim.getObjectPose(self.handles.ik_tip)
        waypoints = interpolate_pose_xyz(start_pose, target_pose, max_step=max_step)
        self.update_collision_collections(attached_object=attached_object, ignored_objects=ignored_objects)
        try:
            reason = self._motion_collision_reason()
            if reason is not None:
                self._set_error(f"collision precheck failed at waypoint 0: {reason}")
                return False

            final_index = len(waypoints) - 1
            for index, waypoint in enumerate(waypoints[1:], start=1):
                config = self.solve_arm_ik(waypoint)
                if config is None:
                    detail = self.last_error
                    self._set_error(f"collision precheck IK failed at waypoint {index}: {detail}")
                    return False

                self.set_arm_config(config)
                include_attached_object = not (allow_final_contact and index == final_index)
                reason = self._motion_collision_reason(include_attached_object=include_attached_object)
                if reason is not None:
                    self._set_error(f"collision precheck failed at waypoint {index}: {reason}")
                    return False

            self._clear_error()
            return True
        finally:
            self.set_arm_config(buffered_config)

    def move_arm_linear_to_pose(
        self,
        target_pose: list[float],
        max_step: float = 0.03,
        joint_step: float = 5.0 * math.pi / 180.0,
        attached_object: int | None = None,
        ignored_objects: list[int] | tuple[int, ...] | None = None,
        allow_final_contact: bool = False,
        validate_collision: bool = True,
    ) -> bool:
        self._clear_error()
        if validate_collision and not self._validate_linear_motion_collision_free(
            target_pose,
            max_step=max_step,
            attached_object=attached_object,
            ignored_objects=ignored_objects,
            allow_final_contact=allow_final_contact,
        ):
            return False

        last_abort_result = None
        for constraints in self._ik_constraint_candidates():
            start_pose = self.sim.getObjectPose(self.handles.ik_tip)
            self.sim.setObjectPose(self.handles.ik_target, start_pose)
            motion = self.sim.moveToPose_init(
                {
                    "ik": {
                        "tip": self.handles.ik_tip,
                        "target": self.handles.ik_target,
                        "base": self.handles.arm_base,
                        "joints": self.handles.arm_joints,
                        "method": self.simIK.method_damped_least_squares,
                        "damping": 0.3,
                        "iterations": 99,
                        "constraints": constraints,
                        "precision": [0.0005, 0.005],
                    },
                    "targetPose": list(target_pose),
                    "maxVel": [0.4, 0.4, 0.4, 1.8],
                    "maxAccel": [0.8, 0.8, 0.8, 0.9],
                    "maxJerk": [0.6, 0.6, 0.6, 0.8],
                }
            )
            try:
                for _ in range(10000):
                    result = self._motion_step_result(self.sim.moveToPose_step(motion))
                    if result < 0:
                        self._set_error(f"sim.moveToPose_step failed with result {result}")
                        return False
                    if result == 2:
                        last_abort_result = result
                        break
                    if result == 1:
                        self.sim.setObjectPose(self.handles.ik_target, list(target_pose))
                        self.ctx.step(10)
                        if not self._tip_reaches_pose(target_pose, constraints):
                            return False
                        self._clear_error()
                        return True
                    self.ctx.step()
                else:
                    self._set_error("sim.moveToPose_step exceeded 10000 controller steps")
                    return False
            finally:
                self.sim.moveToPose_cleanup(motion)
        self._set_error(f"sim.moveToPose_step aborted for pose {target_pose} (result={last_abort_result})")
        return False

    def move_arm_linear_by(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        max_step: float = 0.03,
        joint_step: float = 5.0 * math.pi / 180.0,
        attached_object: int | None = None,
        ignored_objects: list[int] | tuple[int, ...] | None = None,
        allow_final_contact: bool = False,
        validate_collision: bool = True,
    ) -> bool:
        start_pose = self.sim.getObjectPose(self.handles.ik_tip)
        target_pose = [
            start_pose[0] + dx,
            start_pose[1] + dy,
            start_pose[2] + dz,
            *start_pose[3:7],
        ]
        return self.move_arm_linear_to_pose(
            target_pose,
            max_step=max_step,
            joint_step=joint_step,
            attached_object=attached_object,
            ignored_objects=ignored_objects,
            allow_final_contact=allow_final_contact,
            validate_collision=validate_collision,
        )
