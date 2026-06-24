from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .remote import first_object


def interpolate_scalar(start: float, target: float, t: float) -> float:
    return start + (target - start) * t


def interpolate_pose_xyz(start: list[float], target: list[float], max_step: float = 0.03) -> list[list[float]]:
    if max_step <= 0:
        raise ValueError("max_step must be greater than 0")

    distance = math.dist(start[:3], target[:3])
    if distance == 0:
        return [list(target)]

    step_count = max(1, math.ceil(distance / max_step))
    poses = []
    for index in range(step_count + 1):
        t = index / step_count
        poses.append(
            [
                interpolate_scalar(start[0], target[0], t),
                interpolate_scalar(start[1], target[1], t),
                interpolate_scalar(start[2], target[2], t),
                *target[3:],
            ]
        )
    return poses


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
    """Initial robot controller shell. Later tasks add motion behavior."""

    def __init__(self, ctx: object, rail_enabled: bool = True):
        self.ctx = ctx
        self.sim = ctx.sim
        self.simIK = ctx.client.require("simIK")
        self.simOMPL = ctx.client.require("simOMPL")
        self.last_error: str | None = None
        self.rail_enabled = rail_enabled
        self.handles = self.discover_handles()
        self.robot_collection = None
        self.moving_collection = None
        self.obstacle_collection = None
        self.ik_env = None
        self.ik_group = None
        self.ik_element = None
        self.sim_to_ik = None
        self.setup_ik()
        self.prime_ik_worlds()
        self.update_robot_collection()

    def discover_handles(self) -> RobotHandles:
        sim = self.sim
        ik_base = first_object(sim, ["/mobile_arm"])
        rail_joint = first_object(sim, ["/mobile_arm/railJoint", ":/mobile_arm/railJoint", "/railJoint", ":/railJoint"])
        ur10_base = first_object(sim, ["/mobile_arm/railJoint/UR10", "/mobile_arm/UR10", "/UR10"])
        ik_tip = first_object(
            sim,
            ["/mobile_arm/railJoint/UR10/ikTip", "/mobile_arm/UR10/ikTip", "/ikTip", ":/ikTip"],
        )
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

    def active_joints(self, include_rail: bool | None = None) -> list[int]:
        rail_enabled = getattr(self, "rail_enabled", True)
        include_rail = False if include_rail is None else include_rail and rail_enabled
        rail_joint = getattr(self.handles, "rail_joint", None)
        has_rail_joint = rail_joint is not None and self.handles.joints and self.handles.joints[0] == rail_joint
        if include_rail or len(self.handles.joints) <= 1 or not has_rail_joint:
            return list(self.handles.joints)
        return list(self.handles.joints[1:])

    def _arm_ik_base(self) -> int:
        return getattr(self.handles, "ur10_base", self.handles.ik_base)

    def _cartesian_ik_base(self, include_rail: bool = False) -> int:
        return self._arm_ik_base()

    def _ik_object_map(self, include_rail: bool | None = None) -> object:
        sim_to_ik = getattr(self, "sim_to_ik", None)
        if sim_to_ik is None:
            raise RuntimeError("IK environment has not been set up")
        return sim_to_ik

    def _ik_context(self, include_rail: bool | None = None) -> tuple[int, int, int | None, object]:
        if (
            getattr(self, "ik_env", None) is None
            or getattr(self, "ik_group", None) is None
            or getattr(self, "sim_to_ik", None) is None
        ):
            raise RuntimeError("IK environment has not been set up")
        return self.ik_env, self.ik_group, getattr(self, "ik_element", None), self.sim_to_ik

    def _ik_context_base(self, include_rail: bool | None = None) -> int:
        return self._arm_ik_base()

    def _sync_world_pose_to_ik_target(self, pose: list[float], include_rail: bool | None = None) -> None:
        h = self.handles
        env, _group, _element, _sim_to_ik = self._ik_context(include_rail=include_rail)
        base = self._ik_context_base(include_rail=include_rail)
        self.sim.setObjectPose(h.ik_target, pose)
        self.simIK.setObjectPose(
            env,
            self._ik_handle(h.ik_target, include_rail=include_rail),
            self.sim.getObjectPose(h.ik_target, base),
            self._ik_handle(base, include_rail=include_rail),
        )

    def _sync_world_pose_to_arm_ik_target(self, pose: list[float]) -> None:
        self._sync_world_pose_to_ik_target(pose, include_rail=False)

    def get_motion_config(self, include_rail: bool | None = None) -> list[float]:
        return [self.sim.getJointPosition(joint) for joint in self.active_joints(include_rail=include_rail)]

    def _validate_config(self, config: list[float]) -> None:
        expected = len(self.handles.joints)
        actual = len(config)
        if actual != expected:
            raise ValueError(f"Expected {expected} joint values, got {actual}")

    def _validate_motion_config(self, config: list[float], include_rail: bool | None = None) -> None:
        expected = len(self.active_joints(include_rail=include_rail))
        actual = len(config)
        if actual != expected:
            raise ValueError(f"Expected {expected} active joint values, got {actual}")

    def set_config(self, config: list[float]) -> None:
        self._validate_config(config)
        for joint, value in zip(self.handles.joints, config):
            self.sim.setJointPosition(joint, value)

    def set_target_config(self, config: list[float]) -> None:
        self._validate_config(config)
        for joint, value in zip(self.handles.joints, config):
            self.sim.setJointTargetPosition(joint, value)

    def set_motion_config(self, config: list[float], include_rail: bool | None = None) -> None:
        self._validate_motion_config(config, include_rail=include_rail)
        for joint, value in zip(self.active_joints(include_rail=include_rail), config):
            self.sim.setJointPosition(joint, value)

    def set_motion_target_config(self, config: list[float], include_rail: bool | None = None) -> None:
        self._validate_motion_config(config, include_rail=include_rail)
        for joint, value in zip(self.active_joints(include_rail=include_rail), config):
            self.sim.setJointTargetPosition(joint, value)

    def preposition_rail_for_world_x(self, target_x: float, reach_half_width: float = 0.5) -> bool:
        self._clear_error()
        if not getattr(self, "rail_enabled", True):
            return True
        rail_joint = getattr(self.handles, "rail_joint", None)
        ur10_base = getattr(self.handles, "ur10_base", None)
        if rail_joint is None or ur10_base is None:
            self._set_error("rail pre-position requested without rail or UR10 base handle")
            return False

        base_pose = self.sim.getObjectPose(ur10_base)
        relative_x = target_x - base_pose[0]
        if abs(relative_x) <= reach_half_width:
            return True

        base_shift_x = relative_x - reach_half_width if relative_x > reach_half_width else relative_x + reach_half_width
        current_rail = self.sim.getJointPosition(rail_joint)
        target_rail = current_rail - base_shift_x
        self.sim.moveToConfig(
            {
                "joints": [rail_joint],
                "targetPos": [target_rail],
                "maxVel": [0.25],
                "maxAccel": [0.5],
                "maxJerk": [0.5],
            }
        )
        self.sim.setJointPosition(rail_joint, target_rail)
        self.ctx.step(10)
        self._clear_error()
        return True

    def preposition_rail_for_pose(self, pose: list[float], reach_half_width: float = 0.5) -> bool:
        return self.preposition_rail_for_world_x(pose[0], reach_half_width=reach_half_width)

    def setup_ik(self) -> None:
        self.ik_env, self.ik_group, self.ik_element, self.sim_to_ik = self._create_ik_context(self._arm_ik_base())

    def _ik_precision(self) -> list[float]:
        return [0.0005, 0.005]

    def _create_ik_context(self, base: int) -> tuple[int, int, int, object]:
        simIK = self.simIK
        h = self.handles
        ik_env = simIK.createEnvironment()
        ik_group = simIK.createGroup(ik_env)
        simIK.setGroupCalculation(
            ik_env,
            ik_group,
            simIK.method_damped_least_squares,
            0.3,
            99,
        )
        res = simIK.addElementFromScene(
            ik_env,
            ik_group,
            base,
            h.ik_tip,
            h.ik_target,
            simIK.constraint_pose,
        )
        set_precision = getattr(simIK, "setElementPrecision", None)
        if set_precision is not None:
            set_precision(ik_env, ik_group, res[0], self._ik_precision())
        return ik_env, ik_group, res[0], res[1]

    def prime_ik_worlds(self) -> None:
        tip_pose = self.sim.getObjectPose(self.handles.ik_tip)
        self._prime_ik_world(tip_pose, include_rail=False)

    def _prime_ik_world(self, tip_pose: list[float], include_rail: bool) -> None:
        env, group, _element, _sim_to_ik = self._ik_context(include_rail=include_rail)
        self.simIK.syncFromSim(env, [group])
        self._sync_world_pose_to_ik_target(tip_pose, include_rail=include_rail)

    def _ik_handle(self, sim_handle: int, include_rail: bool | None = None) -> int:
        sim_to_ik = self._ik_object_map(include_rail=include_rail)
        if isinstance(sim_to_ik, Sequence):
            raise RuntimeError("Unsupported IK object map shape; expected a mapping keyed by scene handles")
        try:
            return sim_to_ik[sim_handle]
        except KeyError:
            pass
        except TypeError as exc:
            raise RuntimeError("Unsupported IK object map shape; expected a mapping keyed by scene handles") from exc
        try:
            return sim_to_ik[str(sim_handle)]
        except KeyError as exc:
            raise RuntimeError(f"IK object map does not contain scene handle {sim_handle}") from exc
        except TypeError as exc:
            raise RuntimeError("Unsupported IK object map shape; expected a mapping keyed by scene handles") from exc

    def update_robot_collection(self, attached_object: int | None = None) -> None:
        sim = self.sim
        previous_robot_collection = getattr(self, "robot_collection", None)
        previous_moving_collection = getattr(self, "moving_collection", None)
        previous_obstacle_collection = getattr(self, "obstacle_collection", None)
        next_robot_collection = sim.createCollection()
        next_moving_collection = sim.createCollection()
        next_obstacle_collection = sim.createCollection()
        created = [next_robot_collection, next_moving_collection, next_obstacle_collection]
        try:
            sim.addItemToCollection(next_robot_collection, sim.handle_tree, self.handles.ik_base, 0)
            sim.addItemToCollection(next_moving_collection, sim.handle_tree, self.handles.ik_base, 0)
            if attached_object is not None and attached_object >= 0:
                sim.addItemToCollection(next_moving_collection, sim.handle_single, attached_object, 0)
            sim.addItemToCollection(next_obstacle_collection, sim.handle_all, -1, 0)
            sim.addItemToCollection(next_obstacle_collection, sim.handle_tree, self.handles.ik_base, 1)
            if attached_object is not None and attached_object >= 0:
                sim.addItemToCollection(next_obstacle_collection, sim.handle_single, attached_object, 1)
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

    def _path_lengths(self, result: object) -> list[float] | None:
        if (
            isinstance(result, (list, tuple))
            and len(result) >= 2
            and isinstance(result[0], (list, tuple))
            and isinstance(result[1], (int, float))
        ):
            result = result[0]
        if not isinstance(result, (list, tuple)):
            return None
        try:
            return [float(value) for value in result]
        except (TypeError, ValueError):
            return None

    def _motion_step_result(self, result: object) -> int:
        if isinstance(result, (list, tuple)):
            if not result:
                return -1
            return int(result[0])
        return int(result)

    def _clear_error(self) -> None:
        self.last_error = None

    def _set_error(self, reason: str) -> None:
        self.last_error = reason

    def _set_error_with_detail(self, reason: str, detail: str | None) -> None:
        self.last_error = f"{reason}: {detail}" if detail else reason

    def _ik_damping_candidates(self) -> list[float]:
        return [0.1, 0.3, 0.03, 0.6]

    def _ik_constraint_candidates(self) -> list[int]:
        constraints = [self.simIK.constraint_pose]
        position = getattr(self.simIK, "constraint_position", None)
        alpha_beta = getattr(self.simIK, "constraint_alpha_beta", None)
        if position is not None and alpha_beta is not None:
            relaxed = position | alpha_beta
            if relaxed not in constraints:
                constraints.append(relaxed)
        return constraints

    def _ik_pose_candidates(self, pose: list[float]) -> list[list[float]]:
        candidates = [list(pose)]
        multiply_poses = getattr(self.sim, "multiplyPoses", None)
        if multiply_poses is None:
            return candidates
        for yaw in (math.pi / 2.0, -math.pi / 2.0, math.pi):
            local_yaw = [0.0, 0.0, 0.0, 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
            try:
                candidate = multiply_poses(pose, local_yaw)
            except Exception:
                continue
            if isinstance(candidate, (list, tuple)) and len(candidate) >= 7:
                candidates.append([float(value) for value in candidate[:7]])
        return candidates

    def _set_ik_constraint(self, constraints: int, include_rail: bool | None = None) -> None:
        set_constraints = getattr(self.simIK, "setElementConstraints", None)
        if set_constraints is not None:
            env, group, element, _sim_to_ik = self._ik_context(include_rail=include_rail)
            if element is not None:
                set_constraints(env, group, element, constraints)

    def _set_ik_calculation(self, damping: float, include_rail: bool | None = None) -> None:
        set_group_calculation = getattr(self.simIK, "setGroupCalculation", None)
        if set_group_calculation is not None:
            env, group, _element, _sim_to_ik = self._ik_context(include_rail=include_rail)
            set_group_calculation(
                env,
                group,
                self.simIK.method_damped_least_squares,
                damping,
                99,
            )

    def _motion_collision_reason(self) -> str | None:
        sim = self.sim
        if self._collision_count(sim.checkCollision(self.moving_collection, self.obstacle_collection)) > 0:
            return "moving robot or attached object collided with an obstacle"
        if self._collision_count(sim.checkCollision(self.robot_collection, self.robot_collection)) > 0:
            return "robot self-collision"
        return None

    def _pose_error(self, actual: list[float], target: list[float]) -> tuple[float, float]:
        position_error = math.dist(actual[:3], target[:3])
        if len(actual) < 7 or len(target) < 7:
            return position_error, 0.0
        dot = sum(actual[i] * target[i] for i in range(3, 7))
        dot = max(-1.0, min(1.0, abs(dot)))
        orientation_error = 2.0 * math.acos(dot)
        return position_error, orientation_error

    def _validate_tip_reached_pose(
        self,
        target_pose: list[float],
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 2.0 * math.pi / 180.0,
    ) -> bool:
        tip_pose = self.sim.getObjectPose(self.handles.ik_tip)
        position_error, orientation_error = self._pose_error(tip_pose, target_pose)
        if position_error <= position_tolerance and orientation_error <= orientation_tolerance:
            return True
        self._set_error(
            "Cartesian move finished away from target "
            f"(position_error={position_error:.6f}, orientation_error={orientation_error:.6f})"
        )
        return False

    def _config_reaches_pose(
        self,
        config: list[float],
        target_pose: list[float],
        include_rail: bool | None = None,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 2.0 * math.pi / 180.0,
    ) -> bool:
        ik_tip = getattr(self.handles, "ik_tip", None)
        if ik_tip is None:
            return True

        buffered = self.get_motion_config(include_rail=include_rail)
        try:
            self.set_motion_config(config, include_rail=include_rail)
            tip_pose = self.sim.getObjectPose(ik_tip)
            position_error, orientation_error = self._pose_error(tip_pose, target_pose)
        finally:
            self.set_motion_config(buffered, include_rail=include_rail)

        if position_error <= position_tolerance and orientation_error <= orientation_tolerance:
            return True
        self._set_error(
            "IK candidate misses target "
            f"(position_error={position_error:.6f}, orientation_error={orientation_error:.6f})"
        )
        return False

    def collides(
        self,
        configs: list[list[float]],
        attached_object: int | None = None,
        include_rail: bool | None = None,
    ) -> bool:
        sim = self.sim
        buffered = self.get_motion_config(include_rail=include_rail)
        self.update_robot_collection(attached_object=attached_object)
        try:
            for config in configs:
                self.set_motion_config(config, include_rail=include_rail)
                if self._collision_count(sim.checkCollision(self.moving_collection, self.obstacle_collection)) > 0:
                    return True
                if self._collision_count(sim.checkCollision(self.robot_collection, self.robot_collection)) > 0:
                    return True
            return False
        finally:
            self.set_motion_config(buffered, include_rail=include_rail)

    def _sample_joint_path(self, start_config: list[float], goal_config: list[float], max_step: float = 5 * math.pi / 180) -> list[list[float]]:
        max_delta = max(abs(goal - start) for start, goal in zip(start_config, goal_config))
        step_count = max(2, math.ceil(max_delta / max_step))
        samples = []
        for index in range(step_count + 1):
            t = index / step_count
            samples.append([interpolate_scalar(start, goal, t) for start, goal in zip(start_config, goal_config)])
        return samples

    def move_to_config(
        self,
        goal_config: list[float],
        attached_object: int | None = None,
        allow_start_collision: bool = False,
    ) -> bool:
        self._validate_motion_config(goal_config)
        self._clear_error()
        active_joints = self.active_joints()
        start_config = self.get_motion_config()
        samples = self._sample_joint_path(start_config, goal_config)
        if allow_start_collision:
            has_cleared_start_contact = False
            for sample in samples:
                sample_collides = self.collides([sample], attached_object=attached_object)
                if sample_collides and not has_cleared_start_contact:
                    continue
                if sample_collides:
                    self._set_error("direct config trajectory collides")
                    return False
                has_cleared_start_contact = True
        elif self.collides(samples, attached_object=attached_object):
            self._set_error("direct config trajectory collides")
            return False

        rail_vel = 0.25
        rail_accel = 0.5
        arm_vel = math.pi / 3
        arm_accel = math.pi / 3
        max_vel = []
        max_accel = []
        max_jerk = []
        rail_joint = getattr(self.handles, "rail_joint", self.handles.joints[0] if self.handles.joints else None)
        rail_active = getattr(self, "rail_enabled", True) and active_joints and active_joints[0] == rail_joint
        for index in range(len(active_joints)):
            if rail_active and index == 0:
                max_vel.append(rail_vel)
                max_accel.append(rail_accel)
                max_jerk.append(rail_accel)
            else:
                max_vel.append(arm_vel)
                max_accel.append(arm_accel)
                max_jerk.append(arm_accel)

        self.sim.moveToConfig(
            {
                "joints": active_joints,
                "targetPos": goal_config,
                "maxVel": max_vel,
                "maxAccel": max_accel,
                "maxJerk": max_jerk,
            }
        )
        self.set_motion_config(goal_config)
        self.ctx.step(10)
        self._clear_error()
        return True

    def _find_configs_metric(self, include_rail: bool | None = None) -> list[float]:
        base_metric = [1.0, 8.0, 8.0, 8.0, 0.8, 0.6, 0.3]
        dof = len(self.active_joints(include_rail=include_rail))
        if dof <= len(base_metric):
            return base_metric[:dof]
        return base_metric + [1.0] * (dof - len(base_metric))

    def _normalize_ik_configs(self, raw_configs: object, include_rail: bool | None = None) -> list[list[float]]:
        if not isinstance(raw_configs, (list, tuple)):
            return []
        dof = len(self.active_joints(include_rail=include_rail))
        configs = []
        if raw_configs and all(isinstance(value, (int, float)) for value in raw_configs):
            raw_configs = [raw_configs]
        for raw_config in raw_configs:
            if not isinstance(raw_config, (list, tuple)) or len(raw_config) != dof:
                continue
            try:
                configs.append([float(value) for value in raw_config])
            except (TypeError, ValueError):
                continue
        return configs

    def find_configs_for_pose(
        self,
        pose: list[float],
        constraints: int | None = None,
        include_rail: bool | None = None,
    ) -> list[list[float]]:
        include_rail = None
        find_configs = getattr(self.simIK, "findConfigs", None)
        if find_configs is None:
            self._set_error("simIK.findConfigs is not available")
            return []

        self._clear_error()
        env, group, _element, _sim_to_ik = self._ik_context(include_rail=include_rail)
        if constraints is not None:
            self._set_ik_constraint(constraints, include_rail=include_rail)
        self.simIK.syncFromSim(env, [group])
        self._sync_world_pose_to_ik_target(pose, include_rail=include_rail)
        ik_joints = [self._ik_handle(joint, include_rail=include_rail) for joint in self.active_joints(include_rail=include_rail)]
        params = {
            "maxDist": 0.28,
            "maxTime": 1.0,
            "findMultiple": True,
            "cMetric": self._find_configs_metric(include_rail=include_rail),
        }
        configs = self._normalize_ik_configs(
            find_configs(env, group, ik_joints, params),
            include_rail=include_rail,
        )
        if not configs:
            self._set_error(f"simIK.findConfigs returned no usable configs for pose {pose}")
        else:
            self._clear_error()
        return configs

    def _append_unique_config(self, configs: list[list[float]], config: list[float] | None) -> list[list[float]]:
        if config is None:
            return configs
        self._validate_motion_config(config)
        for existing in configs:
            if len(existing) == len(config) and all(abs(a - b) <= 1e-6 for a, b in zip(existing, config)):
                return configs
        return [*configs, list(config)]

    def _first_non_colliding_config(
        self,
        configs: list[list[float]],
        attached_object: int | None = None,
        check_collision: bool = True,
        include_rail: bool | None = None,
        target_pose: list[float] | None = None,
    ) -> list[float] | None:
        for config in configs:
            if target_pose is not None and not self._config_reaches_pose(
                config,
                target_pose,
                include_rail=include_rail,
            ):
                continue
            if check_collision and self._collides_motion_config(
                [config],
                attached_object=attached_object,
                include_rail=include_rail,
            ):
                continue
            return config
        return None

    def _collides_motion_config(
        self,
        configs: list[list[float]],
        attached_object: int | None = None,
        include_rail: bool | None = None,
    ) -> bool:
        if include_rail is None:
            return self.collides(configs, attached_object=attached_object)
        return self.collides(configs, attached_object=attached_object, include_rail=include_rail)

    def plan_pose_path(
        self,
        pose: list[float],
        attached_object: int | None = None,
        fallback_config: list[float] | None = None,
    ) -> list[float] | None:
        self._clear_error()
        configs = []
        for config in self.find_configs_for_pose(pose):
            if self._config_reaches_pose(config, pose):
                configs = self._append_unique_config(configs, config)
        if fallback_config is not None and self._config_reaches_pose(fallback_config, pose):
            configs = self._append_unique_config(configs, fallback_config)
        if not configs:
            solved_config = self.solve_ik_to_pose(pose, attached_object=attached_object, check_collision=False)
            configs = self._append_unique_config(configs, solved_config)
        if not configs:
            self._set_error("plan mode found no IK configs to try")
            return None

        rejected_collisions = 0
        planner_errors = []
        for config in configs:
            if self.collides([config], attached_object=attached_object):
                rejected_collisions += 1
                continue
            path = self.plan_joint_path(config, attached_object=attached_object)
            if path is not None:
                self._clear_error()
                return path
            if self.last_error:
                planner_errors.append(self.last_error)

        details = []
        if rejected_collisions:
            details.append(f"{rejected_collisions} IK config(s) collided")
        if planner_errors:
            details.append(planner_errors[-1])
        self._set_error("; ".join(details) if details else "plan mode found no exact collision-free path")
        return None

    def plan_joint_path(self, goal_config: list[float], attached_object: int | None = None) -> list[float] | None:
        self._validate_motion_config(goal_config)
        self._clear_error()
        sim = self.sim
        simOMPL = self.simOMPL
        active_joints = self.active_joints()
        self.update_robot_collection(attached_object=attached_object)
        task = simOMPL.createTask("python_rail_ur10_path_task")
        try:
            simOMPL.setAlgorithm(task, simOMPL.Algorithm.RRTConnect)
            projection = [1 if i <= 2 else 0 for i in range(len(active_joints))]
            simOMPL.setStateSpaceForJoints(task, active_joints, projection)
            simOMPL.setCollisionPairs(
                task,
                [
                    self.moving_collection,
                    self.obstacle_collection,
                    self.robot_collection,
                    self.robot_collection,
                ],
            )
            simOMPL.setStartState(task, self.get_motion_config())
            simOMPL.setGoalState(task, goal_config)
            simOMPL.setup(task)
            if not self._remote_success(simOMPL.solve(task, 10.0)):
                self._set_error("OMPL could not solve a path to the IK goal")
                return None
            has_exact_solution = getattr(simOMPL, "hasExactSolution", None)
            if has_exact_solution is not None and not self._remote_success(has_exact_solution(task)):
                self._set_error("OMPL returned no exact solution")
                return None
            simOMPL.simplifyPath(task, 10.0)
            path = simOMPL.getPath(task)
            if not path:
                self._set_error("OMPL returned an empty path")
                return None
            self._clear_error()
            return path
        finally:
            simOMPL.destroyTask(task)

    def follow_joint_path(self, path: list[float]) -> bool:
        sim = self.sim
        active_joints = self.active_joints()
        dof = len(active_joints)
        self._clear_error()
        if dof == 0:
            self._set_error("cannot follow joint path without robot joints")
            return False
        if not path:
            self._set_error("cannot follow an empty joint path")
            return False
        if len(path) % dof != 0:
            self._set_error(f"joint path length {len(path)} is not divisible by dof {dof}")
            return False

        rail_vel = 0.25
        rail_accel = 0.5
        arm_vel = math.pi
        arm_accel = 40 * math.pi / 180
        min_max_vel = []
        min_max_accel = []
        rail_joint = getattr(self.handles, "rail_joint", self.handles.joints[0] if self.handles.joints else None)
        rail_active = getattr(self, "rail_enabled", True) and active_joints and active_joints[0] == rail_joint
        for i in range(dof):
            vel = rail_vel if rail_active and i == 0 else arm_vel
            accel = rail_accel if rail_active and i == 0 else arm_accel
            min_max_vel.extend([-vel, vel])
            min_max_accel.extend([-accel, accel])

        path_lengths = self._path_lengths(sim.getPathLengths(path, dof))
        if not path_lengths:
            self._set_error("sim.getPathLengths returned no usable float path lengths")
            return False
        trajectory = sim.generateTimeOptimalTrajectory(
            path,
            path_lengths,
            min_max_vel,
            min_max_accel,
            1000,
            "not-a-knot",
            5,
        )
        if not isinstance(trajectory, (list, tuple)) or len(trajectory) < 2:
            self._set_error("sim.generateTimeOptimalTrajectory returned an invalid result")
            return False
        path_pts, times = trajectory[0], trajectory[1]
        if not path_pts or not times:
            self._set_error("sim.generateTimeOptimalTrajectory returned an empty trajectory")
            return False

        start_time = sim.getSimulationTime()
        while True:
            elapsed = sim.getSimulationTime() - start_time
            if elapsed >= times[-1]:
                break
            self.set_motion_target_config(sim.getPathInterpolatedConfig(path_pts, times, elapsed))
            self.ctx.step()
        self.set_motion_target_config(sim.getPathInterpolatedConfig(path_pts, times, times[-1]))
        self.ctx.step(10)
        self._clear_error()
        return True

    def solve_ik_to_pose(
        self,
        pose: list[float],
        attached_object: int | None = None,
        check_collision: bool = True,
        include_rail: bool | None = None,
    ) -> list[float] | None:
        include_rail = None
        self._clear_error()
        simIK = self.simIK
        h = self.handles
        previous_pose = self.sim.getObjectPose(h.ik_target)
        solved = False
        attempts = 0
        last_result = None
        last_collision_error = None
        try:
            for constraints in self._ik_constraint_candidates():
                for candidate_pose in self._ik_pose_candidates(pose):
                    configs = self.find_configs_for_pose(
                        candidate_pose,
                        constraints=constraints,
                        include_rail=include_rail,
                    )
                    config = self._first_non_colliding_config(
                        configs,
                        attached_object=attached_object,
                        check_collision=check_collision,
                        include_rail=include_rail,
                        target_pose=candidate_pose,
                    )
                    if config is not None:
                        solved = True
                        self._clear_error()
                        return config
                    if configs:
                        last_collision_error = f"all findConfigs solutions collide for pose {candidate_pose}"

            for constraints in self._ik_constraint_candidates():
                self._set_ik_constraint(constraints, include_rail=include_rail)
                for candidate_pose in self._ik_pose_candidates(pose):
                    env, group, _element, _sim_to_ik = self._ik_context(include_rail=include_rail)
                    simIK.syncFromSim(env, [group])
                    self._sync_world_pose_to_ik_target(candidate_pose, include_rail=include_rail)
                    for damping in self._ik_damping_candidates():
                        self._set_ik_calculation(damping, include_rail=include_rail)
                        result = simIK.handleGroup(env, group)
                        attempts += 1
                        result_code = result[0] if isinstance(result, (list, tuple)) else result
                        last_result = result_code
                        if result_code != simIK.result_success:
                            continue
                        config = [
                            simIK.getJointPosition(env, self._ik_handle(joint, include_rail=include_rail))
                            for joint in self.active_joints(include_rail=include_rail)
                        ]
                        if not self._config_reaches_pose(config, candidate_pose, include_rail=include_rail):
                            continue
                        if check_collision and self._collides_motion_config(
                            [config],
                            attached_object=attached_object,
                            include_rail=include_rail,
                        ):
                            last_collision_error = f"IK solution collides for pose {candidate_pose}"
                            continue
                        solved = True
                        self._clear_error()
                        return config
            if last_collision_error is not None:
                self._set_error(last_collision_error)
            else:
                self._set_error(f"IK did not converge for pose {pose} after {attempts} attempts (last_result={last_result})")
            return None
        finally:
            if not solved:
                self.sim.setObjectPose(h.ik_target, previous_pose)

    def validate_cartesian_waypoints(
        self,
        target_pose: list[float],
        attached_object: int | None = None,
        include_rail: bool = False,
        max_step: float = 0.03,
    ) -> bool:
        start_pose = self.sim.getObjectPose(self.handles.ik_tip)
        for index, waypoint in enumerate(interpolate_pose_xyz(start_pose, target_pose, max_step=max_step)):
            if self.solve_ik_to_pose(waypoint, attached_object=attached_object, include_rail=include_rail) is None:
                detail = self.last_error
                self._set_error_with_detail(f"Cartesian waypoint {index} IK failed", detail)
                return False
        self._clear_error()
        return True

    def move_cartesian_linear(
        self,
        target_pose: list[float],
        attached_object: int | None = None,
        include_rail: bool = False,
    ) -> bool:
        include_rail = None
        self._clear_error()
        self.update_robot_collection(attached_object=attached_object)
        ik_joints = self.active_joints(include_rail=include_rail)
        start_pose = self.sim.getObjectPose(self.handles.ik_tip)
        self.sim.setObjectPose(self.handles.ik_target, start_pose)
        motion = self.sim.moveToPose_init(
            {
                "ik": {
                    "tip": self.handles.ik_tip,
                    "target": self.handles.ik_target,
                    "base": self._cartesian_ik_base(include_rail=include_rail),
                    "joints": ik_joints,
                    "method": self.simIK.method_damped_least_squares,
                    "damping": 0.3,
                    "iterations": 99,
                    "constraints": self.simIK.constraint_pose,
                    "precision": [0.001, 0.5 * math.pi / 180.0],
                },
                "targetPose": target_pose,
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
                    self._set_error("sim.moveToPose_step aborted")
                    return False
                collision_reason = self._motion_collision_reason()
                if collision_reason is not None:
                    self._set_error(f"collision during Cartesian move: {collision_reason}")
                    return False
                if result == 1:
                    self.sim.setObjectPose(self.handles.ik_target, target_pose)
                    self.ctx.step(10)
                    if not self._validate_tip_reached_pose(target_pose):
                        return False
                    self._clear_error()
                    return True
                self.ctx.step()
            self._set_error("sim.moveToPose_step exceeded 10000 controller steps")
            return False
        finally:
            self.sim.moveToPose_cleanup(motion)
