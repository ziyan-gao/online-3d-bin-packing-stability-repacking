from __future__ import annotations

from .robot import RobotControllerV2


def object_size(sim: object, handle: int) -> list[float]:
    try:
        bb = sim.getShapeBB(handle)
        if isinstance(bb, list) and len(bb) >= 3:
            return [abs(float(bb[0])), abs(float(bb[1])), abs(float(bb[2]))]
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
    return [abs(float(maxs[index]) - float(mins[index])) for index in range(3)]


def world_pose(sim: object, handle: int) -> list[float]:
    return sim.getObjectPose(handle, getattr(sim, "handle_world", -1))


class PickPlaceControllerV2:
    """Simple v2 pick/place pipeline built on arm-only IK."""

    def __init__(self, robot: RobotControllerV2):
        self.robot = robot
        self.sim = robot.sim
        self.attached_object: int | None = None
        self.pose_before_pre_pick: list[float] | None = None
        self.saved_arm_configs: dict[str, list[float]] = {}
        self.approach = 0.10
        self.contact_offset = 0.005
        self.hold_clearance = 0.50
        self.reach_half_width = 0.50
        self.pallet_boundary_offset = 0.7
        self.last_error: str | None = None

    def _clear_error(self) -> None:
        self.last_error = None

    def _fail(self, reason: str, detail: str | None = None) -> bool:
        self.last_error = f"{reason}: {detail}" if detail else reason
        return False

    def _robot_error(self) -> str | None:
        return getattr(self.robot, "last_error", None)

    def _remember_arm_config(self, name: str) -> list[float]:
        config = self.robot.get_arm_config()
        self.saved_arm_configs[name] = list(config)
        remember = getattr(self.robot, "remember_arm_config", None)
        if remember is not None:
            remember(name, config)
        return list(config)

    def _preposition_rail_for_pose(self, pose: list[float]) -> bool:
        base_pose = self.sim.getObjectPose(self.robot.handles.arm_base)
        relative_x = pose[0] - base_pose[0]
        if abs(relative_x) <= self.reach_half_width:
            self._clear_error()
            return True

        if relative_x > self.reach_half_width:
            base_shift_x = relative_x - self.reach_half_width
        else:
            base_shift_x = relative_x + self.reach_half_width
        target_rail = self.robot.get_rail_position() - base_shift_x
        if self.robot.move_rail_to(target_rail):
            self._clear_error()
            return True
        return self._fail("rail pre-position failed", self._robot_error())

    def _preposition_rail_for_place(self) -> bool:
        base_pose = world_pose(self.sim, self.robot.handles.arm_base)
        tip_pose = world_pose(self.sim, self.robot.handles.ik_tip)
        tip_side_x = tip_pose[0] - base_pose[0]
        if abs(tip_side_x) <= 1.0e-9:
            self._clear_error()
            return True

        pallet_pose = world_pose(self.sim, self.robot.handles.pallet)
        pallet_width_x = object_size(self.sim, self.robot.handles.pallet)[0]
        pallet_min_x = pallet_pose[0] - pallet_width_x * 0.5
        pallet_max_x = pallet_pose[0] + pallet_width_x * 0.5
        if tip_side_x > 0.0:
            target_base_x = pallet_min_x - self.pallet_boundary_offset
        else:
            target_base_x = pallet_max_x + self.pallet_boundary_offset

        base_shift_x = target_base_x - base_pose[0]
        target_rail = self.robot.get_rail_position() - base_shift_x
        if self.robot.move_rail_to(target_rail):
            self._clear_error()
            return True
        return self._fail("rail placement pre-position failed", self._robot_error())

    def _move_tip_to_pose(
        self,
        pose: list[float],
        reason: str,
        attached_object: int | None = None,
        ignored_objects: list[int] | tuple[int, ...] | None = None,
        allow_final_contact: bool = False,
        fallback_config: list[float] | None = None,
    ) -> bool:
        if not self.robot.move_arm_linear_to_pose(
            pose,
            attached_object=attached_object,
            ignored_objects=ignored_objects,
            allow_final_contact=allow_final_contact,
        ):
            linear_error = self._robot_error()
            config_error = None
            if fallback_config is not None:
                move_to_config = getattr(self.robot, "move_arm_ompl_to_config", None)
                if move_to_config is not None and move_to_config(
                    fallback_config,
                    attached_object=attached_object,
                ):
                    self._clear_error()
                    return True
                config_error = self._robot_error()
            if self.robot.move_arm_ompl_to_pose(
                pose,
                attached_object=attached_object,
                ignored_objects=ignored_objects,
                allow_final_contact=allow_final_contact,
            ):
                self._clear_error()
                return True
            ompl_error = self._robot_error()
            detail = f"linear failed: {linear_error}"
            if config_error is not None:
                detail += f"; config OMPL failed: {config_error}"
            detail += f"; OMPL failed: {ompl_error}"
            return self._fail(reason, detail)
        self._clear_error()
        return True

    def pick_poses_for_object(self, handle: int) -> tuple[list[float], list[float], list[float]]:
        pose = world_pose(self.sim, handle)
        size = object_size(self.sim, handle)
        rotated = self.sim.multiplyPoses(pose, [0, 0, 0, 0, 1, 0, 0])
        top_z = pose[2] + size[2] * 0.5
        pick = [pose[0], pose[1], top_z + self.contact_offset, *rotated[3:7]]
        pre = list(pick)
        pre[2] = pick[2] + self.approach
        hold = [pose[0], pose[1], pose[2] + size[2] + self.hold_clearance, *rotated[3:7]]
        return pre, pick, hold

    def place_poses_for_object(self, handle: int, position: list[float] | None) -> tuple[list[float], list[float]]:
        position = [0.0, 0.0, 1.0] if position is None else position
        pallet_pose = world_pose(self.sim, self.robot.handles.pallet)
        size = object_size(self.sim, handle)
        center = [
            position[0] + size[0] * 0.5,
            position[1] + size[1] * 0.5,
            position[2] + size[2] * 0.5,
        ]
        local_place = [center[0], center[1], center[2], 0, 1, 0, 0]
        local_pre = list(local_place)
        local_pre[2] = local_place[2] + self.approach
        place = self.sim.multiplyPoses(pallet_pose, local_place)
        pre = self.sim.multiplyPoses(pallet_pose, local_pre)
        return pre, place

    def set_suction(self, enabled: bool) -> None:
        self.sim.setInt32Signal("suctionPadEnabled", 1 if enabled else 0)

    def attach_object(self, handle: int) -> None:
        self.set_suction(True)
        self.sim.setObjectParent(handle, self.robot.handles.ik_tip, True)
        self.attached_object = handle

    def release_object(self) -> None:
        if self.attached_object is None:
            self.set_suction(False)
            return
        handle = self.attached_object
        self.set_suction(False)
        self.sim.setObjectParent(handle, -1, True)
        reset_dynamic = getattr(self.sim, "resetDynamicObject", None)
        if reset_dynamic is not None:
            reset_dynamic(handle)
        self.attached_object = None

    def pick(self, handle: int) -> bool:
        self._clear_error()
        pre, pick, hold = self.pick_poses_for_object(handle)
        self._remember_arm_config("initial")
        if not self._preposition_rail_for_pose(pre):
            return False

        self.pose_before_pre_pick = world_pose(self.sim, self.robot.handles.ik_tip)
        if not self._move_tip_to_pose(pre, "pick pre-approach failed", ignored_objects=[handle]):
            return False
        self._remember_arm_config("pre_pick")
        if not self._move_tip_to_pose(pick, "pick descend failed", ignored_objects=[handle]):
            return False
        self._remember_arm_config("pick")

        self.attach_object(handle)
        if not self._move_tip_to_pose(hold, "pick lift failed", attached_object=handle):
            detail = self.last_error
            self.release_object()
            return self._fail("pick lift failed", detail)
        self._remember_arm_config("hold")
        self._clear_error()
        return True

    def place(self, position: list[float] | None = None) -> bool:
        self._clear_error()
        if self.attached_object is None:
            return self._fail("place called without an attached object")
        if self.pose_before_pre_pick is None:
            return self._fail("place called without saved pose before pre-pick")

        attached_object = self.attached_object
        if not self._preposition_rail_for_place():
            return False
        pre, place = self.place_poses_for_object(attached_object, position)
        if not self._move_tip_to_pose(pre, "place pre-approach failed", attached_object=attached_object):
            return False
        self._remember_arm_config("pre_place")
        if not self._move_tip_to_pose(
            place,
            "place descend failed",
            attached_object=attached_object,
            allow_final_contact=True,
        ):
            return False
        self._remember_arm_config("place")

        self.release_object()
        if not self._move_tip_to_pose(pre, "place retract failed", ignored_objects=[attached_object]):
            return False
        self._remember_arm_config("pre_place")

        self.pose_before_pre_pick = None
        self._clear_error()
        return True
