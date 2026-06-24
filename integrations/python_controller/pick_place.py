from __future__ import annotations
from .robot import RobotController


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


def world_pose(sim: object, handle: int) -> list[float]:
    return sim.getObjectPose(handle, getattr(sim, "handle_world", -1))


class PickPlaceController:
    def __init__(self, robot: RobotController, belt: object):
        self.robot = robot
        self.belt = belt
        self.sim = robot.sim
        self.attached_object: int | None = None
        self.approach = 0.10
        self.contact_offset = 0.005
        self.hold_clearance = 0.50
        self.last_error: str | None = None

    def _clear_error(self) -> None:
        self.last_error = None

    def _fail(self, reason: str, detail: str | None = None) -> bool:
        self.last_error = f"{reason}: {detail}" if detail else reason
        return False

    def _robot_error(self) -> str | None:
        return getattr(self.robot, "last_error", None)

    def _tip_reached_pose(self, pose: list[float]) -> bool:
        validate = getattr(self.robot, "_validate_tip_reached_pose", None)
        if validate is None:
            return True
        return bool(validate(pose))

    def _optional_object(self, path: str) -> int:
        get_object = getattr(self.sim, "getObject", None)
        if get_object is None:
            return -1
        try:
            handle = get_object(path, {"noError": True})
        except TypeError:
            try:
                handle = get_object(path)
            except Exception:
                return -1
        except Exception:
            return -1
        return handle if isinstance(handle, int) and handle >= 0 else -1

    def loading_home_pose(self) -> list[float] | None:
        handle = self._optional_object("/loading_home_pose")
        if handle < 0:
            return None
        return world_pose(self.sim, handle)

    def _try_cartesian_then_linear_fallback(
        self,
        pose: list[float],
        attached_object: int | None,
        primary_reason: str,
        primary_error: str | None,
        try_linear: bool = True,
    ) -> bool:
        if self.robot.move_cartesian_linear(pose, attached_object=attached_object):
            self._clear_error()
            return True
        cartesian_error = self._robot_error()
        detail = primary_error
        if cartesian_error:
            detail = f"{primary_error}; Cartesian fallback failed: {cartesian_error}" if primary_error else cartesian_error
        return self._fail(primary_reason, detail)

    def linear_move_to_pose(
        self,
        pose: list[float],
        attached_object: int | None = None,
        reason: str = "linear move failed",
    ) -> bool:
        self._clear_error()
        if self.robot.move_cartesian_linear(pose, attached_object=attached_object):
            self._clear_error()
            return True
        linear_error = self._robot_error()
        if self.move_to_pose(pose, attached_object=attached_object, try_linear_fallback=False):
            self._clear_error()
            return True
        planned_error = self.last_error
        detail = linear_error
        if planned_error:
            detail = f"{linear_error}; planned fallback failed: {planned_error}" if linear_error else planned_error
        return self._fail(reason, detail)

    def pick_poses_for_object(self, handle: int) -> tuple[list[float], list[float], list[float]]:
        pose = world_pose(self.sim, handle)
        size = object_size(self.sim, handle)
        rotated = self.sim.multiplyPoses(pose, [0, 0, 0, 0, 1, 0, 0])
        top_z = pose[2] + size[2] * 0.5
        pick = [pose[0], pose[1], top_z + self.contact_offset, *rotated[3:]]
        pre = list(pick)
        pre[2] = pick[2] + self.approach
        hold_z = pose[2] + size[2] + self.hold_clearance
        hold = [pose[0], pose[1], hold_z, *rotated[3:]]
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
        orientation = [0, 1, 0, 0]
        place = self.sim.multiplyPoses(pallet_pose, [center[0], center[1], center[2], *orientation])
        pre = list(place)
        pre[2] = place[2] + self.approach
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

    def move_to_pose(
        self,
        pose: list[float],
        attached_object: int | None = None,
        try_linear_fallback: bool = True,
        allow_cartesian_fallback: bool = True,
        allow_start_collision: bool = False,
    ) -> bool:
        self._clear_error()
        goal = self.robot.solve_ik_to_pose(pose, attached_object=attached_object)
        if goal is None:
            if not allow_cartesian_fallback:
                return self._fail("IK failed while moving to pose", self._robot_error())
            return self._try_cartesian_then_linear_fallback(
                pose,
                attached_object,
                "IK failed while moving to pose",
                self._robot_error(),
                try_linear=try_linear_fallback,
            )
        if self.robot.move_to_config(
            goal,
            attached_object=attached_object,
            allow_start_collision=allow_start_collision,
        ):
            if self._tip_reached_pose(pose):
                self._clear_error()
                return True
            config_error = self._robot_error()
        else:
            config_error = self._robot_error()
        path = self.robot.plan_pose_path(pose, attached_object=attached_object, fallback_config=goal)
        if path is None:
            plan_error = self._robot_error()
            detail = config_error
            if plan_error:
                detail = f"{config_error}; plan mode failed: {plan_error}" if config_error else plan_error
            if not allow_cartesian_fallback:
                return self._fail("joint planning failed while moving to pose", detail)
            return self._try_cartesian_then_linear_fallback(
                pose,
                attached_object,
                "joint planning failed while moving to pose",
                detail,
                try_linear=try_linear_fallback,
            )
        if not self.robot.follow_joint_path(path):
            follow_error = self._robot_error()
            detail = config_error
            if follow_error:
                detail = f"{config_error}; plan mode failed: {follow_error}" if config_error else follow_error
            if not allow_cartesian_fallback:
                return self._fail("joint path following failed while moving to pose", detail)
            return self._try_cartesian_then_linear_fallback(
                pose,
                attached_object,
                "joint path following failed while moving to pose",
                detail,
                try_linear=try_linear_fallback,
            )
        if not self._tip_reached_pose(pose):
            reached_error = self._robot_error()
            detail = config_error
            if reached_error:
                detail = f"{config_error}; plan mode failed: {reached_error}" if config_error else reached_error
            if not allow_cartesian_fallback:
                return self._fail("joint path following failed while moving to pose", detail)
            return self._try_cartesian_then_linear_fallback(
                pose,
                attached_object,
                "joint path following failed while moving to pose",
                detail,
                try_linear=try_linear_fallback,
            )
        self._clear_error()
        return True

    def move_to_reachable_pose(
        self,
        pose: list[float],
        attached_object: int | None = None,
    ) -> bool:
        if not self.robot.preposition_rail_for_pose(pose):
            return self._fail("rail pre-position failed", self._robot_error())
        return self.move_to_pose(pose, attached_object=attached_object)

    def preserve_current_orientation(self, pose: list[float]) -> list[float]:
        try:
            current_pose = world_pose(self.sim, self.robot.handles.ik_tip)
        except Exception:
            return pose
        if not isinstance(current_pose, (list, tuple)) or len(current_pose) < 7:
            return pose
        return [*pose[:3], *current_pose[3:7]]

    def pick(self, handle: int) -> bool:
        self._clear_error()
        pre, pick, hold = self.pick_poses_for_object(handle)
        if not self.move_to_reachable_pose(pre):
            return self._fail("pick pre-approach failed", self.last_error)
        if not self.linear_move_to_pose(pick, reason="pick descend failed"):
            return False

        self.attach_object(handle)
        self.robot.ctx.step(20)
        if not self.linear_move_to_pose(hold, attached_object=handle, reason="pick lift failed"):
            detail = self.last_error
            self.release_object()
            return self._fail("pick lift failed", detail)
        self._clear_error()
        return True

    def place(self, position: list[float] | None = None) -> bool:
        self._clear_error()
        if self.attached_object is None:
            return self._fail("place called without an attached object")

        position = [0.0, 0.0, 1.0] if position is None else position
        attached_object = self.attached_object
        pre, place = self.place_poses_for_object(attached_object, position)
        transfer_pre = self.preserve_current_orientation(pre)
        if not self.move_to_reachable_pose(transfer_pre, attached_object=attached_object):
            return self._fail("place pre-approach failed", self.last_error)
        if not self.linear_move_to_pose(place, attached_object=attached_object, reason="place descend failed"):
            return False

        self.release_object()
        self.robot.ctx.step(20)
        if not self.linear_move_to_pose(pre, reason="place retract failed"):
            return False
        self._clear_error()
        return True
