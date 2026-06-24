from __future__ import annotations

import logging
import math
import random

from common import ManualTestFailure, build_parser, create_robot, print_config, print_pose, run
from python_controller.belt import BeltMonitor
from python_controller_v2.pick_place import PickPlaceControllerV2
from python_controller_v2.robot import quaternion_to_roll_pitch_yaw


def get_object(sim: object, path: str) -> int:
    try:
        handle = sim.getObject(path, {"noError": True})
    except TypeError:
        handle = sim.getObject(path)
    except Exception as exc:
        raise ManualTestFailure(f"object not found: {path}") from exc
    if not isinstance(handle, int) or handle < 0:
        raise ManualTestFailure(f"object not found: {path}")
    return handle


def configure_controller(controller: PickPlaceControllerV2, args) -> None:
    controller.approach = args.approach
    controller.hold_clearance = args.hold_clearance
    controller.reach_half_width = args.reach_half_width


def print_pick_state(controller: PickPlaceControllerV2, handle: int) -> None:
    pre_pick, pick, hold = controller.pick_poses_for_object(handle)
    print_pose("pre_pick", pre_pick)
    print_pose("pick", pick)
    print_pose("hold", hold)


def print_relative_target(label: str, robot: object, pose: list[float]) -> None:
    base_pose = robot.sim.getObjectPose(robot.handles.arm_base)
    roll, pitch, yaw = quaternion_to_roll_pitch_yaw(pose[3:7])
    print_pose(f"{label}_arm_base", base_pose)
    print(
        f"{label}_target_relative_xyz: "
        f"[{pose[0] - base_pose[0]:.9f}, {pose[1] - base_pose[1]:.9f}, {pose[2] - base_pose[2]:.9f}]"
    )
    print(
        f"{label}_target_rpy_deg: "
        f"[{math.degrees(roll):.6f}, {math.degrees(pitch):.6f}, {math.degrees(yaw):.6f}]"
    )


def print_place_state(controller: PickPlaceControllerV2, handle: int, place_position: list[float]) -> None:
    pre_place, place = controller.place_poses_for_object(handle, place_position)
    print_pose("pre_place", pre_place)
    print_pose("place", place)


def run_pick_place(
    controller: PickPlaceControllerV2,
    robot: object,
    handle: int,
    place_position: list[float],
    pick_only: bool,
    belt: BeltMonitor | None = None,
) -> bool:
    pre_pick, _pick, _hold = controller.pick_poses_for_object(handle)
    print_pick_state(controller, handle)
    print_config("arm_config_before_pick", robot.get_arm_config())
    print(f"rail_before_pick: {robot.get_rail_position():.9f}")
    print_relative_target("before_pick", robot, pre_pick)

    pick_ok = controller.pick(handle)
    print(f"pick_ok: {pick_ok}")
    print(f"pick_error: {controller.last_error}")
    print_config("arm_config_after_pick", robot.get_arm_config())
    print(f"rail_after_pick: {robot.get_rail_position():.9f}")
    print_relative_target("after_pick", robot, pre_pick)
    if pick_ok and belt is not None:
        belt.set_robot_picking(False)
    if not pick_ok or pick_only:
        return pick_ok

    print_place_state(controller, handle, place_position)
    place_ok = controller.place(place_position)
    print(f"place_ok: {place_ok}")
    print(f"place_error: {controller.last_error}")
    print_config("arm_config_after_place", robot.get_arm_config())
    return place_ok


def main() -> None:
    parser = build_parser("Test python_controller_v2 pick/place pipeline.")
    parser.add_argument("--object-path", default=None, help="Bypass the belt and pick this object path directly")
    parser.add_argument("--place", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--approach", type=float, default=0.050)
    parser.add_argument("--hold-clearance", type=float, default=0.50)
    parser.add_argument("--reach-half-width", type=float, default=0.5)
    parser.add_argument("--pick-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    live = create_robot(args)
    robot = live.robot
    sim = live.sim
    controller = PickPlaceControllerV2(robot)
    configure_controller(controller, args)
    place_position = list(args.place)

    if args.object_path is not None:
        handle = get_object(sim, args.object_path)
        ok = run_pick_place(controller, robot, handle, place_position, args.pick_only)
        if not ok:
            raise ManualTestFailure(controller.last_error or "pick/place returned False")
        return

    belt = BeltMonitor(sim)
    cycles = 0
    while sim.getSimulationState() != sim.simulation_advancing_abouttostop:
        items = belt.full_transition_items()
        if items:
            item = random.choice(items)
            logging.info("selected Cuboid handle=%s", item.handle)
            belt.set_robot_picking(True)
            ok = False
            try:
                ok = run_pick_place(controller, robot, item.handle, place_position, args.pick_only, belt=belt)
                reason = getattr(controller, "last_error", None)
                logging.info("v2 pick/place handle=%s ok=%s reason=%s", item.handle, ok, reason)
                if not ok and getattr(controller, "attached_object", None) is not None:
                    raise ManualTestFailure(
                        f"pick/place failed with object still attached after hold pose belt release; reason={reason}"
                    )
                if not ok:
                    raise ManualTestFailure(reason or "pick/place returned False")
                cycles += 1
            finally:
                if getattr(controller, "attached_object", None) is None and not ok:
                    belt.set_robot_picking(False)
                elif getattr(controller, "attached_object", None) is not None and not ok:
                    logging.error("object is still attached after hold pose belt release")
        if args.max_cycles and cycles >= args.max_cycles:
            break
        live.ctx.step()


if __name__ == "__main__":
    run(main)
