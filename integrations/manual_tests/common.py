from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python_controller.belt import BeltMonitor
from python_controller.pick_place import PickPlaceController
from python_controller.remote import connect, ensure_started
from python_controller.robot import RobotController


DEFAULT_OBJECT_ALIASES = [
    "/Cuboid",
    ":/Cuboid",
    "/Cuboid[0]",
    ":/Cuboid[0]",
    "/Item",
    ":/Item",
    "/box",
    ":/box",
]
_CLEANUP_CONTEXTS: list[object] = []


class ManualTestFailure(RuntimeError):
    pass


@dataclass
class LiveRobot:
    ctx: object
    sim: object
    robot: RobotController


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--scene", default=None, help="Optional scene path to load before running the test")
    parser.add_argument(
        "--start",
        dest="start",
        action="store_true",
        default=True,
        help="Restart/start the current scene before running (default)",
    )
    parser.add_argument(
        "--no-start",
        dest="start",
        action="store_false",
        help="Attach to the current simulation state without restarting",
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Leave a simulation started by this script running after Python exits",
    )
    parser.add_argument("--disable-rail", action="store_true", help="Keep the rail fixed; arm IK remains arm-only")
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=2.0)
    return parser


def add_delta_args(parser: argparse.ArgumentParser, *, dz: float = 0.05) -> None:
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=dz)


def add_joint_goal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--joint-index", type=int, default=0, help="Index in robot.active_joints(), not scene joint handle")
    parser.add_argument("--delta-deg", type=float, default=5.0)
    parser.add_argument("--follow", action="store_true", help="Follow the planned path after planning")


def add_object_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--object-handle", type=int, default=None)
    parser.add_argument("--object-path", default=None)


def add_place_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=1.0)


def resolve_scene_path(scene_path: str | None) -> str | None:
    if scene_path is None:
        return None
    path = Path(scene_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise ManualTestFailure(f"scene file does not exist: {path}")
    return str(path)


def register_simulation_cleanup(ctx: object) -> None:
    _CLEANUP_CONTEXTS.append(ctx)


def stop_simulation(ctx: object, timeout_s: float = 10.0) -> None:
    sim = ctx.sim
    if sim.getSimulationState() == sim.simulation_stopped:
        return
    sim.stopSimulation()
    deadline = time.time() + timeout_s
    while sim.getSimulationState() != sim.simulation_stopped:
        if time.time() >= deadline:
            raise TimeoutError("Timed out waiting for manual test simulation to stop")
        try:
            ctx.step()
        except Exception:
            time.sleep(0.05)


def stop_registered_simulations() -> None:
    while _CLEANUP_CONTEXTS:
        stop_simulation(_CLEANUP_CONTEXTS.pop())


def create_robot(args: argparse.Namespace) -> LiveRobot:
    ctx = connect(port=args.port)
    scene_path = resolve_scene_path(args.scene)
    started_by_script = bool(scene_path or args.start)
    if started_by_script:
        ensure_started(ctx, scene_path=scene_path)
    if started_by_script and not args.keep_running:
        register_simulation_cleanup(ctx)
    robot = RobotController(ctx, rail_enabled=not args.disable_rail)
    return LiveRobot(ctx=ctx, sim=ctx.sim, robot=robot)


def make_pick_place(robot: RobotController) -> PickPlaceController:
    return PickPlaceController(robot=robot, belt=object())


def shifted_pose(pose: list[float], dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> list[float]:
    return [pose[0] + dx, pose[1] + dy, pose[2] + dz, *pose[3:7]]


def pose_error(actual: list[float], target: list[float]) -> tuple[float, float]:
    position_error = math.dist(actual[:3], target[:3])
    if len(actual) < 7 or len(target) < 7:
        return position_error, 0.0
    dot = sum(actual[i] * target[i] for i in range(3, 7))
    dot = max(-1.0, min(1.0, abs(dot)))
    return position_error, 2.0 * math.acos(dot)


def print_pose(label: str, pose: list[float]) -> None:
    print(f"{label}: {[round(value, 9) for value in pose]}")


def print_config(label: str, config: list[float]) -> None:
    print(f"{label}: {[round(value, 9) for value in config]}")


def check_pose_error(
    label: str,
    actual: list[float],
    target: list[float],
    args: argparse.Namespace,
) -> tuple[float, float]:
    position_error, orientation_error = pose_error(actual, target)
    print(f"{label} position_error_m: {position_error:.9f}")
    print(f"{label} orientation_error_deg: {math.degrees(orientation_error):.9f}")
    if position_error > args.position_tolerance:
        raise ManualTestFailure(
            f"{label} position error {position_error:.6f} exceeds tolerance {args.position_tolerance:.6f}"
        )
    orientation_tolerance = math.radians(args.orientation_tolerance_deg)
    if orientation_error > orientation_tolerance:
        raise ManualTestFailure(
            f"{label} orientation error {math.degrees(orientation_error):.6f} deg exceeds tolerance "
            f"{args.orientation_tolerance_deg:.6f} deg"
        )
    return position_error, orientation_error


def check_config_error(label: str, actual: list[float], target: list[float], tolerance: float = 1e-4) -> float:
    if len(actual) != len(target):
        raise ManualTestFailure(f"{label} config length mismatch: actual={len(actual)} target={len(target)}")
    max_error = max((abs(a - b) for a, b in zip(actual, target)), default=0.0)
    print(f"{label} max_config_error_rad_or_m: {max_error:.9f}")
    if max_error > tolerance:
        raise ManualTestFailure(f"{label} max config error {max_error:.6f} exceeds tolerance {tolerance:.6f}")
    return max_error


def goal_from_joint_delta(robot: RobotController, joint_index: int, delta_deg: float) -> list[float]:
    config = robot.get_motion_config()
    if not config:
        raise ManualTestFailure("robot has no active joints")
    if joint_index < 0 or joint_index >= len(config):
        raise ManualTestFailure(f"joint-index {joint_index} is outside active joint range 0..{len(config) - 1}")
    goal = list(config)
    goal[joint_index] += math.radians(delta_deg)
    return goal


def resolve_path(sim: object, path: str) -> int:
    try:
        handle = sim.getObject(path, {"noError": True})
    except TypeError:
        handle = sim.getObject(path)
    if handle is None or handle < 0:
        raise ManualTestFailure(f"could not resolve object path {path!r}")
    return int(handle)


def tracked_item_handles(sim: object) -> list[int]:
    try:
        return [item.handle for item in BeltMonitor(sim).read_full_items()]
    except Exception:
        return []


def handles_in_tree(sim: object, root: int) -> list[int]:
    get_objects_in_tree = getattr(sim, "getObjectsInTree", None)
    if get_objects_in_tree is None:
        return []
    object_types = [
        getattr(sim, "sceneobject_shape", None),
        getattr(sim, "handle_all", None),
    ]
    seen = []
    for object_type in object_types:
        if object_type is None:
            continue
        try:
            for handle in get_objects_in_tree(root, object_type):
                if isinstance(handle, int) and handle >= 0 and handle not in seen:
                    seen.append(handle)
        except Exception:
            continue
    return seen


def resolve_test_object(
    sim: object,
    args: argparse.Namespace,
    aliases: Iterable[str] = DEFAULT_OBJECT_ALIASES,
    attached_to_tip: int | None = None,
) -> int:
    if args.object_handle is not None:
        return args.object_handle
    if args.object_path:
        return resolve_path(sim, args.object_path)
    if attached_to_tip is not None:
        attached = handles_in_tree(sim, attached_to_tip)
        if attached:
            return attached[0]
    tracked = tracked_item_handles(sim)
    if tracked:
        return tracked[0]
    for alias in aliases:
        try:
            return resolve_path(sim, alias)
        except ManualTestFailure:
            continue
    raise ManualTestFailure("could not find a test object; pass --object-handle or --object-path")


def run(main_func) -> None:
    try:
        main_func()
    except ManualTestFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        stop_registered_simulations()
    print("PASS")
