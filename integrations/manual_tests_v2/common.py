from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python_controller_v2.remote import connect, ensure_started
from python_controller_v2.robot import RobotControllerV2


class ManualTestFailure(RuntimeError):
    pass


@dataclass
class LiveRobot:
    ctx: object
    sim: object
    robot: RobotControllerV2


_CLEANUP_CONTEXTS: list[object] = []


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--scene", default=None, help="Optional scene path to load before running the test")
    parser.add_argument("--start", dest="start", action="store_true", default=True, help="Restart/start before running")
    parser.add_argument("--no-start", dest="start", action="store_false", help="Attach without restarting")
    parser.add_argument("--keep-running", action="store_true", help="Leave a simulation started by this script running")
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=2.0)
    return parser


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
            raise TimeoutError("Timed out waiting for v2 manual test simulation to stop")
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
    robot = RobotControllerV2(ctx)
    return LiveRobot(ctx=ctx, sim=ctx.sim, robot=robot)


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


def check_pose_error(label: str, actual: list[float], target: list[float], args: argparse.Namespace) -> None:
    position_error, orientation_error = pose_error(actual, target)
    print(f"{label} position_error_m: {position_error:.9f}")
    print(f"{label} orientation_error_deg: {math.degrees(orientation_error):.9f}")
    if position_error > args.position_tolerance:
        raise ManualTestFailure(f"{label} position error {position_error:.6f} exceeds {args.position_tolerance:.6f}")
    orientation_tolerance = math.radians(args.orientation_tolerance_deg)
    if orientation_error > orientation_tolerance:
        raise ManualTestFailure(
            f"{label} orientation error {math.degrees(orientation_error):.6f} deg exceeds "
            f"{args.orientation_tolerance_deg:.6f} deg"
        )


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

