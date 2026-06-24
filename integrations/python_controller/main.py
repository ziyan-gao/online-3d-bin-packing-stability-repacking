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
    parser.add_argument("--disable-rail", action="store_true", help="Keep the rail joint fixed during robot motion")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    ctx = connect(port=args.port)
    ensure_started(ctx, scene_path=args.scene)
    belt = BeltMonitor(ctx.sim)
    robot = RobotController(ctx, rail_enabled=not args.disable_rail)
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
                reason = getattr(pick_place, "last_error", None)
                logging.info("pick/place handle=%s ok=%s reason=%s", item.handle, ok, reason)
                if not ok and getattr(pick_place, "attached_object", None) is not None:
                    raise RuntimeError(
                        f"pick/place failed with object still attached; leaving robotPicking=1; reason={reason}"
                    )
                cycles += 1
            finally:
                if getattr(pick_place, "attached_object", None) is None:
                    belt.set_robot_picking(False)
                else:
                    logging.error("leaving robotPicking=1 because object is still attached")
        if args.max_cycles and cycles >= args.max_cycles:
            break
        ctx.step()


if __name__ == "__main__":
    main()
