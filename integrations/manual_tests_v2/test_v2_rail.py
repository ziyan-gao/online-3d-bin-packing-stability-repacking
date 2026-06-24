from __future__ import annotations

from common import build_parser, create_robot, run


def main() -> None:
    parser = build_parser("Test python_controller_v2 rail motion only.")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    live = create_robot(args)
    robot = live.robot
    start = robot.get_rail_position()
    target = start + args.delta

    print(f"start_rail: {start:.9f}")
    print(f"target_rail: {target:.9f}")
    ok = robot.move_rail_to(target)
    print(f"rail_ok: {ok}")
    print(f"rail_error: {robot.last_error}")
    print(f"final_rail: {robot.get_rail_position():.9f}")
    if not ok:
        raise RuntimeError(robot.last_error or "move_rail_to returned False")

    if args.restore:
        robot.move_rail_to(start)


if __name__ == "__main__":
    run(main)

