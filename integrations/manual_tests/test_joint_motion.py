from __future__ import annotations

import math

from common import add_joint_goal_args, build_parser, check_config_error, create_robot, print_config, run


def main() -> None:
    parser = build_parser("Move directly to a nearby joint-space config.")
    add_joint_goal_args(parser)
    parser.add_argument("--restore", action="store_true", help="Move back to the starting config after the check")
    args = parser.parse_args()

    live = create_robot(args)
    robot = live.robot
    start = robot.get_motion_config()
    goal = list(start)
    goal[args.joint_index] += math.radians(args.delta_deg)

    print_config("start_config", start)
    print_config("goal_config", goal)
    ok = robot.move_to_config(goal)
    print(f"joint_move_ok: {ok}")
    print(f"joint_move_error: {robot.last_error}")
    if not ok:
        raise RuntimeError(robot.last_error or "move_to_config returned False")
    check_config_error("joint_move", robot.get_motion_config(), goal)

    if args.restore:
        restored = robot.move_to_config(start)
        print(f"restore_ok: {restored}")
        print(f"restore_error: {robot.last_error}")


if __name__ == "__main__":
    run(main)

