from __future__ import annotations

import math

from common import build_parser, create_robot, print_config, run


def main() -> None:
    parser = build_parser("Test python_controller_v2 arm joint interpolation.")
    parser.add_argument("--joint-index", type=int, default=0)
    parser.add_argument("--delta-deg", type=float, default=5.0)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    live = create_robot(args)
    robot = live.robot
    start = robot.get_arm_config()
    if args.joint_index < 0 or args.joint_index >= len(start):
        raise RuntimeError(f"joint-index must be 0..{len(start) - 1}")
    goal = list(start)
    goal[args.joint_index] += math.radians(args.delta_deg)

    print_config("start_arm_config", start)
    print_config("goal_arm_config", goal)
    ok = robot.move_arm_to_config(goal)
    print(f"joint_ok: {ok}")
    print(f"joint_error: {robot.last_error}")
    print_config("final_arm_config", robot.get_arm_config())
    if not ok:
        raise RuntimeError(robot.last_error or "move_arm_to_config returned False")

    if args.restore:
        robot.move_arm_to_config(start)


if __name__ == "__main__":
    run(main)

