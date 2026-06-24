from __future__ import annotations

from common import build_parser, check_pose_error, create_robot, print_pose, run


def main() -> None:
    parser = build_parser("Test python_controller_v2 Cartesian linear arm motion.")
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.05)
    parser.add_argument("--max-step", type=float, default=0.02)
    args = parser.parse_args()

    live = create_robot(args)
    robot = live.robot
    sim = live.sim
    start = sim.getObjectPose(robot.handles.ik_tip)
    target = [start[0] + args.dx, start[1] + args.dy, start[2] + args.dz, *start[3:7]]

    print_pose("start_tip", start)
    print_pose("target", target)
    ok = robot.move_arm_linear_to_pose(target, max_step=args.max_step)
    print(f"linear_ok: {ok}")
    print(f"linear_error: {robot.last_error}")
    tip = sim.getObjectPose(robot.handles.ik_tip)
    print_pose("final_tip", tip)
    if not ok:
        raise RuntimeError(robot.last_error or "move_arm_linear_to_pose returned False")
    check_pose_error("linear", tip, target, args)


if __name__ == "__main__":
    run(main)
