from __future__ import annotations

from common import add_delta_args, build_parser, check_pose_error, create_robot, print_pose, run, shifted_pose


def main() -> None:
    parser = build_parser("Move the tip linearly from its current pose by an XYZ delta.")
    add_delta_args(parser, dz=0.05)
    parser.add_argument("--restore", action="store_true", help="Move back to the starting tip pose after the check")
    args = parser.parse_args()

    live = create_robot(args)
    robot = live.robot
    sim = live.sim
    start_pose = sim.getObjectPose(robot.handles.ik_tip)
    target = shifted_pose(start_pose, args.dx, args.dy, args.dz)

    print_pose("start_tip", start_pose)
    print_pose("target", target)
    ok = robot.move_cartesian_linear(target)
    print(f"linear_ok: {ok}")
    print(f"linear_error: {robot.last_error}")
    tip = sim.getObjectPose(robot.handles.ik_tip)
    print_pose("final_tip", tip)
    if not ok:
        raise RuntimeError(robot.last_error or "move_cartesian_linear returned False")
    check_pose_error("linear", tip, target, args)

    if args.restore:
        restored = robot.move_cartesian_linear(start_pose)
        print(f"restore_ok: {restored}")
        print(f"restore_error: {robot.last_error}")


if __name__ == "__main__":
    run(main)

