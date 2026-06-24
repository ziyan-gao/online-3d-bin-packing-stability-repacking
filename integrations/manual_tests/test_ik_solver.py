from __future__ import annotations

from common import build_parser, check_pose_error, create_robot, print_config, print_pose, run, shifted_pose


def main() -> None:
    parser = build_parser("Solve IK to a target pose and verify the returned joint config reaches it.")
    parser.add_argument("--target", choices=["ik-target", "tip-delta"], default="tip-delta")
    parser.add_argument("--move", action="store_true", help="Move to the solved config after direct residual validation")
    parser.add_argument("--skip-collision", action="store_true")
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.2)
    args = parser.parse_args()

    live = create_robot(args)
    robot = live.robot
    sim = live.sim
    start_config = robot.get_motion_config()
    if args.target == "tip-delta":
        target = shifted_pose(sim.getObjectPose(robot.handles.ik_tip), args.dx, args.dy, args.dz)
    else:
        target = sim.getObjectPose(robot.handles.ik_target)

    print_pose("target", target)
    config = robot.solve_ik_to_pose(target, check_collision=not args.skip_collision)
    print_config("config", config or [])
    print(f"solve_error: {robot.last_error}")
    if config is None:
        raise RuntimeError(robot.last_error or "solve_ik_to_pose returned None")

    try:
        robot.set_motion_config(config)
        direct_tip = sim.getObjectPose(robot.handles.ik_tip)
        print_pose("direct_tip", direct_tip)
        check_pose_error("direct_config", direct_tip, target, args)
    finally:
        robot.set_motion_config(start_config)

    if args.move:
        ok = robot.move_to_config(config)
        print(f"move_ok: {ok}")
        print(f"move_error: {robot.last_error}")
        moved_tip = sim.getObjectPose(robot.handles.ik_tip)
        print_pose("moved_tip", moved_tip)
        check_pose_error("move_to_config", moved_tip, target, args)


if __name__ == "__main__":
    run(main)

