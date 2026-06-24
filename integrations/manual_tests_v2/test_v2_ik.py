from __future__ import annotations

from common import build_parser, check_pose_error, create_robot, print_config, print_pose, run


def main() -> None:
    parser = build_parser("Test python_controller_v2 arm-only IK.")
    parser.add_argument("--target", choices=["ik-target", "tip-delta"], default="tip-delta")
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.05)
    parser.add_argument("--move", action="store_true", help="Move the arm to the solved config")
    args = parser.parse_args()

    live = create_robot(args)
    robot = live.robot
    sim = live.sim
    if args.target == "ik-target":
        target = sim.getObjectPose(robot.handles.ik_target)
    else:
        tip = sim.getObjectPose(robot.handles.ik_tip)
        target = [tip[0] + args.dx, tip[1] + args.dy, tip[2] + args.dz, *tip[3:7]]

    print_pose("target", target)
    config = robot.solve_arm_ik(target)
    print_config("arm_config", config or [])
    print(f"error: {robot.last_error}")
    if config is None:
        raise RuntimeError(robot.last_error or "solve_arm_ik returned None")

    if args.move:
        ok = robot.move_arm_to_config(config)
        print(f"move_ok: {ok}")
        print(f"move_error: {robot.last_error}")
        tip = sim.getObjectPose(robot.handles.ik_tip)
        print_pose("final_tip", tip)
        check_pose_error("ik_move", tip, target, args)


if __name__ == "__main__":
    run(main)
