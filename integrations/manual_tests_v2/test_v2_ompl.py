from __future__ import annotations

from common import build_parser, check_pose_error, create_robot, print_config, print_pose, run


def main() -> None:
    parser = build_parser("Test python_controller_v2 arm-only OMPL pose motion.")
    parser.add_argument("--target", choices=["ik-target", "tip-delta"], default="tip-delta")
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.05)
    parser.add_argument("--max-configs", type=int, default=8)
    parser.add_argument("--ik-time", type=float, default=1.0)
    parser.add_argument("--planning-time", type=float, default=3.0)
    parser.add_argument("--plan-only", action="store_true")
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
    print_config("start_arm_config", robot.get_arm_config())

    configs = robot.find_arm_configs_for_pose(target, max_configs=args.max_configs, max_time=args.ik_time)
    print(f"ik_config_count: {len(configs)}")
    for index, config in enumerate(configs):
        print_config(f"ik_config_{index}", config)

    if args.plan_only:
        return

    ok = robot.move_arm_ompl_to_pose(
        target,
        max_configs=args.max_configs,
        ik_time=args.ik_time,
        planning_time=args.planning_time,
    )
    print(f"ompl_ok: {ok}")
    print(f"ompl_error: {robot.last_error}")
    print_config("final_arm_config", robot.get_arm_config())
    tip = sim.getObjectPose(robot.handles.ik_tip)
    print_pose("final_tip", tip)
    if not ok:
        raise RuntimeError(robot.last_error or "move_arm_ompl_to_pose returned False")
    check_pose_error("ompl", tip, target, args)


if __name__ == "__main__":
    run(main)
