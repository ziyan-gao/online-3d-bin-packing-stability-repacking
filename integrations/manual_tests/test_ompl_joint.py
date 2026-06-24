from __future__ import annotations

from common import add_joint_goal_args, build_parser, check_config_error, create_robot, goal_from_joint_delta, print_config, run


def main() -> None:
    parser = build_parser("Plan a nearby joint-space path with OMPL, optionally follow it.")
    add_joint_goal_args(parser)
    args = parser.parse_args()

    live = create_robot(args)
    robot = live.robot
    start = robot.get_motion_config()
    goal = goal_from_joint_delta(robot, args.joint_index, args.delta_deg)
    dof = len(robot.active_joints())

    print_config("start_config", start)
    print_config("goal_config", goal)
    path = robot.plan_joint_path(goal)
    print(f"ompl_error: {robot.last_error}")
    if path is None:
        raise RuntimeError(robot.last_error or "plan_joint_path returned None")
    print(f"path_values: {len(path)}")
    print(f"path_points: {len(path) // dof if dof else 0}")

    if args.follow:
        ok = robot.follow_joint_path(path)
        print(f"follow_ok: {ok}")
        print(f"follow_error: {robot.last_error}")
        if not ok:
            raise RuntimeError(robot.last_error or "follow_joint_path returned False")
        check_config_error("ompl_follow", robot.get_motion_config(), goal, tolerance=1e-3)


if __name__ == "__main__":
    run(main)

