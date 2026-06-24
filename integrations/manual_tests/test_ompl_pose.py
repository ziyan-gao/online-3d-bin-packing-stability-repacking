from __future__ import annotations

from common import add_delta_args, build_parser, check_pose_error, create_robot, print_pose, run, shifted_pose


def main() -> None:
    parser = build_parser("Plan an OMPL joint path to a nearby Cartesian pose.")
    add_delta_args(parser, dz=0.05)
    parser.add_argument("--follow", action="store_true", help="Follow the planned path after planning")
    args = parser.parse_args()

    live = create_robot(args)
    robot = live.robot
    sim = live.sim
    start_pose = sim.getObjectPose(robot.handles.ik_tip)
    target = shifted_pose(start_pose, args.dx, args.dy, args.dz)
    dof = len(robot.active_joints())

    print_pose("start_tip", start_pose)
    print_pose("target", target)
    path = robot.plan_pose_path(target)
    print(f"ompl_pose_error: {robot.last_error}")
    if path is None:
        raise RuntimeError(robot.last_error or "plan_pose_path returned None")
    print(f"path_values: {len(path)}")
    print(f"path_points: {len(path) // dof if dof else 0}")

    if args.follow:
        ok = robot.follow_joint_path(path)
        print(f"follow_ok: {ok}")
        print(f"follow_error: {robot.last_error}")
        if not ok:
            raise RuntimeError(robot.last_error or "follow_joint_path returned False")
        tip = sim.getObjectPose(robot.handles.ik_tip)
        print_pose("final_tip", tip)
        check_pose_error("ompl_pose_follow", tip, target, args)


if __name__ == "__main__":
    run(main)

