from __future__ import annotations

from common import add_object_args, build_parser, create_robot, make_pick_place, print_pose, resolve_test_object, run


def main() -> None:
    parser = build_parser("Pick one object and leave it attached to the tool for a later place test.")
    add_object_args(parser)
    args = parser.parse_args()

    live = create_robot(args)
    robot = live.robot
    sim = live.sim
    pick_place = make_pick_place(robot)
    handle = resolve_test_object(sim, args)
    pre, pick, hold = pick_place.pick_poses_for_object(handle)

    print(f"object_handle: {handle}")
    print_pose("pre_pick", pre)
    print_pose("pick", pick)
    print_pose("hold", hold)
    ok = pick_place.pick(handle)
    print(f"pick_ok: {ok}")
    print(f"pick_error: {pick_place.last_error}")
    print(f"robot_error: {robot.last_error}")
    print(f"attached_object: {pick_place.attached_object}")
    if not ok:
        raise RuntimeError(pick_place.last_error or "pick returned False")


if __name__ == "__main__":
    run(main)

