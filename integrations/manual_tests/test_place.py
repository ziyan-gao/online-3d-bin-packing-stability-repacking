from __future__ import annotations

from common import (
    add_object_args,
    add_place_args,
    build_parser,
    create_robot,
    make_pick_place,
    print_pose,
    resolve_test_object,
    run,
)


def main() -> None:
    parser = build_parser("Place the currently attached object at a pallet-relative position.")
    add_object_args(parser)
    add_place_args(parser)
    parser.add_argument("--attach-first", action="store_true", help="Parent the object to ikTip before placing")
    args = parser.parse_args()

    live = create_robot(args)
    robot = live.robot
    sim = live.sim
    pick_place = make_pick_place(robot)
    handle = resolve_test_object(sim, args, attached_to_tip=robot.handles.ik_tip)
    if args.attach_first:
        sim.setObjectParent(handle, robot.handles.ik_tip, True)
        sim.setInt32Signal("suctionPadEnabled", 1)
    pick_place.attached_object = handle
    pre, place = pick_place.place_poses_for_object(handle, [args.x, args.y, args.z])

    print(f"object_handle: {handle}")
    print_pose("pre_place", pre)
    print_pose("place", place)
    ok = pick_place.place([args.x, args.y, args.z])
    print(f"place_ok: {ok}")
    print(f"place_error: {pick_place.last_error}")
    print(f"robot_error: {robot.last_error}")
    print(f"attached_object: {pick_place.attached_object}")
    if not ok:
        raise RuntimeError(pick_place.last_error or "place returned False")


if __name__ == "__main__":
    run(main)

