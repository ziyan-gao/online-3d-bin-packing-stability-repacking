from __future__ import annotations

from types import SimpleNamespace

import pytest

from python_controller_v2.pick_place import PickPlaceControllerV2


class V2PickPlaceFakeSim:
    handle_world = -1

    def __init__(self):
        self.object_poses = {
            42: [2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        }
        self.object_sizes = {42: [0.2, 0.2, 0.2], 40: [1.0, 2.0, 0.2]}
        self.arm_base_pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.pallet_pose = [10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0]
        self.tip_pose = [0.3, -0.2, 0.4, 0.0, 0.0, 0.0, 1.0]
        self.signals = []
        self.parents = []
        self.resets = []

    def getObjectPose(self, handle, relative_to=None):
        if handle == 11:
            return list(self.arm_base_pose)
        if handle == 20:
            return list(self.tip_pose)
        if handle == 40:
            return list(self.pallet_pose)
        if handle in self.object_poses:
            return list(self.object_poses[handle])
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def getShapeBB(self, handle):
        return list(self.object_sizes[handle])

    def multiplyPoses(self, pose, local_pose):
        return [
            pose[0] + local_pose[0],
            pose[1] + local_pose[1],
            pose[2] + local_pose[2],
            *local_pose[3:7],
        ]

    def setInt32Signal(self, name, value):
        self.signals.append((name, value))

    def setObjectParent(self, handle, parent, keep_in_place):
        self.parents.append((handle, parent, keep_in_place))

    def resetDynamicObject(self, handle):
        self.resets.append(handle)


class V2PickPlaceFakeRobot:
    def __init__(self, sim):
        self.sim = sim
        self.ctx = SimpleNamespace(step=lambda count=1: self.actions.append(("step", count)))
        self.handles = SimpleNamespace(arm_base=11, ik_tip=20, pallet=40)
        self.actions = []
        self.rail_position = 0.0
        self.arm_config = [0.1, 0.2, 0.3]
        self.last_error = None
        self.setup_arm_ik_calls = 0
        self.linear_calls = []
        self.ompl_calls = []
        self.ompl_config_calls = []
        self.remembered_configs = []
        self.linear_results = []
        self.ompl_results = []
        self.ompl_config_results = []

    def get_rail_position(self):
        return self.rail_position

    def move_rail_to(self, position, max_step=0.02):
        self.actions.append(("rail", position))
        self.rail_position = position
        return True

    def get_arm_config(self):
        return list(self.arm_config)

    def solve_arm_ik(self, pose):
        self.actions.append(("solve", list(pose)))
        return [round(pose[0], 4), round(pose[1], 4), round(pose[2], 4)]

    def move_arm_to_config(self, config, max_step=0.05):
        self.actions.append(("joint", list(config)))
        self.arm_config = list(config)
        return True

    def move_arm_linear_to_pose(
        self,
        pose,
        max_step=0.03,
        joint_step=0.05,
        attached_object=None,
        ignored_objects=None,
        allow_final_contact=False,
        validate_collision=True,
    ):
        self.actions.append(("linear", list(pose)))
        self.linear_calls.append(
            (
                list(pose),
                attached_object,
                list(ignored_objects or []),
                allow_final_contact,
                validate_collision,
            )
        )
        if self.linear_results:
            result = self.linear_results.pop(0)
            if not result:
                self.last_error = "linear blocked"
                return False
        self.sim.tip_pose = list(pose)
        self.arm_config = [round(float(pose[0]), 4), round(float(pose[1]), 4), round(float(pose[2]), 4)]
        self.last_error = None
        return True

    def move_arm_ompl_to_pose(
        self,
        pose,
        attached_object=None,
        ignored_objects=None,
        allow_final_contact=False,
        max_configs=8,
        ik_time=1.0,
        planning_time=3.0,
        path_state_count=100,
    ):
        self.actions.append(("ompl", list(pose)))
        self.ompl_calls.append(
            (
                list(pose),
                attached_object,
                list(ignored_objects or []),
                allow_final_contact,
                max_configs,
                ik_time,
                planning_time,
                path_state_count,
            )
        )
        if self.ompl_results:
            result = self.ompl_results.pop(0)
            if not result:
                self.last_error = "ompl blocked"
                return False
        self.sim.tip_pose = list(pose)
        self.arm_config = [round(float(pose[0]), 4), round(float(pose[1]), 4), round(float(pose[2]), 4)]
        self.last_error = None
        return True

    def move_arm_ompl_to_config(
        self,
        config,
        attached_object=None,
        planning_time=3.0,
        path_state_count=100,
    ):
        self.actions.append(("ompl_config", list(config)))
        self.ompl_config_calls.append((list(config), attached_object, planning_time, path_state_count))
        if self.ompl_config_results:
            result = self.ompl_config_results.pop(0)
            if not result:
                self.last_error = "ompl config blocked"
                return False
        self.arm_config = list(config)
        self.last_error = None
        return True

    def remember_arm_config(self, name, config=None):
        stored = self.get_arm_config() if config is None else list(config)
        self.remembered_configs.append((name, stored))

    def setup_arm_ik(self):
        self.setup_arm_ik_calls += 1


def test_v2_pick_moves_rail_then_linear_pre_pick_descend_suction_and_lift():
    sim = V2PickPlaceFakeSim()
    robot = V2PickPlaceFakeRobot(sim)
    controller = PickPlaceControllerV2(robot)
    pose_before_pre_pick = list(sim.tip_pose)

    assert controller.pick(42) is True

    pre_pose, pick_pose, hold_pose = controller.pick_poses_for_object(42)

    assert robot.actions[:4] == [
        ("rail", pytest.approx(-0.5)),
        ("linear", pre_pose),
        ("linear", pick_pose),
        ("linear", hold_pose),
    ]
    assert "solve" not in [action[0] for action in robot.actions]
    assert "joint" not in [action[0] for action in robot.actions]
    assert sim.signals == [("suctionPadEnabled", 1)]
    assert sim.parents == [(42, 20, True)]
    assert controller.attached_object == 42
    assert controller.pose_before_pre_pick == pose_before_pre_pick
    assert controller.saved_arm_configs == {
        "initial": pytest.approx([0.1, 0.2, 0.3]),
        "pre_pick": pytest.approx(pre_pose[:3]),
        "pick": pytest.approx(pick_pose[:3]),
        "hold": pytest.approx(hold_pose[:3]),
    }
    assert [name for name, _config in robot.remembered_configs] == ["initial", "pre_pick", "pick", "hold"]


def test_v2_place_moves_linear_to_pre_place_descends_releases_and_retracts():
    sim = V2PickPlaceFakeSim()
    robot = V2PickPlaceFakeRobot(sim)
    controller = PickPlaceControllerV2(robot)
    controller.attached_object = 42
    controller.pose_before_pre_pick = [0.3, -0.2, 0.4, 0.0, 0.0, 0.0, 1.0]

    assert controller.place([0.4, 0.5, 0.6]) is True

    pre_pose, place_pose = controller.place_poses_for_object(42, [0.4, 0.5, 0.6])

    assert robot.actions[:4] == [
        ("rail", pytest.approx(-10.2)),
        ("linear", pre_pose),
        ("linear", place_pose),
        ("linear", pre_pose),
    ]
    assert len(robot.actions) == 4
    assert "solve" not in [action[0] for action in robot.actions]
    assert "joint" not in [action[0] for action in robot.actions]
    assert sim.signals == [("suctionPadEnabled", 0)]
    assert sim.parents == [(42, -1, True)]
    assert sim.resets == [42]
    assert controller.attached_object is None
    assert controller.pose_before_pre_pick is None


def test_v2_pick_place_saves_all_named_arm_configs():
    sim = V2PickPlaceFakeSim()
    robot = V2PickPlaceFakeRobot(sim)
    controller = PickPlaceControllerV2(robot)

    assert controller.pick(42) is True
    assert controller.place([0.4, 0.5, 0.6]) is True

    assert set(controller.saved_arm_configs) == {
        "initial",
        "pre_pick",
        "pick",
        "hold",
        "pre_place",
        "place",
    }
    assert [name for name, _config in robot.remembered_configs] == [
        "initial",
        "pre_pick",
        "pick",
        "hold",
        "pre_place",
        "place",
        "pre_place",
    ]


def test_v2_place_does_not_return_to_saved_initial_config_after_retract():
    sim = V2PickPlaceFakeSim()
    robot = V2PickPlaceFakeRobot(sim)
    robot.linear_results = [True, True, True]
    controller = PickPlaceControllerV2(robot)
    controller.attached_object = 42
    controller.pose_before_pre_pick = list(sim.tip_pose)
    controller.saved_arm_configs["initial"] = [0.1, 0.2, 0.3]

    assert controller.place([0.4, 0.5, 0.6]) is True

    assert ("ompl_config", [0.1, 0.2, 0.3]) not in robot.actions
    assert robot.ompl_calls == []
    assert robot.ompl_config_calls == []


def test_v2_place_repositions_rail_to_offset_past_positive_x_pallet_boundary_when_tip_is_negative_x_of_base():
    sim = V2PickPlaceFakeSim()
    sim.tip_pose[0] = 0.3
    sim.arm_base_pose[0] = 1.0
    robot = V2PickPlaceFakeRobot(sim)
    controller = PickPlaceControllerV2(robot)
    controller.attached_object = 42
    controller.pose_before_pre_pick = list(sim.tip_pose)

    assert controller.place([0.4, 0.5, 0.6]) is True

    assert robot.actions[0] == ("rail", pytest.approx(-10.2))


def test_v2_place_repositions_rail_to_offset_past_negative_x_pallet_boundary_when_tip_is_positive_x_of_base():
    sim = V2PickPlaceFakeSim()
    sim.tip_pose[0] = 2.0
    sim.arm_base_pose[0] = 1.0
    robot = V2PickPlaceFakeRobot(sim)
    controller = PickPlaceControllerV2(robot)
    controller.attached_object = 42
    controller.pose_before_pre_pick = list(sim.tip_pose)

    assert controller.place([0.4, 0.5, 0.6]) is True

    assert robot.actions[0] == ("rail", pytest.approx(-7.8))


def test_v2_place_poses_are_pallet_relative_and_converted_to_world():
    sim = V2PickPlaceFakeSim()
    robot = V2PickPlaceFakeRobot(sim)
    controller = PickPlaceControllerV2(robot)

    pre_pose, place_pose = controller.place_poses_for_object(42, [0.4, 0.5, 0.6])

    assert place_pose == pytest.approx([10.5, 20.6, 30.7, 0.0, 1.0, 0.0, 0.0])
    assert pre_pose == pytest.approx([10.5, 20.6, 30.8, 0.0, 1.0, 0.0, 0.0])


def test_v2_pick_does_not_reconfigure_ik_after_rail_motion():
    sim = V2PickPlaceFakeSim()
    robot = V2PickPlaceFakeRobot(sim)
    controller = PickPlaceControllerV2(robot)

    assert controller.pick(42) is True

    assert robot.setup_arm_ik_calls == 0


def test_v2_pick_place_passes_collision_context_to_linear_moves():
    sim = V2PickPlaceFakeSim()
    robot = V2PickPlaceFakeRobot(sim)
    controller = PickPlaceControllerV2(robot)
    assert controller.pick(42) is True
    assert controller.place([0.4, 0.5, 0.6]) is True

    pre_pick, pick_pose, hold_pose = controller.pick_poses_for_object(42)
    pre_place, place_pose = controller.place_poses_for_object(42, [0.4, 0.5, 0.6])
    assert robot.linear_calls == [
        (pre_pick, None, [42], False, True),
        (pick_pose, None, [42], False, True),
        (hold_pose, 42, [], False, True),
        (pre_place, 42, [], False, True),
        (place_pose, 42, [], True, True),
        (pre_place, None, [42], False, True),
    ]


def test_v2_pick_falls_back_to_ompl_when_linear_pre_pick_fails():
    sim = V2PickPlaceFakeSim()
    robot = V2PickPlaceFakeRobot(sim)
    robot.linear_results = [False, True, True]
    controller = PickPlaceControllerV2(robot)

    assert controller.pick(42) is True

    pre_pose, pick_pose, hold_pose = controller.pick_poses_for_object(42)
    assert robot.actions[:5] == [
        ("rail", pytest.approx(-0.5)),
        ("linear", pre_pose),
        ("ompl", pre_pose),
        ("linear", pick_pose),
        ("linear", hold_pose),
    ]
    assert robot.ompl_calls == [
        (pre_pose, None, [42], False, 8, 1.0, 3.0, 100),
    ]
    assert sim.signals == [("suctionPadEnabled", 1)]
    assert controller.last_error is None


def test_v2_place_fallback_passes_final_contact_context_to_ompl():
    sim = V2PickPlaceFakeSim()
    robot = V2PickPlaceFakeRobot(sim)
    robot.linear_results = [True, False, True, True]
    controller = PickPlaceControllerV2(robot)
    controller.attached_object = 42
    controller.pose_before_pre_pick = list(sim.tip_pose)

    assert controller.place([0.4, 0.5, 0.6]) is True

    pre_pose, place_pose = controller.place_poses_for_object(42, [0.4, 0.5, 0.6])
    assert robot.actions[:5] == [
        ("rail", pytest.approx(-10.2)),
        ("linear", pre_pose),
        ("linear", place_pose),
        ("ompl", place_pose),
        ("linear", pre_pose),
    ]
    assert len(robot.actions) == 5
    assert robot.ompl_calls == [
        (place_pose, 42, [], True, 8, 1.0, 3.0, 100),
    ]
    assert sim.signals == [("suctionPadEnabled", 0)]
    assert controller.attached_object is None


def test_v2_pick_reports_linear_and_ompl_errors_when_both_fail():
    sim = V2PickPlaceFakeSim()
    robot = V2PickPlaceFakeRobot(sim)
    robot.linear_results = [False]
    robot.ompl_results = [False]
    controller = PickPlaceControllerV2(robot)

    assert controller.pick(42) is False

    assert controller.last_error == "pick pre-approach failed: linear failed: linear blocked; OMPL failed: ompl blocked"
    pre_pose, _pick_pose, _hold_pose = controller.pick_poses_for_object(42)
    assert robot.actions[:3] == [
        ("rail", pytest.approx(-0.5)),
        ("linear", pre_pose),
        ("ompl", pre_pose),
    ]
