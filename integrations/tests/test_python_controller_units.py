import logging
from types import SimpleNamespace

import pytest

from python_controller.belt import BeltMonitor
from python_controller.pick_place import PickPlaceController
from python_controller.robot import RobotController, interpolate_pose_xyz, interpolate_scalar


class FakeSim:
    def __init__(self, packed=None):
        self.packed = packed
        self.signals = {}

    def getStringSignal(self, name):
        assert name == "trackedItemsData"
        return self.packed

    def unpackTable(self, packed):
        return packed

    def setInt32Signal(self, name, value):
        self.signals[name] = value


def test_belt_monitor_decodes_utf8_json_signal_without_unpack_table():
    raw_item = {
        "handle": 12,
        "size": [1, 2, 3],
        "pose": [0, 0, 0, 0, 0, 0, 1],
    }
    sim = FakeSim('{"full": true, "items": [{"handle": 12, "size": [1, 2, 3], "pose": [0, 0, 0, 0, 0, 0, 1]}]}')
    sim.unpackTable = lambda packed: (_ for _ in ()).throw(AssertionError("binary unpack should not be used"))

    items = BeltMonitor(sim).read_full_items()

    assert [item.handle for item in items] == [12]
    assert items[0].raw == raw_item


def test_belt_monitor_returns_items_only_when_full():
    raw_item = {
        "handle": 12,
        "size": [1, 2, 3],
        "pose": [0, 0, 0, 0, 0, 0, 1],
    }
    sim = FakeSim(
        {
            "full": True,
            "items": [raw_item],
        }
    )
    items = BeltMonitor(sim).read_full_items()
    assert [item.handle for item in items] == [12]
    assert items[0].size == [1, 2, 3]
    assert items[0].pose == [0, 0, 0, 0, 0, 0, 1]
    assert items[0].raw is raw_item

    sim.packed = {"full": False, "items": [{"handle": 13}]}
    assert BeltMonitor(sim).read_full_items() == []


def test_belt_monitor_writes_robot_picking_signal():
    sim = FakeSim()
    belt = BeltMonitor(sim)
    belt.set_robot_picking(True)
    assert sim.signals["robotPicking"] == 1
    belt.set_robot_picking(False)
    assert sim.signals["robotPicking"] == 0


def test_full_transition_items_only_returns_on_rising_edge():
    sim = FakeSim({"full": True, "items": [{"handle": 12}]})
    belt = BeltMonitor(sim)
    assert [item.handle for item in belt.full_transition_items()] == [12]
    assert belt.full_transition_items() == []
    sim.packed = {"full": False, "items": []}
    assert belt.full_transition_items() == []
    sim.packed = {"full": True, "items": [{"handle": 13}]}
    assert [item.handle for item in belt.full_transition_items()] == [13]


def test_belt_monitor_release_resets_full_transition_state_for_next_pick():
    sim = FakeSim({"full": True, "items": [{"handle": 12}]})
    belt = BeltMonitor(sim)
    assert [item.handle for item in belt.full_transition_items()] == [12]
    assert belt.full_transition_items() == []

    sim.packed = {"full": True, "items": [{"handle": 13}]}
    belt.set_robot_picking(False)

    assert [item.handle for item in belt.full_transition_items()] == [13]


def test_interpolate_pose_xyz_keeps_quaternion_and_interpolates_position():
    start = [0, 0, 0, 0, 0, 0, 1]
    target = [1, 2, 3, 0, 0, 0, 1]
    poses = interpolate_pose_xyz(start, target, max_step=1.0)
    assert poses[0] == start
    assert poses[-1] == target
    assert poses[1][:3] == [0.25, 0.5, 0.75]
    assert all(p[3:] == [0, 0, 0, 1] for p in poses)


def test_interpolate_scalar():
    assert interpolate_scalar(2.0, 6.0, 0.25) == 3.0


def test_interpolate_pose_xyz_rejects_non_positive_step():
    with pytest.raises(ValueError, match="max_step must be greater than 0"):
        interpolate_pose_xyz([0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1], max_step=0)


def test_interpolate_pose_xyz_returns_target_orientation_for_zero_distance():
    start = [1, 2, 3, 0, 0, 0, 1]
    target = [1, 2, 3, 0, 0, 1, 0]

    assert interpolate_pose_xyz(start, target) == [[1, 2, 3, 0, 0, 1, 0]]


class RecordingJointSim:
    def __init__(self):
        self.calls = []

    def setJointPosition(self, joint, value):
        self.calls.append(("position", joint, value))

    def setJointTargetPosition(self, joint, value):
        self.calls.append(("target", joint, value))


def test_robot_controller_rejects_wrong_length_configs_before_writing():
    sim = RecordingJointSim()
    robot = RobotController.__new__(RobotController)
    robot.handles = SimpleNamespace(joints=[1, 2, 3], rail_joint=1)
    robot.sim = sim

    with pytest.raises(ValueError, match="Expected 3 joint values, got 2"):
        robot.set_config([1.0, 2.0])
    assert sim.calls == []

    with pytest.raises(ValueError, match="Expected 3 joint values, got 2"):
        robot.set_target_config([1.0, 2.0])
    assert sim.calls == []

    robot.set_config([1.0, 2.0, 3.0])
    robot.set_target_config([4.0, 5.0, 6.0])
    assert sim.calls == [
        ("position", 1, 1.0),
        ("position", 2, 2.0),
        ("position", 3, 3.0),
        ("target", 1, 4.0),
        ("target", 2, 5.0),
        ("target", 3, 6.0),
    ]


def test_active_joints_excludes_rail_when_rail_is_disabled():
    robot = RobotController.__new__(RobotController)
    robot.handles = SimpleNamespace(joints=[1, 2, 3], rail_joint=1)
    robot.rail_enabled = False

    assert robot.active_joints() == [2, 3]


def test_active_joints_defaults_to_arm_only_when_rail_is_enabled():
    robot = RobotController.__new__(RobotController)
    robot.handles = SimpleNamespace(joints=[1, 2, 3], rail_joint=1)
    robot.rail_enabled = True

    assert robot.active_joints() == [2, 3]


def test_active_joints_can_explicitly_include_rail_when_rail_is_enabled():
    robot = RobotController.__new__(RobotController)
    robot.handles = SimpleNamespace(joints=[1, 2, 3], rail_joint=1)
    robot.rail_enabled = True

    assert robot.active_joints(include_rail=True) == [1, 2, 3]


def test_preposition_rail_for_world_x_uses_opposite_rail_axis_sign():
    sim = SimpleNamespace(
        positions={1: 2.0},
        base_pose=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        move_params=None,
        getJointPosition=lambda joint: sim.positions[joint],
        setJointPosition=lambda joint, value: sim.positions.__setitem__(joint, value),
        getObjectPose=lambda handle: list(sim.base_pose),
        moveToConfig=lambda params: setattr(sim, "move_params", params),
    )
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.ctx = SimpleNamespace(step=lambda count=1: None)
    robot.handles = SimpleNamespace(rail_joint=1, ur10_base=10)
    robot.rail_enabled = True

    assert robot.preposition_rail_for_world_x(2.2, reach_half_width=0.5) is True

    assert sim.move_params["joints"] == [1]
    assert sim.move_params["targetPos"] == pytest.approx([1.3])
    assert sim.positions[1] == pytest.approx(1.3)


def test_preposition_rail_for_world_x_does_not_move_when_target_is_in_window():
    sim = SimpleNamespace(
        positions={1: 2.0},
        base_pose=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        move_params=None,
        getJointPosition=lambda joint: sim.positions[joint],
        setJointPosition=lambda joint, value: sim.positions.__setitem__(joint, value),
        getObjectPose=lambda handle: list(sim.base_pose),
        moveToConfig=lambda params: setattr(sim, "move_params", params),
    )
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.ctx = SimpleNamespace(step=lambda count=1: None)
    robot.handles = SimpleNamespace(rail_joint=1, ur10_base=10)
    robot.rail_enabled = True

    assert robot.preposition_rail_for_world_x(1.4, reach_half_width=0.5) is True

    assert sim.move_params is None
    assert sim.positions[1] == pytest.approx(2.0)


def test_preposition_rail_for_world_x_is_disabled_when_rail_disabled():
    robot = RobotController.__new__(RobotController)
    robot.rail_enabled = False

    assert robot.preposition_rail_for_world_x(2.2) is True


def test_active_joints_can_explicitly_include_all_joints_when_rail_is_enabled():
    robot = RobotController.__new__(RobotController)
    robot.handles = SimpleNamespace(joints=[1, 2, 3], rail_joint=1)
    robot.rail_enabled = True

    assert robot.active_joints(include_rail=True) == [1, 2, 3]


class CollisionRaisingSim:
    def __init__(self):
        self.handle_all = -2
        self.handle_tree = 1
        self.handle_single = 2
        self.positions = {1: 0.1, 2: 0.2}
        self.next_collection = 50
        self.robot_items = []

    def getJointPosition(self, joint):
        return self.positions[joint]

    def setJointPosition(self, joint, value):
        self.positions[joint] = value

    def createCollection(self):
        self.next_collection += 1
        return self.next_collection

    def addItemToCollection(self, collection, mode, handle, options):
        self.robot_items.append((collection, mode, handle, options))

    def checkCollision(self, collection, other):
        raise RuntimeError("remote collision failed")


def test_collides_restores_config_when_collision_check_raises():
    sim = CollisionRaisingSim()
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.handles = SimpleNamespace(ik_base=10, joints=[1, 2])
    robot.robot_collection = None

    with pytest.raises(RuntimeError, match="remote collision failed"):
        robot.collides([[9.0, 8.0]])

    assert sim.positions == {1: 0.1, 2: 0.2}


class CollectionRecordingSim:
    def __init__(self, fail_on_attached=False):
        self.handle_all = -2
        self.handle_tree = 1
        self.handle_single = 2
        self.created = []
        self.destroyed = []
        self.items = []
        self.fail_on_attached = fail_on_attached
        self.next_collection = 100

    def createCollection(self):
        self.next_collection += 1
        self.created.append(self.next_collection)
        return self.next_collection

    def addItemToCollection(self, collection, mode, handle, options):
        if self.fail_on_attached and mode == self.handle_single:
            raise RuntimeError("cannot add attached object")
        self.items.append((collection, mode, handle, options))

    def destroyCollection(self, collection):
        self.destroyed.append(collection)


def test_update_robot_collection_includes_attached_object_after_successful_swap():
    sim = CollectionRecordingSim()
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.handles = SimpleNamespace(ik_base=10)
    robot.robot_collection = 99
    robot.moving_collection = 98
    robot.obstacle_collection = 97

    robot.update_robot_collection(attached_object=42)

    assert robot.robot_collection == 101
    assert robot.moving_collection == 102
    assert robot.obstacle_collection == 103
    assert sim.items == [
        (101, sim.handle_tree, 10, 0),
        (102, sim.handle_tree, 10, 0),
        (102, sim.handle_single, 42, 0),
        (103, sim.handle_all, -1, 0),
        (103, sim.handle_tree, 10, 1),
        (103, sim.handle_single, 42, 1),
    ]
    assert sim.destroyed == [99, 98, 97]


def test_update_robot_collection_keeps_previous_collection_when_rebuild_fails():
    sim = CollectionRecordingSim(fail_on_attached=True)
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.handles = SimpleNamespace(ik_base=10)
    robot.robot_collection = 99
    robot.moving_collection = 98
    robot.obstacle_collection = 97

    with pytest.raises(RuntimeError, match="cannot add attached object"):
        robot.update_robot_collection(attached_object=42)

    assert robot.robot_collection == 99
    assert robot.moving_collection == 98
    assert robot.obstacle_collection == 97
    assert sim.destroyed == [101, 102, 103]


def test_ik_handle_supports_integer_and_string_keyed_maps():
    robot = RobotController.__new__(RobotController)
    robot.sim_to_ik = {7: 70}
    assert robot._ik_handle(7) == 70

    robot.sim_to_ik = {"7": 71}
    assert robot._ik_handle(7) == 71


def test_ik_handle_rejects_unsupported_map_shapes_with_clear_error():
    robot = RobotController.__new__(RobotController)
    robot.sim_to_ik = [10, 11]

    with pytest.raises(RuntimeError, match="Unsupported IK object map"):
        robot._ik_handle(7)


def test_ik_handle_rejects_string_and_bytes_maps_with_clear_error():
    robot = RobotController.__new__(RobotController)

    for unsupported_map in ("not-a-map", b"not-a-map"):
        robot.sim_to_ik = unsupported_map
        with pytest.raises(RuntimeError, match="Unsupported IK object map"):
            robot._ik_handle(0)


class MultiReturnCollisionSim(CollisionRaisingSim):
    def __init__(self, results):
        super().__init__()
        self.results = list(results)

    def checkCollision(self, collection, other):
        return self.results.pop(0)


def test_collides_detects_tuple_and_list_collision_results():
    for collision_result in ((1, 111), [1, 222]):
        sim = MultiReturnCollisionSim([0, collision_result])
        robot = RobotController.__new__(RobotController)
        robot.sim = sim
        robot.handles = SimpleNamespace(ik_base=10, joints=[1, 2])
        robot.robot_collection = None

        assert robot.collides([[9.0, 8.0]]) is True
        assert sim.positions == {1: 0.1, 2: 0.2}


def test_move_to_config_can_tolerate_only_the_initial_attached_start_collision():
    robot = RobotController.__new__(RobotController)
    robot.handles = SimpleNamespace(joints=[1])
    robot.sim = SimpleNamespace(
        moveToConfig=lambda params: setattr(robot, "move_params", params),
        setJointPosition=lambda joint, value: None,
    )
    robot.ctx = SimpleNamespace(step=lambda count=1: None)
    robot.get_motion_config = lambda include_rail=None: [0.0]
    robot.active_joints = lambda include_rail=None: [1]
    checked_paths = []
    collision_results = [True, False]

    def collides(configs, attached_object=None):
        checked_paths.append([list(config) for config in configs])
        return collision_results.pop(0) if collision_results else False

    robot.collides = collides

    assert robot.move_to_config([0.2], attached_object=42, allow_start_collision=True) is True

    assert checked_paths
    assert checked_paths[0] == [[0.0]]
    assert checked_paths[1][0] != [0.0]
    assert robot.move_params["targetPos"] == [0.2]


def test_move_to_config_rejects_collision_after_attached_object_has_cleared_start_contact():
    robot = RobotController.__new__(RobotController)
    robot.handles = SimpleNamespace(joints=[1])
    robot.sim = SimpleNamespace(setJointPosition=lambda joint, value: None)
    robot.get_motion_config = lambda include_rail=None: [0.0]
    robot.active_joints = lambda include_rail=None: [1]
    collision_results = [True, False, True]

    def collides(configs, attached_object=None):
        return collision_results.pop(0)

    robot.collides = collides

    assert robot.move_to_config([0.2], attached_object=42, allow_start_collision=True) is False

    assert "direct config trajectory collides" in robot.last_error


class IkTargetRecordingSim:
    def __init__(self):
        self.target_pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        self.poses_written = []
        self.pose_requests = []

    def getObjectPose(self, handle, relative_to=None):
        self.pose_requests.append((handle, relative_to))
        return list(self.target_pose)

    def setObjectPose(self, handle, pose):
        self.target_pose = list(pose)
        self.poses_written.append((handle, list(pose)))


class FailingIk:
    method_damped_least_squares = 1
    constraint_pose = 2
    constraint_position = 4
    constraint_alpha_beta = 8
    result_success = 1

    def __init__(self):
        self.calculations = []
        self.constraints = []
        self.precisions = []
        self.object_pose_writes = []
        self.syncs = []
        self.handle_results = [0]

    def setGroupCalculation(self, env, group, method, damping, iterations):
        self.calculations.append((env, group, method, damping, iterations))

    def setElementConstraints(self, env, group, element, constraints):
        self.constraints.append((env, group, element, constraints))

    def setElementPrecision(self, env, group, element, precision):
        self.precisions.append((env, group, element, list(precision)))

    def syncFromSim(self, env, groups):
        self.syncs.append((env, list(groups)))


    def setObjectPose(self, env, target, pose, base):
        self.object_pose_writes.append((env, target, list(pose), base))

    def handleGroup(self, env, group):
        return self.handle_results.pop(0) if self.handle_results else 0

    def getJointPosition(self, env, joint):
        return {101: 0.4}.get(joint, 0.0)


class MultiConfigIk(FailingIk):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.find_configs_calls = []

    def findConfigs(self, env, group, joints, params):
        self.find_configs_calls.append((env, group, list(joints), dict(params)))
        return self.configs


class SetupRecordingIk(FailingIk):
    def __init__(self):
        super().__init__()
        self.add_element_calls = []
        self.next_env = 0
        self.next_group = 10

    def createEnvironment(self):
        self.next_env += 1
        return self.next_env

    def createGroup(self, env):
        self.next_group += 1
        return self.next_group

    def addElementFromScene(self, env, group, base, tip, target, constraints):
        self.add_element_calls.append((env, group, base, tip, target, constraints))
        return 55, {base: 100, 2: 102, tip: 200, target: 300}


def test_setup_ik_creates_only_arm_context_even_when_rail_is_enabled():
    ik = SetupRecordingIk()
    robot = RobotController.__new__(RobotController)
    robot.simIK = ik
    robot.rail_enabled = True
    robot.handles = SimpleNamespace(ik_base=10, rail_joint=1, ur10_base=11, ik_tip=20, ik_target=30, joints=[1, 2])

    robot.setup_ik()

    assert ik.add_element_calls == [
        (1, 11, 11, 20, 30, ik.constraint_pose),
    ]
    assert robot.sim_to_ik == {11: 100, 2: 102, 20: 200, 30: 300}


def test_setup_ik_sets_precision_for_arm_context():
    ik = SetupRecordingIk()
    robot = RobotController.__new__(RobotController)
    robot.simIK = ik
    robot.rail_enabled = True
    robot.handles = SimpleNamespace(ik_base=10, rail_joint=1, ur10_base=11, ik_tip=20, ik_target=30, joints=[1, 2])

    robot.setup_ik()

    assert ik.precisions == [
        (1, 11, 55, [0.0005, 0.005]),
    ]


class ConstructorSim(CollectionRecordingSim):
    sceneobject_joint = 99

    def __init__(self):
        super().__init__()
        self.objects = {
            "/mobile_arm": 10,
            "/mobile_arm/railJoint": 1,
            "/mobile_arm/railJoint/UR10": 11,
            "/mobile_arm/railJoint/UR10/ikTip": 20,
            "/ikTarget": 30,
            "/pallet": 40,
        }
        self.tip_pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        self.relative_target_poses = {
            10: [0.9, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            11: [0.1, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        }
        self.pose_writes = []
        self.pose_requests = []

    def getObject(self, path, options=None):
        return self.objects.get(path, -1)

    def getObjectsInTree(self, base, object_type):
        assert base == 11
        assert object_type == self.sceneobject_joint
        return [2]

    def getObjectPose(self, handle, relative_to=None):
        self.pose_requests.append((handle, relative_to))
        if handle == 20:
            return list(self.tip_pose)
        if handle == 30 and relative_to in self.relative_target_poses:
            return list(self.relative_target_poses[relative_to])
        return list(self.tip_pose)

    def setObjectPose(self, handle, pose):
        self.pose_writes.append((handle, list(pose)))


class ConstructorClient:
    def __init__(self, ik):
        self.ik = ik

    def require(self, name):
        if name == "simIK":
            return self.ik
        if name == "simOMPL":
            return SimpleNamespace()
        raise KeyError(name)


def test_robot_controller_primes_ik_target_to_current_tip_pose_on_init():
    sim = ConstructorSim()
    ik = SetupRecordingIk()
    ctx = SimpleNamespace(sim=sim, client=ConstructorClient(ik))

    robot = RobotController(ctx)

    assert robot.handles.ik_target == 30
    assert sim.pose_writes == [(30, sim.tip_pose)]
    assert ik.syncs == [(1, [11])]
    assert ik.object_pose_writes == [
        (1, 300, sim.relative_target_poses[11], 100),
    ]


def test_solve_ik_to_pose_restores_target_pose_when_ik_fails():
    sim = IkTargetRecordingSim()
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = FailingIk()
    robot.ik_env = 1
    robot.ik_group = 2
    robot.sim_to_ik = {10: 100, 20: 200}
    robot.handles = SimpleNamespace(ik_base=10, ik_target=20, joints=[])

    assert robot.solve_ik_to_pose([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]) is None

    assert sim.target_pose == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]


def test_solve_ik_to_pose_retries_damping_values_before_failing():
    sim = IkTargetRecordingSim()
    ik = FailingIk()
    ik.handle_results = [0, ik.result_success]
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = ik
    robot.ik_env = 1
    robot.ik_group = 2
    robot.ik_element = 55
    robot.sim_to_ik = {10: 100, 20: 200, 1: 101}
    robot.handles = SimpleNamespace(ik_base=10, ik_target=20, joints=[1])
    robot.collides = lambda configs, attached_object=None: False

    assert robot.solve_ik_to_pose([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]) == [0.4]

    assert [call[3] for call in ik.calculations[:2]] == [0.1, 0.3]
    assert sim.target_pose == [9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]


def test_solve_ik_to_pose_relaxes_tool_yaw_constraint_after_full_pose_fails():
    sim = IkTargetRecordingSim()
    ik = FailingIk()
    ik.handle_results = [0, 0, 0, 0, ik.result_success]
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = ik
    robot.ik_env = 1
    robot.ik_group = 2
    robot.ik_element = 55
    robot.sim_to_ik = {10: 100, 20: 200, 1: 101}
    robot.handles = SimpleNamespace(ik_base=10, ik_target=20, joints=[1])
    robot.collides = lambda configs, attached_object=None: False

    assert robot.solve_ik_to_pose([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]) == [0.4]

    assert ik.constraints[0] == (1, 2, 55, ik.constraint_pose)
    assert ik.constraints[-1] == (1, 2, 55, ik.constraint_position | ik.constraint_alpha_beta)


def test_solve_ik_to_pose_prefers_non_colliding_find_configs_result():
    sim = IkTargetRecordingSim()
    ik = MultiConfigIk([[0.1], [0.2]])
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = ik
    robot.ik_env = 1
    robot.ik_group = 2
    robot.ik_element = 55
    robot.sim_to_ik = {10: 100, 20: 200, 1: 101}
    robot.handles = SimpleNamespace(ik_base=10, ik_target=20, joints=[1])
    robot.collides = lambda configs, attached_object=None: configs[0] == [0.1]

    assert robot.solve_ik_to_pose([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]) == [0.2]
    assert ik.handle_results == [0]


class TipPoseByConfigSim(IkTargetRecordingSim):
    def __init__(self):
        super().__init__()
        self.joint_positions = {1: 0.0}
        self.tip_pose_checks = []

    def getJointPosition(self, joint):
        return self.joint_positions[joint]

    def setJointPosition(self, joint, value):
        self.joint_positions[joint] = value

    def getObjectPose(self, handle, relative_to=None):
        if handle == 30:
            joint_value = self.joint_positions[1]
            self.tip_pose_checks.append(joint_value)
            if abs(joint_value - 0.4) <= 1e-9:
                return list(self.target_pose)
            return [8.6, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]
        return super().getObjectPose(handle, relative_to=relative_to)


def test_solve_ik_to_pose_rejects_find_configs_candidate_that_misses_target_pose():
    sim = TipPoseByConfigSim()
    ik = MultiConfigIk([[0.1]])
    ik.handle_results = [ik.result_success]
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = ik
    robot.ik_env = 1
    robot.ik_group = 2
    robot.ik_element = 55
    robot.sim_to_ik = {10: 100, 20: 200, 30: 300, 1: 101}
    robot.handles = SimpleNamespace(ik_base=10, ik_target=20, ik_tip=30, joints=[1])
    robot.collides = lambda configs, attached_object=None: False

    target = [9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]

    assert robot.solve_ik_to_pose(target, check_collision=False) == [0.4]
    assert sim.tip_pose_checks[0] == pytest.approx(0.1)
    assert sim.tip_pose_checks[-1] == pytest.approx(0.4)
    assert sim.joint_positions[1] == pytest.approx(0.0)


def test_solve_ik_to_pose_relaxes_constraints_for_find_configs_before_handle_group():
    sim = IkTargetRecordingSim()
    ik = MultiConfigIk([])
    responses = [[], [[0.4]]]

    def find_configs(env, group, joints, params):
        ik.find_configs_calls.append((env, group, list(joints), dict(params)))
        return responses.pop(0)

    ik.findConfigs = find_configs
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = ik
    robot.ik_env = 1
    robot.ik_group = 2
    robot.ik_element = 55
    robot.sim_to_ik = {10: 100, 20: 200, 1: 101}
    robot.handles = SimpleNamespace(ik_base=10, ik_target=20, joints=[1])
    robot.collides = lambda configs, attached_object=None: False

    assert robot.solve_ik_to_pose([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]) == [0.4]
    assert [call[3] for call in ik.constraints] == [ik.constraint_pose, ik.constraint_position | ik.constraint_alpha_beta]
    assert ik.handle_results == [0]


def test_find_configs_for_pose_uses_simik_find_configs_and_filters_malformed_results():
    sim = IkTargetRecordingSim()
    ik = MultiConfigIk([[0.1], ["bad"], [0.2]])
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = ik
    robot.ik_env = 1
    robot.ik_group = 2
    robot.sim_to_ik = {10: 100, 20: 200, 1: 101}
    robot.handles = SimpleNamespace(ik_base=10, ik_target=20, joints=[1])

    configs = robot.find_configs_for_pose([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0])

    assert configs == [[0.1], [0.2]]
    assert ik.find_configs_calls == [(1, 2, [101], {"maxDist": 0.28, "maxTime": 1.0, "findMultiple": True, "cMetric": [1.0]})]
    assert sim.target_pose == [9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]


def test_find_configs_for_pose_sets_target_relative_to_ur10_base_when_present():
    sim = IkTargetRecordingSim()
    ik = MultiConfigIk([[0.1]])
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = ik
    robot.ik_env = 1
    robot.ik_group = 2
    robot.sim_to_ik = {10: 100, 11: 110, 20: 200, 1: 101}
    robot.handles = SimpleNamespace(ik_base=10, ur10_base=11, ik_target=20, joints=[1])

    configs = robot.find_configs_for_pose([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0])

    assert configs == [[0.1]]
    assert sim.pose_requests == [(20, 11)]
    assert ik.object_pose_writes == [(1, 200, [9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0], 110)]


def test_find_configs_for_pose_ignores_include_rail_and_uses_arm_context():
    sim = IkTargetRecordingSim()
    ik = MultiConfigIk([[0.2]])
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = ik
    robot.ik_env = 1
    robot.ik_group = 2
    robot.ik_element = 55
    robot.sim_to_ik = {11: 110, 20: 200, 2: 102}
    robot.handles = SimpleNamespace(ik_base=10, rail_joint=1, ur10_base=11, ik_target=20, joints=[1, 2])
    robot.rail_enabled = True

    configs = robot.find_configs_for_pose([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0], include_rail=True)

    assert configs == [[0.2]]
    assert ik.find_configs_calls == [(1, 2, [102], {"maxDist": 0.28, "maxTime": 1.0, "findMultiple": True, "cMetric": [1.0]})]
    assert sim.pose_requests == [(20, 11)]
    assert ik.object_pose_writes == [(1, 200, [9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0], 110)]


def test_ik_target_sync_accepts_world_pose_and_converts_through_arm_base():
    sim = IkTargetRecordingSim()
    ik = FailingIk()
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = ik
    robot.ik_env = 1
    robot.ik_group = 2
    robot.sim_to_ik = {11: 110, 20: 200}
    robot.handles = SimpleNamespace(ik_base=10, ur10_base=11, ik_target=20)

    robot._sync_world_pose_to_arm_ik_target([9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0])

    assert sim.poses_written == [(20, [9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0])]
    assert sim.pose_requests == [(20, 11)]
    assert ik.object_pose_writes == [(1, 200, [9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0], 110)]


class PlanningSim:
    def __init__(self):
        self.handle_all = -2
        self.joint_positions = {1: 0.1, 2: 0.2, 3: 0.3}

    def getJointPosition(self, joint):
        return self.joint_positions[joint]


class PlanningOMPL:
    class Algorithm:
        RRTConnect = "rrt-connect"

    def __init__(self, solve_result=True, exact_result=True):
        self.solve_result = solve_result
        self.exact_result = exact_result
        self.calls = []
        self.path = [0.1, 0.2, 0.3, 1.0, 1.1, 1.2]

    def createTask(self, name):
        self.calls.append(("create", name))
        return 77

    def setAlgorithm(self, task, algorithm):
        self.calls.append(("algorithm", task, algorithm))

    def setStateSpaceForJoints(self, task, joints, projection):
        self.calls.append(("state_space", task, list(joints), list(projection)))

    def setCollisionPairs(self, task, pairs):
        self.calls.append(("collision_pairs", task, pairs))

    def setStartState(self, task, config):
        self.calls.append(("start", task, config))

    def setGoalState(self, task, config):
        self.calls.append(("goal", task, config))

    def setup(self, task):
        self.calls.append(("setup", task))

    def solve(self, task, timeout):
        self.calls.append(("solve", task, timeout))
        return self.solve_result

    def hasExactSolution(self, task):
        self.calls.append(("exact", task))
        return self.exact_result

    def simplifyPath(self, task, timeout):
        self.calls.append(("simplify", task, timeout))

    def getPath(self, task):
        self.calls.append(("path", task))
        return self.path

    def destroyTask(self, task):
        self.calls.append(("destroy", task))


class SequencedPlanningOMPL(PlanningOMPL):
    def __init__(self, solve_results, exact_results=None):
        super().__init__(solve_result=True, exact_result=True)
        self.solve_results = list(solve_results)
        self.exact_results = list(exact_results if exact_results is not None else [True] * len(self.solve_results))

    def solve(self, task, timeout):
        self.calls.append(("solve", task, timeout))
        return self.solve_results.pop(0)

    def hasExactSolution(self, task):
        self.calls.append(("exact", task))
        return self.exact_results.pop(0)


def test_plan_joint_path_configures_ompl_and_destroys_task_on_success():
    sim = PlanningSim()
    ompl = PlanningOMPL(solve_result=(True, "ok"), exact_result=[1])
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simOMPL = ompl
    robot.handles = SimpleNamespace(joints=[1, 2, 3], rail_joint=1)
    robot.robot_collection = 44
    robot.moving_collection = 45
    robot.obstacle_collection = 46
    robot.update_robot_collection = lambda attached_object=None: setattr(robot, "updated_attached", attached_object)

    path = robot.plan_joint_path([1.1, 1.2], attached_object=99)

    assert path == ompl.path
    assert robot.updated_attached == 99
    assert ("state_space", 77, [2, 3], [1, 1]) in ompl.calls
    assert ("start", 77, [0.2, 0.3]) in ompl.calls
    assert ("goal", 77, [1.0, 1.1, 1.2]) not in ompl.calls
    assert ("goal", 77, [1.1, 1.2]) in ompl.calls
    assert ("collision_pairs", 77, [45, 46, 44, 44]) in ompl.calls
    assert ompl.calls[-1] == ("destroy", 77)


def test_plan_joint_path_excludes_rail_from_state_space_when_disabled():
    sim = PlanningSim()
    ompl = PlanningOMPL(solve_result=True, exact_result=True)
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simOMPL = ompl
    robot.handles = SimpleNamespace(joints=[1, 2, 3], rail_joint=1)
    robot.rail_enabled = False
    robot.robot_collection = 44
    robot.moving_collection = 45
    robot.obstacle_collection = 46
    robot.update_robot_collection = lambda attached_object=None: None

    path = robot.plan_joint_path([1.1, 1.2])

    assert path == ompl.path
    assert ("state_space", 77, [2, 3], [1, 1]) in ompl.calls
    assert ("start", 77, [0.2, 0.3]) in ompl.calls
    assert ("goal", 77, [1.1, 1.2]) in ompl.calls


def test_plan_joint_path_returns_none_without_exact_solution_and_still_destroys_task():
    sim = PlanningSim()
    ompl = PlanningOMPL(solve_result=True, exact_result=False)
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simOMPL = ompl
    robot.handles = SimpleNamespace(joints=[1, 2, 3], rail_joint=1)
    robot.robot_collection = 44
    robot.moving_collection = 45
    robot.obstacle_collection = 46
    robot.update_robot_collection = lambda attached_object=None: None

    assert robot.plan_joint_path([1.1, 1.2]) is None
    assert ("simplify", 77, 10.0) not in ompl.calls
    assert ompl.calls[-1] == ("destroy", 77)


def test_plan_pose_path_tries_multiple_ik_configs_until_ompl_finds_exact_path():
    sim = PlanningSim()
    ik = MultiConfigIk([[0.5, 0.6], [1.1, 1.2]])
    ompl = SequencedPlanningOMPL(solve_results=[False, True])
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = ik
    robot.simOMPL = ompl
    robot.ik_env = 1
    robot.ik_group = 2
    robot.sim_to_ik = {10: 100, 20: 200, 1: 101, 2: 102, 3: 103}
    robot.handles = SimpleNamespace(ik_base=10, ik_target=20, joints=[1, 2, 3], rail_joint=1)
    robot.robot_collection = 44
    robot.moving_collection = 45
    robot.obstacle_collection = 46
    robot.update_robot_collection = lambda attached_object=None: None
    robot.collides = lambda configs, attached_object=None: False
    sim.target_pose = [0, 0, 0, 0, 0, 0, 1]
    sim.getObjectPose = lambda handle, relative_to=None: list(sim.target_pose)
    sim.setObjectPose = lambda handle, pose: setattr(sim, "target_pose", list(pose))

    path = robot.plan_pose_path([9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0], attached_object=99)

    assert path == ompl.path
    assert ("goal", 77, [0.5, 0.6]) in ompl.calls
    assert ("goal", 77, [1.1, 1.2]) in ompl.calls


def test_plan_pose_path_rejects_find_configs_candidate_that_misses_target_pose():
    sim = TipPoseByConfigSim()
    ik = MultiConfigIk([[0.1]])
    ik.handle_results = [ik.result_success]
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = ik
    robot.ik_env = 1
    robot.ik_group = 2
    robot.ik_element = 55
    robot.sim_to_ik = {10: 100, 20: 200, 30: 300, 1: 101}
    robot.handles = SimpleNamespace(ik_base=10, ik_target=20, ik_tip=30, joints=[1])
    robot.collides = lambda configs, attached_object=None: False
    planned_goals = []

    def plan_joint_path(config, attached_object=None):
        planned_goals.append(list(config))
        return [0.0, *config]

    robot.plan_joint_path = plan_joint_path
    target = [9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]

    assert robot.plan_pose_path(target, attached_object=99) == [0.0, 0.4]
    assert planned_goals == [[0.4]]


class FollowPathSim:
    def __init__(self):
        self.time = 0.0
        self.targets = []
        self.path_lengths_args = None
        self.trajectory_args = None

    def getPathLengths(self, path, dof):
        self.path_lengths_args = (path, dof)
        return [0.0, 1.0]

    def generateTimeOptimalTrajectory(self, path, path_lengths, min_max_vel, min_max_accel, samples, method, point_count):
        self.trajectory_args = (path, path_lengths, min_max_vel, min_max_accel, samples, method, point_count)
        return path, [0.0, 0.2]

    def getSimulationTime(self):
        return self.time

    def getPathInterpolatedConfig(self, path_pts, times, elapsed):
        return [elapsed, elapsed + 1.0]

    def setJointTargetPosition(self, joint, value):
        self.targets.append((joint, value))


class SteppingContext:
    def __init__(self, sim):
        self.sim = sim
        self.steps = []

    def step(self, count=1):
        self.steps.append(count)
        self.sim.time += 0.1 * count


def test_follow_joint_path_rejects_empty_and_misaligned_paths():
    robot = RobotController.__new__(RobotController)
    robot.handles = SimpleNamespace(joints=[1, 2])
    robot.sim = FollowPathSim()

    assert robot.follow_joint_path([]) is False
    assert robot.follow_joint_path([0.0, 1.0, 2.0]) is False
    assert "path length" in robot.last_error


def test_follow_joint_path_generates_time_optimal_trajectory_and_steps_to_end():
    sim = FollowPathSim()
    ctx = SteppingContext(sim)
    robot = RobotController.__new__(RobotController)
    robot.handles = SimpleNamespace(joints=[1, 2])
    robot.sim = sim
    robot.ctx = ctx

    assert robot.follow_joint_path([0.0, 1.0, 2.0, 3.0]) is True

    assert sim.path_lengths_args == ([0.0, 1.0, 2.0, 3.0], 2)
    assert sim.trajectory_args[2] == [-0.25, 0.25, -3.141592653589793, 3.141592653589793]
    assert ctx.steps[-1] == 10
    assert sim.targets[-2:] == [(1, 0.2), (2, 1.2)]


class MultiReturnPathLengthsSim(FollowPathSim):
    def getPathLengths(self, path, dof):
        self.path_lengths_args = (path, dof)
        return [0.0, 1.0], 1.0


def test_follow_joint_path_uses_first_value_from_remote_path_lengths():
    sim = MultiReturnPathLengthsSim()
    ctx = SteppingContext(sim)
    robot = RobotController.__new__(RobotController)
    robot.handles = SimpleNamespace(joints=[1, 2])
    robot.sim = sim
    robot.ctx = ctx

    assert robot.follow_joint_path([0.0, 1.0, 2.0, 3.0]) is True

    assert sim.trajectory_args[1] == [0.0, 1.0]


class CartesianMoveSim:
    def __init__(self, step_results=None, collision_results=None, final_tip_pose=None):
        self.handle_all = -2
        self.handle_tree = 1
        self.handle_single = 2
        self.start_pose = [0, 0, 0, 0, 0, 0, 1]
        self.final_tip_pose = final_tip_pose
        self.step_results = list(step_results or [(0, {"step": 1}), (1, {"step": 2})])
        self.collision_results = list(collision_results or [])
        self.pose_requests = []
        self.pose_writes = []
        self.move_to_pose_params = None
        self.step_calls = []
        self.cleaned = []
        self.items = []
        self.destroyed = []
        self.collision_checks = []
        self.next_collection = 50
        self.time = 0.0

    def getObjectPose(self, handle, relative_to=None):
        self.pose_requests.append((handle, relative_to))
        if handle == 20 and self.final_tip_pose is not None and self.step_calls:
            return list(self.final_tip_pose)
        return list(self.start_pose)

    def setObjectPose(self, handle, pose):
        self.pose_writes.append((handle, list(pose)))

    def moveToPose_init(self, params):
        self.move_to_pose_params = params
        return {"motion": 77}

    def moveToPose_step(self, motion):
        self.step_calls.append(motion)
        return self.step_results.pop(0)

    def moveToPose_cleanup(self, motion):
        self.cleaned.append(motion)

    def createCollection(self):
        self.next_collection += 1
        return self.next_collection

    def addItemToCollection(self, collection, mode, handle, options):
        self.items.append((collection, mode, handle, options))

    def destroyCollection(self, collection):
        self.destroyed.append(collection)

    def checkCollision(self, collection, other):
        self.collision_checks.append((collection, other))
        if self.collision_results:
            return self.collision_results.pop(0)
        return 0


def make_cartesian_robot(sim=None):
    sim = sim or CartesianMoveSim()
    ctx = SteppingContext(sim)
    robot = RobotController.__new__(RobotController)
    robot.sim = sim
    robot.simIK = SimpleNamespace(method_damped_least_squares=11, constraint_pose=22)
    robot.ctx = ctx
    robot.handles = SimpleNamespace(ik_base=10, ur10_base=11, ik_tip=20, ik_target=30, joints=[1, 2], rail_joint=1)
    robot.robot_collection = None
    robot.moving_collection = None
    robot.obstacle_collection = None
    return robot, sim, ctx


def test_move_cartesian_linear_uses_coppeliasim_move_to_pose_with_ik():
    target_pose = [0.06, 0, 0, 0, 0, 0, 1]
    robot, sim, ctx = make_cartesian_robot(CartesianMoveSim(final_tip_pose=target_pose))

    assert robot.move_cartesian_linear(target_pose, attached_object=99) is True

    assert sim.pose_requests == [(20, None), (20, None)]
    assert sim.pose_writes[0] == (30, sim.start_pose)
    assert sim.pose_writes[-1] == (30, target_pose)
    assert sim.move_to_pose_params["targetPose"] == target_pose
    assert sim.move_to_pose_params["ik"] == {
        "tip": 20,
        "target": 30,
        "base": 11,
        "joints": [2],
        "method": 11,
        "damping": 0.3,
        "iterations": 99,
        "constraints": 22,
        "precision": [0.001, pytest.approx(0.5 * 3.141592653589793 / 180.0)],
    }
    assert len(sim.move_to_pose_params["maxVel"]) == 4
    assert len(sim.move_to_pose_params["maxAccel"]) == 4
    assert len(sim.move_to_pose_params["maxJerk"]) == 4
    assert sim.step_calls == [{"motion": 77}, {"motion": 77}]
    assert sim.cleaned == [{"motion": 77}]
    assert ctx.steps == [1, 10]
    assert sim.items[:6] == [
        (51, sim.handle_tree, 10, 0),
        (52, sim.handle_tree, 10, 0),
        (52, sim.handle_single, 99, 0),
        (53, sim.handle_all, -1, 0),
        (53, sim.handle_tree, 10, 1),
        (53, sim.handle_single, 99, 1),
    ]
    assert sim.collision_checks == [(52, 53), (51, 51), (52, 53), (51, 51)]


def test_move_cartesian_linear_ignores_include_rail_for_arm_only_ik():
    target_pose = [0.06, 0, 0, 0, 0, 0, 1]
    robot, sim, _ctx = make_cartesian_robot(CartesianMoveSim(final_tip_pose=target_pose))

    assert robot.move_cartesian_linear(target_pose, include_rail=True) is True

    assert sim.move_to_pose_params["ik"]["joints"] == [2]
    assert sim.move_to_pose_params["ik"]["base"] == 11


def test_move_cartesian_linear_cannot_include_rail_when_rail_is_disabled():
    target_pose = [0.06, 0, 0, 0, 0, 0, 1]
    robot, sim, _ctx = make_cartesian_robot(CartesianMoveSim(final_tip_pose=target_pose))
    robot.rail_enabled = False

    assert robot.move_cartesian_linear(target_pose, include_rail=True) is True

    assert sim.move_to_pose_params["ik"]["joints"] == [2]


def test_move_cartesian_linear_stops_when_step_collision_is_detected():
    robot, sim, ctx = make_cartesian_robot(CartesianMoveSim(collision_results=[1]))

    assert robot.move_cartesian_linear([0.06, 0, 0, 0, 0, 0, 1]) is False

    assert "collision during Cartesian move" in robot.last_error
    assert sim.cleaned == [{"motion": 77}]
    assert ctx.steps == []


def test_move_cartesian_linear_rejects_success_result_when_tip_misses_target_pose():
    target_pose = [0.06, 0, 0, 0, 0, 0, 1]
    robot, sim, ctx = make_cartesian_robot(CartesianMoveSim(final_tip_pose=[0.20, 0, 0, 0, 0, 0, 1]))

    assert robot.move_cartesian_linear(target_pose) is False

    assert "Cartesian move finished away from target" in robot.last_error
    assert "position_error" in robot.last_error
    assert sim.cleaned == [{"motion": 77}]
    assert ctx.steps == [1, 10]


def test_move_cartesian_linear_starts_without_waypoint_prevalidation():
    target_pose = [0.09, 0, 0, 0, 0, 0, 1]
    robot, sim, ctx = make_cartesian_robot(CartesianMoveSim(final_tip_pose=target_pose))

    def solve_ik_to_pose(pose, attached_object=None, include_rail=None):
        raise AssertionError("linear mode should not pre-validate IK waypoints")

    robot.solve_ik_to_pose = solve_ik_to_pose

    assert robot.move_cartesian_linear(target_pose, attached_object=99) is True

    assert sim.move_to_pose_params["targetPose"] == target_pose
    assert sim.step_calls == [{"motion": 77}, {"motion": 77}]
    assert ctx.steps == [1, 10]


def test_move_cartesian_linear_cleans_up_and_reports_aborted_motion():
    robot, sim, ctx = make_cartesian_robot(CartesianMoveSim(step_results=[(2, {"aborted": True})]))

    assert robot.move_cartesian_linear([0.06, 0, 0, 0, 0, 0, 1]) is False

    assert "aborted" in robot.last_error
    assert sim.cleaned == [{"motion": 77}]
    assert ctx.steps == []


def test_move_cartesian_linear_reports_negative_motion_step_result():
    robot, sim, ctx = make_cartesian_robot(CartesianMoveSim(step_results=[(-1, {"error": True})]))

    assert robot.move_cartesian_linear([0.06, 0, 0, 0, 0, 0, 1]) is False

    assert "failed with result -1" in robot.last_error
    assert sim.cleaned == [{"motion": 77}]
    assert ctx.steps == []


class PickPlaceSim:
    objfloatparam_objbbox_min_x = 0
    objfloatparam_objbbox_min_y = 1
    objfloatparam_objbbox_min_z = 2
    objfloatparam_objbbox_max_x = 3
    objfloatparam_objbbox_max_y = 4
    objfloatparam_objbbox_max_z = 5

    def __init__(self, shape_bb=None):
        self.shape_bb = shape_bb if shape_bb is not None else [0.2, 0.4, 0.6]
        self.float_params = {
            self.objfloatparam_objbbox_min_x: -0.1,
            self.objfloatparam_objbbox_min_y: -0.2,
            self.objfloatparam_objbbox_min_z: -0.3,
            self.objfloatparam_objbbox_max_x: 0.1,
            self.objfloatparam_objbbox_max_y: 0.2,
            self.objfloatparam_objbbox_max_z: 0.3,
        }
        self.poses = {
            42: [1.0, 2.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            7: [10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0],
            8: [3.0, 4.0, 1.8, 0.0, 0.0, 0.0, 1.0],
            20: [1.0, 2.0, 1.4, 0.2, 0.3, 0.4, 0.8],
        }
        self.objects = {"/loading_home_pose": 8}
        self.signals = {}
        self.multiplied = []
        self.parents = []
        self.pose_requests = []

    def getObject(self, path, options=None):
        handle = self.objects.get(path, -1)
        if handle >= 0 or options and options.get("noError"):
            return handle
        raise RuntimeError(f"object not found: {path}")

    def getShapeBB(self, handle):
        if isinstance(self.shape_bb, Exception):
            raise self.shape_bb
        return self.shape_bb

    def getObjectFloatParam(self, handle, param):
        return self.float_params[param]

    def getObjectPose(self, handle, relative_to=None):
        self.pose_requests.append((handle, relative_to))
        return list(self.poses[handle])

    def multiplyPoses(self, pose, local_pose):
        self.multiplied.append((list(pose), list(local_pose)))
        return [
            pose[0] + local_pose[0],
            pose[1] + local_pose[1],
            pose[2] + local_pose[2],
            *local_pose[3:],
        ]

    def setInt32Signal(self, name, value):
        self.signals[name] = value

    def setObjectParent(self, handle, parent, keep_in_place):
        self.parents.append((handle, parent, keep_in_place))

    def resetDynamicObject(self, handle):
        self.parents.append(("reset", handle))


class PickPlaceCtx:
    def __init__(self):
        self.steps = []

    def step(self, count=1):
        self.steps.append(count)


class PickPlaceRobot:
    def __init__(self, sim, cartesian_results=None, config_results=None):
        self.sim = sim
        self.ctx = PickPlaceCtx()
        self.handles = SimpleNamespace(pallet=7, ik_tip=20)
        self.cartesian_results = list(cartesian_results or [])
        self.config_results = list(config_results or [])
        self.pose_reached_results = []
        self.ik_calls = []
        self.config_calls = []
        self.plan_pose_calls = []
        self.plan_calls = []
        self.follow_calls = []
        self.cartesian_calls = []
        self.rail_preposition_calls = []
        self.action_log = []

    def solve_ik_to_pose(self, pose, attached_object=None):
        self.action_log.append(("ik", list(pose), attached_object))
        self.ik_calls.append((list(pose), attached_object))
        return [pose[0], pose[1], pose[2]]

    def move_to_config(self, goal, attached_object=None, allow_start_collision=False):
        self.action_log.append(("config", list(goal), attached_object, allow_start_collision))
        self.config_calls.append((list(goal), attached_object, allow_start_collision))
        if self.config_results:
            return self.config_results.pop(0)
        return True

    def plan_joint_path(self, goal, attached_object=None):
        self.plan_calls.append((list(goal), attached_object))
        return [*goal, 9.0]

    def plan_pose_path(self, pose, attached_object=None, fallback_config=None):
        self.plan_pose_calls.append((list(pose), attached_object, None if fallback_config is None else list(fallback_config)))
        return [*(fallback_config or pose[:3]), 9.0]

    def follow_joint_path(self, path):
        self.follow_calls.append(list(path))
        return True

    def move_cartesian_linear(self, pose, attached_object=None, include_rail=False):
        self.action_log.append(("linear", list(pose), attached_object, include_rail))
        self.cartesian_calls.append((list(pose), attached_object, include_rail))
        if self.cartesian_results:
            return self.cartesian_results.pop(0)
        return True

    def preposition_rail_for_pose(self, pose):
        self.action_log.append(("rail", list(pose)))
        self.rail_preposition_calls.append(list(pose))
        return True

    def _validate_tip_reached_pose(self, pose):
        if self.pose_reached_results:
            result = self.pose_reached_results.pop(0)
            if not result:
                self.last_error = "tip did not reach requested pose"
            return result
        return True


def test_object_size_falls_back_to_bbox_float_params():
    from python_controller.pick_place import object_size

    sim = PickPlaceSim(shape_bb=RuntimeError("not available"))

    assert object_size(sim, 42) == [0.2, 0.4, 0.6]


def test_pick_pre_pose_is_directly_above_pick_pose():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim)
    controller = PickPlaceController(robot, belt=object())

    pre, pick, _hold = controller.pick_poses_for_object(42)

    assert sim.pose_requests[0] == (42, -1)
    assert pre[:2] == pytest.approx(pick[:2])
    assert pre[2] == pytest.approx(pick[2] + controller.approach)
    assert pre[3:] == pytest.approx(pick[3:])


def test_pick_toggles_suction_and_sets_attached_object_on_success():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim)
    controller = PickPlaceController(robot, belt=object())

    assert controller.pick(42) is True

    assert sim.signals["suctionPadEnabled"] == 1
    assert controller.attached_object == 42
    assert sim.parents == [(42, 20, True)]
    assert robot.ctx.steps == [20]
    assert robot.cartesian_calls[-1][1] == 42
    assert robot.ik_calls[0][0] == pytest.approx([1.0, 2.0, 0.705, 0.0, 1.0, 0.0, 0.0])
    assert robot.cartesian_calls[-1][0] == pytest.approx([1.0, 2.0, 1.4, 0.0, 1.0, 0.0, 0.0])
    assert robot.plan_calls == []


def test_pick_uses_item_relative_hold_height_without_intermediate_move():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim)
    controller = PickPlaceController(robot, belt=object())

    assert controller.pick(42) is True

    assert robot.cartesian_calls[0][0][:3] == pytest.approx([1.0, 2.0, 0.605])
    assert robot.cartesian_calls[1][0] == pytest.approx([1.0, 2.0, 1.4, 0.0, 1.0, 0.0, 0.0])
    assert robot.cartesian_calls[1][1:] == (42, False)
    assert len(robot.cartesian_calls) == 2
    assert len(robot.ik_calls) == 1
    assert robot.config_calls[0][:2] == ([1.0, 2.0, 0.705], None)
    assert len(robot.config_calls) == 1


def test_pick_prepositions_rail_before_arm_moves_to_pre_pick():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim)
    controller = PickPlaceController(robot, belt=object())

    assert controller.pick(42) is True

    assert robot.rail_preposition_calls[0] == pytest.approx([1.0, 2.0, 0.705, 0.0, 1.0, 0.0, 0.0])
    assert robot.action_log[0][0] == "rail"
    assert robot.action_log[1][0] == "ik"


def test_pick_lift_uses_linear_motion_without_ompl():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim, config_results=[True, True, False])
    controller = PickPlaceController(robot, belt=object())

    assert controller.pick(42) is True

    assert robot.config_calls == [([1.0, 2.0, 0.705], None, False)]
    assert robot.plan_pose_calls == []
    assert robot.follow_calls == []
    assert robot.cartesian_calls[-1] == ([1.0, 2.0, 1.4, 0.0, 1.0, 0.0, 0.0], 42, False)


def test_pick_cleans_suction_and_attachment_when_hold_move_fails():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim, cartesian_results=[True, False, False], config_results=[True, False])
    controller = PickPlaceController(robot, belt=object())

    def follow_joint_path(path):
        robot.follow_calls.append(list(path))
        robot.last_error = "joint path following failed"
        return False

    robot.follow_joint_path = follow_joint_path

    assert controller.pick(42) is False

    assert sim.signals["suctionPadEnabled"] == 0
    assert controller.attached_object is None
    assert sim.parents == [(42, 20, True), (42, -1, True), ("reset", 42)]
    assert robot.ctx.steps == [20]
    assert "pick lift failed" in controller.last_error


def test_pick_reports_pre_approach_failure_with_robot_reason():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim, cartesian_results=[False])
    controller = PickPlaceController(robot, belt=object())

    def solve_ik_to_pose(pose, attached_object=None):
        robot.ik_calls.append((list(pose), attached_object))
        robot.last_error = "IK did not converge"
        return None

    robot.solve_ik_to_pose = solve_ik_to_pose

    def move_cartesian_linear(pose, attached_object=None, include_rail=False):
        robot.cartesian_calls.append((list(pose), attached_object, include_rail))
        robot.last_error = "fallback Cartesian move failed"
        return False

    robot.move_cartesian_linear = move_cartesian_linear

    assert controller.pick(42) is False

    assert "pick pre-approach failed" in controller.last_error
    assert "IK did not converge" in controller.last_error
    assert "fallback Cartesian move failed" in controller.last_error


def test_move_to_pose_falls_back_to_cartesian_move_when_seed_ik_fails():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim)
    controller = PickPlaceController(robot, belt=object())

    def solve_ik_to_pose(pose, attached_object=None):
        robot.ik_calls.append((list(pose), attached_object))
        robot.last_error = "IK did not converge"
        return None

    robot.solve_ik_to_pose = solve_ik_to_pose

    assert controller.move_to_pose([1, 2, 3, 0, 0, 0, 1], attached_object=42) is True

    assert robot.plan_calls == []
    assert robot.follow_calls == []
    assert robot.cartesian_calls == [([1, 2, 3, 0, 0, 0, 1], 42, False)]


def test_move_to_pose_reports_both_ik_and_cartesian_fallback_errors():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim, cartesian_results=[False])
    controller = PickPlaceController(robot, belt=object())

    def solve_ik_to_pose(pose, attached_object=None):
        robot.ik_calls.append((list(pose), attached_object))
        robot.last_error = "IK did not converge"
        return None

    def move_cartesian_linear(pose, attached_object=None, include_rail=False):
        robot.cartesian_calls.append((list(pose), attached_object, include_rail))
        robot.last_error = "fallback Cartesian move failed"
        return False

    robot.solve_ik_to_pose = solve_ik_to_pose
    robot.move_cartesian_linear = move_cartesian_linear

    assert controller.move_to_pose([1, 2, 3, 0, 0, 0, 1], attached_object=42) is False

    assert "IK did not converge" in controller.last_error
    assert "fallback Cartesian move failed" in controller.last_error


def test_move_to_pose_uses_plan_mode_before_cartesian_when_config_move_fails():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim, config_results=[False])
    controller = PickPlaceController(robot, belt=object())

    assert controller.move_to_pose([1, 2, 3, 0, 0, 0, 1], attached_object=42) is True

    assert robot.config_calls == [([1, 2, 3], 42, False)]
    assert robot.plan_pose_calls == [([1, 2, 3, 0, 0, 0, 1], 42, [1, 2, 3])]
    assert robot.plan_calls == []
    assert robot.follow_calls == [[1, 2, 3, 9.0]]
    assert robot.cartesian_calls == []


def test_move_to_pose_replans_when_config_motion_finishes_away_from_pose():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim)
    robot.pose_reached_results = [False, True]
    controller = PickPlaceController(robot, belt=object())

    assert controller.move_to_pose([1, 2, 3, 0, 0, 0, 1], attached_object=42) is True

    assert robot.config_calls == [([1, 2, 3], 42, False)]
    assert robot.plan_pose_calls == [([1, 2, 3, 0, 0, 0, 1], 42, [1, 2, 3])]
    assert robot.follow_calls == [[1, 2, 3, 9.0]]
    assert robot.cartesian_calls == []


def test_move_to_pose_can_disable_cartesian_fallback_for_planned_lifts():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim, config_results=[False])
    controller = PickPlaceController(robot, belt=object())

    def plan_pose_path(pose, attached_object=None, fallback_config=None):
        robot.plan_pose_calls.append((list(pose), attached_object, None if fallback_config is None else list(fallback_config)))
        robot.last_error = "OMPL returned no exact solution"
        return None

    robot.plan_pose_path = plan_pose_path

    assert controller.move_to_pose(
        [1, 2, 3, 0, 0, 0, 1],
        attached_object=42,
        allow_cartesian_fallback=False,
    ) is False

    assert robot.cartesian_calls == []
    assert "OMPL returned no exact solution" in controller.last_error


def test_move_to_pose_uses_arm_only_cartesian_after_config_and_plan_fail():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim, cartesian_results=[True], config_results=[False])
    controller = PickPlaceController(robot, belt=object())

    def plan_pose_path(pose, attached_object=None, fallback_config=None):
        robot.plan_pose_calls.append((list(pose), attached_object, None if fallback_config is None else list(fallback_config)))
        robot.last_error = "OMPL returned no exact solution"
        return None

    robot.plan_pose_path = plan_pose_path

    assert controller.move_to_pose([1, 2, 3, 0, 0, 0, 1], attached_object=42) is True

    assert robot.config_calls == [([1, 2, 3], 42, False)]
    assert robot.follow_calls == []
    assert robot.cartesian_calls == [([1, 2, 3, 0, 0, 0, 1], 42, False)]


def test_move_to_pose_reports_ompl_and_cartesian_fallback_errors():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim, config_results=[False])
    controller = PickPlaceController(robot, belt=object())

    def plan_pose_path(pose, attached_object=None, fallback_config=None):
        robot.plan_pose_calls.append((list(pose), attached_object, None if fallback_config is None else list(fallback_config)))
        robot.last_error = "OMPL returned no exact solution"
        return None

    def move_cartesian_linear(pose, attached_object=None, include_rail=False):
        robot.cartesian_calls.append((list(pose), attached_object, include_rail))
        robot.last_error = "collision during Cartesian move"
        return False

    robot.plan_pose_path = plan_pose_path
    robot.move_cartesian_linear = move_cartesian_linear

    assert controller.move_to_pose([1, 2, 3, 0, 0, 0, 1], attached_object=42) is False

    assert "OMPL returned no exact solution" in controller.last_error
    assert "collision during Cartesian move" in controller.last_error


def test_place_pre_pose_is_directly_above_place_pose():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim)
    controller = PickPlaceController(robot, belt=object())

    pre, place = controller.place_poses_for_object(42, [0.0, 0.0, 1.0])

    assert sim.pose_requests[0] == (7, -1)
    assert pre[:2] == pytest.approx(place[:2])
    assert pre[2] == pytest.approx(place[2] + controller.approach)
    assert pre[3:] == pytest.approx(place[3:])


def test_place_defaults_position_and_clears_suction_and_attachment_after_release():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim)
    controller = PickPlaceController(robot, belt=object())
    controller.attached_object = 42

    assert controller.place() is True

    assert sim.signals["suctionPadEnabled"] == 0
    assert controller.attached_object is None
    assert sim.parents == [(42, -1, True), ("reset", 42)]
    assert robot.ctx.steps == [20]
    assert robot.config_calls[0][0] == pytest.approx([10.1, 20.2, 31.4])
    assert robot.config_calls[0][1] == 42
    assert robot.cartesian_calls[0][0][:3] == pytest.approx([10.1, 20.2, 31.3])
    assert robot.cartesian_calls[0][1] == 42
    assert robot.cartesian_calls[1][0][:3] == pytest.approx([10.1, 20.2, 31.4])
    assert robot.cartesian_calls[1][1] is None
    assert sim.multiplied[-1][1][:3] == pytest.approx([0.1, 0.2, 1.3])


def test_place_prepositions_rail_and_preserves_hold_orientation_for_pre_place_transfer():
    sim = PickPlaceSim()
    sim.poses[20] = [1.0, 2.0, 1.4, 0.2, 0.3, 0.4, 0.8]
    robot = PickPlaceRobot(sim)
    controller = PickPlaceController(robot, belt=object())
    controller.attached_object = 42

    assert controller.place() is True

    assert robot.rail_preposition_calls[0][:3] == pytest.approx([10.1, 20.2, 31.4])
    assert robot.rail_preposition_calls[0][3:] == pytest.approx([0.2, 0.3, 0.4, 0.8])
    assert robot.action_log[0][0] == "rail"
    assert robot.action_log[1][0] == "ik"
    assert robot.ik_calls[0][0][3:] == pytest.approx([0.2, 0.3, 0.4, 0.8])


def test_next_pick_moves_from_pre_place_to_next_pre_pick_with_config_motion():
    sim = PickPlaceSim()
    robot = PickPlaceRobot(sim)
    controller = PickPlaceController(robot, belt=object())
    controller.attached_object = 42

    assert controller.place() is True
    assert robot.ik_calls[-1][0] == pytest.approx([10.1, 20.2, 31.4, 0.2, 0.3, 0.4, 0.8])

    assert controller.pick(42) is True

    assert robot.ik_calls[-1][0] == pytest.approx([1.0, 2.0, 0.705, 0.0, 1.0, 0.0, 0.0])
    assert robot.config_calls[-1] == ([1.0, 2.0, 0.705], None, False)


class MainLoopSim:
    simulation_advancing_abouttostop = 99

    def getSimulationState(self):
        return 1


class MainLoopContext:
    def __init__(self):
        self.sim = MainLoopSim()
        self.steps = 0

    def step(self):
        self.steps += 1


def install_main_loop_fakes(monkeypatch, place_result=True, attached_after_place=None):
    from python_controller import main as main_module

    records = {"pick_calls": [], "place_calls": [], "picking_signals": []}

    def fake_connect(port):
        records["port"] = port
        records["ctx"] = MainLoopContext()
        return records["ctx"]

    def fake_ensure_started(ctx, scene_path=None):
        records["started"] = (ctx, scene_path)

    class FakeBelt:
        def __init__(self, sim):
            records["belt_sim"] = sim

        def full_transition_items(self):
            return [SimpleNamespace(handle=42), SimpleNamespace(handle=43)]

        def set_robot_picking(self, enabled):
            records["picking_signals"].append(enabled)

    class FakeRobot:
        def __init__(self, ctx, rail_enabled=True):
            records["robot_ctx"] = ctx
            records["rail_enabled"] = rail_enabled

    class FakePickPlace:
        def __init__(self, robot, belt):
            records["pick_place_args"] = (robot, belt)
            self.attached_object = None
            self.last_error = None

        def pick(self, handle):
            records["pick_calls"].append(handle)
            return True

        def place(self, position):
            records["place_calls"].append(position)
            self.attached_object = attached_after_place
            if isinstance(place_result, Exception):
                raise place_result
            if place_result is False:
                self.last_error = "fake place failure"
            return place_result

    monkeypatch.setattr(main_module, "connect", fake_connect)
    monkeypatch.setattr(main_module, "ensure_started", fake_ensure_started)
    monkeypatch.setattr(main_module, "BeltMonitor", FakeBelt)
    monkeypatch.setattr(main_module, "RobotController", FakeRobot)
    monkeypatch.setattr(main_module, "PickPlaceController", FakePickPlace)
    monkeypatch.setattr(main_module.random, "choice", lambda items: items[0])
    return main_module, records


def test_main_loop_runs_bounded_cycle_with_fake_controllers(monkeypatch):
    main_module, records = install_main_loop_fakes(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "python_controller",
            "--port",
            "23001",
            "--scene",
            "scene.ttt",
            "--place",
            "1.0",
            "2.0",
            "3.0",
            "--max-cycles",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    main_module.main()

    assert records["port"] == 23001
    assert records["started"] == (records["ctx"], "scene.ttt")
    assert records["belt_sim"] is records["ctx"].sim
    assert records["robot_ctx"] is records["ctx"]
    assert records["rail_enabled"] is True
    assert records["pick_calls"] == [42]
    assert records["place_calls"] == [[1.0, 2.0, 3.0]]
    assert records["picking_signals"] == [True, False]


def test_main_loop_can_disable_rail_from_cli(monkeypatch):
    main_module, records = install_main_loop_fakes(monkeypatch)
    monkeypatch.setattr("sys.argv", ["python_controller", "--max-cycles", "1", "--disable-rail"])

    main_module.main()

    assert records["rail_enabled"] is False


def test_main_loop_resets_robot_picking_when_place_raises(monkeypatch):
    failure = RuntimeError("place failed")
    main_module, records = install_main_loop_fakes(monkeypatch, place_result=failure)
    monkeypatch.setattr("sys.argv", ["python_controller", "--max-cycles", "1"])

    with pytest.raises(RuntimeError, match="place failed"):
        main_module.main()

    assert records["pick_calls"] == [42]
    assert records["picking_signals"] == [True, False]


def test_main_loop_keeps_belt_paused_when_place_fails_with_attached_object(monkeypatch):
    main_module, records = install_main_loop_fakes(monkeypatch, place_result=False, attached_after_place=42)
    monkeypatch.setattr("sys.argv", ["python_controller", "--max-cycles", "1"])

    with pytest.raises(RuntimeError, match="object still attached"):
        main_module.main()

    assert records["pick_calls"] == [42]
    assert records["place_calls"] == [[0.0, 0.0, 1.0]]
    assert records["picking_signals"] == [True]


def test_main_loop_logs_pick_place_failure_reason(monkeypatch, caplog):
    main_module, records = install_main_loop_fakes(monkeypatch, place_result=False)
    monkeypatch.setattr("sys.argv", ["python_controller", "--max-cycles", "1"])

    with caplog.at_level(logging.INFO):
        main_module.main()

    assert records["pick_calls"] == [42]
    assert "reason=fake place failure" in caplog.text
