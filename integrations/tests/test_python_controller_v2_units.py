from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from python_controller_v2.robot import RobotControllerV2, interpolate_pose_xyz


class V2FakeSim:
    sceneobject_joint = 99
    handle_tree = 100
    handle_single = 101
    handle_all = -2

    def __init__(self, tip_pose=None):
        self.objects = {
            "/mobile_arm": 10,
            "/mobile_arm/railJoint": 1,
            ":/mobile_arm/railJoint": 1,
            "/mobile_arm/railJoint/UR10": 11,
            "/mobile_arm/railJoint/UR10/ikTip": 20,
            "/ikTarget": 30,
            "/pallet": 40,
            ":/pallet": 40,
        }
        self.positions = {1: 0.0, 2: 0.1, 3: 0.2}
        self.target_pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        self.tip_pose = list(tip_pose) if tip_pose is not None else [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        self.solved_tip_pose = None
        self.tip_pose_by_config = {}
        self.joint_writes = []
        self.pose_requests = []
        self.pose_writes = []
        self.move_to_pose_params = None
        self.move_to_pose_step_results = [1]
        self.move_to_pose_steps = []
        self.move_to_pose_cleanups = []
        self.move_to_pose_done = False
        self.next_collection = 50
        self.collection_items = []
        self.destroyed_collections = []
        self.collision_checks = []
        self.collision_results = []

    def getObject(self, path, options=None):
        return self.objects.get(path, -1)

    def getObjectsInTree(self, base, object_type):
        assert base == 11
        assert object_type == self.sceneobject_joint
        return [2, 3]

    def getJointPosition(self, joint):
        return self.positions[joint]

    def setJointPosition(self, joint, value):
        self.positions[joint] = value
        self.joint_writes.append((joint, value))

    def getObjectPose(self, handle, relative_to=None):
        self.pose_requests.append((handle, relative_to))
        if handle == 20:
            if self.move_to_pose_done and self.move_to_pose_params is not None:
                return list(self.move_to_pose_params["targetPose"])
            if self.positions[2] == 0.4 and self.positions[3] == 0.5:
                if self.solved_tip_pose is not None:
                    return list(self.solved_tip_pose)
                return list(self.target_pose)
            key = tuple(round(self.positions[joint], 6) for joint in (2, 3))
            if key in self.tip_pose_by_config:
                return list(self.tip_pose_by_config[key])
            return list(self.tip_pose)
        if handle == 30:
            return list(self.target_pose)
        return list(self.tip_pose)

    def setObjectPose(self, handle, pose):
        assert handle == 30
        self.target_pose = list(pose)
        self.pose_writes.append((handle, list(pose)))

    def moveToPose_init(self, params):
        self.move_to_pose_params = params
        return {"motion": 77}

    def moveToPose_step(self, motion):
        self.move_to_pose_steps.append(motion)
        result = self.move_to_pose_step_results.pop(0)
        if result == 1:
            self.move_to_pose_done = True
        return result

    def moveToPose_cleanup(self, motion):
        self.move_to_pose_cleanups.append(motion)

    def createCollection(self):
        self.next_collection += 1
        return self.next_collection

    def addItemToCollection(self, collection, mode, handle, options):
        self.collection_items.append((collection, mode, handle, options))

    def destroyCollection(self, collection):
        self.destroyed_collections.append(collection)

    def checkCollision(self, collection, other):
        self.collision_checks.append((collection, other))
        if self.collision_results:
            return self.collision_results.pop(0)
        return 0


class V2FakeIK:
    method_damped_least_squares = 1
    constraint_position = 7
    constraint_alpha_beta = 8
    constraint_gamma = 16
    constraint_pose = 31
    result_success = 1

    def __init__(self):
        self.add_element_calls = []
        self.calculations = []
        self.constraints = []
        self.object_pose_writes = []
        self.syncs = []
        self.precisions = []
        self.handle_results = [self.result_success]
        self.find_configs_result = [[0.4, 0.5]]
        self.find_configs_calls = []
        self.generate_path_result = None
        self.generate_path_calls = []

    def createEnvironment(self):
        return 101

    def createGroup(self, env):
        return 202

    def setGroupCalculation(self, env, group, method, damping, iterations):
        self.calculations.append((env, group, method, damping, iterations))

    def setElementConstraints(self, env, group, element, constraints):
        self.constraints.append((env, group, element, constraints))

    def addElementFromScene(self, env, group, base, tip, target, constraints):
        self.add_element_calls.append((env, group, base, tip, target, constraints))
        return 303, {base: 110, tip: 120, target: 130, 2: 102, 3: 103}, "extra-live-api-value"

    def setElementPrecision(self, env, group, element, precision):
        self.precisions.append((env, group, element, list(precision)))

    def syncFromSim(self, env, groups):
        self.syncs.append((env, list(groups)))

    def setObjectPose(self, env, target, pose, base):
        self.object_pose_writes.append((env, target, list(pose), base))

    def handleGroup(self, env, group):
        return self.handle_results.pop(0) if self.handle_results else self.result_success

    def getJointPosition(self, env, joint):
        return {102: 0.4, 103: 0.5}[joint]

    def findConfigs(self, env, group, joints, params=None, configs=None):
        self.find_configs_calls.append((env, group, list(joints), dict(params or {}), configs))
        return [list(config) for config in self.find_configs_result]

    def generatePath(self, env, group, joints, tip, path_point_count, validation_callback=None, aux_data=None):
        self.generate_path_calls.append((env, group, list(joints), tip, path_point_count, validation_callback, aux_data))
        return None if self.generate_path_result is None else list(self.generate_path_result)


class V2FakeOMPL:
    class Algorithm:
        RRTConnect = 88
        RRTstar = 89

    def __init__(self):
        self.calls = []
        self.solve_results = [True]
        self.exact_results = [True]
        self.path = [0.1, 0.2, 0.4, 0.5]
        self.next_task = 70

    def createTask(self, name):
        self.next_task += 1
        self.calls.append(("create", name))
        return self.next_task

    def setAlgorithm(self, task, algorithm):
        self.calls.append(("algorithm", task, algorithm))

    def setStateSpaceForJoints(self, task, joints, projection):
        self.calls.append(("state_space", task, list(joints), list(projection)))

    def setCollisionPairs(self, task, pairs):
        self.calls.append(("collision_pairs", task, list(pairs)))

    def setStartState(self, task, config):
        self.calls.append(("start", task, list(config)))

    def setGoalState(self, task, config):
        self.calls.append(("goal", task, list(config)))

    def setGoalStates(self, task, configs):
        self.calls.append(("goals", task, [list(config) for config in configs]))

    def setup(self, task):
        self.calls.append(("setup", task))

    def solve(self, task, timeout):
        self.calls.append(("solve", task, timeout))
        return self.solve_results.pop(0)

    def hasExactSolution(self, task):
        self.calls.append(("exact", task))
        return self.exact_results.pop(0)

    def simplifyPath(self, task, timeout):
        self.calls.append(("simplify", task, timeout))

    def interpolatePath(self, task, state_count=0):
        self.calls.append(("interpolate", task, state_count))

    def getPath(self, task):
        self.calls.append(("path", task))
        return list(self.path)

    def destroyTask(self, task):
        self.calls.append(("destroy", task))


class V2FakeClient:
    def __init__(self, ik, ompl):
        self.ik = ik
        self.ompl = ompl
        self.required = []

    def require(self, name):
        self.required.append(name)
        if name == "simIK":
            return self.ik
        if name == "simOMPL":
            return self.ompl
        raise KeyError(name)


def make_robot(tip_pose=None):
    sim = V2FakeSim(tip_pose=tip_pose)
    ik = V2FakeIK()
    ompl = V2FakeOMPL()
    ctx = SimpleNamespace(sim=sim, client=V2FakeClient(ik, ompl), step=lambda count=1: None)
    return RobotControllerV2(ctx), sim, ik, ompl, ctx


def assert_pose_close(actual, expected):
    assert actual[:3] == pytest.approx(expected[:3])
    assert actual[3:7] == pytest.approx(expected[3:7])


def test_v2_setup_uses_arm_only_ik_and_requires_ompl():
    robot, _sim, ik, _ompl, ctx = make_robot()

    assert ctx.client.required == ["simIK", "simOMPL"]
    assert robot.handles.rail_joint == 1
    assert robot.handles.pallet == 40
    assert robot.handles.arm_joints == [2, 3]
    assert ik.add_element_calls == [(101, 202, 11, 20, 30, ik.constraint_pose)]
    assert ik.precisions == [(101, 202, 303, [0.0005, 0.005])]


def test_v2_controls_rail_and_arm_joints_separately():
    robot, sim, _ik, _ompl, _ctx = make_robot()

    robot.move_rail_to(0.3, max_step=1.0)
    robot.move_arm_to_config([0.7, 0.8], max_step=1.0)

    assert sim.positions[1] == pytest.approx(0.3)
    assert sim.positions[2] == pytest.approx(0.7)
    assert sim.positions[3] == pytest.approx(0.8)
    assert (1, 0.3) in sim.joint_writes
    assert (2, 0.7) in sim.joint_writes
    assert (3, 0.8) in sim.joint_writes


def test_v2_solve_arm_ik_returns_arm_config_only():
    robot, sim, ik, _ompl, _ctx = make_robot()
    target = [9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0]

    config = robot.solve_arm_ik(target)

    assert config == [0.4, 0.5]
    assert sim.positions[1] == pytest.approx(0.0)
    assert sim.pose_writes[-1] == (30, target)
    assert ik.object_pose_writes[-1] == (101, 130, target, 110)
    assert ik.find_configs_calls == []


def test_v2_solve_arm_ik_uses_exact_target_pose():
    robot, sim, ik, _ompl, _ctx = make_robot()
    target = [9.0, 8.0, 7.0, 0.5, 0.5, 0.5, 0.5]
    sim.tip_pose_by_config[(0.4, 0.5)] = target

    config = robot.solve_arm_ik(target)

    assert config == [0.4, 0.5]
    assert_pose_close(sim.pose_writes[-1][1], target)
    assert_pose_close(ik.object_pose_writes[-1][2], target)
    assert ik.find_configs_calls == []


def test_v2_solve_arm_ik_uses_direct_handle_group_not_find_configs():
    robot, sim, ik, _ompl, _ctx = make_robot()
    target = [9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0]
    ik.handle_results = [ik.result_success]

    config = robot.solve_arm_ik(target)

    assert config == [0.4, 0.5]
    assert robot.last_error is None
    assert ik.constraints == [(101, 202, 303, ik.constraint_pose)]
    assert ik.find_configs_calls == []


def test_v2_linear_tip_motion_uses_sim_move_to_pose():
    robot, sim, _ik, _ompl, _ctx = make_robot()
    sim.tip_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    assert robot.move_arm_linear_by(dz=0.05, max_step=0.025) is True

    params = sim.move_to_pose_params
    assert params["targetPose"] == pytest.approx([0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0])
    assert params["ik"]["tip"] == 20
    assert params["ik"]["target"] == 30
    assert params["ik"]["base"] == 11
    assert params["ik"]["joints"] == [2, 3]
    assert params["ik"]["constraints"] == _ik.constraint_pose
    assert params["maxVel"] == [0.4, 0.4, 0.4, 1.8]
    assert sim.move_to_pose_steps == [{"motion": 77}]
    assert sim.move_to_pose_cleanups == [{"motion": 77}]
    assert _ik.find_configs_calls == []


def test_v2_linear_collision_precheck_blocks_before_move_to_pose_and_restores_config():
    robot, sim, _ik, _ompl, _ctx = make_robot()
    sim.tip_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    sim.collision_results = [1]
    start_config = robot.get_arm_config()

    assert robot.move_arm_linear_by(dz=0.05, max_step=0.025) is False

    assert sim.move_to_pose_params is None
    assert robot.get_arm_config() == pytest.approx(start_config)
    assert "collision precheck" in robot.last_error
    assert sim.collision_checks == [(52, 53)]


def test_v2_linear_collision_precheck_seeds_each_ik_solve_from_previous_waypoint_config():
    robot, sim, _ik, _ompl, _ctx = make_robot()
    sim.tip_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    start_config = robot.get_arm_config()
    seeds = []
    solved_configs = [[0.4, 0.5], [0.6, 0.7]]

    def solve_from_current_seed(_pose):
        seeds.append(robot.get_arm_config())
        return solved_configs.pop(0)

    robot.solve_arm_ik = solve_from_current_seed

    assert robot.move_arm_linear_by(dz=0.06, max_step=0.03) is True

    assert seeds == [
        pytest.approx(start_config),
        pytest.approx([0.4, 0.5]),
    ]
    assert robot.get_arm_config() == pytest.approx(start_config)


def test_v2_linear_collision_precheck_builds_collections_with_attached_and_ignored_objects():
    robot, sim, _ik, _ompl, _ctx = make_robot()
    sim.tip_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    assert robot.move_arm_linear_by(dz=0.03, attached_object=99, ignored_objects=[42]) is True

    assert sim.collection_items[:7] == [
        (51, sim.handle_tree, 11, 0),
        (52, sim.handle_tree, 11, 0),
        (52, sim.handle_single, 99, 0),
        (53, sim.handle_all, -1, 0),
        (53, sim.handle_tree, 11, 1),
        (53, sim.handle_single, 99, 1),
        (53, sim.handle_single, 42, 1),
    ]
    assert _ik.find_configs_calls == []


def test_v2_linear_allow_final_contact_still_checks_robot_arm_against_obstacles():
    robot, sim, _ik, _ompl, _ctx = make_robot()
    sim.tip_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    sim.collision_results = [0, 0, 1]

    assert robot.move_arm_linear_by(
        dz=0.03,
        max_step=0.03,
        attached_object=99,
        allow_final_contact=True,
    ) is False

    assert sim.move_to_pose_params is None
    assert sim.collision_checks == [(52, 53), (51, 51), (51, 53)]
    assert "robot arm collided with an obstacle" in robot.last_error


def test_v2_find_arm_configs_uses_waypoint_ik_pipeline_without_find_configs():
    robot, sim, ik, _ompl, _ctx = make_robot()
    target = [1.0, 2.0, 3.06, 0.0, 0.0, 0.0, 1.0]
    sim.tip_pose_by_config[(0.4, 0.5)] = target
    start_config = robot.get_arm_config()

    configs = robot.find_arm_configs_for_pose(target, max_configs=2, max_time=0.25)

    assert configs == [pytest.approx([0.4, 0.5])]
    assert robot.get_arm_config() == pytest.approx(start_config)
    assert ik.find_configs_calls == []
    assert len(ik.syncs) >= 3
    assert sim.pose_writes[-1] == (30, target)


def test_v2_find_arm_configs_adds_generate_path_final_config_as_second_goal():
    robot, sim, ik, _ompl, _ctx = make_robot()
    target = [1.0, 2.0, 3.03, 0.0, 0.0, 0.0, 1.0]
    ik.generate_path_result = [0.1, 0.2, 0.6, 0.7]
    sim.tip_pose_by_config[(0.4, 0.5)] = target
    sim.tip_pose_by_config[(0.6, 0.7)] = target

    configs = robot.find_arm_configs_for_pose(target, max_configs=2, max_time=0.25)

    assert configs == [pytest.approx([0.4, 0.5]), pytest.approx([0.6, 0.7])]
    assert ik.generate_path_calls[-1][:5] == (101, 202, [102, 103], 120, 12)
    assert ik.find_configs_calls == []


def test_v2_find_arm_configs_uses_remembered_arm_configs_as_ik_seeds():
    robot, _sim, _ik, _ompl, _ctx = make_robot()
    target = [1.0, 2.0, 3.03, 0.0, 0.0, 0.0, 1.0]
    robot.remember_arm_config("pre_pick", [0.6, 0.7])
    seeds = []

    def solve_from_seed(_pose):
        seed = robot.get_arm_config()
        seeds.append(seed)
        if seed == pytest.approx([0.6, 0.7]):
            return [0.6, 0.7]
        robot._set_error("seed did not converge")
        return None

    robot.solve_arm_ik = solve_from_seed

    configs = robot.find_arm_configs_for_pose(target, max_configs=1, max_time=0.25)

    assert configs == [pytest.approx([0.6, 0.7])]
    assert seeds == [pytest.approx([0.1, 0.2]), pytest.approx([0.6, 0.7])]
    assert robot.get_arm_config() == pytest.approx([0.1, 0.2])


def test_v2_find_arm_configs_continues_after_failed_online_waypoint_ik():
    robot, sim, ik, _ompl, _ctx = make_robot()
    target = [1.0, 2.0, 3.03, 0.0, 0.0, 0.0, 1.0]
    ik.handle_results = [2, ik.result_success]
    sim.tip_pose_by_config[(0.4, 0.5)] = target

    configs = robot.find_arm_configs_for_pose(target, max_configs=2, max_time=0.25)

    assert configs == [pytest.approx([0.4, 0.5])]
    assert ik.find_configs_calls == []
    assert len(sim.pose_writes) >= 2


def test_v2_plan_arm_joint_path_configures_arm_only_ompl_and_destroys_task():
    robot, sim, _ik, ompl, _ctx = make_robot()

    path = robot.plan_arm_joint_path([0.4, 0.5], attached_object=99, planning_time=1.5, path_state_count=12)

    assert path == ompl.path
    assert ("state_space", 71, [2, 3], [1, 1]) in ompl.calls
    assert ("collision_pairs", 71, [52, 53, 51, 51]) in ompl.calls
    assert ("start", 71, [0.1, 0.2]) in ompl.calls
    assert ("goal", 71, [0.4, 0.5]) in ompl.calls
    assert ("solve", 71, 1.5) in ompl.calls
    assert ("interpolate", 71, 12) in ompl.calls
    assert ompl.calls[-1] == ("destroy", 71)
    assert sim.positions[1] == pytest.approx(0.0)


def test_v2_plan_arm_joint_path_uses_multi_goal_states_when_given_multiple_configs():
    robot, sim, _ik, ompl, _ctx = make_robot()

    path = robot.plan_arm_joint_path([[0.4, 0.5], [0.6, 0.7]], attached_object=99, planning_time=1.5)

    assert path == ompl.path
    assert ("goals", 71, [[0.4, 0.5], [0.6, 0.7]]) in ompl.calls
    assert ("goal", 71, [0.4, 0.5]) not in ompl.calls
    assert sim.positions[1] == pytest.approx(0.0)


def test_v2_move_arm_ompl_to_pose_uses_waypoint_ik_target_config_and_follows_arm_path():
    robot, sim, ik, ompl, _ctx = make_robot()
    target = [1.0, 2.0, 3.03, 0.0, 0.0, 0.0, 1.0]
    sim.tip_pose_by_config[(0.4, 0.5)] = target
    ompl.solve_results = [True]
    ompl.exact_results = [True]
    ompl.path = [0.1, 0.2, 0.4, 0.5]

    assert robot.move_arm_ompl_to_pose(target, max_configs=2, planning_time=0.5) is True

    assert ("goal", 71, [0.4, 0.5]) in ompl.calls
    assert ik.find_configs_calls == []
    assert sim.positions[1] == pytest.approx(0.0)
    assert sim.positions[2] == pytest.approx(0.4)
    assert sim.positions[3] == pytest.approx(0.5)


def test_v2_move_arm_ompl_to_pose_passes_multiple_goal_configs_to_one_ompl_task():
    robot, sim, _ik, ompl, _ctx = make_robot()
    target = [1.0, 2.0, 3.03, 0.0, 0.0, 0.0, 1.0]
    robot.find_arm_configs_for_pose = lambda target_pose, max_configs=8, max_time=10.0: [[0.4, 0.5], [0.6, 0.7]]
    robot.arm_configs_collide = lambda configs, attached_object=None, ignored_objects=None, allow_final_contact=False: False
    ompl.path = [0.1, 0.2, 0.6, 0.7]

    assert robot.move_arm_ompl_to_pose(target, max_configs=2, planning_time=0.5) is True

    assert ("goals", 71, [[0.4, 0.5], [0.6, 0.7]]) in ompl.calls
    assert sim.positions[2] == pytest.approx(0.6)
    assert sim.positions[3] == pytest.approx(0.7)


def test_v2_move_arm_ompl_to_pose_allows_final_contact_for_goal_collision_filter():
    robot, _sim, _ik, _ompl, _ctx = make_robot()
    target = [9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0]
    calls = []
    robot.find_arm_configs_for_pose = lambda target_pose, max_configs=8, max_time=10.0: [[0.4, 0.5]]
    robot.plan_arm_joint_path = lambda config, attached_object=None, planning_time=3.0, path_state_count=100: [0.1, 0.2, *config]
    robot.follow_arm_path = lambda path: True

    def arm_configs_collide(configs, attached_object=None, ignored_objects=None, allow_final_contact=False):
        calls.append((configs, attached_object, ignored_objects, allow_final_contact))
        return False

    robot.arm_configs_collide = arm_configs_collide

    assert robot.move_arm_ompl_to_pose(target, attached_object=99, allow_final_contact=True) is True

    assert calls == [([[0.4, 0.5]], 99, None, True)]


def test_v2_move_arm_ompl_to_pose_omits_attached_object_from_planner_when_final_contact_is_allowed():
    robot, _sim, _ik, _ompl, _ctx = make_robot()
    target = [9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0]
    planned_attached_objects = []
    robot.find_arm_configs_for_pose = lambda target_pose, max_configs=8, max_time=10.0: [[0.4, 0.5]]
    robot.arm_configs_collide = lambda configs, attached_object=None, ignored_objects=None, allow_final_contact=False: False
    robot.follow_arm_path = lambda path: True

    def plan_arm_joint_path(config, attached_object=None, planning_time=3.0, path_state_count=100):
        planned_attached_objects.append(attached_object)
        return [0.1, 0.2, *config]

    robot.plan_arm_joint_path = plan_arm_joint_path

    assert robot.move_arm_ompl_to_pose(target, attached_object=99, allow_final_contact=True) is True

    assert planned_attached_objects == [None]


def test_interpolate_pose_xyz_keeps_target_orientation():
    poses = interpolate_pose_xyz(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.05, 0.0, 1.0, 0.0, 0.0],
        max_step=0.025,
    )

    assert [pose[2] for pose in poses] == pytest.approx([0.0, 0.025, 0.05])
    assert all(pose[3:] == [0.0, 1.0, 0.0, 0.0] for pose in poses)
