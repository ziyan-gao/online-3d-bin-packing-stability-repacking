from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def test_controller_modules_exist_and_use_remote_api_boundaries():
    assert (ROOT / "python_controller" / "__init__.py").exists()
    assert "RemoteAPIClient" in read("python_controller/remote.py")
    assert "client.setStepping(True)" in read("python_controller/remote.py")
    assert "client.step()" in read("python_controller/remote.py")
    assert "trackedItemsData" in read("python_controller/belt.py")
    assert "robotPicking" in read("python_controller/belt.py")
    assert "require(\"simIK\")" in read("python_controller/robot.py")
    assert "require(\"simOMPL\")" in read("python_controller/robot.py")
    assert "def pick(" in read("python_controller/pick_place.py")
    assert "def place(" in read("python_controller/pick_place.py")


def test_robot_controller_discovers_scene_handles_and_configs():
    text = read("python_controller/robot.py")
    assert "class RobotHandles" in text
    assert "def discover_handles(" in text
    assert "def get_config(" in text
    assert "def set_config(" in text
    assert "def set_target_config(" in text
    assert "sim.getObjectsInTree" in text
    assert "sim.sceneobject_joint" in text


def test_remote_api_client_import_is_lazy():
    text = read("python_controller/remote.py")
    assert "RemoteAPIClient" in text
    assert "from coppeliasim_zmqremoteapi_client import RemoteAPIClient" not in text.split("def connect(")[0]


def test_robot_controller_sets_up_ik_and_collision_collection():
    text = read("python_controller/robot.py")
    assert "def setup_ik(" in text
    assert "simIK.createEnvironment" in text
    assert "simIK.addElementFromScene" in text
    assert "simIK.constraint_pose" in text
    assert "def solve_ik_to_pose(" in text
    assert "simIK.handleGroup" in text
    assert "def update_robot_collection(" in text
    assert "sim.createCollection" in text
    assert "sim.checkCollision" in text


def test_robot_controller_plans_and_follows_joint_paths():
    text = read("python_controller/robot.py")
    assert "def plan_joint_path(" in text
    assert "simOMPL.createTask" in text
    assert "simOMPL.setStateSpaceForJoints" in text
    assert "simOMPL.setCollisionPairs" in text
    assert "simOMPL.solve" in text
    assert "def follow_joint_path(" in text
    assert "sim.generateTimeOptimalTrajectory" in text
    assert "sim.getPathInterpolatedConfig" in text


def test_robot_controller_has_cartesian_move():
    text = read("python_controller/robot.py")
    assert "def move_cartesian_linear(" in text
    assert "interpolate_pose_xyz" in text
    assert "solve_ik_to_pose" in text


def test_pick_place_controller_uses_robot_and_suction_signals():
    text = read("python_controller/pick_place.py")
    assert "def object_size(" in text
    assert "def pick_poses_for_object(" in text
    assert "def place_poses_for_object(" in text
    assert "suctionPadEnabled" in text
    assert "setObjectParent" in text
    assert "move_to_pose" in text
    assert "position = [0.0, 0.0, 1.0]" in text


def test_main_loop_wires_belt_robot_and_pick_place():
    text = read("python_controller/main.py")
    assert "connect(" in text
    assert "ensure_started(" in text
    assert "BeltMonitor" in text
    assert "RobotController" in text
    assert "PickPlaceController" in text
    assert "full_transition_items" in text
    assert "random.choice" in text


def test_controller_readme_documents_scene_script_contract():
    text = read("python_controller/README.md")
    assert "Keep only tiny loader scripts embedded in the scene" in text
    for script_name in [
        "coppeliaSim/external_api.lua",
        "coppeliaSim/loading_orchastrator.lua",
        "coppeliaSim/collision_free_trajectory.lua",
        "coppeliaSim/ur10_joint.lua",
        "coppeliaSim/ur10_linear.lua",
        "coppeliaSim/belt.lua",
    ]:
        assert script_name in text
    assert "trackedItemsData" in text
    assert "robotPicking" in text
    assert "cd /home/gao/online-3d-bin-packing-stability-repacking/integrations" in text
    assert "python -m python_controller.main" in text


def test_zero_buffer_belt_ui_is_optional_for_headless_mode():
    text = read("coppeliaSim/belt.lua")
    assert "pcall(require, 'simUI')" in text
    assert "uiEnabled = false" in text
    assert "if not uiEnabled then" in text


def test_zero_buffer_belt_publishes_data_without_threaded_script_mode():
    text = read("coppeliaSim/belt.lua")
    assert "function stepBelt()" in text
    assert "publishTrackedItemsData()" in text.split("function sysCall_init()")[1].split("function initConfig()")[0]
    assert "function sysCall_actuation()" in text
    assert "stepBelt()" in text.split("function sysCall_actuation()")[1].split("function sysCall_thread()")[0]


def test_zero_buffer_belt_publishes_utf8_json_for_remote_api():
    text = read("coppeliaSim/belt.lua")
    publish_body = text.split("function publishTrackedItemsData()")[1].split("function initItemTrackingUi()")[0]
    assert "json.encode" in publish_body
    assert "sim.packTable" not in publish_body
