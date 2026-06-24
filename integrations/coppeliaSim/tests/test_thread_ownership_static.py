from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]


def test_only_belt_and_external_api_are_threaded_scripts():
    threaded_scripts = []
    for script in sorted(SCRIPT_DIR.glob("*.lua")):
        if script.name == "demo.lua":
            continue
        text = script.read_text()
        if "function sysCall_thread" in text:
            threaded_scripts.append(script.name)

    assert threaded_scripts == ["belt.lua", "external_api.lua"]


def test_robot_motion_helpers_are_non_threaded():
    helper_scripts = [
        "collision_free_trajectory.lua",
        "loading_orchastrator.lua",
        "railApi.lua",
        "ur10_joint.lua",
        "ur10_linear.lua",
    ]

    for script_name in helper_scripts:
        text = (SCRIPT_DIR / script_name).read_text()
        assert "function sysCall_thread" not in text
