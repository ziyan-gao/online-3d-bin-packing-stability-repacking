from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def test_rail_ur10_reuses_time_optimal_trajectory_script_handle():
    text = read("coppeliaSim/collision_free_trajectory.lua")

    assert "followPathScript = -1" in text
    assert "pathPts, times, followPathScript = sim.generateTimeOptimalTrajectory" in text
    assert "followPathScript = nil" in text


def test_collision_free_trajectory_standalone_script_is_legacy_disabled():
    text = read("coppeliaSim/collision_free_trajectory.lua")

    assert "collisionFreeTrajectoryDisabled = true" in text
    assert "setStatus('disabled')" in text
    assert "external_api.lua owns planned rail+UR10 motion" in text
    assert "function sysCall_thread" not in text
