from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "manual_tests"
SCRIPT_NAMES = [
    "test_ik_solver.py",
    "test_linear_motion.py",
    "test_joint_motion.py",
    "test_ompl_joint.py",
    "test_ompl_pose.py",
    "test_pick.py",
    "test_place.py",
]


def load_common_module():
    spec = importlib.util.spec_from_file_location("manual_test_common", SCRIPT_DIR / "common.py")
    assert spec is not None and spec.loader is not None
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    return common


def test_manual_diagnostic_scripts_are_present_and_runnable_as_vscode_files():
    assert (SCRIPT_DIR / "common.py").exists()
    common_text = (SCRIPT_DIR / "common.py").read_text()
    assert "sys.path.insert" in common_text
    assert "python_controller.robot" in common_text

    for script_name in SCRIPT_NAMES:
        path = SCRIPT_DIR / script_name
        assert path.exists(), f"missing {path}"
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert "main" in functions
        assert 'if __name__ == "__main__"' in text
        assert "from common import" in text
        assert "from .common import" not in text


def test_vscode_launch_config_points_at_manual_tests():
    launch = ROOT / ".vscode" / "launch.json"
    assert launch.exists()
    text = launch.read_text()
    for script_name in SCRIPT_NAMES:
        assert f"manual_tests/{script_name}" in text


def test_scene_paths_are_resolved_to_absolute_workspace_paths_before_loading():
    common = load_common_module()

    scene = common.resolve_scene_path("coppeliaSim/zeroPressureBuffer_v5.ttt")

    assert scene == str((ROOT / "coppeliaSim" / "zeroPressureBuffer_v5.ttt").resolve())


def test_manual_debug_scripts_start_simulation_by_default_and_allow_attach_mode():
    common = load_common_module()
    parser = common.build_parser("manual test")

    assert parser.parse_args([]).start is True
    assert parser.parse_args(["--start"]).start is True
    assert parser.parse_args(["--no-start"]).start is False
    assert parser.parse_args([]).keep_running is False
    assert parser.parse_args(["--keep-running"]).keep_running is True


def test_registered_started_simulations_are_stopped_during_cleanup():
    common = load_common_module()

    class FakeSim:
        simulation_stopped = 0

        def __init__(self):
            self.state = 1
            self.stop_calls = 0

        def getSimulationState(self):
            return self.state

        def stopSimulation(self):
            self.stop_calls += 1
            self.state = self.simulation_stopped

    class FakeCtx:
        def __init__(self):
            self.sim = FakeSim()
            self.steps = 0

        def step(self):
            self.steps += 1

    ctx = FakeCtx()

    common.register_simulation_cleanup(ctx)
    common.stop_registered_simulations()

    assert ctx.sim.stop_calls == 1
    assert common._CLEANUP_CONTEXTS == []
