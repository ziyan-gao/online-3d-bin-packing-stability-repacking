from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "manual_tests_v2"
SCRIPT_NAMES = [
    "test_v2_ik.py",
    "test_v2_joint.py",
    "test_v2_linear.py",
    "test_v2_ompl.py",
    "test_v2_pick_place.py",
    "test_v2_rail.py",
]


def test_v2_manual_debug_scripts_are_direct_vscode_files():
    assert (SCRIPT_DIR / "common.py").exists()
    common_text = (SCRIPT_DIR / "common.py").read_text()
    assert "sys.path.insert" in common_text
    assert "python_controller_v2.robot" in common_text

    for script_name in SCRIPT_NAMES:
        path = SCRIPT_DIR / script_name
        assert path.exists(), f"missing {path}"
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert "main" in functions
        assert 'if __name__ == "__main__"' in text
        assert "from common import" in text


def test_vscode_launch_config_has_v2_manual_entries():
    launch = json.loads((ROOT / ".vscode" / "launch.json").read_text())
    programs = [config["program"] for config in launch["configurations"]]

    for script_name in SCRIPT_NAMES:
        assert f"${{workspaceFolder}}/manual_tests_v2/{script_name}" in programs


def test_v2_pick_place_manual_test_matches_controller_test_belt_loop():
    path = SCRIPT_DIR / "test_v2_pick_place.py"
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))

    assert "BeltMonitor" in text
    assert "full_transition_items" in text
    assert "set_robot_picking(True)" in text
    assert "set_robot_picking(False)" in text
    assert "belt.set_robot_picking(False)\n    if not pick_ok or pick_only" in text
    assert "random.choice(items)" in text
    assert "max-cycles" in text
    assert "--place" in text

    reach_defaults = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "--reach-half-width":
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                reach_defaults.append(keyword.value.value)
    assert reach_defaults == [0.50]
