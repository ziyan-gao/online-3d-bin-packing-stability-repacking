from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "ur10_joint.lua"


def test_ur10_joint_validates_ik_config_before_move_to_config():
    text = SCRIPT.read_text()

    assert "local function configIsFinite(config)" in text
    assert "value ~= value" in text
    assert "value == math.huge or value == -math.huge" in text
    assert "if not configIsFinite(targetConfig) then" in text
    assert "setStatus('ik_failed')" in text
    assert "invalid IK target config; refusing sim.moveToConfig" in text
    assert text.index("if not configIsFinite(targetConfig) then") < text.index("sim.moveToConfig({")
