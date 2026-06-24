from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "railApi.lua"


def test_rail_api_exposes_direct_position_motion_for_other_scripts():
    text = SCRIPT.read_text()

    assert "function moveRailToPosition(inData)" in text
    assert "pendingRailTargetPosition = targetPosition" in text
    assert "return {ok = true, accepted = true, targetPosition = targetPosition}" in text
    assert "local function moveRailToPositionValue(targetPosition)" in text
    assert "moveRailToPositionValue(targetPosition)" in text
    assert "sim.moveToConfig" in text
    assert "go_to_rail_target" in text
    assert "local readySignal = 'railApiReady'" in text
    assert "sim.setInt32Signal(readySignal, 1)" in text
    assert "sim.setInt32Signal(readySignal, 0)" in text
    assert "railApiTargetPosition" in text
    assert "local function optionalObject(paths)" in text
    assert "railTarget = optionalObject({" in text
    assert "railTarget and railTarget >= 0" in text
    assert "rail_target is required when go_to_rail_target has no railApiTargetPosition" in text
    assert "local function stepRailMotion()" in text
    assert "function sysCall_actuation()" in text
    assert "function sysCall_thread" not in text
    assert "stepRailMotion()" in text.split("function sysCall_actuation()")[1].split("function sysCall_cleanup()")[0]
    assert "sim.getSimulationTimeStep()" in text
    assert "sim.setJointPosition(railJoint, nextPosition)" in text
    assert "local requestedTargetPosition = sim.getFloatSignal(targetPositionSignal)" in text
    assert "if requestedTargetPosition ~= nil then" in text
    assert "return requestedTargetPosition" in text
    assert "moveRailToPositionValue(targetPosition)" in text
    custom_function_body = text.split("function moveRailToPosition(inData)")[1].split("local function moveToRailTarget()")[0]
    assert "sim.moveToConfig" not in custom_function_body


def test_rail_api_does_not_use_pcall():
    text = SCRIPT.read_text()

    assert "pcall" not in text
