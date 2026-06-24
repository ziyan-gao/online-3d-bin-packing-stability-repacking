from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "loading_orchastrator.lua"


def test_loading_test_randomly_picks_unprocessed_cuboids_to_height_one():
    text = SCRIPT.read_text()

    assert "local fullItems = itemsIfFull()" in text
    assert "local items = fullItems and trackedCuboidItems(fullItems)" in text
    assert "if items and not lastFull then" in text
    assert "lastFull = fullItems ~= nil" in text
    assert "local function trackedCuboidItems(trackedItems)" in text
    assert "alias and (alias:match('[^/]+$') or alias)" in text
    assert "localAlias:match('^Cuboid')" in text
    assert "placedItems[itemHandle(item)] = true" in text
    assert "api('pick', {object = itemHandle(item)})" in text
    assert "api('place', {position = {0, 0, 1}})" in text
    assert "function sysCall_actuation()" in text
    assert "pollActiveCommand()" in text
    assert "maybeStartPickAndPlace()" in text
    assert "sim.step()" not in text
    assert "function sysCall_thread" not in text
    assert "local function setRobotPicking(enabled)" in text
    assert "sim.setInt32Signal('robotPicking', enabled and 1 or 0)" in text
    assert "setRobotPicking(true)" in text
    assert "setRobotPicking(false)" in text
    assert "moveRailAndTip" not in text
    assert "moveTip" not in text


def test_loading_orchestrator_reports_external_api_failure_detail():
    text = SCRIPT.read_text()

    assert "local function statusFailureReason(prefix, status, state, resultField)" in text
    assert "local result = state and state[resultField]" in text
    assert "result.reason" in text
    assert "finishActiveCommand(false, statusFailureReason('pick', pickStatus, state, 'lastPickResult'))" in text
    assert "finishActiveCommand(false, statusFailureReason('place', placeStatus, state, 'lastPlaceResult'))" in text


def test_loading_orchestrator_does_not_use_pcall():
    text = SCRIPT.read_text()

    assert "pcall" not in text
