from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "external_api.lua"


def test_pick_and_place_try_planned_fallback_before_stopping():
    text = SCRIPT.read_text()

    assert "automatic collision-free planned fallback is disabled" not in text
    assert "local function failMotionAndStop" in text
    assert "sim.stopSimulation()" in text
    assert "local function dispatchAndWaitWithPlannedFallback" in text
    assert "normalMotion = resultSummary(normal)" in text
    assert "plannedMotion = resultSummary(planned)" in text
    assert "planned fallback succeeded for " in text
    assert "local pickRailPreposition = prepositionUr10BaseForPose(prePickPose, pickReachHalfWidth)" in text
    assert text.index("local pickRailPreposition = prepositionUr10BaseForPose(prePickPose, pickReachHalfWidth)") < text.index(
        "local approach = dispatchAndWaitWithPlannedFallback(prePickDummy, 'config', nil, nil, 60, 120"
    )
    assert "dispatchAndWaitWithPlannedFallback(prePickDummy, 'config', nil, nil, 60, 120, 'pre-pick config move')" in text
    assert "dispatchAndWaitWithPlannedFallback(pickDummy, 'linear', nil, nil, 30, 120, 'pick linear descent')" in text
    assert "simIK = require('simIK')" in text
    assert "simOMPL = require('simOMPL')" in text
    assert "local function moveToTargetByConfig()" in text
    assert "local function moveToTargetLinearly()" in text
    assert "sim.moveToConfig({" in text
    assert "sim.moveToPose({" in text
    assert "ur10MotionMode" not in text
    assert "local holdLiftHeight = " in text
    assert "local holdClearanceAbovePrePick = 0.05" in text
    assert "local holdPose = table.clone(rotatedPose)" in text
    assert "holdPose[3] = math.max(objectPose[3] + holdLiftHeight, prePickPose[3] + holdClearanceAbovePrePick)" in text
    assert "return prePickPose, pickPose, holdPose, size" in text
    assert "local prePickPose, pickPose, holdPose, size = pickPosesForObject(objectHandle)" in text
    assert "holdDummy = createDummyAtPose('holdDummy', holdPose)" in text
    assert "dispatchAndWaitWithPlannedFallback(holdDummy, 'linear', nil, objectHandle, 30, 120, 'hold lift')" in text
    assert text.index("local descend = dispatchAndWaitWithPlannedFallback(pickDummy, 'linear', nil, nil, 30, 120") < text.index(
        "local suction = setSuction({enabled = true})"
    )
    assert text.index("local suction = setSuction({enabled = true})") < text.index(
        "local lift = dispatchAndWaitWithPlannedFallback(holdDummy, 'linear', nil, objectHandle, 30, 120"
    )
    assert "reason = 'failed linear lift to holding pose'" in text
    assert "dispatchAndWait(intermediateDummy, 'config', nil, 60)" not in text
    assert "sim.setStringSignal(placeStatusSignal, 'approaching_config')" in text
    assert "dispatchAndWaitWithPlannedFallback(prePlaceDummy, 'config', objectHandle, objectHandle, 60, 120, 'pre-place config move')" in text
    assert "dispatchAndWaitWithPlannedFallback(prePlaceDummy, 'linear', objectHandle, objectHandle, 60, 120, 'pre-place linear move')" not in text
    assert "dispatchAndWaitWithPlannedFallback(placeDummy, 'linear', objectHandle, objectHandle, 30, 120, 'place linear descent')" in text
    assert "local retreat = dispatchAndWaitWithPlannedFallback(prePlaceDummy, 'linear', nil, nil, 30, 120, 'place retreat')" in text
    assert text.index("local suction = setSuction({enabled = false})") < text.index(
        "local retreat = dispatchAndWaitWithPlannedFallback(prePlaceDummy, 'linear', nil, nil, 30, 120"
    )
    assert "reason = 'failed linear lift from place pose'" in text


def test_place_prepositions_ur10_base_before_pre_place_config_move():
    text = SCRIPT.read_text()

    assert "local placeBaseBoundaryOffset = " in text
    assert "local railMaxVel = 0.25" in text
    assert "local railMaxAccel = 0.5" in text
    assert "local railMaxJerk = 0.5" in text
    assert "local railMotionTimeout = 20" in text
    assert "local railTargetTolerance = 0.001" in text
    assert "local function prepositionUr10BaseForPlace" in text
    assert "local function prepositionUr10BaseForPose" in text
    assert "local function clampRailTargetPosition(targetRail)" in text
    assert "local cyclic, interval = sim.getJointInterval(railJoint)" in text
    assert "local function moveRailTo(position)" in text
    assert "sim.setStringSignal('railApiStatus', 'moving')" in text
    assert "sim.setJointTargetPosition(railJoint, targetRail)" in text
    assert "local currentRail = sim.getJointPosition(railJoint)" in text
    assert "math.abs(currentRail - targetRail) <= railTargetTolerance" in text
    assert "sim.setStringSignal('railApiStatus', 'done')" in text
    rail_body = text.split("local function moveRailTo(position)")[1].split("local function railFailureReason")[0]
    assert "sim.setJointPosition(railJoint" not in rail_body
    assert "sim.moveToConfig({" not in rail_body
    assert "railApi is not initialized" not in text
    assert "railApiReady" not in text
    assert "railApiTargetPosition" not in text
    assert "go_to_rail_target" not in text
    assert "sim.callScriptFunction('moveRailToPosition'" not in text
    assert "tipSideX = tipPose[1] - basePose[1]" in text
    assert "targetRail = sim.getJointPosition(railJoint) - baseShiftX" in text
    assert "local railPreposition = prepositionUr10BaseForPlace()" in text
    assert text.index("local railPreposition = prepositionUr10BaseForPlace()") < text.index(
        "local approach = dispatchAndWaitWithPlannedFallback(prePlaceDummy, 'config', objectHandle, objectHandle, 60, 120"
    )


def test_pick_and_place_preorient_first_arm_joint_with_move_to_config():
    text = SCRIPT.read_text()

    assert "local prePickFirstJointAngle = 0.0" in text
    assert "local prePlaceFirstJointAngle = 0.5 * math.pi" in text
    assert "local function nearestEquivalentAngle(targetAngle, currentAngle)" in text
    assert "local function moveFirstArmJointTo(targetAngle, label, allowOppositeSign)" in text

    helper_body = text.split("local function moveFirstArmJointTo(targetAngle, label, allowOppositeSign)")[1].split(
        "local function moveToTargetByConfig()"
    )[0]
    assert "local currentAngle = sim.getJointPosition(joints[1])" in helper_body
    assert "local wrappedTargetAngle = nearestEquivalentAngle(targetAngle, currentAngle)" in helper_body
    assert "if allowOppositeSign then" in helper_body
    assert "local oppositeTargetAngle = nearestEquivalentAngle(-targetAngle, currentAngle)" in helper_body
    assert "math.abs(oppositeTargetAngle - currentAngle) < math.abs(wrappedTargetAngle - currentAngle)" in helper_body
    assert "sim.moveToConfig({" in helper_body
    assert "joints = {joints[1]}" in helper_body
    assert "targetPos = {wrappedTargetAngle}" in helper_body
    assert "maxVel = {configMaxVel[1]}" in helper_body
    assert "maxAccel = {configMaxAccel[1]}" in helper_body
    assert "maxJerk = {configMaxJerk[1]}" in helper_body
    assert "sim.setJointTargetPosition" not in helper_body

    pick_body = text.split("local function runPickCommand(command)")[1].split(
        "local function runPlaceCommand(command)"
    )[0]
    assert pick_body.index("moveFirstArmJointTo(prePickFirstJointAngle, 'pre-pick')") < pick_body.index(
        "local approach = dispatchAndWaitWithPlannedFallback(prePickDummy, 'config'"
    )

    place_body = text.split("local function runPlaceCommand(command)")[1].split(
        "function moveTo(targetOrData, mode, tcpObject)"
    )[0]
    assert place_body.index("moveFirstArmJointTo(prePlaceFirstJointAngle, 'pre-place', true)") < place_body.index(
        "local approach = dispatchAndWaitWithPlannedFallback(prePlaceDummy, 'config'"
    )


def test_rail_preposition_failures_preserve_nested_reason():
    text = SCRIPT.read_text()

    assert "local function railFailureReason(prefix, railResult)" in text
    assert "railResult.reason" in text
    assert "reason = railFailureReason('failed prepositioning UR10 base before pre-pick pose', pickRailPreposition)" in text
    assert "reason = railFailureReason('failed prepositioning UR10 base before pre-place pose', railPreposition)" in text
    assert "currentStatus == nil or currentStatus == 'stopped'" not in text


def test_external_api_does_not_use_pcall():
    text = SCRIPT.read_text()

    assert "pcall" not in text


def test_linear_motion_settle_warning_does_not_fail_completed_move():
    text = SCRIPT.read_text()

    linear_body = text.split("local function moveToTargetLinearly()")[1].split("local function dispatchAndWait")[0]
    assert "local settleOk, settleStatus = settleAtTargetPose(targetPose)" in linear_body
    assert "linear settle did not reach tight target tolerance" in linear_body
    assert "setMotionStatus('linear', 'done')" in linear_body
    assert "return true, 'done'" in linear_body
    assert "return false, 'ik_failed'" not in linear_body


def test_motion_logs_current_collision_state_before_executing_arm():
    text = SCRIPT.read_text()

    assert "local function objectLabel(handle)" in text
    assert "local function fmtCollisionPair(pair)" in text
    assert "local function collectionCollisionPair(collection)" in text
    assert "sim.getObjectAlias(handle, 5)" in text
    assert "sim.checkCollision(collection, sim.handle_all)" in text
    assert "local function allowedAdjacentSelfCollision(pair)" not in text
    assert "local function manualSelfCollisionPair(collection)" not in text
    assert "local function objectIsCollidableShape(handle)" not in text
    collision_pair_body = text.split("local function collectionCollisionPair(collection)")[1].split(
        "local function setMotionStatus"
    )[0]
    assert "sim.checkCollision(collection, collection)" not in collision_pair_body
    assert "manualSelfCollisionPair(collection)" not in collision_pair_body
    assert "local function logCurrentUr10CollisionState(label)" in text
    assert "current UR10 collision before " in text
    assert "': pair=' .. fmtCollisionPair(pair)" in text
    assert "current UR10 collision-free before " in text
    assert "local function logCurrentPlannedCollisionState(label)" in text
    assert "current planned robot collision before " in text
    assert "current planned robot collision-free before " in text

    config_body = text.split("local function moveToTargetByConfig()")[1].split(
        "local function moveToPoseCallback"
    )[0]
    assert config_body.index("logCurrentUr10CollisionState('config move')") < config_body.index(
        "sim.moveToConfig({"
    )

    linear_body = text.split("local function moveToTargetLinearly()")[1].split(
        "local function getPlannedConfig()"
    )[0]
    assert linear_body.index("updateRobotCollection()") < linear_body.index(
        "logCurrentUr10CollisionState('linear move')"
    )
    assert linear_body.index("logCurrentUr10CollisionState('linear move')") < linear_body.index(
        "sim.moveToPose({"
    )

    planned_body = text.split("local function followPlannedPath(path)")[1].split(
        "local function moveToTargetWithPlanner()"
    )[0]
    assert planned_body.index("logCurrentPlannedCollisionState('planned path playback')") < planned_body.index(
        "plannedDebug('path playback start:"
    )


def test_motion_logs_current_collision_state_before_ik_solves():
    text = SCRIPT.read_text()

    config_body = text.split("local function moveToTargetByConfig()")[1].split(
        "local function moveToPoseCallback"
    )[0]
    assert config_body.index("updateRobotCollection()") < config_body.index(
        "logCurrentUr10CollisionState('config IK solve')"
    )
    assert config_body.index("logCurrentUr10CollisionState('config IK solve')") < config_body.index(
        "local targetConfig = solveTargetConfig()"
    )

    linear_body = text.split("local function moveToTargetLinearly()")[1].split(
        "local function getPlannedConfig()"
    )[0]
    assert linear_body.index("updateRobotCollection()") < linear_body.index(
        "logCurrentUr10CollisionState('linear IK solve')"
    )
    assert linear_body.index("logCurrentUr10CollisionState('linear IK solve')") < linear_body.index(
        "simIK.syncFromSim(linearIkEnv"
    )

    planned_body = text.split("local function moveToTargetWithPlanner()")[1].split(
        "local function dispatchAndWait(target, mode, tcpObject, timeout)"
    )[0]
    assert planned_body.index("updatePlannedRobotCollection()") < planned_body.index(
        "logCurrentPlannedCollisionState('planned IK solve')"
    )
    assert planned_body.index("logCurrentPlannedCollisionState('planned IK solve')") < planned_body.index(
        "local goalConfig, goalStatus = findPlannedTargetConfig()"
    )


def test_external_api_state_results_are_shallow_to_avoid_recursive_table_growth():
    text = SCRIPT.read_text()

    assert "local function stateSnapshot()" in text
    assert "local function resultSummary(result)" in text
    assert "state = stateSnapshot()" in text
    assert "state = getState({})" not in text

    get_state_body = text.split("function getState(inData)")[1].split("function sysCall_thread()")[0]
    assert "lastPickResult = resultSummary(lastPickResult)" in get_state_body
    assert "lastPlaceResult = resultSummary(lastPlaceResult)" in get_state_body
    assert "lastPickResult = lastPickResult" not in get_state_body
    assert "lastPlaceResult = lastPlaceResult" not in get_state_body

    summary_body = text.split("local function resultSummary(result)")[1].split("local function logMotionInfo")[0]
    assert "state =" not in summary_body
    assert "approach =" not in summary_body
    assert "descend =" not in summary_body


def test_explicit_planned_motion_is_owned_by_external_api():
    text = SCRIPT.read_text()

    assert "local function moveToTargetWithPlanner()" in text
    assert "local function initPlannedMotion()" in text
    assert "local function findPlannedPath(goalConfig)" in text
    assert "local function followPlannedPath(path)" in text
    assert "simOMPL.createTask('external_api_planned_motion_task')" in text
    assert "simOMPL.setStateSpaceForJoints(task, plannedJoints, useForProjection)" in text
    assert "pathPts, times, followPathScript = sim.generateTimeOptimalTrajectory" in text
    assert "requestedMode == 'plan' or requestedMode == 'planned'" in text
    assert "pendingMoveCommand = {" in text
    assert "mode = 'planned'" in text
    assert "setMotionStatus('planned', 'queued')" in text
    assert "lastMoveResult = runMoveCommand(command)" in text
    assert "return dispatchAndWait(data.target, 'planned', data.tcpObject, 120)" not in text
    assert "collision-free planned motion is temporarily disabled" not in text


def test_planned_move_runs_from_external_api_thread_not_call_script_function():
    text = SCRIPT.read_text()

    assert "local function runMoveCommand(command)" in text
    assert "dispatchAndWait(command.target, command.mode, command.tcpObject, command.timeout)" in text
    thread_body = text.split("function sysCall_thread()")[1].split("function sysCall_cleanup()")[0]
    assert "pendingMoveCommand and not moveBusy" in thread_body
    assert thread_body.index("pendingMoveCommand and not moveBusy") < thread_body.index(
        "pendingPickCommand and not pickBusy"
    )
    assert "moveBusy = true" in thread_body
    assert "moveBusy = false" in thread_body


def test_planned_motion_uses_all_joint_ik_before_ompl_planning():
    text = SCRIPT.read_text()

    assert "plannedIkEnv = simIK.createEnvironment()" in text
    assert "plannedIkGroup = simIK.createGroup(plannedIkEnv)" in text
    assert "simIK.addElementFromScene(plannedIkEnv" in text
    assert "plannedIkBase," in text
    assert "simIK.syncFromSim(plannedIkEnv, {plannedIkGroup})" in text
    assert "plannedSimToIkObjectMap[plannedJoints[i]]" in text
    assert "local result, flags, precision = simIK.handleGroup(plannedIkEnv, plannedIkGroup)" in text
    assert "simIK.getJointPosition(plannedIkEnv," in text
    assert "all-joint IK failed: result=" in text
    assert "all-joint IK success" in text
    assert "simIK.findConfigs" in text
    assert "findMultiple = true" in text
    assert "findAlt = true" not in text
    assert "local function plannedRailTargetCandidates()" not in text
    assert "local function solvePlannedUr10ConfigAtRail(railPosition)" not in text


def test_ik_find_configs_fallback_accepts_only_pose_accurate_collision_free_configs():
    text = SCRIPT.read_text()

    assert "local findConfigsMaxDist = 0.28" in text
    assert "local findConfigsMaxTime = 1.0" in text
    assert "local findConfigsLinearTolerance = 0.005" in text
    assert "local findConfigsAngularTolerance = 2.0 * math.pi / 180.0" in text
    assert "local findConfigsArmMetric = {8.0, 8.0, 8.0, 0.8, 0.6, 0.3}" in text
    assert "local function poseError(actualPose, targetPose)" in text
    assert "local function findConfigsParams(jointCount)" in text
    assert "maxDist = findConfigsMaxDist" in text
    assert "maxTime = findConfigsMaxTime" in text
    assert "findMultiple = true" in text
    assert "metric[1] = 1.0" in text
    assert "metric[i] = findConfigsArmMetric[i]" in text
    assert "local function findFeasibleIkConfig(" in text

    fallback_body = text.split("local function findFeasibleIkConfig(")[1].split(
        "local function solveTargetConfig()"
    )[0]
    assert "simIK.findConfigs(env, group, ikJoints, findConfigsParams(#ikJoints))" in fallback_body
    assert "local bufferedConfig = getMotionConfig()" in fallback_body
    assert "setMotionConfig(candidateConfig)" in fallback_body
    assert "local tipPose = sim.getObjectPose(ikTip, baseObject)" in fallback_body
    assert "local positionError, orientationError = poseError(tipPose, targetPose)" in fallback_body
    assert "positionError <= findConfigsLinearTolerance" in fallback_body
    assert "orientationError <= findConfigsAngularTolerance" in fallback_body
    assert "collisionCheck(candidateConfig)" in fallback_body
    assert "setMotionConfig(bufferedConfig)" in fallback_body

    config_body = text.split("local function solveTargetConfig()")[1].split(
        "local function moveToTargetByConfig()"
    )[0]
    assert config_body.index("local result, flags, precision = simIK.handleGroup(configIkEnv, configIkGroup)") < config_body.index(
        "findFeasibleIkConfig('config'"
    )
    assert "findFeasibleIkConfig('config', configIkEnv, configIkGroup, configSimToIkObjectMap, joints, targetPose, ur10Base" in config_body

    planned_body = text.split("local function findPlannedTargetConfig()")[1].split(
        "local function findPlannedPath(goalConfig)"
    )[0]
    assert planned_body.index("local result, flags, precision = simIK.handleGroup(plannedIkEnv, plannedIkGroup)") < planned_body.index(
        "findFeasibleIkConfig('planned'"
    )
    assert "findFeasibleIkConfig('planned', plannedIkEnv, plannedIkGroup, plannedSimToIkObjectMap, plannedJoints, targetPose, plannedIkBase" in planned_body


def test_planned_motion_uses_single_simik_goal_for_ompl():
    text = SCRIPT.read_text()

    assert "local function findPlannedTargetConfig()" in text
    assert "local goalConfig, goalStatus = findPlannedTargetConfig()" in text
    assert "local path, pathStatus = findPlannedPath(goalConfig)" in text
    assert "return false, pathStatus" in text

    planner_body = text.split("local function moveToTargetWithPlanner()")[1].split(
        "local function dispatchAndWait(target, mode, tcpObject, timeout)"
    )[0]
    assert "local goalConfigs, goalStatus = findPlannedTargetConfigs()" not in planner_body
    assert "for i = 1, #goalConfigs do" not in planner_body
    assert "OMPL trying goal candidate" not in planner_body


def test_planned_collision_collection_uses_ur10_tree_not_whole_mobile_arm():
    text = SCRIPT.read_text()

    collection_body = text.split("local function updatePlannedRobotCollection()")[1].split(
        "local function logCurrentPlannedCollisionState(label)"
    )[0]
    assert "sim.addItemToCollection(plannedRobotCollection, sim.handle_tree, ur10Base, 0)" in collection_body
    assert "sim.addItemToCollection(plannedRobotCollection, sim.handle_tree, plannedIkBase, 0)" not in collection_body


def test_planned_motion_preflights_ompl_start_and_goal_states():
    text = SCRIPT.read_text()

    assert "start_state_invalid = true" in text
    assert "goal_state_invalid = true" in text
    assert "local function plannedStateValidity(task, label, config)" in text

    validity_body = text.split("local function plannedStateValidity(task, label, config)")[1].split(
        "local function findPlannedPath(goalConfig)"
    )[0]
    assert "simOMPL.isStateWithinBounds(task, config)" in validity_body
    assert "simOMPL.isStateValid(task, config)" in validity_body
    ok_line = next(
        line.strip()
        for line in validity_body.splitlines()
        if line.strip().startswith("local ok =")
    )
    assert ok_line == "local ok = withinBounds and stateValid"
    assert "OMPL preflight ' .. label" in validity_body
    assert "withinBounds=" in validity_body
    assert "stateValid=" in validity_body
    assert "scriptCollides=" not in validity_body

    path_body = text.split("local function findPlannedPath(goalConfig)")[1].split(
        "local function followPlannedPath(path)"
    )[0]
    assert "local startConfig = getPlannedConfig()" in path_body
    assert "plannedRobotCollection, sim.handle_all" in path_body
    assert "plannedRobotCollection, plannedRobotCollection" in path_body
    assert "simOMPL.setStateValidationCallback" not in path_body
    assert path_body.index("simOMPL.setup(task)") < path_body.index(
        "local startValidity = plannedStateValidity(task, 'start', startConfig)"
    )
    assert path_body.index("local startValidity = plannedStateValidity(task, 'start', startConfig)") < path_body.index(
        "simOMPL.solve(task, pathPlanningMaxTime)"
    )
    assert "return nil, 'start_state_invalid'" in path_body
    assert "return nil, 'goal_state_invalid'" in path_body
    assert "plannedDebug('OMPL solve start: start=' .. fmtConfig(startConfig, 8)" in path_body


def test_planned_motion_uses_vanilla_ompl_without_custom_sampling_callbacks():
    text = SCRIPT.read_text()

    assert "local function plannedStateValidationCallback" not in text
    assert "local function plannedStateFromCallbackInput" not in text
    assert "local function plannedPathCollides" not in text
    assert "simOMPL.setStateValidationCallback" not in text
    path_body = text.split("local function findPlannedPath(goalConfig)")[1].split(
        "local function followPlannedPath(path)"
    )[0]
    assert "beginPlannedSamplingDisplaySuppression" not in path_body
    assert "endPlannedSamplingDisplaySuppression" not in path_body
    assert "planned returned path failed filtered collision validation" not in path_body


def test_planned_motion_logs_copyable_debug_trace():
    text = SCRIPT.read_text()

    assert "local plannedDebugEnabled = true" in text
    assert "local function plannedDebug(message)" in text
    assert "[external_api_planned_debug] " in text
    assert "planned request start:" in text
    assert "all-joint IK targetInBase=" in text
    assert "accepted target config" in text
    assert "OMPL preflight" in text
    assert "OMPL solve start:" in text
    assert "OMPL solve failed" in text
    assert "trajectory timing:" in text
    assert "path playback start:" in text
    assert "path playback progress:" in text
    assert "path playback stalled:" in text
    assert "path playback final target:" in text
    assert "path playback final actual:" in text
    assert "maxError=" in text
