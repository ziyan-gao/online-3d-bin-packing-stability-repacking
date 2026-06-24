sim = require('sim')

--[[
Small public motion API.

Supported calls:
  moveTo(dummyOrPose, mode, tcpObject)
  moveTo({target = dummyOrPose, mode = 'config', tcpObject = object})
  linearMoveTo(dummyOrPose, tcpObject)
  linearMoveTo({target = dummyOrPose, tcpObject = object})
  pick(objectAlias)
  pick({object = objectAlias})
  place({x, y, z})
  place({position = {x, y, z}})
  setSuction(true|false)
  setSuction({enabled = true|false})

`dummyOrPose` can be an object handle, object alias/path, or a 7-value pose.
`tcpObject` can be nil, an object handle, or an object alias/path.
]]

local modeSignal = 'ur10MotionMode'
local tcpObjectSignal = 'motionTcpObjectHandle'
local pickStatusSignal = 'externalApiPickStatus'
local placeStatusSignal = 'externalApiPlaceStatus'

local delayAfterApproach = 0.5
local delayAfterDescend = 0.2
local delayAfterSuction = 0.5
local delayAfterRetreat = 0.2
local placeBaseBoundaryOffset = 0.7
local pickReachHalfWidth = 0.5
local railMaxVel = 0.25
local railMaxAccel = 0.5
local railMaxJerk = 0.5
local railMotionTimeout = 20
local railTargetTolerance = 0.001
local holdLiftHeight = 0.5
local holdClearanceAbovePrePick = 0.05

local motionStatusSignals = {
    config = 'ur10JointMotionStatus',
    linear = 'ur10LinearMotionStatus',
}

local motionTerminalStatuses = {
    config = {
        done = true,
        ik_failed = true,
        collision_failed = true,
        stopped = true,
    },
    linear = {
        done = true,
        stopped = true,
    },
}

local function asBool(value)
    if type(value) == 'table' then
        value = value.enabled
    end
    if type(value) == 'string' then
        return value == 'true' or value == '1' or value == 'yes' or value == 'on'
    end
    return value == true or value == 1
end

local function motionDelay(seconds)
    if seconds and seconds > 0 then
        local startTime = sim.getSimulationTime()
        while sim.getSimulationTime() - startTime < seconds and
              sim.getSimulationState() ~= sim.simulation_advancing_abouttostop do
            sim.wait(0.01)
        end
    end
end

local function optionalObject(alias)
    local handle = sim.getObject(alias, {noError = true})
    if handle and handle >= 0 then
        return handle
    end
    return -1
end

local function firstObject(paths, label)
    for _, path in ipairs(paths) do
        local handle = optionalObject(path)
        if handle >= 0 then
            return handle
        end
    end
    error('Could not find ' .. label .. '. Tried: ' .. table.concat(paths, ', '))
end

local function resolveObject(value, label)
    if value == nil then
        return -1
    end
    if type(value) == 'number' then
        return value
    end
    if type(value) == 'string' then
        local handle = optionalObject(value)
        if handle >= 0 then
            return handle
        end
        error('Could not find ' .. label .. ': ' .. value)
    end
    error(label .. ' must be an object handle or alias/path')
end

local function isPose(value)
    return type(value) == 'table' and #value >= 7
end

local function poseFromTarget(target)
    if isPose(target) then
        return target
    end

    local handle = resolveObject(target, 'target')
    if handle < 0 then
        error('target is required')
    end
    return sim.getObjectPose(handle)
end

local function objectSize(handle)
    local function floatParam(param)
        return sim.getObjectFloatParam(handle, param) or 0
    end

    local minX = floatParam(sim.objfloatparam_objbbox_min_x)
    local maxX = floatParam(sim.objfloatparam_objbbox_max_x)
    local minY = floatParam(sim.objfloatparam_objbbox_min_y)
    local maxY = floatParam(sim.objfloatparam_objbbox_max_y)
    local minZ = floatParam(sim.objfloatparam_objbbox_min_z)
    local maxZ = floatParam(sim.objfloatparam_objbbox_max_z)

    return {
        math.abs(maxX - minX),
        math.abs(maxY - minY),
        math.abs(maxZ - minZ),
    }
end

local function commandData(targetOrData, mode, tcpObject)
    if type(targetOrData) == 'table' and not isPose(targetOrData) and
       (targetOrData.target or targetOrData.dummy or targetOrData.pose or
        targetOrData.mode or targetOrData.tcpObject or targetOrData.tcp_object) then
        return {
            target = targetOrData.target or targetOrData.dummy or targetOrData.pose,
            mode = targetOrData.mode or mode,
            tcpObject = targetOrData.tcpObject or targetOrData.tcp_object or tcpObject,
        }
    end

    return {
        target = targetOrData,
        mode = mode,
        tcpObject = tcpObject,
    }
end

local function setTcpObject(tcpObject)
    local handle = resolveObject(tcpObject, 'tcpObject')
    sim.setInt32Signal(tcpObjectSignal, handle)
    return handle
end

local function setIkTargetPose(target)
    local pose = poseFromTarget(target)
    sim.setObjectPose(ikTarget, pose)
    return pose
end

local function dispatchMotion(targetOrData, requestedMode, tcpObject, signalMode)
    local data = commandData(targetOrData, requestedMode, tcpObject)
    local pose = setIkTargetPose(data.target)
    local tcpHandle = setTcpObject(data.tcpObject)

    sim.setStringSignal(modeSignal, signalMode)
    lastCommand = {
        mode = signalMode,
        pose = pose,
        tcpObject = tcpHandle,
    }

    return {
        ok = true,
        accepted = true,
        mode = signalMode,
        targetPose = pose,
        tcpObject = tcpHandle,
        state = getState({}),
    }
end

local function waitForMotion(signalMode, timeout)
    timeout = timeout or 60
    local statusSignal = motionStatusSignals[signalMode]
    local terminals = motionTerminalStatuses[signalMode] or {}
    local startTime = sim.getSimulationTime()

    while true do
        if sim.getSimulationState() == sim.simulation_advancing_abouttostop then
            return false, 'stopped'
        end
        local status = sim.getStringSignal(statusSignal)
        if terminals[status] then
            return status == 'done', status
        end
        if sim.getSimulationTime() - startTime > timeout then
            return false, 'timeout'
        end
        sim.wait(0.01)
    end
end

local function motionHandlerAvailable(signalMode)
    local statusSignal = motionStatusSignals[signalMode]
    if not statusSignal then
        return false
    end

    local status = sim.getStringSignal(statusSignal)
    return status ~= nil and status ~= 'stopped'
end

local function dispatchAndWait(target, mode, tcpObject, timeout)
    local statusSignal = motionStatusSignals[mode]
    if not motionHandlerAvailable(mode) then
        return {
            ok = false,
            accepted = false,
            completed = false,
            motionStatus = 'handler_unavailable',
            reason = 'motion handler unavailable for mode: ' .. tostring(mode),
            state = getState({}),
        }
    end

    if statusSignal then
        sim.setStringSignal(statusSignal, 'requested')
    end

    local result = dispatchMotion(target, mode, tcpObject, mode)
    if not result.ok then
        return result
    end

    local ok, status = waitForMotion(mode, timeout)
    result.completed = ok
    result.motionStatus = status
    return result
end

local function failMotionAndStop(statusSignal, result)
    result.ok = false
    sim.setStringSignal(statusSignal, 'failed')
    sim.addLog(sim.verbosity_warnings,
               '[external_api] ' .. tostring(result.reason) ..
               '; stopping simulation because collision-free planned motion is temporarily disabled')
    sim.stopSimulation()
    return result
end

local function removePickDummies()
    if prePickDummy and prePickDummy >= 0 then
        sim.removeObjects({prePickDummy})
        prePickDummy = -1
    end
    if pickDummy and pickDummy >= 0 then
        sim.removeObjects({pickDummy})
        pickDummy = -1
    end
    if holdDummy and holdDummy >= 0 then
        sim.removeObjects({holdDummy})
        holdDummy = -1
    end
    if intermediateDummy and intermediateDummy >= 0 then
        sim.removeObjects({intermediateDummy})
        intermediateDummy = -1
    end

    local stalePrePick = optionalObject('/prePickDummy')
    if stalePrePick >= 0 then
        sim.removeObjects({stalePrePick})
    end
    local stalePick = optionalObject('/pickDummy')
    if stalePick >= 0 then
        sim.removeObjects({stalePick})
    end
    local staleHold = optionalObject('/holdDummy')
    if staleHold >= 0 then
        sim.removeObjects({staleHold})
    end
    local staleIntermediate = optionalObject('/intermediateDummy')
    if staleIntermediate >= 0 then
        sim.removeObjects({staleIntermediate})
    end
end

local function removePlaceDummies()
    if prePlaceDummy and prePlaceDummy >= 0 then
        sim.removeObjects({prePlaceDummy})
        prePlaceDummy = -1
    end
    if placeDummy and placeDummy >= 0 then
        sim.removeObjects({placeDummy})
        placeDummy = -1
    end

    local stalePrePlace = optionalObject('/prePlaceDummy')
    if stalePrePlace >= 0 then
        sim.removeObjects({stalePrePlace})
    end
    local stalePlace = optionalObject('/placeDummy')
    if stalePlace >= 0 then
        sim.removeObjects({stalePlace})
    end
end

local function createDummyAtPose(alias, pose)
    local dummy = sim.createDummy(0.03)
    sim.setObjectAlias(dummy, alias)
    sim.setObjectPose(dummy, pose)
    return dummy
end

local function clampRailTargetPosition(targetRail)
    local cyclic, interval = sim.getJointInterval(railJoint)
    if cyclic == false and type(interval) == 'table' and
       type(interval[1]) == 'number' and type(interval[2]) == 'number' and
       interval[2] > 0 then
        local minRail = interval[1]
        local maxRail = interval[1] + interval[2]
        return math.min(math.max(targetRail, minRail), maxRail)
    end
    return targetRail
end

local function moveRailTo(position)
    local requestedRail = position
    local targetRail = clampRailTargetPosition(requestedRail)
    local startTime = sim.getSimulationTime()
    sim.setStringSignal('railApiStatus', 'moving')

    if math.abs(targetRail - requestedRail) > railTargetTolerance then
        sim.addLog(sim.verbosity_warnings,
                   '[external_api] clamped rail target from ' ..
                   tostring(requestedRail) .. ' to ' .. tostring(targetRail))
    end

    sim.setJointTargetPosition(railJoint, targetRail)

    while true do
        if sim.getSimulationState() == sim.simulation_advancing_abouttostop then
            sim.setStringSignal('railApiStatus', 'stopped')
            return {
                ok = false,
                accepted = true,
                requestedPosition = requestedRail,
                targetPosition = targetRail,
                completed = false,
                railStatus = 'stopped',
                reason = 'rail motion stopped',
            }
        end

        local currentRail = sim.getJointPosition(railJoint)
        if math.abs(currentRail - targetRail) <= railTargetTolerance then
            sim.setStringSignal('railApiStatus', 'done')
            return {
                ok = true,
                accepted = true,
                requestedPosition = requestedRail,
                targetPosition = targetRail,
                completed = true,
                railStatus = 'done',
            }
        end

        if sim.getSimulationTime() - startTime > railMotionTimeout then
            sim.setStringSignal('railApiStatus', 'failed')
            return {
                ok = false,
                accepted = true,
                requestedPosition = requestedRail,
                targetPosition = targetRail,
                completed = false,
                railStatus = 'timeout',
                reason = 'rail target position timeout',
            }
        end

        sim.wait(0.01)
    end
end

local function railFailureReason(prefix, railResult)
    if railResult and railResult.reason then
        return prefix .. ': ' .. tostring(railResult.reason)
    end
    if railResult and railResult.railStatus then
        return prefix .. ': rail status ' .. tostring(railResult.railStatus)
    end
    return prefix
end

local function prepositionUr10BaseForPose(pose, reachHalfWidth)
    local basePose = sim.getObjectPose(ur10Base)
    local relativeX = pose[1] - basePose[1]

    if math.abs(relativeX) <= reachHalfWidth then
        return {
            ok = true,
            moved = false,
            relativeX = relativeX,
        }
    end

    local baseShiftX
    if relativeX > reachHalfWidth then
        baseShiftX = relativeX - reachHalfWidth
    else
        baseShiftX = relativeX + reachHalfWidth
    end

    local targetRail = sim.getJointPosition(railJoint) - baseShiftX
    local railMove = moveRailTo(targetRail)
    if not railMove.ok then
        railMove.targetRail = targetRail
        return railMove
    end

    return {
        ok = true,
        moved = true,
        relativeX = relativeX,
        targetRail = targetRail,
        railMove = railMove,
    }
end

local function prepositionUr10BaseForPlace()
    local basePose = sim.getObjectPose(ur10Base)
    local tipPose = sim.getObjectPose(ikTip)
    local palletPose = sim.getObjectPose(pallet)
    local palletSize = objectSize(pallet)
    local tipSideX = tipPose[1] - basePose[1]

    if math.abs(tipSideX) <= 1.0e-9 then
        return {
            ok = true,
            moved = false,
            reason = 'tip is centered on UR10 base x',
        }
    end

    local palletMinX = palletPose[1] - palletSize[1] * 0.5
    local palletMaxX = palletPose[1] + palletSize[1] * 0.5
    local targetBaseX
    if tipSideX > 0 then
        targetBaseX = palletMinX - placeBaseBoundaryOffset
    else
        targetBaseX = palletMaxX + placeBaseBoundaryOffset
    end

    local baseShiftX = targetBaseX - basePose[1]
    local targetRail = sim.getJointPosition(railJoint) - baseShiftX
    local railMove = moveRailTo(targetRail)
    if not railMove.ok then
        railMove.targetRail = targetRail
        return railMove
    end

    return {
        ok = true,
        moved = true,
        tipSideX = tipSideX,
        targetBaseX = targetBaseX,
        targetRail = targetRail,
        railMove = railMove,
    }
end

local function finishPlaceCommand(result)
    return result
end

local function finishPickCommand(result)
    removePickDummies()
    return result
end

local function pickPosesForObject(objectHandle)
    local objectPose = sim.getObjectPose(objectHandle)
    local rotatedPose = sim.multiplyPoses(objectPose, {0, 0, 0, 0, 1, 0, 0})
    local size = objectSize(objectHandle)

    local topZ = objectPose[3] + size[3] * 0.5

    local prePickPose = table.clone(rotatedPose)
    prePickPose[1] = objectPose[1]
    prePickPose[2] = objectPose[2]
    prePickPose[3] = topZ + 0.10

    local pickPose = table.clone(rotatedPose)
    pickPose[1] = objectPose[1]
    pickPose[2] = objectPose[2]
    pickPose[3] = topZ + 0.005

    local holdPose = table.clone(rotatedPose)
    holdPose[1] = objectPose[1]
    holdPose[2] = objectPose[2]
    holdPose[3] = math.max(objectPose[3] + holdLiftHeight, prePickPose[3] + holdClearanceAbovePrePick)

    return prePickPose, pickPose, holdPose, size
end

local function placePositionFromData(positionOrData)
    local position = positionOrData
    if type(positionOrData) == 'table' and positionOrData.position then
        position = positionOrData.position
    end
    if type(positionOrData) == 'table' and positionOrData.x then
        position = {positionOrData.x, positionOrData.y, positionOrData.z}
    end
    if type(position) ~= 'table' or #position < 3 then
        error('place position must be {x, y, z}')
    end
    return position
end

local function attachedObject()
    if attachedObjectHandle and attachedObjectHandle >= 0 then
        return attachedObjectHandle
    end
    if lastPickResult and lastPickResult.ok and lastPickResult.object then
        return lastPickResult.object
    end
    return -1
end

local function placePosesForObject(objectHandle, position)
    local pallet = resolveObject('/pallet', 'pallet')
    local palletPose = sim.getObjectPose(pallet)
    local size = objectSize(objectHandle)
    local center = {
        position[1] + size[1] * 0.5,
        position[2] + size[2] * 0.5,
        position[3] + size[3] * 0.5,
    }
    local orientation = {0, 0, 0, 0, 1, 0, 0}

    local placePose = sim.multiplyPoses(palletPose, {
        center[1], center[2], center[3],
        orientation[4], orientation[5], orientation[6], orientation[7],
    })

    local prePlacePose = sim.multiplyPoses(palletPose, {
        center[1], center[2], center[3] + 0.10,
        orientation[4], orientation[5], orientation[6], orientation[7],
    })

    return prePlacePose, placePose, size
end

function sysCall_init()
    ikTarget = resolveObject('/ikTarget', 'ikTarget')
    railJoint = firstObject({
        '/mobile_arm/railJoint',
        '/railJoint',
        ':/railJoint',
    }, 'rail joint')
    ur10Base = firstObject({
        '/mobile_arm/railJoint/UR10',
        '/mobile_arm/UR10',
        '/UR10',
    }, 'UR10 base')
    ikTip = firstObject({
        '/mobile_arm/railJoint/UR10/ikTip',
        '/mobile_arm/UR10/ikTip',
        '/ikTip',
        ':/ikTip',
    }, 'ikTip')
    pallet = resolveObject('/pallet', 'pallet')

    lastCommand = nil
    pendingPickCommand = nil
    pendingPlaceCommand = nil
    pickBusy = false
    placeBusy = false
    lastPickResult = nil
    lastPlaceResult = nil
    attachedObjectHandle = -1
    sim.setInt32Signal(tcpObjectSignal, -1)
    sim.setInt32Signal('externalApiReady', 1)
    sim.setStringSignal(pickStatusSignal, 'ready')
    sim.setStringSignal(placeStatusSignal, 'ready')
end

function moveTo(targetOrData, mode, tcpObject)
    local data = commandData(targetOrData, mode, tcpObject)
    local requestedMode = data.mode or 'config'

    if requestedMode == 'config' then
        return dispatchMotion(data.target, requestedMode, data.tcpObject, 'config')
    end
    if requestedMode == 'plan' or requestedMode == 'planned' then
        return {
            ok = false,
            reason = 'collision-free planned motion is temporarily disabled',
            state = getState({}),
        }
    end

    return {
        ok = false,
        reason = 'unknown moveTo mode: ' .. tostring(requestedMode),
        state = getState({}),
    }
end

function linearMoveTo(targetOrData, tcpObject)
    local data = commandData(targetOrData, 'linear', tcpObject)
    return dispatchMotion(data.target, 'linear', data.tcpObject, 'linear')
end

local function runPickCommand(command)
    sim.setStringSignal(pickStatusSignal, 'creating_dummies')

    local objectHandle = command.objectHandle
    local prePickPose, pickPose, holdPose, size = pickPosesForObject(objectHandle)

    removePickDummies()
    prePickDummy = createDummyAtPose('prePickDummy', prePickPose)
    pickDummy = createDummyAtPose('pickDummy', pickPose)
    holdDummy = createDummyAtPose('holdDummy', holdPose)

    sim.setStringSignal(pickStatusSignal, 'prepositioning_rail')
    local pickRailPreposition = prepositionUr10BaseForPose(prePickPose, pickReachHalfWidth)
    if not pickRailPreposition.ok then
        return failMotionAndStop(pickStatusSignal, finishPickCommand({
            ok = false,
            reason = railFailureReason('failed prepositioning UR10 base before pre-pick pose', pickRailPreposition),
            object = objectHandle,
            objectSize = size,
            prePickDummy = prePickDummy,
            pickDummy = pickDummy,
            holdDummy = holdDummy,
            pickRailPreposition = pickRailPreposition,
            state = getState({}),
        }))
    end

    sim.setStringSignal(pickStatusSignal, 'approaching_config')
    local approach = dispatchAndWait(prePickDummy, 'config', nil, 60)
    if not approach.completed then
        return failMotionAndStop(pickStatusSignal, finishPickCommand({
            ok = false,
            reason = 'failed moving to pre-pick pose',
            object = objectHandle,
            objectSize = size,
            prePickDummy = prePickDummy,
            pickDummy = pickDummy,
            holdDummy = holdDummy,
            pickRailPreposition = pickRailPreposition,
            approach = approach,
            state = getState({}),
        }))
    end
    motionDelay(delayAfterApproach)

    sim.setStringSignal(pickStatusSignal, 'descending')
    local descend = dispatchAndWait(pickDummy, 'linear', nil, 30)
    if not descend.completed then
        return failMotionAndStop(pickStatusSignal, finishPickCommand({
            ok = false,
            reason = 'failed linear move to pick pose',
            object = objectHandle,
            objectSize = size,
            prePickDummy = prePickDummy,
            pickDummy = pickDummy,
            holdDummy = holdDummy,
            pickRailPreposition = pickRailPreposition,
            approach = approach,
            descend = descend,
            state = getState({}),
        }))
    end
    motionDelay(delayAfterDescend)

    sim.setStringSignal(pickStatusSignal, 'suction_on')
    local suction = setSuction({enabled = true})
    if not suction.ok then
        return finishPickCommand(suction)
    end
    attachedObjectHandle = objectHandle
    motionDelay(delayAfterSuction)

    sim.setStringSignal(pickStatusSignal, 'lifting_to_hold')
    local lift = dispatchAndWait(holdDummy, 'linear', nil, 30)
    if not lift.completed then
        return failMotionAndStop(pickStatusSignal, finishPickCommand({
            ok = false,
            reason = 'failed linear lift to holding pose',
            object = objectHandle,
            objectSize = size,
            prePickDummy = prePickDummy,
            pickDummy = pickDummy,
            holdDummy = holdDummy,
            pickRailPreposition = pickRailPreposition,
            approach = approach,
            descend = descend,
            suction = suction,
            lift = lift,
            state = getState({}),
        }))
    end
    motionDelay(delayAfterRetreat)

    lastCommand = {
        mode = 'pick',
        object = objectHandle,
        objectSize = size,
        prePickDummy = prePickDummy,
        pickDummy = pickDummy,
        holdDummy = holdDummy,
        pickRailPreposition = pickRailPreposition,
    }

    return finishPickCommand({
        ok = true,
        object = objectHandle,
        objectSize = size,
        prePickDummy = prePickDummy,
        pickDummy = pickDummy,
        holdDummy = holdDummy,
        pickRailPreposition = pickRailPreposition,
        approach = approach,
        descend = descend,
        suction = suction,
        lift = lift,
        state = getState({}),
    })
end

local function runPlaceCommand(command)
    local objectHandle = command.objectHandle
    local size = objectSize(objectHandle)
    sim.setStringSignal(placeStatusSignal, 'creating_dummies')

    local prePlacePose, placePose
    prePlacePose, placePose, size = placePosesForObject(objectHandle, command.position)

    sim.setStringSignal(placeStatusSignal, 'prepositioning_rail')
    local railPreposition = prepositionUr10BaseForPlace()
    if not railPreposition.ok then
        return failMotionAndStop(placeStatusSignal, finishPlaceCommand({
            ok = false,
            reason = railFailureReason('failed prepositioning UR10 base before pre-place pose', railPreposition),
            object = objectHandle,
            objectSize = size,
            railPreposition = railPreposition,
            state = getState({}),
        }))
    end

    removePlaceDummies()
    prePlaceDummy = createDummyAtPose('prePlaceDummy', prePlacePose)
    placeDummy = createDummyAtPose('placeDummy', placePose)

    sim.setStringSignal(placeStatusSignal, 'approaching_linear')
    local approach = dispatchAndWait(prePlaceDummy, 'linear', objectHandle, 60)
    if not approach.completed then
        return failMotionAndStop(placeStatusSignal, finishPlaceCommand({
            ok = false,
            reason = 'failed moving to pre-place pose',
            object = objectHandle,
            objectSize = size,
            prePlaceDummy = prePlaceDummy,
            placeDummy = placeDummy,
            railPreposition = railPreposition,
            approach = approach,
            state = getState({}),
        }))
    end
    motionDelay(delayAfterApproach)

    sim.setStringSignal(placeStatusSignal, 'descending')
    local descend = dispatchAndWait(placeDummy, 'linear', objectHandle, 30)
    if not descend.completed then
        return failMotionAndStop(placeStatusSignal, finishPlaceCommand({
            ok = false,
            reason = 'failed linear move to place pose',
            object = objectHandle,
            objectSize = size,
            prePlaceDummy = prePlaceDummy,
            placeDummy = placeDummy,
            railPreposition = railPreposition,
            approach = approach,
            descend = descend,
            state = getState({}),
        }))
    end
    motionDelay(delayAfterDescend)

    sim.setStringSignal(placeStatusSignal, 'suction_off')
    local suction = setSuction({enabled = false})
    if not suction.ok then
        return finishPlaceCommand(suction)
    end
    attachedObjectHandle = -1
    motionDelay(delayAfterSuction)

    sim.setStringSignal(placeStatusSignal, 'retreating')
    local retreat = dispatchAndWait(prePlaceDummy, 'linear', nil, 30)
    if not retreat.completed then
        return failMotionAndStop(placeStatusSignal, finishPlaceCommand({
            ok = false,
            reason = 'failed linear lift from place pose',
            object = objectHandle,
            objectSize = size,
            prePlaceDummy = prePlaceDummy,
            placeDummy = placeDummy,
            railPreposition = railPreposition,
            approach = approach,
            descend = descend,
            suction = suction,
            retreat = retreat,
            state = getState({}),
        }))
    end
    motionDelay(delayAfterRetreat)

    lastCommand = {
        mode = 'place',
        object = objectHandle,
        objectSize = size,
        prePlaceDummy = prePlaceDummy,
        placeDummy = placeDummy,
        railPreposition = railPreposition,
        retreat = retreat,
    }

    return finishPlaceCommand({
        ok = true,
        object = objectHandle,
        objectSize = size,
        prePlaceDummy = prePlaceDummy,
        placeDummy = placeDummy,
        railPreposition = railPreposition,
        approach = approach,
        descend = descend,
        suction = suction,
        retreat = retreat,
        state = getState({}),
    })
end

function pick(objectAliasOrData)
    if pickBusy or placeBusy or pendingPickCommand or pendingPlaceCommand then
        return {
            ok = false,
            reason = 'external_api is already busy',
            state = getState({}),
        }
    end

    local objectAlias = objectAliasOrData
    if type(objectAliasOrData) == 'table' then
        objectAlias = objectAliasOrData.object or
                      objectAliasOrData.objectAlias or
                      objectAliasOrData.target
    end

    local objectHandle = resolveObject(objectAlias, 'pick object')
    if objectHandle < 0 then
        error('pick object is required')
    end

    pendingPickCommand = {
        objectHandle = objectHandle,
    }
    lastCommand = {
        mode = 'pick',
        object = objectHandle,
    }
    sim.setStringSignal(pickStatusSignal, 'queued')

    return {
        ok = true,
        accepted = true,
        object = objectHandle,
        state = getState({}),
    }
end

function place(positionOrData)
    if pickBusy or placeBusy or pendingPickCommand or pendingPlaceCommand then
        return {
            ok = false,
            reason = 'external_api is already busy',
            state = getState({}),
        }
    end

    local position = placePositionFromData(positionOrData)
    local objectHandle = attachedObject()
    if objectHandle < 0 then
        return {
            ok = false,
            reason = 'no attached object is known',
            state = getState({}),
        }
    end

    pendingPlaceCommand = {
        objectHandle = objectHandle,
        position = position,
    }
    lastCommand = {
        mode = 'place',
        object = objectHandle,
        position = position,
    }
    sim.setStringSignal(placeStatusSignal, 'queued')

    return {
        ok = true,
        accepted = true,
        object = objectHandle,
        position = position,
        state = getState({}),
    }
end

function setSuction(inData)
    local enabled = asBool(inData)
    sim.setInt32Signal('suctionPadEnabled', enabled and 1 or 0)
    if not enabled then
        attachedObjectHandle = -1
    end

    return {
        ok = true,
        suctionEnabled = enabled,
        state = getState({}),
    }
end

function getState(inData)
    return {
        ok = true,
        ready = sim.getInt32Signal('externalApiReady') == 1,
        lastCommand = lastCommand,
        pickBusy = pickBusy == true,
        placeBusy = placeBusy == true,
        pickStatus = sim.getStringSignal(pickStatusSignal),
        placeStatus = sim.getStringSignal(placeStatusSignal),
        lastPickResult = lastPickResult,
        lastPlaceResult = lastPlaceResult,
        attachedObject = attachedObjectHandle,
        tcpObject = sim.getInt32Signal(tcpObjectSignal) or -1,
        suctionEnabled = sim.getInt32Signal('suctionPadEnabled') == 1,
    }
end

function sysCall_thread()
    while sim.getSimulationState() ~= sim.simulation_advancing_abouttostop do
        if pendingPickCommand and not pickBusy then
            local command = pendingPickCommand
            pendingPickCommand = nil
            pickBusy = true
            sim.setStringSignal(pickStatusSignal, 'running')

            lastPickResult = runPickCommand(command)

            pickBusy = false
            sim.setStringSignal(pickStatusSignal,
                                lastPickResult.ok and 'done' or 'failed')
        elseif pendingPlaceCommand and not placeBusy then
            local command = pendingPlaceCommand
            pendingPlaceCommand = nil
            placeBusy = true
            sim.setStringSignal(placeStatusSignal, 'running')

            lastPlaceResult = runPlaceCommand(command)

            placeBusy = false
            sim.setStringSignal(placeStatusSignal,
                                lastPlaceResult.ok and 'done' or 'failed')
        else
            sim.wait(0.01)
        end
        
    end
end

function sysCall_cleanup()
    sim.setInt32Signal('externalApiReady', 0)
    sim.setInt32Signal(tcpObjectSignal, -1)
    sim.setStringSignal(pickStatusSignal, 'stopped')
    sim.setStringSignal(placeStatusSignal, 'stopped')
end
