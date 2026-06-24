sim = require('sim')

--[[
Threaded rail motion.

How to use:
  1. Put a dummy named `/rail_target` in the world.
  2. Start simulation.
  3. Trigger this script with:
       sim.setInt32Signal('go_to_rail_target', 1)

What this script does:
  - It watches the boolean-style signal `go_to_rail_target`.
  - When the signal is 1/true, it immediately sets the signal back to 0/false.
  - It reads the current position of `/rail_target`.
  - It converts that target position into a prismatic-joint position.
  - It moves the rail joint with `sim.moveToConfig`.

Important:
  In CoppeliaSim, a prismatic joint moves along its local joint axis.  The
  joint coordinate used here is therefore the target dummy position expressed
  in the rail joint frame, along that local axis.
]]

local triggerSignal = 'go_to_rail_target'
local statusSignal = 'railApiStatus'
local targetPositionSignal = 'railApiTargetPosition'
local readySignal = 'railApiReady'
local pendingRailTargetPosition = nil
local activeRailTargetPosition = nil

local function firstObject(paths)
    for _, path in ipairs(paths) do
        local handle = sim.getObject(path, {noError = true})
        if handle and handle >= 0 then
            return handle
        end
    end
    error('Could not find object. Tried: ' .. table.concat(paths, ', '))
end

local function optionalObject(paths)
    for _, path in ipairs(paths) do
        local handle = sim.getObject(path, {noError = true})
        if handle and handle >= 0 then
            return handle
        end
    end
    return -1
end

local function getRailTargetPosition()
    if not (railTarget and railTarget >= 0) then
        error('rail_target is required when go_to_rail_target has no railApiTargetPosition')
    end
    local positionInRailFrame = sim.getObjectPosition(railTarget, rail_base)
    -- CoppeliaSim joint motion is along the joint's local z-axis.  If the rail
    -- visually points along world x/y, the rail joint object itself should be
    -- rotated so its local z-axis is aligned with the physical rail direction.
    return -positionInRailFrame[1]
end

local function moveRailToPositionValue(targetPosition)
    lastTargetPosition = targetPosition
    sim.setStringSignal(statusSignal, 'moving')

    sim.moveToConfig({
        joints = {railJoint},
        targetPos = {targetPosition},
        maxVel = {railMaxVel},
        maxAccel = {railMaxAccel},
        maxJerk = {railMaxJerk},
    })
    sim.setStringSignal(statusSignal, 'done')
end

local function requestedRailTargetPosition()
    if pendingRailTargetPosition ~= nil then
        local targetPosition = pendingRailTargetPosition
        pendingRailTargetPosition = nil
        return targetPosition
    end

    if sim.getInt32Signal(triggerSignal) == 1 then
        sim.setInt32Signal(triggerSignal, 0)
        local requestedTargetPosition = sim.getFloatSignal(targetPositionSignal)
        if requestedTargetPosition ~= nil then
            return requestedTargetPosition
        end
        return getRailTargetPosition()
    end

    return nil
end

local function startRailMotion(targetPosition)
    activeRailTargetPosition = targetPosition
    lastTargetPosition = targetPosition
    sim.setStringSignal(statusSignal, 'moving')
end

local function stepActiveRailMotion()
    if activeRailTargetPosition == nil then
        return false
    end

    local currentPosition = sim.getJointPosition(railJoint)
    local delta = activeRailTargetPosition - currentPosition
    local maxStep = railMaxVel * sim.getSimulationTimeStep()

    if math.abs(delta) <= maxStep or maxStep <= 0 then
        sim.setJointPosition(railJoint, activeRailTargetPosition)
        activeRailTargetPosition = nil
        sim.setStringSignal(statusSignal, 'done')
        return true
    end

    local direction = delta > 0 and 1 or -1
    local nextPosition = currentPosition + direction * maxStep
    sim.setJointPosition(railJoint, nextPosition)
    return true
end

local function stepRailMotion()
    if stepActiveRailMotion() then
        return true
    end

    local targetPosition = requestedRailTargetPosition()
    if targetPosition == nil then
        return false
    end

    startRailMotion(targetPosition)
    stepActiveRailMotion()
    return true
end

function moveRailToPosition(inData)
    local targetPosition = inData
    if type(inData) == 'table' then
        targetPosition = inData.position or inData.targetPosition
    end
    if type(targetPosition) ~= 'number' then
        return {
            ok = false,
            reason = 'moveRailToPosition expects a numeric position',
        }
    end

    pendingRailTargetPosition = targetPosition
    sim.setStringSignal(statusSignal, 'requested')
    return {ok = true, accepted = true, targetPosition = targetPosition}
end

local function moveToRailTarget()
    local targetPosition = getRailTargetPosition()
    moveRailToPositionValue(targetPosition)
end

function sysCall_init()
    sim.setStepping(true)
    pendingRailTargetPosition = nil
    activeRailTargetPosition = nil
    rail_base = sim.getObject('/mobile_arm')
    

    railJoint = firstObject({
        '/mobile_arm/railJoint',
        '/railJoint',
        ':/railJoint',
    })
    railTarget = optionalObject({
        '/rail_target',
        ':/rail_target',
    })

    railMaxVel = 0.25
    railMaxAccel = 0.5
    railMaxJerk = 0.5
    sim.setInt32Signal(readySignal, 1)
    sim.setStringSignal(statusSignal, 'ready')
end

function sysCall_actuation()
    stepRailMotion()
end

function sysCall_thread()
    while sim.getSimulationState() ~= sim.simulation_advancing_abouttostop do
        local targetPosition = requestedRailTargetPosition()
        if targetPosition ~= nil then
            moveRailToPositionValue(targetPosition)
        else
            sim.step()
        end
    end
end

function sysCall_cleanup()
    sim.setInt32Signal(readySignal, 0)
    sim.setStringSignal(statusSignal, 'stopped')
end
