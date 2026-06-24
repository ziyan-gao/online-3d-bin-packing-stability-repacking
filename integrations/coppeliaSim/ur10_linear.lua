sim = require('sim')
simIK = require('simIK')

--[[
Threaded UR10 Cartesian-linear motion.

How to use:
  1. Put a dummy named `/ikTarget` in the world.
  2. Start simulation.
  3. Trigger this script with:
       sim.setStringSignal('ur10MotionMode', 'linear')

What this script does:
  - It watches the string signal `ur10MotionMode`.
  - When the signal value is `linear`, it reads the current pose of `/ikTarget`.
  - It interpolates the tool pose from the current `ikTip` pose to `/ikTarget`.
  - At each interpolated pose, it solves IK and updates the UR10 joints.

Important:
  This is Cartesian-space motion.  The tip should follow a straight-line path
  from its current pose to the target pose, as long as IK can solve each step.
]]

local modeSignal = 'ur10MotionMode'
local myMode = 'linear'
local statusSignal = 'ur10LinearMotionStatus'
local helperDisabled = true

local function setStatus(status)
    sim.setStringSignal(statusSignal, status)
end

local function logWarning(message)
    sim.addLog(sim.verbosity_warnings, '[ur10_linear_motion] ' .. message)
end

local function firstObject(paths)
    for _, path in ipairs(paths) do
        local handle = sim.getObject(path, {noError = true})
        if handle and handle >= 0 then
            return handle
        end
    end
    error('Could not find object. Tried: ' .. table.concat(paths, ', '))
end

local function moveToPoseCallback(data)
    sim.setObjectPose(data.auxData.target, data.pose, data.auxData.base)
    simIK.setObjectPose(data.auxData.env,
                        data.auxData.ikTarget,
                        data.pose,
                        data.auxData.ikBase)
    simIK.handleGroup(data.auxData.env, data.auxData.group, {syncWorlds = true})
end

local function getTipTargetError()
    local finalTipPose = sim.getObjectPose(ikTip, robotBase)
    local finalTargetPose = sim.getObjectPose(ikTarget, robotBase)
    local dx = finalTipPose[1] - finalTargetPose[1]
    local dy = finalTipPose[2] - finalTargetPose[2]
    local dz = finalTipPose[3] - finalTargetPose[3]
    local distance = math.sqrt(dx * dx + dy * dy + dz * dz)

    return dx, dy, dz, distance
end

local function logFinalPoseError(label)
    local dx, dy, dz, distance = getTipTargetError()

    sim.addLog(sim.verbosity_scriptinfos,
        string.format('[ur10_linear_motion] %s tip-target xyz error: %.6f %.6f %.6f, norm: %.6f',
            label, dx, dy, dz, distance))
end

local function settleAtTargetPose(targetPose)
    local tolerance = 0.0005
    local maxIterations = 20

    sim.setObjectPose(ikTarget, targetPose, robotBase)
    simIK.setObjectPose(ikEnv,
                        simToIkObjectMap[ikTarget],
                        targetPose,
                        simToIkObjectMap[robotBase])

    for _ = 1, maxIterations, 1 do
        local result = simIK.handleGroup(ikEnv, ikGroup, {syncWorlds = true})
        if result ~= simIK.result_success then
            logWarning('settle IK handleGroup failed with result: ' .. tostring(result))
            return false
        end

        local _, _, _, distance = getTipTargetError()
        if distance <= tolerance then
            return true
        end
    end

    return false
end

local function moveToTargetLinearly()
    simIK.syncFromSim(ikEnv, {ikGroup})

    local startPose = sim.getObjectPose(ikTip, robotBase)
    local targetPose = sim.getObjectPose(ikTarget, robotBase)

    lastStartPose = startPose
    lastTargetPose = targetPose

    setStatus('moving')
    sim.moveToPose({
        pose = startPose,
        targetPose = targetPose,
        maxVel = maxVel,
        maxAccel = maxAccel,
        maxJerk = maxJerk,
        callback = moveToPoseCallback,
        auxData = ikData,
    })
    logFinalPoseError('before settle')
    settleAtTargetPose(targetPose)
    logFinalPoseError('after settle')
    setStatus('done')
end

function sysCall_init()
    if helperDisabled then
        setStatus('disabled')
        logWarning('disabled because external_api.lua now owns UR10 Cartesian-linear motion')
        return
    end

    robotBase = firstObject({
        '/mobile_arm/railJoint/UR10',
        '/mobile_arm/UR10',
        '/UR10',
    })
    ikTip = firstObject({
        '/mobile_arm/railJoint/UR10/ikTip',
        '/mobile_arm/UR10/ikTip',
        '/ikTip',
        ':/ikTip',
    })
    ikTarget = firstObject({
        '/ikTarget',
        ':/ikTarget',
    })

    maxVel = {0.4, 0.4, 0.4, 1.8}
    maxAccel = {0.8, 0.8, 0.8, 0.9}
    maxJerk = {0.6, 0.6, 0.6, 0.8}

    ikEnv = simIK.createEnvironment()
    ikGroup = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv, ikGroup, simIK.method_damped_least_squares, 0.3, 99)
    local _
    _, simToIkObjectMap = simIK.addElementFromScene(ikEnv,
                                                    ikGroup,
                                                    robotBase,
                                                    ikTip,
                                                    ikTarget,
                                                    simIK.constraint_pose)

    ikData = {
        env = ikEnv,
        group = ikGroup,
        target = ikTarget,
        base = robotBase,
        ikTarget = simToIkObjectMap[ikTarget],
        ikBase = simToIkObjectMap[robotBase],
    }
    setStatus('ready')
end

function sysCall_cleanup()
    if ikEnv then
        simIK.eraseEnvironment(ikEnv)
        ikEnv = nil
    end
    setStatus('stopped')
end
