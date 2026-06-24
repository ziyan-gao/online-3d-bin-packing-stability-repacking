sim = require('sim')
simIK = require('simIK')

--[[
Threaded UR10 joint-space motion.

How to use:
  1. Put a dummy named `/ikTarget` in the world.
  2. Start simulation.
  3. Trigger this script with:
       sim.setStringSignal('ur10MotionMode', 'config')

What this script does:
  - It watches the string signal `ur10MotionMode`.
  - When the signal value is `config`, it reads the current pose of `/ikTarget`.
  - It solves IK with the same direct scene-element flow as `moveToConfig.lua`.
  - It samples the joint-space trajectory to the solved configuration.
  - It moves the UR10 joints to the solved configuration with `sim.moveToConfig`.

Important:
  This is joint-space motion.  The final tip pose should match `/ikTarget`,
  but the path of the tip is not guaranteed to be a straight line.
]]

local modeSignal = 'ur10MotionMode'
local myMode = 'config'
local statusSignal = 'ur10JointMotionStatus'
local ikConfigCountSignal = 'ur10JointMotionIkConfigCount'
local commandCountSignal = 'ur10JointMotionCommandCount'
local tcpObjectSignal = 'motionTcpObjectHandle'
local helperDisabled = true

local function logInfo(message)
    sim.addLog(sim.verbosity_scriptinfos, '[ur10_joint_motion] ' .. message)
end

local function logWarning(message)
    sim.addLog(sim.verbosity_warnings, '[ur10_joint_motion] ' .. message)
end

local function setStatus(status)
    sim.setStringSignal(statusSignal, status)
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

local function makeJointVector(value)
    local values = {}
    for i = 1, #joints do
        values[i] = value
    end
    return values
end

local function tipTargetError()
    local tipPose = sim.getObjectPose(ikTip, robotBase)
    local targetPose = sim.getObjectPose(ikTarget, robotBase)
    local dx = tipPose[1] - targetPose[1]
    local dy = tipPose[2] - targetPose[2]
    local dz = tipPose[3] - targetPose[3]
    local distance = math.sqrt(dx * dx + dy * dy + dz * dz)

    return dx, dy, dz, distance
end

local function logTipTargetError(label)
    local dx, dy, dz, distance = tipTargetError()
    logInfo(string.format('%s tip-target xyz error: %.6f %.6f %.6f, norm: %.6f',
                          label, dx, dy, dz, distance))
end

local function updateRobotCollection()
    if robotCollection then
        sim.destroyCollection(robotCollection)
    end

    robotCollection = sim.createCollection()
    sim.addItemToCollection(robotCollection, sim.handle_tree, robotBase, 0)

    local tcpObject = sim.getInt32Signal(tcpObjectSignal) or -1
    if tcpObject >= 0 then
        sim.addItemToCollection(robotCollection, sim.handle_single, tcpObject, 0)
        logInfo('including TCP object in collision collection: ' .. tostring(tcpObject))
    end
end

local function getCurrentConfig()
    local config = {}
    for i = 1, #joints do
        config[i] = sim.getJointPosition(joints[i])
    end
    return config
end

local function setConfig(config)
    for i = 1, #joints do
        sim.setJointPosition(joints[i], config[i])
    end
end

local function sampleJointPath(startConfig, targetConfig)
    local maxDelta = 0
    for i = 1, #joints do
        maxDelta = math.max(maxDelta, math.abs(targetConfig[i] - startConfig[i]))
    end

    local steps = math.max(2, math.ceil(maxDelta / collisionCheckResolution))
    local configs = {}
    for step = 0, steps do
        local t = step / steps
        local config = {}
        for i = 1, #joints do
            config[i] = startConfig[i] + (targetConfig[i] - startConfig[i]) * t
        end
        configs[#configs + 1] = config
    end
    return configs
end

local function collides(configs)
    local bufferedConfig = getCurrentConfig()

    for i = 1, #configs do
        setConfig(configs[i])
        if sim.checkCollision(robotCollection, sim.handle_all) > 0 or
           sim.checkCollision(robotCollection, robotCollection) > 0 then
            setConfig(bufferedConfig)
            return true
        end
    end

    setConfig(bufferedConfig)
    return false
end

local function trajectoryIsCollisionFree(targetConfig)
    local startConfig = getCurrentConfig()
    local pathConfigs = sampleJointPath(startConfig, targetConfig)
    return not collides(pathConfigs)
end

local function configError(targetConfig)
    local maxError = 0
    for i = 1, #joints do
        local error = math.abs(sim.getJointPosition(joints[i]) - targetConfig[i])
        maxError = math.max(maxError, error)
    end
    return maxError
end

local function configIsFinite(config)
    if type(config) ~= 'table' or #config ~= #joints then
        return false
    end

    for i = 1, #joints do
        local value = config[i]
        if type(value) ~= 'number' or value ~= value or
           value == math.huge or value == -math.huge then
            return false
        end
    end

    return true
end

local function applyFinalConfig(targetConfig)
    setConfig(targetConfig)
    logInfo(string.format('after final config apply max joint error: %.6f',
                          configError(targetConfig)))
end

local function solveTargetConfig()
    simIK.syncFromSim(ikEnv, {ikGroup})

    local targetPose = sim.getObjectPose(ikTarget, robotBase)
    simIK.setObjectPose(ikEnv,
                        simToIkObjectMap[ikTarget],
                        targetPose,
                        simToIkObjectMap[robotBase])

    local result, flags, precision = simIK.handleGroup(ikEnv, ikGroup)
    if precision then
        logInfo(string.format('IK precision: linear=%.6f, angular=%.6f',
                              precision[1] or -1,
                              precision[2] or -1))
    end
    if result ~= simIK.result_success then
        sim.setInt32Signal(ikConfigCountSignal, 0)
        logWarning('IK handleGroup failed with result: ' .. tostring(result) ..
                   ', flags: ' .. tostring(flags))
        return nil
    end

    local targetConfig = {}
    for i = 1, #joints do
        targetConfig[i] = simIK.getJointPosition(ikEnv, simToIkObjectMap[joints[i]])
    end

    sim.setInt32Signal(ikConfigCountSignal, 1)
    logInfo('IK direct solve produced one target config')
    return targetConfig
end

local function moveToTargetByConfig()
    updateRobotCollection()
    setStatus('solving_ik')
    local targetConfig = solveTargetConfig()
    if not targetConfig then
        setStatus('ik_failed')
        logWarning('IK failed for /ikTarget')
        return
    end
    if not configIsFinite(targetConfig) then
        setStatus('ik_failed')
        logWarning('invalid IK target config; refusing sim.moveToConfig')
        return
    end

    setStatus('checking_collision')
    if not trajectoryIsCollisionFree(targetConfig) then
        setStatus('collision_failed')
        logWarning('no collision-free joint path to /ikTarget')
        return
    end

    lastTargetPose = sim.getObjectPose(ikTarget, robotBase)
    setStatus('moving')
    logInfo('executing sim.moveToConfig')

    sim.moveToConfig({
        joints = joints,
        targetPos = targetConfig,
        maxVel = maxVel,
        maxAccel = maxAccel,
        maxJerk = maxJerk,
    })

    logTipTargetError('after moveToConfig')
    applyFinalConfig(targetConfig)
    logTipTargetError('after final config apply')
    setStatus('done')
    logInfo('sim.moveToConfig finished')
end

function sysCall_init()
    commandCount = 0
    if helperDisabled then
        setStatus('disabled')
        sim.setInt32Signal(ikConfigCountSignal, 0)
        sim.setInt32Signal(commandCountSignal, commandCount)
        logInfo('disabled because external_api.lua now owns UR10 joint-space motion')
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
    logInfo('handles: robotBase=' .. tostring(robotBase) ..
            ', ikTip=' .. tostring(ikTip) ..
            ', ikTarget=' .. tostring(ikTarget))

    joints = sim.getObjectsInTree(robotBase, sim.sceneobject_joint)
    logInfo('joint count: ' .. tostring(#joints))
    maxVel = makeJointVector(math.pi / 3)
    maxAccel = makeJointVector(math.pi / 3)
    maxJerk = makeJointVector(math.pi / 3)
    collisionCheckResolution = 5 * math.pi / 180
    ikLinearPrecision = 0.0005
    ikAngularPrecision = 0.005

    updateRobotCollection()

    ikEnv = simIK.createEnvironment()
    ikGroup = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv,
                              ikGroup,
                              simIK.method_damped_least_squares,
                              0.3,
                              99)
    logInfo('IK method set to damped least squares')

    ikElement, simToIkObjectMap = simIK.addElementFromScene(ikEnv,
                                                            ikGroup,
                                                            robotBase,
                                                            ikTip,
                                                            ikTarget,
                                                            simIK.constraint_pose)
    simIK.setElementPrecision(ikEnv,
                              ikGroup,
                              ikElement,
                              {ikLinearPrecision, ikAngularPrecision})
    logInfo(string.format('IK element precision set: linear=%.6f, angular=%.6f',
                          ikLinearPrecision,
                          ikAngularPrecision))

    for i = 1, #joints do
        local ikJoint = simToIkObjectMap[joints[i]]
        if not ikJoint then
            error('UR10 joint was not mapped into the IK environment: ' ..
                  tostring(joints[i]))
        end
    end
    setStatus('ready')
    sim.setInt32Signal(ikConfigCountSignal, 0)
    sim.setInt32Signal(commandCountSignal, commandCount)
end

function sysCall_cleanup()
    if ikEnv then
        simIK.eraseEnvironment(ikEnv)
        ikEnv = nil
    end
    if robotCollection then
        sim.destroyCollection(robotCollection)
        robotCollection = nil
    end
    setStatus('stopped')
end
