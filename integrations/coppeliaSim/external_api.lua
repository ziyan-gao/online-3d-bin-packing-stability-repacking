sim = require('sim')
simIK = require('simIK')
simOMPL = require('simOMPL')

--[[
Small public motion API.

Supported calls:
  moveTo(dummyOrPose, mode, tcpObject)
  moveTo({target = dummyOrPose, mode = 'config', tcpObject = object})
  moveTo({target = dummyOrPose, mode = 'planned', tcpObject = object})
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

local tcpObjectSignal = 'motionTcpObjectHandle'
local pickStatusSignal = 'externalApiPickStatus'
local placeStatusSignal = 'externalApiPlaceStatus'

local delayAfterApproach = 0.5
local delayAfterDescend = 0.2
local delayAfterSuction = 0.5
local delayAfterRetreat = 0.2
local placeBaseBoundaryOffset = 0.
local pickReachHalfWidth = 0.7
local railMaxVel = 0.25
local railMaxAccel = 0.5
local railMaxJerk = 0.5
local railMotionTimeout = 20
local railTargetTolerance = 0.001
local holdLiftHeight = 1.3
local holdClearanceAbovePrePick = 0.05
local prePickFirstJointAngle = 0.0
local prePlaceFirstJointAngle = 0.5 * math.pi
local pathPlanningMaxTime = 10.0
local pathPlanningMaxSimplificationTime = 10.0
local plannedRailSearchHalfWidth = 2.0
local plannedRailSearchStep = 0.25
local plannedRailUniformSamples = 17
local plannedDebugEnabled = true
local plannedPlaybackMaxStallIterations = 200
local findConfigsMaxDist = 0.28
local findConfigsMaxTime = 1.0
local findConfigsLinearTolerance = 0.005
local findConfigsAngularTolerance = 2.0 * math.pi / 180.0
local findConfigsArmMetric = {8.0, 8.0, 8.0, 0.8, 0.6, 0.3}

local motionStatusSignals = {
    config = 'ur10JointMotionStatus',
    linear = 'ur10LinearMotionStatus',
    planned = 'railUr10MotionStatus',
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
    planned = {
        done = true,
        ik_failed = true,
        collision_failed = true,
        start_state_invalid = true,
        goal_state_invalid = true,
        planning_failed = true,
        trajectory_failed = true,
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

local function stateSnapshot()
    return {
        ok = true,
        ready = sim.getInt32Signal('externalApiReady') == 1,
        lastCommand = lastCommand,
        moveBusy = moveBusy == true,
        pickBusy = pickBusy == true,
        placeBusy = placeBusy == true,
        moveStatus = sim.getStringSignal(motionStatusSignals.planned),
        pickStatus = sim.getStringSignal(pickStatusSignal),
        placeStatus = sim.getStringSignal(placeStatusSignal),
        attachedObject = attachedObjectHandle,
        tcpObject = sim.getInt32Signal(tcpObjectSignal) or -1,
        suctionEnabled = sim.getInt32Signal('suctionPadEnabled') == 1,
    }
end

local function resultSummary(result)
    if not result then
        return nil
    end

    return {
        ok = result.ok == true,
        reason = result.reason,
        object = result.object,
        objectSize = result.objectSize,
        completed = result.completed,
        mode = result.mode,
        motionStatus = result.motionStatus,
        railStatus = result.railStatus,
    }
end

local function logMotionInfo(message)
    sim.addLog(sim.verbosity_scriptinfos, '[external_api_motion] ' .. message)
end

local function logMotionWarning(message)
    sim.addLog(sim.verbosity_warnings, '[external_api_motion] ' .. message)
end

local function fmt(value)
    if type(value) ~= 'number' then
        return tostring(value)
    end
    return string.format('%.5f', value)
end

local function fmtPoseXYZ(pose)
    if type(pose) ~= 'table' then
        return tostring(pose)
    end
    return '(' .. fmt(pose[1]) .. ', ' .. fmt(pose[2]) .. ', ' .. fmt(pose[3]) .. ')'
end

local function fmtConfig(config, maxCount)
    if type(config) ~= 'table' then
        return tostring(config)
    end

    maxCount = maxCount or #config
    local values = {}
    local count = math.min(#config, maxCount)
    for i = 1, count do
        values[#values + 1] = fmt(config[i])
    end
    if #config > count then
        values[#values + 1] = '...'
    end
    return '{' .. table.concat(values, ', ') .. '}'
end

local function plannedDebug(message)
    if plannedDebugEnabled then
        sim.addLog(sim.verbosity_warnings, '[external_api_planned_debug] ' .. message)
    end
end

local function objectLabel(handle)
    if type(handle) ~= 'number' or handle < 0 then
        return tostring(handle)
    end
    return sim.getObjectAlias(handle, 5) .. '[' .. tostring(handle) .. ']'
end

local function fmtCollisionPair(pair)
    if type(pair) == 'table' and #pair >= 2 then
        return objectLabel(pair[1]) .. ' <-> ' .. objectLabel(pair[2])
    end
    return 'unknown'
end

local function collectionCollisionPair(collection)
    local result, pair = sim.checkCollision(collection, sim.handle_all)
    if result and result > 0 then
        return pair, 'scene'
    end

    return nil, nil
end

local function setMotionStatus(mode, status)
    local statusSignal = motionStatusSignals[mode]
    if statusSignal then
        sim.setStringSignal(statusSignal, status)
    end
end

local function makeJointVector(value)
    local values = {}
    for i = 1, #joints do
        values[i] = value
    end
    return values
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

local function updateRobotCollection()
    if robotCollection then
        sim.destroyCollection(robotCollection)
    end

    robotCollection = sim.createCollection()
    sim.addItemToCollection(robotCollection, sim.handle_tree, ur10Base, 0)

    local tcpObject = sim.getInt32Signal(tcpObjectSignal) or -1
    if tcpObject >= 0 then
        sim.addItemToCollection(robotCollection, sim.handle_single, tcpObject, 0)
        logMotionInfo('including TCP object in collision collection: ' .. tostring(tcpObject))
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
        local pair, pairKind = collectionCollisionPair(robotCollection)
        if pair then
            setConfig(bufferedConfig)
            return true, pair, pairKind
        end
    end

    setConfig(bufferedConfig)
    return false, nil, nil
end

local function logCurrentUr10CollisionState(label)
    local currentConfig = getCurrentConfig()
    local inCollision, pair, pairKind = collides({currentConfig})
    if inCollision then
        logMotionWarning('current UR10 collision before ' .. label ..
                         ': pair=' .. fmtCollisionPair(pair) ..
                         ', pairKind=' .. tostring(pairKind) ..
                         ', config=' .. fmtConfig(currentConfig, #currentConfig))
        return true
    end

    logMotionInfo('current UR10 collision-free before ' .. label ..
                  ': config=' .. fmtConfig(currentConfig, #currentConfig))
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
        local jointError = math.abs(sim.getJointPosition(joints[i]) - targetConfig[i])
        maxError = math.max(maxError, jointError)
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

local function poseError(actualPose, targetPose)
    local dx = actualPose[1] - targetPose[1]
    local dy = actualPose[2] - targetPose[2]
    local dz = actualPose[3] - targetPose[3]
    local positionError = math.sqrt(dx * dx + dy * dy + dz * dz)
    local orientationError = 0.0

    if #actualPose >= 7 and #targetPose >= 7 then
        local dot = actualPose[4] * targetPose[4] +
                    actualPose[5] * targetPose[5] +
                    actualPose[6] * targetPose[6] +
                    actualPose[7] * targetPose[7]
        dot = math.max(-1.0, math.min(1.0, math.abs(dot)))
        orientationError = 2.0 * math.acos(dot)
    end

    return positionError, orientationError
end

local function findConfigsParams(jointCount)
    local metric = {}
    if #joints > 0 and jointCount == #joints + 1 then
        metric[1] = 1.0
        for i = 1, #findConfigsArmMetric do
            metric[i + 1] = findConfigsArmMetric[i]
        end
    else
        for i = 1, jointCount do
            metric[i] = findConfigsArmMetric[i] or 1.0
        end
    end

    return {
        maxDist = findConfigsMaxDist,
        maxTime = findConfigsMaxTime,
        findMultiple = true,
        cMetric = metric,
    }
end

local function normalizeIkConfigs(rawConfigs, jointCount)
    local configs = {}
    if type(rawConfigs) ~= 'table' then
        return configs
    end

    local singleConfig = (#rawConfigs == jointCount)
    if singleConfig then
        for i = 1, jointCount do
            if type(rawConfigs[i]) ~= 'number' then
                singleConfig = false
                break
            end
        end
    end
    if singleConfig then
        rawConfigs = {rawConfigs}
    end

    for _, rawConfig in ipairs(rawConfigs) do
        if type(rawConfig) == 'table' and #rawConfig == jointCount then
            local config = {}
            local usable = true
            for i = 1, jointCount do
                local value = rawConfig[i]
                if type(value) ~= 'number' then
                    usable = false
                    break
                end
                config[i] = value
            end
            if usable then
                configs[#configs + 1] = config
            end
        end
    end

    return configs
end

local function findFeasibleIkConfig(label, env, group, simToIkObjectMap, motionJoints, targetPose, baseObject, getMotionConfig, setMotionConfig, configIsUsable, collisionCheck, logFn)
    if simIK.findConfigs == nil then
        logFn(label .. ' findConfigs fallback unavailable: simIK.findConfigs is missing')
        return nil
    end

    local ikTargetObject = simToIkObjectMap[ikTarget]
    local ikBaseObject = simToIkObjectMap[baseObject]
    if not ikTargetObject or not ikBaseObject then
        logFn(label .. ' findConfigs fallback missing IK target/base mapping')
        return nil
    end

    local ikJoints = {}
    for i = 1, #motionJoints do
        ikJoints[i] = simToIkObjectMap[motionJoints[i]]
        if not ikJoints[i] then
            logFn(label .. ' findConfigs fallback missing IK joint mapping for ' ..
                  tostring(motionJoints[i]))
            return nil
        end
    end

    simIK.syncFromSim(env, {group})
    simIK.setObjectPose(env, ikTargetObject, targetPose, ikBaseObject)

    local rawConfigs = simIK.findConfigs(env, group, ikJoints, findConfigsParams(#ikJoints))
    local candidateConfigs = normalizeIkConfigs(rawConfigs, #motionJoints)
    logFn(label .. ' findConfigs returned ' .. tostring(#candidateConfigs) ..
          ' usable candidate(s)')

    local bufferedConfig = getMotionConfig()
    local rejectedInvalid = 0
    local rejectedPose = 0
    local rejectedCollision = 0

    for i = 1, #candidateConfigs do
        local candidateConfig = candidateConfigs[i]
        if not configIsUsable(candidateConfig) then
            rejectedInvalid = rejectedInvalid + 1
        else
            setMotionConfig(candidateConfig)
            local tipPose = sim.getObjectPose(ikTip, baseObject)
            local positionError, orientationError = poseError(tipPose, targetPose)
            if positionError <= findConfigsLinearTolerance and
               orientationError <= findConfigsAngularTolerance then
                local inCollision, pair, pairKind = collisionCheck(candidateConfig)
                if not inCollision then
                    setMotionConfig(bufferedConfig)
                    logFn(label .. ' findConfigs accepted candidate ' ..
                          tostring(i) ..
                          string.format(' with pose error linear=%.6f, angular=%.6f',
                                        positionError,
                                        orientationError))
                    return candidateConfig
                end

                rejectedCollision = rejectedCollision + 1
                logFn(label .. ' findConfigs rejected colliding candidate ' ..
                      tostring(i) .. ': pair=' .. fmtCollisionPair(pair) ..
                      ', pairKind=' .. tostring(pairKind))
            else
                rejectedPose = rejectedPose + 1
                logFn(label .. ' findConfigs rejected candidate ' ..
                      tostring(i) ..
                      string.format(' with pose error linear=%.6f, angular=%.6f',
                                    positionError,
                                    orientationError))
            end
        end
    end

    setMotionConfig(bufferedConfig)
    logFn(label .. ' findConfigs found no feasible candidate' ..
          ': rejectedInvalid=' .. tostring(rejectedInvalid) ..
          ', rejectedPose=' .. tostring(rejectedPose) ..
          ', rejectedCollision=' .. tostring(rejectedCollision))
    return nil
end

local function applyFinalConfig(targetConfig)
    setConfig(targetConfig)
    logMotionInfo(string.format('after final config apply max joint error: %.6f',
                                configError(targetConfig)))
end

local function nearestEquivalentAngle(targetAngle, currentAngle)
    local wrapped = targetAngle
    local twoPi = 2.0 * math.pi
    while wrapped - currentAngle > math.pi do
        wrapped = wrapped - twoPi
    end
    while wrapped - currentAngle < -math.pi do
        wrapped = wrapped + twoPi
    end
    return wrapped
end

local function moveFirstArmJointTo(targetAngle, label, allowOppositeSign)
    if not joints or not joints[1] then
        logMotionWarning('cannot pre-orient first arm joint before ' ..
                         tostring(label) .. ': missing first UR10 joint')
        return false
    end

    local currentAngle = sim.getJointPosition(joints[1])
    local wrappedTargetAngle = nearestEquivalentAngle(targetAngle, currentAngle)
    if allowOppositeSign then
        local oppositeTargetAngle = nearestEquivalentAngle(-targetAngle, currentAngle)
        if math.abs(oppositeTargetAngle - currentAngle) < math.abs(wrappedTargetAngle - currentAngle) then
            wrappedTargetAngle = oppositeTargetAngle
        end
    end

    logMotionInfo(string.format('pre-orienting first arm joint before %s from %.6f to %.6f rad',
                                tostring(label),
                                currentAngle,
                                wrappedTargetAngle))
    sim.moveToConfig({
        joints = {joints[1]},
        targetPos = {wrappedTargetAngle},
        maxVel = {configMaxVel[1]},
        maxAccel = {configMaxAccel[1]},
        maxJerk = {configMaxJerk[1]},
    })
    logMotionInfo('first arm joint pre-orientation finished before ' ..
                  tostring(label))
    return true
end

local function tipTargetError()
    local tipPose = sim.getObjectPose(ikTip, ur10Base)
    local targetPose = sim.getObjectPose(ikTarget, ur10Base)
    local dx = tipPose[1] - targetPose[1]
    local dy = tipPose[2] - targetPose[2]
    local dz = tipPose[3] - targetPose[3]
    local distance = poseError(tipPose, targetPose)

    return dx, dy, dz, distance
end

local function logTipTargetError(label)
    local dx, dy, dz, distance = tipTargetError()
    logMotionInfo(string.format('%s tip-target xyz error: %.6f %.6f %.6f, norm: %.6f',
                                label, dx, dy, dz, distance))
end

local function solveTargetConfig()
    simIK.syncFromSim(configIkEnv, {configIkGroup})

    local targetPose = sim.getObjectPose(ikTarget, ur10Base)
    simIK.setObjectPose(configIkEnv,
                        configSimToIkObjectMap[ikTarget],
                        targetPose,
                        configSimToIkObjectMap[ur10Base])

    local result, flags, precision = simIK.handleGroup(configIkEnv, configIkGroup)
    if precision then
        logMotionInfo(string.format('IK precision: linear=%.6f, angular=%.6f',
                                    precision[1] or -1,
                                    precision[2] or -1))
    end
    if result ~= simIK.result_success then
        sim.setInt32Signal('ur10JointMotionIkConfigCount', 0)
        logMotionWarning('IK handleGroup failed with result: ' .. tostring(result) ..
                         ', flags: ' .. tostring(flags))
        local fallbackConfig = findFeasibleIkConfig('config', configIkEnv, configIkGroup, configSimToIkObjectMap, joints, targetPose, ur10Base,
                                                    getCurrentConfig, setConfig, configIsFinite,
                                                    function(candidateConfig)
                                                        return collides({candidateConfig})
                                                    end,
                                                    logMotionWarning)
        if fallbackConfig then
            sim.setInt32Signal('ur10JointMotionIkConfigCount', 1)
            logMotionInfo('IK findConfigs fallback produced one feasible target config')
            return fallbackConfig
        end
        return nil
    end

    local targetConfig = {}
    for i = 1, #joints do
        targetConfig[i] = simIK.getJointPosition(configIkEnv,
                                                 configSimToIkObjectMap[joints[i]])
    end

    sim.setInt32Signal('ur10JointMotionIkConfigCount', 1)
    logMotionInfo('IK direct solve produced one target config')
    return targetConfig
end

local function moveToTargetByConfig()
    updateRobotCollection()
    logCurrentUr10CollisionState('config IK solve')
    setMotionStatus('config', 'solving_ik')
    local targetConfig = solveTargetConfig()
    if not targetConfig then
        setMotionStatus('config', 'ik_failed')
        logMotionWarning('IK failed for /ikTarget')
        return false, 'ik_failed'
    end
    if not configIsFinite(targetConfig) then
        setMotionStatus('config', 'ik_failed')
        logMotionWarning('invalid IK target config; refusing sim.moveToConfig')
        return false, 'ik_failed'
    end

    setMotionStatus('config', 'checking_collision')
    if not trajectoryIsCollisionFree(targetConfig) then
        setMotionStatus('config', 'collision_failed')
        logMotionWarning('no collision-free joint path to /ikTarget')
        return false, 'collision_failed'
    end

    setMotionStatus('config', 'moving')
    logCurrentUr10CollisionState('config move')
    logMotionInfo('executing sim.moveToConfig')
    sim.moveToConfig({
        joints = joints,
        targetPos = targetConfig,
        maxVel = configMaxVel,
        maxAccel = configMaxAccel,
        maxJerk = configMaxJerk,
    })

    logTipTargetError('after moveToConfig')
    applyFinalConfig(targetConfig)
    logTipTargetError('after final config apply')
    setMotionStatus('config', 'done')
    logMotionInfo('sim.moveToConfig finished')
    return true, 'done'
end

local function moveToPoseCallback(data)
    sim.setObjectPose(data.auxData.target, data.pose, data.auxData.base)
    simIK.setObjectPose(data.auxData.env,
                        data.auxData.ikTarget,
                        data.pose,
                        data.auxData.ikBase)
    simIK.handleGroup(data.auxData.env, data.auxData.group, {syncWorlds = true})
end

local function settleAtTargetPose(targetPose)
    local tolerance = 0.0005
    local maxIterations = 20

    sim.setObjectPose(ikTarget, targetPose, ur10Base)
    simIK.setObjectPose(linearIkEnv,
                        linearSimToIkObjectMap[ikTarget],
                        targetPose,
                        linearSimToIkObjectMap[ur10Base])

    for _ = 1, maxIterations, 1 do
        local result = simIK.handleGroup(linearIkEnv, linearIkGroup, {syncWorlds = true})
        if result ~= simIK.result_success then
            logMotionWarning('settle IK handleGroup failed with result: ' .. tostring(result))
            return false, 'ik_failed'
        end

        local _, _, _, distance = tipTargetError()
        if distance <= tolerance then
            return true, 'done'
        end
    end

    return false, 'tolerance_not_reached'
end

local function moveToTargetLinearly()
    updateRobotCollection()
    logCurrentUr10CollisionState('linear IK solve')
    simIK.syncFromSim(linearIkEnv, {linearIkGroup})

    local startPose = sim.getObjectPose(ikTip, ur10Base)
    local targetPose = sim.getObjectPose(ikTarget, ur10Base)

    setMotionStatus('linear', 'moving')
    logCurrentUr10CollisionState('linear move')
    sim.moveToPose({
        pose = startPose,
        targetPose = targetPose,
        maxVel = linearMaxVel,
        maxAccel = linearMaxAccel,
        maxJerk = linearMaxJerk,
        callback = moveToPoseCallback,
        auxData = linearIkData,
    })
    logTipTargetError('before settle')
    local settleOk, settleStatus = settleAtTargetPose(targetPose)
    if not settleOk then
        logMotionWarning('linear settle did not reach tight target tolerance: ' ..
                         tostring(settleStatus))
        logTipTargetError('after settle warning')
    else
        logTipTargetError('after settle')
    end
    setMotionStatus('linear', 'done')
    return true, 'done'
end

local function getPlannedConfig()
    local config = {}
    for i = 1, #plannedJoints do
        config[i] = sim.getJointPosition(plannedJoints[i])
    end
    return config
end

local function setPlannedConfig(config)
    for i = 1, #plannedJoints do
        sim.setJointPosition(plannedJoints[i], config[i])
    end
end

local function setPlannedTargetConfig(config)
    for i = 1, #plannedJoints do
        sim.setJointTargetPosition(plannedJoints[i], config[i])
    end
end

local function plannedConfigIsFinite(config)
    if type(config) ~= 'table' or #config ~= #plannedJoints then
        return false
    end

    for i = 1, #plannedJoints do
        local value = config[i]
        if type(value) ~= 'number' or value ~= value or
           value == math.huge or value == -math.huge then
            return false
        end
    end

    return true
end

local function plannedConfigMaxError(actualConfig, targetConfig)
    if type(actualConfig) ~= 'table' or type(targetConfig) ~= 'table' then
        return math.huge
    end

    local count = math.min(#actualConfig, #targetConfig)
    local maxError = 0
    for i = 1, count do
        maxError = math.max(maxError, math.abs(actualConfig[i] - targetConfig[i]))
    end
    if #actualConfig ~= #targetConfig then
        return math.huge
    end
    return maxError
end

local function updatePlannedRobotCollection()
    if plannedRobotCollection then
        sim.destroyCollection(plannedRobotCollection)
    end

    plannedRobotCollection = sim.createCollection()
    sim.addItemToCollection(plannedRobotCollection, sim.handle_tree, ur10Base, 0)
    plannedDebug('collision collection: ur10Base=' .. tostring(ur10Base) ..
                 ', plannedIkBase=' .. tostring(plannedIkBase))

    local tcpObject = sim.getInt32Signal(tcpObjectSignal) or -1
    if tcpObject >= 0 then
        sim.addItemToCollection(plannedRobotCollection, sim.handle_single, tcpObject, 0)
        logMotionInfo('including TCP object in planned collision collection: ' ..
                      tostring(tcpObject))
        plannedDebug('collision collection includes tcpObject=' .. tostring(tcpObject))
    end
end

local function plannedCollides(configs)
    local bufferedConfig = getPlannedConfig()

    for i = 1, #configs do
        if not plannedConfigIsFinite(configs[i]) then
            plannedDebug('invalid planned collision-check config: ' ..
                         fmtConfig(configs[i], 8))
            return true, nil, 'invalid'
        end

        setPlannedConfig(configs[i])
        local pair, pairKind = collectionCollisionPair(plannedRobotCollection)
        if pair then
            setPlannedConfig(bufferedConfig)
            plannedDebug('collision check rejected config=' .. fmtConfig(configs[i], 8) ..
                         ', pair=' .. fmtCollisionPair(pair) ..
                         ', pairKind=' .. tostring(pairKind))
            return true, pair, pairKind
        end
    end

    setPlannedConfig(bufferedConfig)
    return false, nil, nil
end

local function logCurrentPlannedCollisionState(label)
    local currentConfig = getPlannedConfig()
    local inCollision, pair, pairKind = plannedCollides({currentConfig})
    if inCollision then
        plannedDebug('current planned robot collision before ' .. label ..
                     ': pair=' .. fmtCollisionPair(pair) ..
                     ', pairKind=' .. tostring(pairKind) ..
                     ', config=' .. fmtConfig(currentConfig, #currentConfig))
        return true
    end

    plannedDebug('current planned robot collision-free before ' .. label ..
                 ': config=' .. fmtConfig(currentConfig, #currentConfig))
    return false
end

local function plannedStateValidity(task, label, config)
    local withinBounds = simOMPL.isStateWithinBounds(task, config)
    local stateValid = simOMPL.isStateValid(task, config)
    local ok = withinBounds and stateValid

    plannedDebug('OMPL preflight ' .. label ..
                 ': config=' .. fmtConfig(config, 8) ..
                 ', withinBounds=' .. tostring(withinBounds) ..
                 ', stateValid=' .. tostring(stateValid))

    return {
        ok = ok,
        withinBounds = withinBounds,
        stateValid = stateValid,
    }
end

local function findPlannedTargetConfig()
    setMotionStatus('planned', 'solving_ik')
    local bufferedConfig = getPlannedConfig()
    simIK.syncFromSim(plannedIkEnv, {plannedIkGroup})

    local targetPose = sim.getObjectPose(ikTarget, plannedIkBase)
    plannedDebug('all-joint IK targetInBase=' .. fmtPoseXYZ(targetPose) ..
                 ', current=' .. fmtConfig(bufferedConfig, 8))
    simIK.setObjectPose(plannedIkEnv,
                        plannedSimToIkObjectMap[ikTarget],
                        targetPose,
                        plannedSimToIkObjectMap[plannedIkBase])

    local result, flags, precision = simIK.handleGroup(plannedIkEnv, plannedIkGroup)
    if precision then
        plannedDebug('all-joint IK precision: linear=' ..
                     fmt(precision[1] or -1) ..
                     ', angular=' .. fmt(precision[2] or -1))
    end

    local plannedConfig = {}
    for i = 1, #plannedJoints do
        plannedConfig[i] = simIK.getJointPosition(plannedIkEnv,
                                                  plannedSimToIkObjectMap[plannedJoints[i]])
    end
    setPlannedConfig(bufferedConfig)

    if result ~= simIK.result_success then
        plannedDebug('all-joint IK failed: result=' .. tostring(result) ..
                     ', flags=' .. tostring(flags))
        local fallbackConfig = findFeasibleIkConfig('planned', plannedIkEnv, plannedIkGroup, plannedSimToIkObjectMap, plannedJoints, targetPose, plannedIkBase,
                                                    getPlannedConfig, setPlannedConfig, plannedConfigIsFinite,
                                                    function(candidateConfig)
                                                        return plannedCollides({candidateConfig})
                                                    end,
                                                    plannedDebug)
        if fallbackConfig then
            plannedDebug('all-joint findConfigs fallback accepted target config=' ..
                         fmtConfig(fallbackConfig, 8))
            return fallbackConfig, 'done'
        end
        setMotionStatus('planned', 'ik_failed')
        return nil, 'ik_failed'
    end

    if not plannedConfigIsFinite(plannedConfig) then
        setMotionStatus('planned', 'ik_failed')
        plannedDebug('all-joint IK produced invalid target config=' ..
                     fmtConfig(plannedConfig, 8))
        return nil, 'ik_failed'
    end

    plannedDebug('all-joint IK success: accepted target config=' ..
                 fmtConfig(plannedConfig, 8))
    return plannedConfig, 'done'
end

local function findPlannedPath(goalConfig)
    setMotionStatus('planned', 'planning')
    local startConfig = getPlannedConfig()
    local useForProjection = {}
    for i = 1, #plannedJoints do
        useForProjection[i] = (i <= 3 and 1 or 0)
    end
    plannedDebug('OMPL solve start: start=' .. fmtConfig(startConfig, 8) ..
                 ', goal=' .. fmtConfig(goalConfig, 8) ..
                 ', planningTime=' .. fmt(pathPlanningMaxTime))

    local task = simOMPL.createTask('external_api_planned_motion_task')
    simOMPL.setAlgorithm(task, plannedPathPlanningAlgo)
    simOMPL.setStateSpaceForJoints(task, plannedJoints, useForProjection)
    simOMPL.setCollisionPairs(task, {
        plannedRobotCollection, sim.handle_all,
        plannedRobotCollection, plannedRobotCollection,
    })
    simOMPL.setStartState(task, startConfig)
    simOMPL.setGoalState(task, goalConfig)
    simOMPL.setup(task)

    local startValidity = plannedStateValidity(task, 'start', startConfig)
    if not startValidity.ok then
        plannedDebug('OMPL preflight rejected start state; skipping solve')
        simOMPL.destroyTask(task)
        return nil, 'start_state_invalid'
    end

    local goalValidity = plannedStateValidity(task, 'goal', goalConfig)
    if not goalValidity.ok then
        plannedDebug('OMPL preflight rejected goal state; skipping solve')
        simOMPL.destroyTask(task)
        return nil, 'goal_state_invalid'
    end

    local path = nil
    if simOMPL.solve(task, pathPlanningMaxTime) and simOMPL.hasExactSolution(task) then
        simOMPL.simplifyPath(task, pathPlanningMaxSimplificationTime)
        path = simOMPL.getPath(task)
        plannedDebug('OMPL solve success: pathValues=' .. tostring(path and #path or 0) ..
                     ', dof=' .. tostring(#plannedJoints))
    else
        plannedDebug('OMPL solve failed: no exact solution')
    end

    simOMPL.destroyTask(task)
    return path, path and 'done' or 'planning_failed'
end

local function followPlannedPath(path)
    setMotionStatus('planned', 'moving')
    local minMaxVel = {}
    local minMaxAccel = {}

    for i = 1, #plannedJoints do
        minMaxVel[#minMaxVel + 1] = -plannedMaxVel[i]
        minMaxVel[#minMaxVel + 1] = plannedMaxVel[i]
        minMaxAccel[#minMaxAccel + 1] = -plannedMaxAccel[i]
        minMaxAccel[#minMaxAccel + 1] = plannedMaxAccel[i]
    end

    local pathLengths = sim.getPathLengths(path, #plannedJoints)
    if followPathScript == nil then
        followPathScript = -1
    end

    local pathPts, times
    pathPts, times, followPathScript = sim.generateTimeOptimalTrajectory(path,
                                                                         pathLengths,
                                                                         minMaxVel,
                                                                         minMaxAccel,
                                                                         1000,
                                                                         'not-a-knot',
                                                                         5,
                                                                         followPathScript)
    if not pathPts or not times or #times == 0 then
        setMotionStatus('planned', 'trajectory_failed')
        logMotionWarning('planned trajectory timing failed')
        plannedDebug('trajectory timing failed: pathValues=' .. tostring(path and #path or 0))
        return false, 'trajectory_failed'
    end
    plannedDebug('trajectory timing: pointValues=' .. tostring(#pathPts) ..
                 ', timeCount=' .. tostring(#times) ..
                 ', duration=' .. fmt(times[#times]))

    local startTime = sim.getSimulationTime()
    local elapsed = 0
    local rawStepTime = sim.getSimulationTimeStep() or 0
    local stepTime = math.max(rawStepTime, 0.01)
    local nextProgressLogTime = 0
    local stalledIterations = 0
    logCurrentPlannedCollisionState('planned path playback')
    plannedDebug('path playback start: duration=' .. fmt(times[#times]) ..
                 ', rawStepTime=' .. fmt(rawStepTime) ..
                 ', waitStep=' .. fmt(stepTime))

    while elapsed < times[#times] do
        if sim.getSimulationState() == sim.simulation_advancing_abouttostop then
            setMotionStatus('planned', 'stopped')
            return false, 'stopped'
        end

        setPlannedTargetConfig(sim.getPathInterpolatedConfig(pathPts, times, elapsed))
        sim.wait(stepTime)
        local nextElapsed = sim.getSimulationTime() - startTime
        if nextElapsed <= elapsed + 1.0e-9 then
            stalledIterations = stalledIterations + 1
        else
            stalledIterations = 0
        end
        elapsed = nextElapsed

        if elapsed >= nextProgressLogTime then
            plannedDebug('path playback progress: elapsed=' .. fmt(elapsed) ..
                         '/' .. fmt(times[#times]) ..
                         ', actual=' .. fmtConfig(getPlannedConfig(), 8))
            nextProgressLogTime = nextProgressLogTime + 0.5
        end

        if stalledIterations >= plannedPlaybackMaxStallIterations then
            setMotionStatus('planned', 'trajectory_failed')
            plannedDebug('path playback stalled: simulation time did not advance, elapsed=' ..
                         fmt(elapsed) .. ', waitStep=' .. fmt(stepTime) ..
                         ', stallIterations=' .. tostring(stalledIterations))
            return false, 'playback_stalled'
        end
    end

    local finalTargetConfig = sim.getPathInterpolatedConfig(pathPts, times, times[#times])
    setPlannedTargetConfig(finalTargetConfig)
    local finalActualConfig = getPlannedConfig()
    plannedDebug('path playback final target: ' .. fmtConfig(finalTargetConfig, 8))
    plannedDebug('path playback final actual: ' .. fmtConfig(finalActualConfig, 8) ..
                 ', maxError=' .. fmt(plannedConfigMaxError(finalActualConfig, finalTargetConfig)))
    setMotionStatus('planned', 'done')
    return true, 'done'
end

local function moveToTargetWithPlanner()
    plannedDebug('planned request start: targetWorld=' ..
                 fmtPoseXYZ(sim.getObjectPose(ikTarget)) ..
                 ', currentPlannedConfig=' .. fmtConfig(getPlannedConfig(), 8) ..
                 ', tcpObject=' .. tostring(sim.getInt32Signal(tcpObjectSignal) or -1))
    updatePlannedRobotCollection()
    logCurrentPlannedCollisionState('planned IK solve')

    local goalConfig, goalStatus = findPlannedTargetConfig()
    if not goalConfig then
        return false, goalStatus
    end

    local path, pathStatus = findPlannedPath(goalConfig)
    if path then
        return followPlannedPath(path)
    end

    setMotionStatus('planned', pathStatus)
    if pathStatus == 'start_state_invalid' then
        logMotionWarning('OMPL start state is invalid; see planned preflight debug log')
    elseif pathStatus == 'goal_state_invalid' then
        logMotionWarning('OMPL goal state is invalid; see planned preflight debug log')
    else
        logMotionWarning('OMPL found no exact collision-free path')
    end
    return false, pathStatus
end

local function dispatchAndWait(target, mode, tcpObject, timeout)
    local pose = setIkTargetPose(target)
    local tcpHandle = setTcpObject(tcpObject)
    local statusSignal = motionStatusSignals[mode]
    if statusSignal then
        sim.setStringSignal(statusSignal, 'requested')
    end
    ur10MotionCommandCount = (ur10MotionCommandCount or 0) + 1
    sim.setInt32Signal('ur10JointMotionCommandCount', ur10MotionCommandCount)

    lastCommand = {
        mode = mode,
        pose = pose,
        tcpObject = tcpHandle,
    }

    local ok, status
    if mode == 'config' then
        ok, status = moveToTargetByConfig()
    elseif mode == 'linear' then
        ok, status = moveToTargetLinearly()
    elseif mode == 'planned' then
        ok, status = moveToTargetWithPlanner()
    else
        ok = false
        status = 'unknown_mode'
    end

    return {
        ok = ok,
        accepted = true,
        completed = ok,
        mode = mode,
        targetPose = pose,
        tcpObject = tcpHandle,
        motionStatus = status,
        reason = ok and nil or ('motion failed with status: ' .. tostring(status)),
        state = stateSnapshot(),
    }
end

local function dispatchAndWaitWithPlannedFallback(target, normalMode, normalTcpObject, plannedTcpObject, normalTimeout, plannedTimeout, label)
    local normal = dispatchAndWait(target, normalMode, normalTcpObject, normalTimeout)
    if normal.completed then
        return normal
    end

    logMotionWarning('normal ' .. label .. ' failed with ' ..
                     tostring(normal.motionStatus) ..
                     '; trying planned fallback')

    local planned = dispatchAndWait(target, 'planned', plannedTcpObject, plannedTimeout or 120)
    if planned.completed then
        local plannedMotion = resultSummary(planned)
        planned.fallbackUsed = true
        planned.fallbackLabel = label
        planned.normalMotion = resultSummary(normal)
        planned.plannedMotion = plannedMotion
        logMotionInfo('planned fallback succeeded for ' .. label)
        return planned
    end

    return {
        ok = false,
        accepted = true,
        completed = false,
        mode = 'planned',
        targetPose = planned.targetPose or normal.targetPose,
        tcpObject = planned.tcpObject,
        motionStatus = planned.motionStatus,
        reason = 'normal ' .. label .. ' failed with status: ' ..
                 tostring(normal.motionStatus) ..
                 '; planned fallback failed with status: ' ..
                 tostring(planned.motionStatus),
        fallbackUsed = true,
        fallbackLabel = label,
        normalMotion = resultSummary(normal),
        plannedMotion = resultSummary(planned),
        state = stateSnapshot(),
    }
end

local function failMotionAndStop(statusSignal, result)
    result.ok = false
    sim.setStringSignal(statusSignal, 'failed')
    sim.addLog(sim.verbosity_warnings,
               '[external_api] ' .. tostring(result.reason) ..
               '; stopping simulation')
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

local function initUr10Motion()
    joints = sim.getObjectsInTree(ur10Base, sim.sceneobject_joint)
    logMotionInfo('UR10 joint count: ' .. tostring(#joints))

    configMaxVel = makeJointVector(math.pi / 3)
    configMaxAccel = makeJointVector(math.pi / 3)
    configMaxJerk = makeJointVector(math.pi / 3)
    collisionCheckResolution = 5 * math.pi / 180
    ikLinearPrecision = 0.0005
    ikAngularPrecision = 0.005

    updateRobotCollection()

    configIkEnv = simIK.createEnvironment()
    configIkGroup = simIK.createGroup(configIkEnv)
    simIK.setGroupCalculation(configIkEnv,
                              configIkGroup,
                              simIK.method_damped_least_squares,
                              0.3,
                              99)

    configIkElement, configSimToIkObjectMap =
        simIK.addElementFromScene(configIkEnv,
                                  configIkGroup,
                                  ur10Base,
                                  ikTip,
                                  ikTarget,
                                  simIK.constraint_pose)
    simIK.setElementPrecision(configIkEnv,
                              configIkGroup,
                              configIkElement,
                              {ikLinearPrecision, ikAngularPrecision})

    for i = 1, #joints do
        local ikJoint = configSimToIkObjectMap[joints[i]]
        if not ikJoint then
            error('UR10 joint was not mapped into config IK environment: ' ..
                  tostring(joints[i]))
        end
    end

    linearMaxVel = {0.4, 0.4, 0.4, 1.8}
    linearMaxAccel = {0.8, 0.8, 0.8, 0.9}
    linearMaxJerk = {0.6, 0.6, 0.6, 0.8}

    linearIkEnv = simIK.createEnvironment()
    linearIkGroup = simIK.createGroup(linearIkEnv)
    simIK.setGroupCalculation(linearIkEnv,
                              linearIkGroup,
                              simIK.method_damped_least_squares,
                              0.3,
                              99)
    local _
    _, linearSimToIkObjectMap =
        simIK.addElementFromScene(linearIkEnv,
                                  linearIkGroup,
                                  ur10Base,
                                  ikTip,
                                  ikTarget,
                                  simIK.constraint_pose)

    linearIkData = {
        env = linearIkEnv,
        group = linearIkGroup,
        target = ikTarget,
        base = ur10Base,
        ikTarget = linearSimToIkObjectMap[ikTarget],
        ikBase = linearSimToIkObjectMap[ur10Base],
    }

    sim.setInt32Signal('ur10JointMotionIkConfigCount', 0)
    sim.setInt32Signal('ur10JointMotionCommandCount', 0)
    setMotionStatus('config', 'ready')
    setMotionStatus('linear', 'ready')
end

local function cleanupUr10Motion()
    if configIkEnv then
        simIK.eraseEnvironment(configIkEnv)
        configIkEnv = nil
    end
    if linearIkEnv then
        simIK.eraseEnvironment(linearIkEnv)
        linearIkEnv = nil
    end
    if robotCollection then
        sim.destroyCollection(robotCollection)
        robotCollection = nil
    end
    setMotionStatus('config', 'stopped')
    setMotionStatus('linear', 'stopped')
end

local function initPlannedMotion()
    plannedIkBase = firstObject({
        '/mobile_arm',
        ':/mobile_arm',
    }, 'mobile arm base')

    plannedJoints = {railJoint}
    for i = 1, #joints do
        plannedJoints[#plannedJoints + 1] = joints[i]
    end

    updatePlannedRobotCollection()

    plannedIkEnv = simIK.createEnvironment()
    plannedIkGroup = simIK.createGroup(plannedIkEnv)
    simIK.setGroupCalculation(plannedIkEnv,
                              plannedIkGroup,
                              simIK.method_damped_least_squares,
                              0.3,
                              99)
    local _
    _, plannedSimToIkObjectMap =
        simIK.addElementFromScene(plannedIkEnv,
                                  plannedIkGroup,
                                  plannedIkBase,
                                  ikTip,
                                  ikTarget,
                                  simIK.constraint_pose)

    for i = 1, #plannedJoints do
        local ikJoint = plannedSimToIkObjectMap[plannedJoints[i]]
        if not ikJoint then
            error('Planned motion joint was not mapped into the IK environment: ' ..
                  tostring(plannedJoints[i]))
        end
    end

    plannedPathPlanningAlgo = simOMPL.Algorithm.RRTConnect
    plannedMaxVel = {railMaxVel}
    plannedMaxAccel = {railMaxAccel}
    plannedMaxJerk = {railMaxJerk}
    for i = 2, #plannedJoints do
        plannedMaxVel[i] = math.pi
        plannedMaxAccel[i] = 40 * math.pi / 180
        plannedMaxJerk[i] = 40 * math.pi / 180
    end

    setMotionStatus('planned', 'ready')
end

local function cleanupPlannedMotion()
    if plannedIkEnv then
        simIK.eraseEnvironment(plannedIkEnv)
        plannedIkEnv = nil
    end
    if plannedRobotCollection then
        sim.destroyCollection(plannedRobotCollection)
        plannedRobotCollection = nil
    end
    followPathScript = nil
    setMotionStatus('planned', 'stopped')
end

local function runMoveCommand(command)
    return dispatchAndWait(command.target, command.mode, command.tcpObject, command.timeout)
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
    sim.setInt32Signal(tcpObjectSignal, -1)
    initUr10Motion()
    initPlannedMotion()

    lastCommand = nil
    pendingMoveCommand = nil
    pendingPickCommand = nil
    pendingPlaceCommand = nil
    moveBusy = false
    pickBusy = false
    placeBusy = false
    lastMoveResult = nil
    lastPickResult = nil
    lastPlaceResult = nil
    attachedObjectHandle = -1
    sim.setInt32Signal('externalApiReady', 1)
    sim.setStringSignal(pickStatusSignal, 'ready')
    sim.setStringSignal(placeStatusSignal, 'ready')
end

function moveTo(targetOrData, mode, tcpObject)
    if moveBusy or pickBusy or placeBusy or
       pendingMoveCommand or pendingPickCommand or pendingPlaceCommand then
        return {
            ok = false,
            reason = 'external_api is already busy',
            state = stateSnapshot(),
        }
    end

    local data = commandData(targetOrData, mode, tcpObject)
    local requestedMode = data.mode or 'config'

    if requestedMode == 'config' then
        return dispatchAndWait(data.target, 'config', data.tcpObject, 60)
    end
    if requestedMode == 'plan' or requestedMode == 'planned' then
        pendingMoveCommand = {
            target = data.target,
            mode = 'planned',
            tcpObject = data.tcpObject,
            timeout = 120,
        }
        lastCommand = {
            mode = 'planned',
            target = data.target,
            tcpObject = data.tcpObject,
        }
        setMotionStatus('planned', 'queued')
        return {
            ok = true,
            accepted = true,
            mode = 'planned',
            state = stateSnapshot(),
        }
    end

    return {
        ok = false,
        reason = 'unknown moveTo mode: ' .. tostring(requestedMode),
        state = stateSnapshot(),
    }
end

function linearMoveTo(targetOrData, tcpObject)
    local data = commandData(targetOrData, 'linear', tcpObject)
    return dispatchAndWait(data.target, 'linear', data.tcpObject, 60)
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
            state = stateSnapshot(),
        }))
    end

    sim.setStringSignal(pickStatusSignal, 'preorienting_joint1')
    moveFirstArmJointTo(prePickFirstJointAngle, 'pre-pick')

    sim.setStringSignal(pickStatusSignal, 'approaching_config')
    local approach = dispatchAndWaitWithPlannedFallback(prePickDummy, 'config', nil, nil, 60, 120, 'pre-pick config move')
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
            state = stateSnapshot(),
        }))
    end
    motionDelay(delayAfterApproach)

    sim.setStringSignal(pickStatusSignal, 'descending')
    local descend = dispatchAndWaitWithPlannedFallback(pickDummy, 'linear', nil, nil, 30, 120, 'pick linear descent')
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
            state = stateSnapshot(),
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
    local lift = dispatchAndWaitWithPlannedFallback(holdDummy, 'linear', nil, objectHandle, 30, 120, 'hold lift')
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
            state = stateSnapshot(),
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
        state = stateSnapshot(),
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
            state = stateSnapshot(),
        }))
    end

    removePlaceDummies()
    prePlaceDummy = createDummyAtPose('prePlaceDummy', prePlacePose)
    placeDummy = createDummyAtPose('placeDummy', placePose)

    sim.setStringSignal(placeStatusSignal, 'preorienting_joint1')
    moveFirstArmJointTo(prePlaceFirstJointAngle, 'pre-place', true)

    sim.setStringSignal(placeStatusSignal, 'approaching_config')
    local approach = dispatchAndWaitWithPlannedFallback(prePlaceDummy, 'config', objectHandle, objectHandle, 60, 120, 'pre-place config move')
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
            state = stateSnapshot(),
        }))
    end
    motionDelay(delayAfterApproach)

    sim.setStringSignal(placeStatusSignal, 'descending')
    local descend = dispatchAndWaitWithPlannedFallback(placeDummy, 'linear', objectHandle, objectHandle, 30, 120, 'place linear descent')
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
            state = stateSnapshot(),
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
    local retreat = dispatchAndWaitWithPlannedFallback(prePlaceDummy, 'linear', nil, nil, 30, 120, 'place retreat')
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
            state = stateSnapshot(),
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
        state = stateSnapshot(),
    })
end

function pick(objectAliasOrData)
    if moveBusy or pickBusy or placeBusy or
       pendingMoveCommand or pendingPickCommand or pendingPlaceCommand then
        return {
            ok = false,
            reason = 'external_api is already busy',
            state = stateSnapshot(),
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
        state = stateSnapshot(),
    }
end

function place(positionOrData)
    if moveBusy or pickBusy or placeBusy or
       pendingMoveCommand or pendingPickCommand or pendingPlaceCommand then
        return {
            ok = false,
            reason = 'external_api is already busy',
            state = stateSnapshot(),
        }
    end

    local position = placePositionFromData(positionOrData)
    local objectHandle = attachedObject()
    if objectHandle < 0 then
        return {
            ok = false,
            reason = 'no attached object is known',
            state = stateSnapshot(),
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
        state = stateSnapshot(),
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
        state = stateSnapshot(),
    }
end

function getState(inData)
    local state = stateSnapshot()
    state.lastMoveResult = resultSummary(lastMoveResult)
    state.lastPickResult = resultSummary(lastPickResult)
    state.lastPlaceResult = resultSummary(lastPlaceResult)
    return state
end

function sysCall_thread()
    while sim.getSimulationState() ~= sim.simulation_advancing_abouttostop do
        if pendingMoveCommand and not moveBusy then
            local command = pendingMoveCommand
            pendingMoveCommand = nil
            moveBusy = true
            setMotionStatus(command.mode, 'running')

            lastMoveResult = runMoveCommand(command)

            moveBusy = false
            setMotionStatus(command.mode,
                            lastMoveResult.ok and 'done' or 'failed')
        elseif pendingPickCommand and not pickBusy then
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
    cleanupPlannedMotion()
    cleanupUr10Motion()
    sim.setInt32Signal('externalApiReady', 0)
    sim.setInt32Signal(tcpObjectSignal, -1)
    sim.setStringSignal(pickStatusSignal, 'stopped')
    sim.setStringSignal(placeStatusSignal, 'stopped')
end
