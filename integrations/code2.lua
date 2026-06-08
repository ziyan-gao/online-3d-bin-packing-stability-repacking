--[[ Control the arm tip motion as follows

sim.setStringSignal('armCommand', sim.packTable({
    move = true,
    linear = true,
}))

sim.setStringSignal('armCommand', sim.packTable({
    move = true,
    plannedCartesian = true,
}))

--]]

sim=require'sim'
simIK=require'simIK'

function getObjectIfExists(alias)
    local handle = sim.getObject(alias, {noError = true})
    if handle and handle >= 0 then
        return handle
    end
    return nil
end

function createCollisionCollections()
    cartesianPlanner.collisionCollectionsReady = false
    robotCollection = nil
    obstacleCollection = nil

    local robotRoot = getObjectIfExists(cartesianPlanner.robotRootAlias)
    if not robotRoot then
        return false, 'missing robot root: ' .. cartesianPlanner.robotRootAlias
    end

    local newRobotCollection = sim.createCollection()
    sim.addItemToCollection(newRobotCollection, sim.handle_tree, robotRoot, 0)

    local newObstacleCollection = sim.createCollection()

    local obstacleCount = 0
    for i = 1, #cartesianPlanner.obstacleRootAliases, 1 do
        local obstacleRoot = getObjectIfExists(cartesianPlanner.obstacleRootAliases[i])
        if obstacleRoot then
            sim.addItemToCollection(newObstacleCollection, sim.handle_tree, obstacleRoot, 0)
            obstacleCount = obstacleCount + 1
        end
    end

    local cuboidIndex = 0
    while true do
        local cuboid = sim.getObject(cartesianPlanner.cuboidAliasPrefix .. '*', {index = cuboidIndex, noError = true})
        if not cuboid or cuboid < 0 then
            break
        end
        sim.addItemToCollection(newObstacleCollection, sim.handle_single, cuboid, 0)
        obstacleCount = obstacleCount + 1
        cuboidIndex = cuboidIndex + 1
    end

    if obstacleCount == 0 then
        return false, 'no obstacles found for collision checking'
    end

    robotCollection = newRobotCollection
    obstacleCollection = newObstacleCollection
    cartesianPlanner.collisionCollectionsReady = true

    return true, nil
end

function sysCall_init()
    sim.setStepping(true)
    sim.clearStringSignal('armCommand')

    -- Take a few handles from the scene:
    simBase=sim.getObject('..')
    simTip=sim.getObject(':/ikTip')
    simTarget=sim.getObject('../ikTarget')
    simJoints = sim.getObjectsInTree(simBase, sim.sceneobject_joint)
    arm_target_handle = sim.getObject('/arm_target')
    ikEnv=simIK.createEnvironment()
    ikGroup_damped=simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv,ikGroup_damped,simIK.method_undamped_pseudo_inverse,0.01,99)
-----------------------------------------------------------------------------
--Add ik element from the scene. Check the api: simIK.addElementFromScene
--ADD your code here
-----------------------------------------------------------------------------
    --ikConstraint = simIK.constraint_position + simIK.constraint_alpha_beta
    ikElement, simToIkObjectMap, ikToSimObjectMap = simIK.addElementFromScene(ikEnv,ikGroup_damped,simBase,simTip,simTarget,simIK.constraint_pose)
    --ikElement, simToIkObjectMap, ikToSimObjectMap = simIK.addElementFromScene(ikEnv,ikGroup_damped,simBase,simTip,simTarget,ikConstraint)
    ikJoints = simIK.getGroupJoints(ikEnv, ikGroup_damped)

    -- Execute the movement here:
    ik_data={}
    ik_data.ikEnv=ikEnv
    ik_data.ikGroup=ikGroup_damped
    ik_data.tip=simTip
    ik_data.target=simTarget
    ik_data.base = simBase
    ik_data.joints=simJoints
    
    maxVel={math.pi/3,math.pi/3,math.pi/3,math.pi/3,math.pi/3,math.pi/3}
    maxAccel={math.pi/3,math.pi/3,math.pi/3,math.pi/3,math.pi/3,math.pi/3}
    maxJerk= {math.pi/3,math.pi/3,math.pi/3,math.pi/3,math.pi/3,math.pi/3}
    maxIkVel={0.5,0.5,0.5,0.5} -- vx,vy,vz in m/s, Vtheta is rad/s
    maxIkAccel={5,5,5,1} -- ax,ay,az in m/s^2, Atheta is rad/s^2
    maxIkJerk={5,5,5,1} -- is ignored (i.e. infinite) with RML type 2
    cartesianPlanner = {}
    cartesianPlanner.stepSize = 0.05 -- meters between interpolated Cartesian waypoints
    cartesianPlanner.minSteps = 2
    cartesianPlanner.maxSteps = 200
    cartesianPlanner.jointSegmentStep = math.rad(5)
    cartesianPlanner.maxJointSegmentSamples = 100
    cartesianPlanner.robotRootAlias = '/mobile_arm'
    cartesianPlanner.obstacleRootAliases = {'/ZeroPressureBelt', '/Pallet'}
    cartesianPlanner.cuboidAliasPrefix = '/Cuboid'
    cartesianPlanner.collisionCollectionsReady = false

    local collectionsOk, collectionsError = createCollisionCollections()
    if not collectionsOk then
        sim.addLog(sim.verbosity_warnings, 'Cartesian planner collision setup failed: ' .. collectionsError)
    end

end


function normalizeAngle(angle)
    while angle > math.pi do
        angle = angle - 2 * math.pi
    end
    while angle < -math.pi do
        angle = angle + 2 * math.pi
    end
    return angle
end

function interpolateAngle(startAngle, targetAngle, t)
    return startAngle + normalizeAngle(targetAngle - startAngle) * t
end

function poseToTaskPose4d(pose)
    local matrix = sim.poseToMatrix(pose)
    local euler = sim.getEulerAnglesFromMatrix(matrix)
    return {
        x = pose[1],
        y = pose[2],
        z = pose[3],
        roll = euler[1],
        pitch = euler[2],
        yaw = euler[3],
    }
end

function taskPose4dToPose(taskPose, fixedRoll, fixedPitch, baseHandle)
    return sim.buildPose({taskPose.x, taskPose.y, taskPose.z}, {fixedRoll, fixedPitch, taskPose.yaw})
end

function interpolateTaskPose4d(startTaskPose, targetTaskPose, fixedRoll, fixedPitch, t)
    return {
        x = startTaskPose.x + (targetTaskPose.x - startTaskPose.x) * t,
        y = startTaskPose.y + (targetTaskPose.y - startTaskPose.y) * t,
        z = startTaskPose.z + (targetTaskPose.z - startTaskPose.z) * t,
        roll = fixedRoll,
        pitch = fixedPitch,
        yaw = interpolateAngle(startTaskPose.yaw, targetTaskPose.yaw, t),
    }
end

function isPositiveFiniteNumber(value)
    return type(value) == 'number' and value > 0 and value == value and value < math.huge
end

function getJointPositions(joints)
    local positions = {}
    for i = 1, #joints, 1 do
        positions[i] = sim.getJointPosition(joints[i])
    end
    return positions
end

function setJointPositions(joints, positions)
    for i = 1, #joints, 1 do
        sim.setJointPosition(joints[i], positions[i])
    end
end

function syncIkEnvironmentFromSim()
    if simIK.syncFromSim then
        simIK.syncFromSim(ikEnv, {ikGroup_damped})
        return
    end

    for i = 1, #simJoints, 1 do
        local joint = simJoints[i]
        simIK.setJointPosition(ikEnv,
                               simToIkObjectMap[joint],
                               sim.getJointPosition(joint))
    end
end

function solveIkForPose(targetPose)
    syncIkEnvironmentFromSim()

    simIK.setObjectPose(ikEnv,
                        simToIkObjectMap[simTarget],
                        targetPose,
                        simToIkObjectMap[simBase])

    local result = simIK.handleGroup(ikEnv, ikGroup_damped)
    if result ~= simIK.result_success then
        return false, nil, 'IK failed'
    end

    local targetAngles = {}
    for i = 1, #simJoints, 1 do
        local joint = simJoints[i]
        targetAngles[i] = simIK.getJointPosition(ikEnv, simToIkObjectMap[joint])
    end

    return true, targetAngles, nil
end

function isConfigCollisionFree(targetAngles)
    if not cartesianPlanner.collisionCollectionsReady or not robotCollection or not obstacleCollection then
        return false, 'collision collections are not initialized'
    end

    local originalAngles = getJointPositions(simJoints)
    local ok, collisionResult = pcall(function()
        setJointPositions(simJoints, targetAngles)
        return sim.checkCollision(robotCollection, obstacleCollection)
    end)
    setJointPositions(simJoints, originalAngles)

    if not ok then
        return false, 'collision check failed: ' .. tostring(collisionResult)
    end

    if collisionResult ~= 0 then
        return false, 'candidate configuration is in collision'
    end

    return true, nil
end

function interpolateJointConfig(startAngles, targetAngles, t)
    local result = {}
    for i = 1, #startAngles, 1 do
        result[i] = startAngles[i] + (targetAngles[i] - startAngles[i]) * t
    end
    return result
end

function computeJointSegmentSampleCount(startAngles, targetAngles)
    local maxDelta = 0
    for i = 1, #startAngles, 1 do
        local delta = math.abs(targetAngles[i] - startAngles[i])
        if delta > maxDelta then
            maxDelta = delta
        end
    end
    local sampleStep = cartesianPlanner.jointSegmentStep or math.rad(5)
    local samples = math.ceil(maxDelta / sampleStep)
    if samples < 1 then
        samples = 1
    end
    if samples > cartesianPlanner.maxJointSegmentSamples then
        return nil, 'joint segment sample count exceeds limit'
    end
    return samples, nil
end

function validateJointSegmentCollisionFree(startAngles, targetAngles)
    local samples, sampleError = computeJointSegmentSampleCount(startAngles, targetAngles)
    if not samples then
        return false, sampleError
    end
    for sampleIndex = 1, samples, 1 do
        local t = sampleIndex / samples
        local sampleAngles = interpolateJointConfig(startAngles, targetAngles, t)
        local ok, err = isConfigCollisionFree(sampleAngles)
        if not ok then
            return false, 'joint segment sample ' .. sampleIndex .. ': ' .. err
        end
    end
    return true, nil
end

function computeWaypointStepCount(startTaskPose, targetTaskPose)
    local dx = targetTaskPose.x - startTaskPose.x
    local dy = targetTaskPose.y - startTaskPose.y
    local dz = targetTaskPose.z - startTaskPose.z
    local distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    local steps = math.ceil(distance / cartesianPlanner.stepSize)
    local minSteps = math.ceil(cartesianPlanner.minSteps)
    if steps < minSteps then
        steps = minSteps
    end
    if steps ~= steps or steps < 1 then
        steps = 1
    end
    return steps
end

function generateCartesianWaypoints()
    local originalAngles = getJointPositions(simJoints)
    local function finish(ok, waypoints, configs, errorMessage)
        setJointPositions(simJoints, originalAngles)
        syncIkEnvironmentFromSim()
        return ok, waypoints, configs, errorMessage
    end

    if not isPositiveFiniteNumber(cartesianPlanner.stepSize) or
       not isPositiveFiniteNumber(cartesianPlanner.minSteps) or
       not isPositiveFiniteNumber(cartesianPlanner.maxSteps) or
       not isPositiveFiniteNumber(cartesianPlanner.jointSegmentStep) or
       not isPositiveFiniteNumber(cartesianPlanner.maxJointSegmentSamples) then
        return finish(false, nil, nil, 'invalid Cartesian planner step configuration')
    end

    local startPose = sim.getObjectPose(ik_data.tip, ik_data.base)
    local targetPose = sim.getObjectPose(arm_target_handle, ik_data.base)
    local startTaskPose = poseToTaskPose4d(startPose)
    local targetTaskPose = poseToTaskPose4d(targetPose)
    local fixedRoll = startTaskPose.roll
    local fixedPitch = startTaskPose.pitch
    local steps = computeWaypointStepCount(startTaskPose, targetTaskPose)
    if steps > cartesianPlanner.maxSteps then
        return finish(false, nil, nil, 'Cartesian waypoint count exceeds limit')
    end

    local waypoints = {}
    local configs = {}
    local previousAngles = {}
    for i = 1, #originalAngles, 1 do
        previousAngles[i] = originalAngles[i]
    end

    for i = 1, steps, 1 do
        local t = i / steps
        local taskPose = interpolateTaskPose4d(startTaskPose, targetTaskPose, fixedRoll, fixedPitch, t)
        local pose = taskPose4dToPose(taskPose, fixedRoll, fixedPitch, ik_data.base)
        local ikOk, targetAngles, ikError = solveIkForPose(pose)
        if not ikOk then
            return finish(false, nil, nil, 'waypoint ' .. i .. ': ' .. ikError)
        end

        local segmentOk, segmentError = validateJointSegmentCollisionFree(previousAngles, targetAngles)
        if not segmentOk then
            return finish(false, nil, nil, 'waypoint ' .. i .. ': ' .. segmentError)
        end

        waypoints[i] = taskPose
        configs[i] = targetAngles
        setJointPositions(simJoints, targetAngles)
        previousAngles = targetAngles
    end

    return finish(true, waypoints, configs, nil)
end

function executeCartesianWaypointConfigs(configs)
    for i = 1, #configs, 1 do
        local params = {
            joints = simJoints,
            targetPos = configs[i],
            maxVel = maxVel,
            maxAccel = maxAccel,
            maxJerk = maxJerk,
        }
        sim.moveToConfig(params)
    end
    syncIkEnvironmentFromSim()
end

function moveToPlannedCartesian()
    local originalAngles = getJointPositions(simJoints)
    local ok, waypoints, configs, err = generateCartesianWaypoints()
    setJointPositions(simJoints, originalAngles)

    if not ok then
        sim.addLog(sim.verbosity_warnings, 'Cartesian waypoint planning failed: ' .. err)
        return false
    end

    executeCartesianWaypointConfigs(configs)
    sim.addLog(sim.verbosity_scriptinfos, 'Executed ' .. #waypoints .. ' planned Cartesian waypoints')
    return true
end

function moveToPose(targetPose)
-----------------------------------------------------------------------------
--SP I:Using a threaded script to move the robot end-effector 
--through three target poses based on the simIK plugin and sim.moveToConfig(). 
-----------------------------------------------------------------------------
--1. Call simIK.setObjectPose() to specify the target pose in ik world
--2. Call simIK.handleGroup() to solve ik
    simIK.setObjectPose(ikEnv, 
                        simToIkObjectMap[simTarget], 
                        targetPose, 
                        simToIkObjectMap[simBase])
    simIK.handleGroup(ikEnv, ikGroup_damped)
    targetAngles = {}
    for i=1, #simJoints, 1 do
        tmp = simJoints[i]
        targetAngle = simIK.getJointPosition(ikEnv, simToIkObjectMap[tmp])
        table.insert(targetAngles, targetAngle)
    end
    local params = {
        joints = simJoints,
        targetPos = targetAngles,
        maxVel = maxVel,
        maxAccel = maxAccel,
        maxJerk = maxJerk,
    }
    sim.moveToConfig(params)
end

function moveTo(mode)
    pose = sim.getObjectPose(arm_target_handle, ik_data.base)
    if mode == 'linear' then
        return moveToPose_viaIK(pose)
    else
        return moveToPose(pose)
    end
end


function moveToPoseCallback(data)
    --easier way
    
    sim.setObjectPose(data.auxData.target, data.pose, data.auxData.base)
    simIK.handleGroup(data.auxData.ikEnv, data.auxData.ikGroup, {syncWorlds = true, callback = jacobianCallback})

end

function moveToPose_viaIK(targetPose)
    local params = {
        pose = sim.getObjectPose(ik_data.tip, ik_data.base),--current pose
        targetPose = targetPose,
        maxVel = maxIkVel,
        maxAccel = maxIkAccel,
        maxJerk = maxIkJerk,
        callback = moveToPoseCallback,
        auxData = ik_data
    }
    sim.moveToPose(params)
end


function sysCall_thread()
    while sim.getSimulationState() ~= sim.simulation_advancing_abouttostop do
        --currentTargetPose = sim.getObjectPose(wp1, simBase)
        local packed = sim.getStringSignal('armCommand')
        if packed then
            sim.clearStringSignal('armCommand')

            local ok, cmd = pcall(sim.unpackTable, packed)

            if ok and cmd ~= nil and cmd.move then
                if cmd.plannedCartesian then
                    moveToPlannedCartesian()
                elseif cmd.linear then
                    moveTo('linear')
                else
                    moveTo('config')
                end
            end
        else
            sim.step()
        end
    end
end

function sysCall_cleanup()
    sim.clearStringSignal('armCommand')
    simIK.eraseEnvironment(ikEnv)
end
