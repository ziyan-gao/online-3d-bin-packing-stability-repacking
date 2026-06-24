sim = require('sim')
simIK = require('simIK')
simOMPL = require('simOMPL')

--[[
Legacy rail + UR10 collision-free path motion with simplified IK target solve.

How to use:
  Legacy standalone script. It stays disabled because external_api.lua owns
  planned rail+UR10 motion in the two-thread architecture. To use the old
  standalone version instead, set `collisionFreeTrajectoryDisabled` to `false`,
  then:
  1. Put a dummy named `/ikTarget` in the world.
  2. Start simulation.
  3. Trigger this script with:
       sim.setStringSignal('ur10MotionMode', 'planned')

What this script does:
  - It watches the string signal `ur10MotionMode`.
  - When the signal value is `planned`, it clears the signal and starts one
    planning request.
  - The planning state is seven-dimensional:
        rail joint + six UR10 joints
  - It solves one IK configuration where `ikTip` matches `/ikTarget`, using
    direct `simIK.handleGroup` instead of searching for many candidates.
  - It visualizes that target robot configuration as a passive red copy.
  - It asks OMPL for a collision-free joint-space path from the current
    7-DOF configuration to that target.
  - It follows the path with a time-optimal trajectory.

Important:
  This is not linear Cartesian motion.  The planner searches in joint space,
  with the rail treated as one extra joint.
]]

local modeSignal = 'ur10MotionMode'
local myMode = 'planned'
local tcpObjectSignal = 'motionTcpObjectHandle'
local statusSignal = 'railUr10MotionStatus'
local collisionFreeTrajectoryDisabled = true

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

local function getConfig()
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

local function setTargetConfig(config)
    for i = 1, #joints do
        sim.setJointTargetPosition(joints[i], config[i])
    end
end

local function updateRobotCollection()
    if robotCollection then
        sim.destroyCollection(robotCollection)
    end

    robotCollection = sim.createCollection()
    sim.addItemToCollection(robotCollection, sim.handle_tree, ikBase, 0)

    local tcpObject = sim.getInt32Signal(tcpObjectSignal) or -1
    if tcpObject >= 0 then
        sim.addItemToCollection(robotCollection, sim.handle_single, tcpObject, 0)
        sim.addLog(sim.verbosity_scriptinfos,
                   '[rail_ur10_motion_simplify] including TCP object in collision collection: ' ..
                   tostring(tcpObject))
    end
end

local function removeTargetVisualization()
    if targetVisualizationShape then
        sim.removeObjects({targetVisualizationShape})
        targetVisualizationShape = nil
    end
end

local function createTargetVisualizationShape(config)
    local bufferedConfig = getConfig()
    setConfig(config)

    local visibleShapes = {}
    local shapes = sim.getObjectsInTree(railJoint, sim.sceneobject_shape)
    for i = 1, #shapes do
        if sim.getBoolProperty(shapes[i], 'visible') then
            visibleShapes[#visibleShapes + 1] = shapes[i]
        end
    end

    if #visibleShapes == 0 then
        setConfig(bufferedConfig)
        return nil
    end

    local copiedShapes = sim.copyPasteObjects(visibleShapes)
    local passiveVizShape = sim.groupShapes(copiedShapes, true)
    sim.setBoolProperty(passiveVizShape, 'respondable', false)
    sim.setBoolProperty(passiveVizShape, 'dynamic', false)
    sim.setBoolProperty(passiveVizShape, 'collidable', false)
    sim.setBoolProperty(passiveVizShape, 'measurable', false)
    sim.setBoolProperty(passiveVizShape, 'detectable', false)

    local meshes = sim.getIntArrayProperty(passiveVizShape, 'meshes') or {}
    for i = 1, #meshes do
        sim.setColorProperty(meshes[i], 'color.diffuse', {1, 0, 0})
    end
    sim.setObjectAlias(passiveVizShape, 'targetRobotConfigVisualization')

    setConfig(bufferedConfig)
    return passiveVizShape
end

local function visualizeTargetConfig(config)
    removeTargetVisualization()
    targetVisualizationShape = createTargetVisualizationShape(config)
    if targetVisualizationShape then
        sim.addLog(sim.verbosity_scriptinfos,
                   '[rail_ur10_motion_simplify] visualized target robot config')
    else
        sim.addLog(sim.verbosity_warnings,
                   '[rail_ur10_motion_simplify] no visible shapes for target config visualization')
    end
end

local function collides(configs)
    local bufferedConfig = getConfig()

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

local function findTargetConfig()
    setStatus('solving_ik')
    lastTargetPose = sim.getObjectPose(ikTarget, ikBase)
    simIK.syncFromSim(ikEnv, {ikGroup})

    simIK.setObjectPose(ikEnv,
                        simToIkObjectMap[ikTarget],
                        lastTargetPose,
                        simToIkObjectMap[ikBase])

    local result = simIK.handleGroup(ikEnv, ikGroup)
    if result ~= simIK.result_success then
        setStatus('ik_failed')
        sim.addLog(sim.verbosity_warnings,
                   '[rail_ur10_motion_simplify] IK handleGroup failed: ' ..
                   tostring(result))
        return nil
    end

    local config = {}
    for i = 1, #joints do
        config[i] = simIK.getJointPosition(ikEnv, simToIkObjectMap[joints[i]])
    end

    if collides({config}) then
        setStatus('collision_failed')
        sim.addLog(sim.verbosity_warnings,
                   '[rail_ur10_motion_simplify] direct IK target config collides')
        return nil
    end

    return config
end

local function findPath(goalConfig)
    setStatus('planning')
    local useForProjection = {}
    for i = 1, #joints do
        useForProjection[i] = (i <= 3 and 1 or 0)
    end

    local task = simOMPL.createTask('rail_ur10_simplify_path_task')
    simOMPL.setAlgorithm(task, pathPlanningAlgo)
    simOMPL.setStateSpaceForJoints(task, joints, useForProjection)
    simOMPL.setCollisionPairs(task, {
        robotCollection, sim.handle_all,
        robotCollection, robotCollection,
    })
    simOMPL.setStartState(task, getConfig())
    simOMPL.setGoalState(task, goalConfig)
    simOMPL.setup(task)

    local path = nil
    if simOMPL.solve(task, pathPlanningMaxTime) and simOMPL.hasExactSolution(task) then
        simOMPL.simplifyPath(task, pathPlanningMaxSimplificationTime)
        path = simOMPL.getPath(task)
    end

    simOMPL.destroyTask(task)
    return path
end

local function followPath(path)
    setStatus('moving')
    local minMaxVel = {}
    local minMaxAccel = {}

    for i = 1, #joints do
        minMaxVel[#minMaxVel + 1] = -fkMaxVel[i]
        minMaxVel[#minMaxVel + 1] = fkMaxVel[i]
        minMaxAccel[#minMaxAccel + 1] = -fkMaxAccel[i]
        minMaxAccel[#minMaxAccel + 1] = fkMaxAccel[i]
    end

    local pathLengths = sim.getPathLengths(path, #joints)
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
        setStatus('trajectory_failed')
        sim.addLog(sim.verbosity_warnings,
                   '[rail_ur10_motion_simplify] trajectory timing failed')
        return
    end

    local startTime = sim.getSimulationTime()
    local elapsed = 0

    while elapsed < times[#times] do
        setTargetConfig(sim.getPathInterpolatedConfig(pathPts, times, elapsed))
        sim.step()
        elapsed = sim.getSimulationTime() - startTime
    end

    setTargetConfig(sim.getPathInterpolatedConfig(pathPts, times, times[#times]))
    setStatus('done')
end

local function planAndMoveToTarget()
    updateRobotCollection()

    local goalConfig = findTargetConfig()
    if not goalConfig then
        sim.addLog(sim.verbosity_warnings,
                   '[rail_ur10_motion_simplify] no direct rail+UR10 IK config for /ikTarget')
        return
    end

    visualizeTargetConfig(goalConfig)

    local path = findPath(goalConfig)
    if not path then
        setStatus('planning_failed')
        sim.addLog(sim.verbosity_warnings,
                   '[rail_ur10_motion_simplify] OMPL found no exact collision-free path')
        return
    end

    followPath(path)
end

function sysCall_init()
    if collisionFreeTrajectoryDisabled then
        setStatus('disabled')
        sim.addLog(sim.verbosity_warnings,
                   '[rail_ur10_motion_simplify] legacy standalone planner disabled; external_api.lua owns planned rail+UR10 motion')
        return
    end

    ikBase = firstObject({
        '/mobile_arm',
    })
    railJoint = firstObject({
        '/mobile_arm/railJoint',
        '/railJoint',
        ':/railJoint',
    })
    ur10Base = firstObject({
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

    joints = {railJoint}
    local ur10Joints = sim.getObjectsInTree(ur10Base, sim.sceneobject_joint)
    for i = 1, #ur10Joints do
        joints[#joints + 1] = ur10Joints[i]
    end

    updateRobotCollection()

    ikEnv = simIK.createEnvironment()
    ikGroup = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv,
                          ikGroup,
                          simIK.method_damped_least_squares,
                          0.3,
                          99)

    local _
    _, simToIkObjectMap = simIK.addElementFromScene(ikEnv,
                                                    ikGroup,
                                                    ikBase,
                                                    ikTip,
                                                    ikTarget,
                                                    simIK.constraint_pose)

    for i = 1, #joints do
        local ikJoint = simToIkObjectMap[joints[i]]
        if not ikJoint then
            error('Planning joint was not mapped into the IK environment: ' ..
                  tostring(joints[i]))
        end
    end

    pathPlanningMaxTime = 10.0
    pathPlanningMaxSimplificationTime = 10.0
    pathPlanningAlgo = simOMPL.Algorithm.RRTConnect

    local railVel = 0.25
    local railAccel = 0.5
    local armVel = math.pi
    local armAccel = 40 * math.pi / 180

    fkMaxVel = {railVel}
    fkMaxAccel = {railAccel}
    for i = 2, #joints do
        fkMaxVel[i] = armVel
        fkMaxAccel[i] = armAccel
    end
    setStatus('ready')
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
    followPathScript = nil
    setStatus('stopped')
end
