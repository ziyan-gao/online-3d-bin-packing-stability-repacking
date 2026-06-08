# Cartesian Collision-Free Waypoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `plannedCartesian` command to `code2.lua` that generates and executes collision-free Cartesian end-effector waypoints from the current pose to `/arm_target`.

**Architecture:** Keep the existing linear and configuration movement modes unchanged. Add small helper functions for collision collection setup, task-pose conversion, IK solving, collision validation, waypoint generation, and waypoint execution. The first planner is straight-line Cartesian interpolation in `[x, y, z, yaw]`; it fails safely if any candidate waypoint has no IK solution or collides.

**Tech Stack:** CoppeliaSim Lua script, `sim` regular API, `simIK` plugin, CoppeliaSim collision collections, existing `sim.moveToPose` / `sim.moveToConfig` style.

---

## File Structure

- Modify `code2.lua`
  - Add `sim.getObject` lookup helpers for optional obstacle objects.
  - Add collision collection creation in `sysCall_init()`.
  - Add pose conversion helpers for `[x, y, z, yaw]`.
  - Add IK solve and collision validation helpers.
  - Add waypoint generation and execution helpers.
  - Add command dispatch for `plannedCartesian`.
- Keep `docs/superpowers/specs/2026-06-08-cartesian-collision-free-waypoints-design.md` unchanged.
- Keep this implementation plan in `docs/superpowers/plans/2026-06-08-cartesian-collision-free-waypoints.md`.

## Task 1: Add Planner Configuration And Collection Setup

**Files:**
- Modify: `code2.lua`

- [ ] **Step 1: Add planner configuration globals in `sysCall_init()`**

Insert this block near the existing velocity configuration in `sysCall_init()` after `maxIkJerk`:

```lua
    cartesianPlanner = {}
    cartesianPlanner.stepSize = 0.05 -- meters between interpolated Cartesian waypoints
    cartesianPlanner.minSteps = 2
    cartesianPlanner.robotRootAlias = '/mobile_arm'
    cartesianPlanner.obstacleRootAliases = {'/ZeroPressureBelt', '/Pallet'}
    cartesianPlanner.cuboidAliasPrefix = '/Cuboid'
```

- [ ] **Step 2: Add safe object lookup helper**

Insert this function above `sysCall_init()`:

```lua
function getObjectIfExists(alias)
    local handle = sim.getObject(alias, {noError = true})
    if handle and handle >= 0 then
        return handle
    end
    return nil
end
```

- [ ] **Step 3: Add collision collection creation helper**

Insert this function above `sysCall_init()` after `getObjectIfExists()`:

```lua
function createCollisionCollections()
    local robotRoot = getObjectIfExists(cartesianPlanner.robotRootAlias)
    if not robotRoot then
        return false, 'missing robot root: ' .. cartesianPlanner.robotRootAlias
    end

    robotCollection = sim.createCollection()
    sim.addItemToCollection(robotCollection, sim.handle_tree, robotRoot, 0)

    obstacleCollection = sim.createCollection()

    local obstacleCount = 0
    for i = 1, #cartesianPlanner.obstacleRootAliases, 1 do
        local obstacleRoot = getObjectIfExists(cartesianPlanner.obstacleRootAliases[i])
        if obstacleRoot then
            sim.addItemToCollection(obstacleCollection, sim.handle_tree, obstacleRoot, 0)
            obstacleCount = obstacleCount + 1
        end
    end

    local cuboidIndex = 0
    while true do
        local cuboid = sim.getObject(cartesianPlanner.cuboidAliasPrefix .. '*', {index = cuboidIndex, noError = true})
        if not cuboid or cuboid < 0 then
            break
        end
        sim.addItemToCollection(obstacleCollection, sim.handle_single, cuboid, 0)
        obstacleCount = obstacleCount + 1
        cuboidIndex = cuboidIndex + 1
    end

    if obstacleCount == 0 then
        return false, 'no obstacles found for collision checking'
    end

    return true, nil
end
```

- [ ] **Step 4: Call collection setup in `sysCall_init()`**

Insert this after the `cartesianPlanner` configuration block:

```lua
    local collectionsOk, collectionsError = createCollisionCollections()
    if not collectionsOk then
        sim.addLog(sim.verbosity_warnings, 'Cartesian planner collision setup failed: ' .. collectionsError)
    end
```

- [ ] **Step 5: Run Lua syntax check**

Run:

```bash
luac -p code2.lua
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 6: Commit**

Run:

```bash
git add code2.lua
git commit -m "Add Cartesian planner collision collections"
```

## Task 2: Add Task-Space Pose Helpers

**Files:**
- Modify: `code2.lua`

- [ ] **Step 1: Add angle interpolation helpers**

Insert these functions above `moveToPose()`:

```lua
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
```

- [ ] **Step 2: Add pose-to-task-pose helper**

Insert this function after `interpolateAngle()`:

```lua
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
```

- [ ] **Step 3: Add task-pose-to-pose helper**

Insert this function after `poseToTaskPose4d()`:

```lua
function taskPose4dToPose(taskPose, fixedRoll, fixedPitch, baseHandle)
    return sim.buildPose({taskPose.x, taskPose.y, taskPose.z}, {fixedRoll, fixedPitch, taskPose.yaw})
end
```

The `baseHandle` argument is intentionally accepted for call-site clarity; CoppeliaSim poses are already relative to the chosen base when passed into this helper. `sim.buildPose()` returns the 7-value pose format used by `simIK.setObjectPose()`.

- [ ] **Step 4: Add interpolation helper**

Insert this function after `taskPose4dToPose()`:

```lua
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
```

- [ ] **Step 5: Run Lua syntax check**

Run:

```bash
luac -p code2.lua
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 6: Commit**

Run:

```bash
git add code2.lua
git commit -m "Add Cartesian task pose helpers"
```

## Task 3: Add IK Solve And Collision Validation Helpers

**Files:**
- Modify: `code2.lua`

- [ ] **Step 1: Add joint position snapshot helpers**

Insert these functions above `moveToPose()` after the task-pose helpers:

```lua
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
```

- [ ] **Step 2: Add IK solver helper**

Insert this function after `setJointPositions()`:

```lua
function solveIkForPose(targetPose)
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
```

- [ ] **Step 3: Add collision check helper**

Insert this function after `solveIkForPose()`:

```lua
function isConfigCollisionFree(targetAngles)
    if not robotCollection or not obstacleCollection then
        return false, 'collision collections are not initialized'
    end

    local originalAngles = getJointPositions(simJoints)
    setJointPositions(simJoints, targetAngles)
    local collisionResult = sim.checkCollision(robotCollection, obstacleCollection)
    setJointPositions(simJoints, originalAngles)

    if collisionResult ~= 0 then
        return false, 'candidate configuration is in collision'
    end

    return true, nil
end
```

- [ ] **Step 4: Run Lua syntax check**

Run:

```bash
luac -p code2.lua
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 5: Commit**

Run:

```bash
git add code2.lua
git commit -m "Add IK and collision validation helpers"
```

## Task 4: Add Waypoint Generation

**Files:**
- Modify: `code2.lua`

- [ ] **Step 1: Add interpolation count helper**

Insert this function above `moveToPose()` after `isConfigCollisionFree()`:

```lua
function computeWaypointStepCount(startTaskPose, targetTaskPose)
    local dx = targetTaskPose.x - startTaskPose.x
    local dy = targetTaskPose.y - startTaskPose.y
    local dz = targetTaskPose.z - startTaskPose.z
    local distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    local steps = math.ceil(distance / cartesianPlanner.stepSize)
    if steps < cartesianPlanner.minSteps then
        steps = cartesianPlanner.minSteps
    end
    return steps
end
```

- [ ] **Step 2: Add Cartesian waypoint generator**

Insert this function after `computeWaypointStepCount()`:

```lua
function generateCartesianWaypoints()
    local startPose = sim.getObjectPose(ik_data.tip, ik_data.base)
    local targetPose = sim.getObjectPose(arm_target_handle, ik_data.base)
    local startTaskPose = poseToTaskPose4d(startPose)
    local targetTaskPose = poseToTaskPose4d(targetPose)
    local fixedRoll = startTaskPose.roll
    local fixedPitch = startTaskPose.pitch
    local steps = computeWaypointStepCount(startTaskPose, targetTaskPose)

    local waypoints = {}
    local configs = {}

    for i = 1, steps, 1 do
        local t = i / steps
        local taskPose = interpolateTaskPose4d(startTaskPose, targetTaskPose, fixedRoll, fixedPitch, t)
        local pose = taskPose4dToPose(taskPose, fixedRoll, fixedPitch, ik_data.base)
        local ikOk, targetAngles, ikError = solveIkForPose(pose)
        if not ikOk then
            return false, nil, nil, 'waypoint ' .. i .. ': ' .. ikError
        end

        local collisionOk, collisionError = isConfigCollisionFree(targetAngles)
        if not collisionOk then
            return false, nil, nil, 'waypoint ' .. i .. ': ' .. collisionError
        end

        waypoints[i] = taskPose
        configs[i] = targetAngles
    end

    return true, waypoints, configs, nil
end
```

- [ ] **Step 3: Run Lua syntax check**

Run:

```bash
luac -p code2.lua
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 4: Commit**

Run:

```bash
git add code2.lua
git commit -m "Generate Cartesian collision-free waypoints"
```

## Task 5: Execute Planned Cartesian Waypoints

**Files:**
- Modify: `code2.lua`

- [ ] **Step 1: Add execution helper**

Insert this function after `generateCartesianWaypoints()`:

```lua
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
end
```

- [ ] **Step 2: Add planned Cartesian move function**

Insert this function after `executeCartesianWaypointConfigs()`:

```lua
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
```

- [ ] **Step 3: Run Lua syntax check**

Run:

```bash
luac -p code2.lua
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 4: Commit**

Run:

```bash
git add code2.lua
git commit -m "Execute planned Cartesian waypoints"
```

## Task 6: Add Command Dispatch

**Files:**
- Modify: `code2.lua`

- [ ] **Step 1: Update the command example comment**

Extend the top comment with this example:

```lua
sim.setStringSignal('armCommand', sim.packTable({
    move = true,
    plannedCartesian = true,
}))
```

- [ ] **Step 2: Update `sysCall_thread()` dispatch**

Replace the current command dispatch block:

```lua
            if ok and cmd ~= nil and cmd.move then
                if cmd.linear then
                    moveTo('linear')
                else
                    moveTo('config')
                end
            end
```

with:

```lua
            if ok and cmd ~= nil and cmd.move then
                if cmd.plannedCartesian then
                    moveToPlannedCartesian()
                elseif cmd.linear then
                    moveTo('linear')
                else
                    moveTo('config')
                end
            end
```

- [ ] **Step 3: Run Lua syntax check**

Run:

```bash
luac -p code2.lua
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 4: Commit**

Run:

```bash
git add code2.lua
git commit -m "Add planned Cartesian command dispatch"
```

## Task 7: Manual CoppeliaSim Verification

**Files:**
- Verify: `code2.lua`

- [ ] **Step 1: Start the scene in CoppeliaSim**

Open the scene that contains `/mobile_arm`, `/arm_target`, `/ZeroPressureBelt`, `/Pallet`, and `/Cuboid*`.

Expected:

- The script initializes without errors.
- If obstacle objects are missing, the script logs a warning and `plannedCartesian` fails safely.

- [ ] **Step 2: Verify existing linear command**

Run this Lua command from the CoppeliaSim sandbox or an existing control panel:

```lua
sim.setStringSignal('armCommand', sim.packTable({
    move = true,
    linear = true,
}))
```

Expected:

- Existing linear motion still executes.
- No planned Cartesian log message appears.

- [ ] **Step 3: Verify existing configuration command**

Run:

```lua
sim.setStringSignal('armCommand', sim.packTable({
    move = true,
    linear = false,
}))
```

Expected:

- Existing configuration motion still executes.
- No planned Cartesian log message appears.

- [ ] **Step 4: Verify planned Cartesian command on an unobstructed route**

Move `/arm_target` to an unobstructed reachable pose, then run:

```lua
sim.setStringSignal('armCommand', sim.packTable({
    move = true,
    plannedCartesian = true,
}))
```

Expected:

- The arm moves through planned waypoints.
- The log includes `Executed N planned Cartesian waypoints`.
- The rail joint remains fixed.
- End-effector roll and pitch remain visually fixed from the starting pose.
- End-effector yaw approaches `/arm_target` yaw.

- [ ] **Step 5: Verify planned Cartesian failure on a blocked route**

Move `/arm_target` so the straight Cartesian route intersects the conveyor, pallet, or a `Cuboid*` object, then run:

```lua
sim.setStringSignal('armCommand', sim.packTable({
    move = true,
    plannedCartesian = true,
}))
```

Expected:

- The arm does not force motion through the obstacle.
- The log includes `Cartesian waypoint planning failed: waypoint N: candidate configuration is in collision` or an IK failure message.
- The robot returns to the original joint configuration before the failed command.

- [ ] **Step 6: Commit verification notes if code changes were needed**

Only if verification required code changes, run:

```bash
git add code2.lua
git commit -m "Fix planned Cartesian verification issues"
```

Expected:

- No commit is needed if verification passes without code changes.

## Self-Review

- Spec coverage: The plan covers opt-in command dispatch, Cartesian `[x, y, z, yaw]` waypoints, fixed current roll/pitch, `/arm_target` relative to arm base, UR10-only motion with fixed rail, collision collections for `/mobile_arm`, `/ZeroPressureBelt`, `/Pallet`, and `/Cuboid*`, fail-fast behavior, and manual verification.
- Completeness scan: No unresolved markers or unspecified implementation steps remain.
- Type consistency: Helper names and data shapes are consistent across tasks: task poses use `{x, y, z, roll, pitch, yaw}`, waypoint output uses task pose tables, executable configurations use joint angle arrays, and command dispatch uses `plannedCartesian`.
