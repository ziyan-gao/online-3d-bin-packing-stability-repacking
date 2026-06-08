# Cartesian Collision-Free Waypoints Design

## Goal

Add an opt-in motion command to `code2.lua` that generates collision-free Cartesian end-effector waypoints from the current configuration to `/arm_target`.

The generated waypoints are task-space waypoints expressed relative to the arm base. Each waypoint controls position and yaw only:

```lua
{x, y, z, yaw}
```

Roll and pitch are taken from the current end-effector pose and held constant for the full waypoint sequence.

## Existing Context

`code2.lua` already supports two movement modes through the `armCommand` signal:

- linear Cartesian motion with `sim.moveToPose` and IK
- direct configuration motion with `sim.moveToConfig`

The new feature should not replace those modes. It should add a third opt-in mode for planned Cartesian waypoints.

The target pose is read from `/arm_target` relative to the arm base. Only the UR10 manipulator should move. The mobile rail stays fixed.

## Collision Model

Create collision collections once during `sysCall_init()` and reuse them during waypoint validation.

Robot collection:

- root object: `/mobile_arm`
- includes the mobile arm tree

Obstacle collection:

- `/ZeroPressureBelt` tree
- `/Pallet` tree
- every object matching `/Cuboid*`

Waypoint validation uses:

```lua
sim.checkCollision(robotCollection, obstacleCollection)
```

If the `/mobile_arm` tree causes false positives because it includes fixed rail geometry, the implementation should narrow the robot collection to the UR10 moving geometry while still keeping the rail joint fixed.

## Planning Flow

The first implementation should use conservative straight-line Cartesian interpolation:

1. Read the current tip pose relative to the arm base.
2. Read the target pose from `/arm_target` relative to the arm base.
3. Convert both poses into task-space values.
4. Hold current roll and pitch constant.
5. Interpolate position and yaw from current values to target values.
6. For each candidate waypoint:
   - build a full pose from interpolated position, fixed roll/pitch, and interpolated yaw
   - solve IK for the UR10 joints
   - temporarily apply the candidate joint configuration
   - check robot-vs-obstacle collision
   - restore state when needed
   - keep the waypoint and joint configuration if valid
7. If every interpolated waypoint is valid, execute the valid waypoint sequence.
8. If any waypoint fails IK or collision validation, report failure and do not force the motion.

This first version intentionally fails fast when the direct Cartesian route is blocked. A future version can add a task-space RRT in `[x, y, z, yaw]` if obstacle avoidance around blocked routes is needed.

## Command Interface

Keep the existing command behavior unchanged. Add a new command flag such as:

```lua
sim.setStringSignal('armCommand', sim.packTable({
    move = true,
    plannedCartesian = true,
}))
```

When `plannedCartesian` is true, the script uses the new waypoint planner. Existing `linear` and configuration modes remain available.

## Helper Functions

The implementation should keep the feature separated into small helpers:

- `createCollisionCollections()`
- `getCurrentTaskPose4d()`
- `getTargetTaskPose4d()`
- `pose4dToPose()`
- `solveIkForPose()`
- `isConfigCollisionFree()`
- `generateCartesianWaypoints()`
- `executeCartesianWaypoints()`

These helpers should avoid changing existing movement functions except where command dispatch needs to call the new mode.

## Error Handling

The planner should return a clear failure result when:

- required scene objects are missing
- IK cannot solve a candidate waypoint
- a candidate waypoint is in collision
- no valid waypoint sequence is generated

Failures should leave the robot in its original configuration.

## Verification

Manual verification in CoppeliaSim should cover:

- existing `linear` command still works
- existing configuration command still works
- `plannedCartesian` succeeds for an unobstructed route
- `plannedCartesian` fails without moving into collision for a blocked route
- roll and pitch remain fixed from the starting end-effector pose
- yaw changes toward `/arm_target`
- the rail joint does not move
