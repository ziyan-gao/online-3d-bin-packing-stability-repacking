# UR10 Motion Debugging Summary

Date: 2026-06-15

## Issues

1. `pick()` created the pre-pick and pick dummies, but the robot did not move.
   - The pick sequence needed to run as a threaded command because it waits for motion scripts with `sim.wait()`.
   - Wrapping yielding code in `pcall()` caused simulation aborts.

2. `place()` showed small but visible placement error.
   - Initial suspicion was object pose or pick/place dummy construction.
   - Checking the picked object and tip pose showed very small XY deviation, so the object attachment transform was not the main cause.

3. `ur10_linear_motion.lua` finished with millimeter-level `ikTip` to `ikTarget` error.
   - Diagnostic logs showed final linear residuals around 5-8 mm, mostly in Z.
   - Extra settle calls did not reduce the error, which indicated the issue was not simply missing one more IK update.

4. `ur10_joint_motion.lua` also did not align `ikTip` to `ikTarget` accurately after motion.
   - `simIK.handleGroup()` sometimes solved accurately, but after `sim.moveToConfig()` the scene still had about 3 mm residual.
   - Directly applying the final solved joint config helped separate IK accuracy from joint execution/settling error.

5. IK sometimes failed even when the target pose was achievable.
   - The log showed very large IK precision error, e.g. around 0.88 m, meaning the local IK solve was stuck rather than the target being truly unreachable.
   - The rail+UR10 planned fallback also failed because it used the same fragile direct IK setup.

## Fixes

1. `external_api.lua`
   - Implemented `pick()` and `place()` as queued threaded commands.
   - Avoided `pcall()` around functions that call `sim.wait()`.
   - Added optional delays after approach, descent, suction, and retreat:
     - `delayAfterApproach`
     - `delayAfterDescend`
     - `delayAfterSuction`
     - `delayAfterRetreat`
   - Kept latest `prePlaceDummy` and `placeDummy` in the scene for visual inspection.

2. `ur10_joint_motion.lua`
   - Added IK precision logging from `simIK.handleGroup()`.
   - Added final `ikTip` to `ikTarget` error logging.
   - Tightened IK element precision with `simIK.setElementPrecision()`.
   - Switched IK calculation to damped least squares, matching the stable pattern from `moveToConfig.lua`.
   - Added final solved-config application after `sim.moveToConfig()` to remove residual joint settling error.

3. `rail_ur10_motion_simplify.lua`
   - Added final settling after following the planned joint path before reporting `done`.
   - Added planned-motion tip-target error logs before and after final settling.
   - Switched rail+UR10 IK to damped least squares.
   - Added strict IK element precision and IK precision logging.

4. `ur10_linear_motion.lua`
   - Added final linear motion tip-target error logs.
   - Tested extra IK settle behavior, which helped show that the main issue was upstream IK/joint execution rather than the linear command alone.

## Main Takeaway

The final root cause was not the pick/place dummy pose calculation. The main problem was the IK and motion execution stack:

- one-shot undamped pseudo-inverse IK could fail or converge poorly for achievable poses;
- `sim.moveToConfig()` could finish with small joint residuals that became millimeter-level tool error;
- planned rail+UR10 motion needed the same robust IK settings as UR10-only motion.

Switching to damped least squares, tightening IK precision, logging residuals, and applying/settling the final configuration solved the practical motion accuracy problem.

## Useful Logs To Watch

```text
[ur10_joint_motion] IK precision: linear=..., angular=...
[ur10_joint_motion] after moveToConfig tip-target xyz error: ...
[ur10_joint_motion] after final config apply tip-target xyz error: ...
[rail_ur10_motion_simplify] IK precision: linear=..., angular=...
[rail_ur10_motion_simplify] before final settle tip-target xyz error: ...
[rail_ur10_motion_simplify] after final settle tip-target xyz error: ...
[ur10_linear_motion] before settle tip-target xyz error: ...
[ur10_linear_motion] after settle tip-target xyz error: ...
```
