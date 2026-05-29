# Layered Achievability Baseline Design

## Purpose

The current simple-block baseline can achieve reasonable bin utilization, but high utilization does not guarantee that a robot arm can execute the resulting packing sequence. Tall simpleblocks placed in central regions can obstruct later robot motions, especially when the robot must reach down into lower free spaces after the bin already contains taller structures.

This feature adds a configurable layered achievability constraint for the existing largest-block simple-block baseline. The trained policy and true EMS update logic remain unchanged. The baseline instead sees a temporary z-windowed EMS view, so it prefers large blocks that are usable within the currently robot-achievable vertical band.

## Scope

The initial implementation applies only when all of these are true:

- `policy_mode == "largest_block_baseline"`
- `use_simple_blocks == true`
- `layered_achievability == true`

The cascaded block selector is out of scope for this first experiment because it has different candidate-selection semantics. MCTS and replay support should continue to work through the existing packing APIs, but the layered stage policy is part of online baseline observation and action selection.

## Configuration

Add two runtime configuration fields:

- `layered_achievability`, default `false`
- `layered_num_chunks`, default `3`

`layered_num_chunks` must be a positive integer. The value should be accepted through the environment constructor and test YAML config loading so experiments can compare values such as 2, 3, 4, and 5 without editing code.

## Layer Rule

Let the container height be `H` and the configured chunk count be `N`. Stage indexes are 1-based.

For stage 1, the active z-window is chunk 1:

```text
[0, H / N]
```

For stage `i`, where `1 < i <= N`, the active z-window is chunks `i - 1` and `i`:

```text
[(i - 2) * H / N, i * H / N]
```

For `N = 3`, this gives:

```text
stage 1: [0, H/3]
stage 2: [0, 2H/3]
stage 3: [H/3, H]
```

This sliding two-chunk window prevents the baseline from reaching down into old lower regions after the process moves upward.

## EMS Clipping

The true EMS list remains owned by `self.heu_ems`. Before selecting a simpleblock or building the baseline action mask, the environment derives a temporary policy-visible EMS list by intersecting each raw EMS with the active stage z-window.

For each raw EMS:

```text
raw_z0 = ems.FLB.z
raw_z1 = ems.FLB.z + ems.Dim.dz
new_z0 = max(raw_z0, active_z_min)
new_z1 = min(raw_z1, active_z_max)
```

If `new_z1 <= new_z0`, the EMS has no overlap with the active window and is hidden from the policy. Otherwise, create a clipped `EmptyMaximalSpace` with:

```text
FLB.x = raw.FLB.x
FLB.y = raw.FLB.y
FLB.z = new_z0
Dim.dx = raw.Dim.dx
Dim.dy = raw.Dim.dy
Dim.dz = new_z1 - new_z0
```

Truncation is intentional. A raw EMS that crosses a chunk boundary should remain usable inside the active window instead of being discarded. If clipping raises `FLB.z`, the existing stability check must still reject unsupported floating placements.

## Baseline Selection Flow

The simple-block baseline should choose the largest usable block from the layered EMS view, not from the full raw EMS list.

The observation flow becomes:

```text
true EMS list
    -> clip to current stage z-window
    -> select largest usable simpleblock in clipped EMS list
    -> build vectorized EMS and action mask from clipped EMS list
    -> policy chooses placement
```

This preserves the baseline principle, but changes the definition of usable to mean usable within the current robot-achievable vertical band.

## Stage Progression

The environment keeps an internal 1-based stage:

```text
self.layered_stage = 1
```

On each baseline observation when layered achievability is enabled:

1. Clip EMSs to the current stage window.
2. Try to select the largest usable simpleblock in that clipped EMS list.
3. If a block exists, keep the current stage and expose that clipped EMS list.
4. If no block exists and `stage < N`, test exactly `stage + 1`.
5. If the next stage has a usable block, advance one stage and expose the next stage's clipped EMS list.
6. If the next stage also has no usable block, set the episode as done.

The environment must not skip across multiple empty stages. For example, if stage 2 fails and stage 3 fails, it should terminate instead of jumping directly to stage 4. This keeps the layered rule conservative and aligned with collision avoidance.

## Clipped-To-Source EMS Mapping

A clipped EMS may differ from its source raw EMS. The clipped EMS supplies the actual placement `FLB`, because the robot is placing within the active z-window. The EMS manager, however, should update against the source raw EMS that represents the true geometric free space.

The environment should maintain a temporary mapping for policy-visible EMSs:

```text
clipped EMS key -> source raw EMS
```

When decoding an action, `idx2pos()` should return the clipped placement position and retain enough information for `step()` to pass the corresponding source raw EMS to `heu_ems.update(...)`. If a policy-visible EMS is identical to the source EMS, the mapping still points to that same source object.

## Error Handling

Invalid `layered_num_chunks` values should raise `ValueError` during environment construction.

If layered achievability is enabled outside the supported baseline/simple-block mode, the environment should raise a clear `ValueError` so experiments do not accidentally run without the intended constraint.

If a clipped EMS placement is not physically supported, the existing stability mask should mark it invalid. No special floating-placement exception path is required.

## Testing

Add focused tests for:

- Stage windows for `N = 3`.
- EMS clipping that truncates crossing EMSs, discards EMSs outside the active window, and preserves EMSs fully inside the window.
- Largest usable simpleblock selection using the clipped EMS list rather than the full raw EMS list.
- Stage transition that advances by exactly one stage when the current stage has no usable block and the next stage does.
- Termination when both the current and next stage have no usable block.
- No multi-stage skipping.
- Correct source EMS update mapping when packing from a clipped EMS.

The tests should use small deterministic containers and synthetic EMS/block cases where possible, rather than relying on long stochastic packing episodes.
