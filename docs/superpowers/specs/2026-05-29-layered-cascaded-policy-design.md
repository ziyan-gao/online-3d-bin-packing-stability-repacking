# Layered Cascaded Policy Design

## Goal

Allow `layered_achievability` to work with `policy_mode: cascaded_block_selector` as well as `largest_block_baseline`.

The layered approach should keep its current semantics: at stage 1, only chunk 1 is active; at stage `i > 1`, chunks `i - 1` and `i` are active. If no stable candidate exists in the current stage, the environment may activate exactly the next stage. It must not skip multiple stages in one observation.

## Current Behavior

`PackingEnv` currently rejects `layered_achievability=True` unless `policy_mode` is `largest_block_baseline`.

Most of the EMS clipping machinery is already policy-neutral:

- `_layered_stage_window(...)` computes the active z window.
- `_clip_ems_to_layer_window(...)` truncates raw EMSs into the active window.
- `_get_policy_ems_list(...)` returns clipped policy EMSs when layered mode is enabled.
- `resolve_policy_ems_source(...)` maps a clipped policy EMS back to the raw EMS used by the EMS update.

The cascaded policy already builds candidates through `_get_item_fit_ems_list(...)`, so it can reuse the same clipped EMS view.

## Design

Layered achievability is supported for these policy modes:

- `largest_block_baseline`
- `cascaded_block_selector`

When `policy_mode` is `cascaded_block_selector`, the environment still forces `stack_only=True` and `use_simple_blocks=True`.

The cascaded candidate path will use a helper that can evaluate a specific layered stage without permanently mutating `layered_stage`. Candidate generation will:

1. Try the current `layered_stage`.
2. If there is at least one stable oriented-block/EMS pair, return those candidates.
3. If there is no candidate and `layered_stage < layered_num_chunks`, try `layered_stage + 1`.
4. If the next stage has candidates, set `layered_stage` to that next stage and return them.
5. If the next stage also has no candidates, keep the original stage and return no candidates.

This matches the baseline rule and avoids jumping ahead through several chunks.

## EMS Source Resolution

The cascaded policy will continue to receive clipped policy EMSs in observations. Any selected EMS must resolve back to its raw EMS before `heu_ems.update(...)`.

This is already handled by `PackingEnv.step(...)`, `_step(...)`, and `pack(...)` through `resolve_policy_ems_source(...)`. The change should preserve that behavior for cascaded actions.

## Config Behavior

Configs may freely choose:

- `policy_mode: largest_block_baseline`
- `policy_mode: cascaded_block_selector`

and independently set:

- `layered_achievability: true`
- `layered_num_chunks: N`

The only invalid layered policy mode remains any unknown or future policy mode that has not explicitly been added to the supported set.

## Tests

Add or update tests for:

- cascaded policy can be constructed with `layered_achievability=True`;
- cascaded candidate generation uses clipped EMSs from the active layer;
- cascaded candidate generation advances exactly one stage when the current stage has no valid stable candidate;
- cascaded candidate generation does not skip multiple stages;
- cascaded `step(...)` resolves selected clipped EMSs back to raw EMSs before EMS update.

Existing layered baseline tests should continue to pass unchanged.
