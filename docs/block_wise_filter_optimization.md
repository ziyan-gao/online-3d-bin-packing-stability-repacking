# Block-Wise Filter Optimization Notes

This note summarizes the optimization work around CJ-style block-wise packing,
especially the slow filter path for `SimpleBlock` candidates and EMS selection.

## Problem

After replacing Plotly with Three.js, visualization was still slow. Profiling
showed that browser rendering was no longer the main bottleneck.

The slow path was mostly in the packing-side filter process:

- `SimpleBlock` candidates were checked against many EMSs.
- Each `(block, EMS, rotation)` combination could call the stability heuristic.
- The raw EMS list could grow to several thousand EMSs.
- The visualizer also repeated some of the policy EMS/block selection work.

For seed `108`, a late packing state had about `5608` raw EMSs. Before the
filter optimization, the full packing loop took roughly `39-43 s`, and late
block selection could take multiple seconds per step.

## Important Semantics

The CJ-trained policy expects this behavior:

1. Generate block candidates from the buffer.
2. Prefer the largest packable block.
3. Expose policy-visible EMS candidates for that selected block.
4. Let the policy choose a placement among those EMS/action candidates.

So the optimization keeps the policy semantics focused on:

> largest block that is usable in the policy-visible EMS set

It does not try to globally solve all block/EMS combinations.

## Optimization 1: Batch Stability Checks Per Block

Old behavior:

- For each block:
- For each EMS:
- For each x/y orientation:
- Call `heu_stable(...)`

That meant `heu_stable` was called many times with one candidate at a time.
Each call rebuilds or accesses sliding-window views, so this became expensive
when EMS count grew.

New behavior:

- For each block orientation:
- First collect all EMSs that can fit the block.
- Call `heu_stable(...)` once with all candidate FLBs.
- Check returned stable placements against EMS FLBs.

Main code:

- `packing_env/data_type/buffer.py`
- `Buffer._is_block_usable(...)`

Effect:

- Same acceptance rule.
- Much fewer `heu_stable(...)` calls.
- Less Python-loop overhead.

## Optimization 2: Select Largest Usable Block Directly

Old behavior:

- Filter every generated block into `buffer.simple_blocks`.
- Then sample the largest remaining block.

This was unnecessary for CJ behavior, because the policy only needs the largest
usable block.

New behavior:

- Sort all block candidates by volume, consumed count, and dimensions.
- Try candidates from largest to smallest.
- Stop immediately when the first usable block is found.
- Keep only that selected block in `buffer.simple_blocks`.

Main code:

- `packing_env/data_type/buffer.py`
- `Buffer.select_largest_usable(...)`
- `packing_env/gym_env.py`
- `PackingEnv.select_largest_policy_block(...)`

Effect:

- Avoids proving smaller blocks usable when a larger block is already usable.
- Better matches the CJ training assumption.

## Optimization 3: Use Policy-Visible EMSs For Block Selection

A key realization:

The policy does not act on every raw EMS. It acts on the capped/ranked EMS list.

So block usability should be tested against the EMS set that the policy would
actually see for that block, not thousands of raw EMSs.

New behavior:

- For each candidate block:
- Build its policy-visible EMS list with `_get_item_fit_ems_list([block])`.
- Test usability only against that list.
- Stop at the first usable block.

Main code:

- `packing_env/gym_env.py`
- `PackingEnv.select_largest_policy_block(...)`

Effect:

- Aligns block filtering with policy exposure.
- Avoids expensive checks against irrelevant raw EMSs.

## Optimization 4: Vectorized Single-Item EMS Fit/Rank

The hot path repeatedly asks:

> Which EMSs can fit this one block?

Old behavior:

- Python loop over every EMS.
- Python call to `_ems_can_fit_item(...)`.
- Python sort over filtered EMSs.

At around `5051` raw EMSs, this took about `250 ms` per block in a late seed
`108` state, even for blocks that had zero feasible EMSs.

New behavior:

- Convert EMS FLB/dimensions to NumPy arrays.
- Check normal and x/y-rotated fit with vectorized comparisons.
- Collapse duplicate FLBs when capped.
- Rank using `np.lexsort`.

Main code:

- `packing_env/gym_env.py`
- `PackingEnv._get_single_item_fit_ems_list(...)`

Effect:

- Same ranking rule:
  - lower `z`
  - lower `y`
  - lower `x`
  - larger EMS volume as tie-breaker
- Much faster for the single-block policy path.

Measured late-state improvement:

- Before: about `250 ms` per block EMS fit/rank
- After: about `9-10 ms` per block EMS fit/rank

## Optimization 5: Remove Duplicate Pre-Pass In `get_next_observation`

`get_next_observation()` still had an old pre-pass that computed EMS candidates
for all blocks before selecting the actual block.

New behavior:

- In simple-block mode, select the largest policy block first.
- Then compute the EMS/mask only for that selected block.

Main code:

- `packing_env/gym_env.py`
- `PackingEnv.get_next_observation(...)`

Effect:

- Avoids unnecessary all-block EMS ranking in normal env stepping.

## Visualization Impact

After the Three.js replacement, scene construction was already cheap.

For seed `108` blocked state:

- Raw EMS count: about `5608`
- Three.js scene payload: about `6.4 KB`
- Scene build: about `6-8 ms`
- JSON serialization: about `0.1 ms`

The remaining delay was not mainly browser visualization. It was filter and
candidate preparation.

After filter optimization:

- Visual refresh at blocked state became effectively negligible.
- Full seed `108` packing loop dropped from roughly `39-43 s` to about `6.7 s`.

## Measurement Summary

Seed: `108`

Before optimization:

- Full packing loop: about `39-43 s`
- Late block-selection total: about `20-24 s`
- Single block EMS fit/rank near `5000` EMS: about `250 ms`

After optimization:

- Full packing loop: about `6.7 s`
- Late block-selection total: about `1.2 s`
- Single block EMS fit/rank near `5000` EMS: about `9-10 ms`
- Focused tests: `12 passed`

## Files Changed

Core filter changes:

- `packing_env/data_type/buffer.py`
- `packing_env/gym_env.py`
- `packing/test_utils.py`
- `packing/visualizer.py`

Visualization-related Three.js changes:

- `packing/three_scene.py`
- `packing/live_plot.py`
- `packing/interactive_replay.py`
- `interactive_simulator_app/static/simulator.js`
- `interactive_simulator_app/static/index.html`
- `interactive_simulator_app/static/style.css`

## Remaining Possible Optimizations

The remaining visible cost is mostly from:

- EMS generation/update after placement
- Height-map and feasible-map updates
- Stability mask generation for the selected block

Possible next steps:

- Cache EMS NumPy arrays and invalidate only when EMS changes.
- Cache per-block policy EMS fit results within a single decision step.
- Prune EMSs more aggressively before ranking, while preserving policy behavior.
- Reuse already-computed selected block and EMS data in the visualizer instead
  of recomputing it for display.
