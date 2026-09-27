# LP Migration Implementation Plan

> Execute inline with executing-plans; preserve the target repository's existing edits.

**Goal:** Run global LP analysis and load visualization independently in this repository.
**Architecture:** Reuse Item and cached LBCP geometry; port solver and Three.js package; expose a small environment adapter and deterministic CLI demo.
**Tech Stack:** Python 3.11, NumPy, SciPy/HiGHS, Shapely, Three.js.

- [x] Inspect both repositories and run target baseline: `python -m pytest tests -q` (88 passed).
- [x] Copy tests/test_lbcp_load_bounds.py, test_loading_vis.py, test_threejs_replay.py and test_threejs_live.py from source; add tests/test_lp_analysis.py. Run to establish missing-module failures.
- [x] Migrate packing_env/lbcp/*.py and packing/threejs_visualization excluding node_modules and caches. Extend support_vis.py and add missing fields to VisualConfig while preserving existing defaults.
- [x] Add Heu_Stable.lbcp_polygon_map() from source. Implement packing.lp_analysis.solve_packing_loads(env, material_density, gravity) with strict cache/physical-input checks and build_lp_frame returning the solve result and visualization frame. Convert ground and item interfaces to LoadContactPatchVis, preserving vertex force witnesses.
- [x] Add packing.lp_demo CLI: stack and bridge scenes, deterministic seed, replay export, optional localhost live view, density and force-view options. Fail clearly if a demo LP is not optimal.
- [x] Document brief LP model, units, independent-bound interpretation, demo and existing-env API in README; declare Shapely dependency.
- [x] Run `python -m pytest tests -q`, frontend `npm test` and `npm run build`, stack and bridge CLI exports, and inspect exported frames/assets. Review diff and existing edits before completion.

## Validation results

- Original target baseline: 88 passed.
- Final Python suite: 173 passed (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/gao/anaconda3/envs/packing-toolkit/bin/python -m pytest tests -q --tb=short`). Plugin autoload disabled to avoid unrelated ROS launch-test hooks.
- Frontend: 23 passed; `npm run build` succeeded with independent npm-installed dependencies.
- Bridge demo: optimal, 4 interfaces / 16 variables; stack demo: optimal, 3 interfaces / 12 variables.
- Chrome headless rendered the offline replay using bundled assets; live HTTP smoke verified HTML, LP frame, resultant-only arrows and renderer bundle.
- Independent review identified that feasible witnesses overrode bound views. Fixed with regression coverage for all three views, force magnitudes, moment-equivalent resultants and legends; follow-up review found no further important issues.
- `git diff --check` and Python compilation passed. Existing configuration/CoppeliaSim edits preserved.

## Main integration validation

The LP change was cherry-picked onto main (base 859b8e8) without the unrelated blockwise training commits. Main's Item lacks round_buffered_dim, so buffer previews now obtain dimensions through Item.Virtual_Dim. Frontend pretest generates ignored vendor assets, supporting npm ci followed directly by npm test in a fresh checkout.

Main validation: 87 Python tests and 23 frontend tests passed; frontend build succeeded; bridge and stack CLI demos returned optimal and exported offline replays. Independent review found no important remaining issues. Earlier 173-test results above refer to the blockwise branch and include its separate training tests.
