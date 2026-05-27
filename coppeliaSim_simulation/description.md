# CoppeliaSim Strict Stability Experiments

This folder reproduces the strict convex-hull CoppeliaSim experiment. Results
are written inside this folder in directories named like:

```text
coppeliaSim_simulation/experiment_results_strict_convex_compare_<RUN_ID>/
```

## Files

```text
coppeliaSim_bin_packing.ttt
```

CoppeliaSim scene used by the experiment.

```text
coppeliaSimEnv.py
```

Remote API wrapper for loading the scene, selecting the physics engine, adding
objects, stepping simulation, waiting for quasi-static states, and checking
physical stability.

```text
stability_experiment.py
```

Single-process experiment runner. It assumes CoppeliaSim is already running on
`--remote-api-port`.

```text
run_stability_sweep_strict_convex_compare_3parallel.sh
```

Top-level reproducer. It starts three CoppeliaSim processes in parallel and
runs the strict sweep for Bullet 2.83, ODE, and Newton.

## Baselines

The experiment imports baselines from:

```text
../packing_env/stable_heuristics_baselines/
```

Available heuristic names:

```text
convex_hull
convex_hull_old
convex_hull_plain
combined_rules
adaptive_tree
```

The default strict sweep uses:

```bash
HEURISTICS=(convex_hull convex_hull_old)
```

## Main Command

Run the full strict sweep from the project root:

```bash
cd /home/gao/online-3d-bin-packing-stability-repacking
./coppeliaSim_simulation/run_stability_sweep_strict_convex_compare_3parallel.sh
```

Quick smoke run:

```bash
cd /home/gao/online-3d-bin-packing-stability-repacking
EPOCHS=1 MAX_ITEMS=20 SAVE_EVERY=1 \
./coppeliaSim_simulation/run_stability_sweep_strict_convex_compare_3parallel.sh
```

## Shell Variables

These can be set before running the shell script:

```bash
PYTHON_BIN=/home/gao/anaconda3/envs/packing-toolkit/bin/python
```

Python interpreter used for the experiment.

```bash
COPPELIASIM_DIR=/home/gao/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04
```

CoppeliaSim installation folder.

```bash
EPOCHS=20
MAX_ITEMS=500
SAVE_EVERY=5
SIMULATION_TIME_STEP=0.01
```

Experiment size, checkpoint frequency, and physics step size.

```bash
COPPELIASIM_ISOLATED_SETTINGS=0
```

Set to `1` to use per-process CoppeliaSim settings folders.

## Default Sweep Settings

The shell runner uses:

```bash
PORTS=(23000 23001 23002)
ENGINES=(bullet_2_83 ode newton)
BIN_Z_VALUES=(0.6 1.2 1.8)
Z_MODES=(same variable)
CONVEX_THRESHOLDS=(0.1 0.2 0.3 0.4 0.5)
HEURISTICS=(convex_hull convex_hull_old)
```

Strict stability settings:

```bash
--stepping
--settle-by-velocity
--linear-velocity-threshold 0.01
--angular-velocity-threshold 0.05
--settle-stable-duration 0.5
--max-settle-wait-time 5.0
--settle-time 0.5
--sample-time 0.5
--drift-tolerance 0.001
--sim-object-xy-scale 0.999
```

## Output Layout

Each run creates:

```text
coppeliaSim_simulation/experiment_results_strict_convex_compare_<RUN_ID>/
  results/
    results_<RUN_ID>_<engine>_bin<z>_<same|variable>.csv
  summaries/
    summary_<RUN_ID>_<engine>_bin<z>_<same|variable>.csv
  logs/
    coppeliasim_<engine>_port<port>.log
    experiment_<engine>_port<port>.log
```

## Running One Condition Manually

Start CoppeliaSim separately on a port, then run:

```bash
cd /home/gao/online-3d-bin-packing-stability-repacking
/home/gao/anaconda3/envs/packing-toolkit/bin/python \
  coppeliaSim_simulation/stability_experiment.py \
  --epochs 5 \
  --max-items 100 \
  --bin-z 1.8 \
  --same-object-z \
  --remote-api-port 23000 \
  --engines bullet_2_83 \
  --heuristics convex_hull convex_hull_old \
  --convex-thresholds 0.1 0.2 0.3 \
  --drop-heights 0.005 \
  --load-delay 0.0 \
  --sim-object-xy-scale 0.999 \
  --simulation-time-step 0.01 \
  --stepping \
  --settle-by-velocity \
  --settle-stable-duration 0.5 \
  --max-settle-wait-time 5.0 \
  --settle-time 0.5 \
  --sample-time 0.5 \
  --drift-tolerance 0.001 \
  --output-csv /tmp/results.csv \
  --summary-csv /tmp/summary.csv
```

Usually, prefer the shell runner because it starts and stops CoppeliaSim for you.
