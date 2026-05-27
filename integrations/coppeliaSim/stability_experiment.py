from __future__ import annotations

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from packing_env.data_type.container import Container
from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item
from packing_env.data_type.maps import HeightMap
from packing_env.stable_heuristics_baselines import BASELINES, build_height_bound_candidates

from coppeliaSimEnv import PackingEnv


HEURISTICS = {
    "convex_hull": BASELINES["convex_hull"],
    "convex_hull_old": BASELINES["convex_hull_old"],
    "convex_hull_plain": BASELINES["convex_hull_plain"],
    "combined_rules": BASELINES["combined_rules"],
    "adaptive_tree": BASELINES["adaptive_tree"],
}
CONVEX_HEURISTICS = {"convex_hull", "convex_hull_old", "convex_hull_plain"}


def write_csv(path, rows):
    if not rows:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = "{}.tmp".format(path)
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def make_state(container_size, heuristic_name):
    container = Container(*container_size)
    hm = HeightMap(dx=container_size[0], dy=container_size[1], zmax=container_size[2])
    heuristic = HEURISTICS[heuristic_name](dx=container_size[0], dy=container_size[1])
    return container, hm, heuristic


def sample_action(o3d, hm, heuristic, rng, convex_threshold=None, anchor_step=60):
    candidates = build_height_bound_candidates(o3d, hm, anchor_step=anchor_step)
    t0 = time.time()
    kwargs = {}
    if convex_threshold is not None:
        kwargs["scale"] = convex_threshold
    stable_coords, stable_flags = heuristic(o3d=o3d, hm=hm, candidates=candidates, **kwargs)
    action_time = time.time() - t0
    coords = [coord for coord, stable in zip(stable_coords, stable_flags) if stable and coord is not None]
    if not coords:
        return None, action_time, len(candidates)
    min_z = min(coord.z for coord in coords)
    min_z_coords = [coord for coord in coords if coord.z == min_z]
    return min_z_coords[int(rng.integers(0, len(min_z_coords)))], action_time, len(candidates)


def update_heuristic(heuristic, hm, box):
    heuristic.update(hm=hm, box=box)


def item_to_sim(box, sim_object_xy_scale):
    obj_m = box.Dim.raw().astype(np.float64) / 1000.0
    obj_vis = obj_m.copy()
    obj_vis[:2] *= sim_object_xy_scale
    p_m = box.True_FLB.numpy().astype(np.float64) / 1000.0
    p_m[:2] += (obj_m[:2] - obj_vis[:2]) / 2
    return obj_vis, p_m


def run_epoch(
    objects,
    heuristic_name,
    physics_engine,
    env,
    drop_height,
    load_delay,
    container_size,
    same_object_z,
    stability_kwargs,
    settle_by_velocity_kwargs,
    convex_threshold,
    sim_object_xy_scale,
    seed,
    anchor_step,
):
    rng = np.random.default_rng(seed)
    container, hm, heuristic = make_state(container_size, heuristic_name)
    accepted_items = 0
    attempted_items = 0
    accepted_volume = 0.0
    total_action_time = 0.0
    total_checked_count = 0
    max_position_delta = 0.0
    max_rotation_delta = 0.0
    max_settle_drift = 0.0
    max_linear_speed = 0.0
    max_angular_speed = 0.0
    total_settle_wait_time = 0.0
    settle_timeout_count = 0
    failure_reason = ""
    t0 = time.time()

    try:
        for obj_raw in objects:
            o3d = Orthogonal3D(*map(int, obj_raw))
            coord, action_time, checked_count = sample_action(
                o3d,
                hm,
                heuristic,
                rng,
                convex_threshold=convex_threshold,
                anchor_step=anchor_step,
            )
            total_action_time += action_time
            total_checked_count += checked_count
            if coord is None:
                failure_reason = "no_stable_action"
                break

            attempted_items += 1
            box = Item(FLB=coord, Dim=o3d)
            try:
                update_heuristic(heuristic, hm, box)
                container.add(box)
                hm.update(box)
            except Exception as exc:
                failure_reason = "state_update_failed:{}".format(exc)
                break

            obj_vis, p_vis = item_to_sim(box, sim_object_xy_scale=sim_object_xy_scale)
            env.add_object(obj_vis, p_vis, drop_height=drop_height)
            if load_delay > 0:
                env.run_for_sim_time(load_delay)
            if settle_by_velocity_kwargs is not None:
                settle_status = env.wait_until_quasi_static(**settle_by_velocity_kwargs)
                total_settle_wait_time += settle_status["settle_wait_time"]
                max_linear_speed = max(max_linear_speed, settle_status["max_linear_speed"])
                max_angular_speed = max(max_angular_speed, settle_status["max_angular_speed"])
                if not settle_status["settled"]:
                    settle_timeout_count += 1
            stability = env.check_stability(**stability_kwargs)
            max_position_delta = max(max_position_delta, stability["max_position_delta"])
            max_rotation_delta = max(max_rotation_delta, stability["max_rotation_delta"])
            max_settle_drift = max(max_settle_drift, stability["max_settle_drift"])
            if not stability["stable"]:
                failure_reason = "physics_unstable"
                break
            accepted_items += 1
            accepted_volume += float(box.Dim.Volume)
    finally:
        env.reset()

    return {
        "heuristic": heuristic_name,
        "physics_engine": physics_engine,
        "physics_engine_selection": env.get_physics_engine_selection(),
        "simulation_time_step": env.get_simulation_time_step(),
        "stepping": env.stepping,
        "drop_height": drop_height,
        "load_delay": load_delay,
        "convex_threshold": "" if convex_threshold is None else convex_threshold,
        "sim_object_xy_scale": sim_object_xy_scale,
        "bin_z": container_size[2] / 1000.0,
        "same_object_z": same_object_z,
        "stable_epoch": failure_reason not in ("physics_unstable", "state_update_failed"),
        "failure_reason": failure_reason,
        "attempted_items": attempted_items,
        "accepted_items": accepted_items,
        "accepted_utilization": accepted_volume / float(np.prod(container_size)),
        "elapsed_time": time.time() - t0,
        "mean_action_time_ms": 1000 * total_action_time / attempted_items if attempted_items else 0.0,
        "checked_candidates": total_checked_count,
        "max_position_delta": max_position_delta,
        "max_rotation_delta": max_rotation_delta,
        "max_settle_drift": max_settle_drift,
        "velocity_settle_enabled": settle_by_velocity_kwargs is not None,
        "total_settle_wait_time": total_settle_wait_time,
        "mean_settle_wait_time": total_settle_wait_time / attempted_items if attempted_items else 0.0,
        "settle_timeout_count": settle_timeout_count,
        "max_linear_speed": max_linear_speed,
        "max_angular_speed": max_angular_speed,
    }


def summarize(rows):
    summary = []
    keys = sorted({
        (
            row["heuristic"],
            row["physics_engine"],
            row.get("simulation_time_step", ""),
            row.get("stepping", False),
            row["drop_height"],
            row["load_delay"],
            row["convex_threshold"],
            row["bin_z"],
            row["same_object_z"],
        )
        for row in rows
    })
    for heuristic_name, physics_engine, simulation_time_step, stepping, drop_height, load_delay, convex_threshold, bin_z, same_object_z in keys:
        group = [
            row for row in rows
            if (
                row["heuristic"] == heuristic_name
                and row["physics_engine"] == physics_engine
                and row.get("simulation_time_step", "") == simulation_time_step
                and row.get("stepping", False) == stepping
                and row["drop_height"] == drop_height
                and row["load_delay"] == load_delay
                and row["convex_threshold"] == convex_threshold
                and row["bin_z"] == bin_z
                and row["same_object_z"] == same_object_z
            )
        ]
        accepted_counts = np.array([row["accepted_items"] for row in group], dtype=float)
        stable_count = sum(row["stable_epoch"] for row in group)
        summary.append({
            "heuristic": heuristic_name,
            "physics_engine": physics_engine,
            "simulation_time_step": simulation_time_step,
            "stepping": stepping,
            "drop_height": drop_height,
            "load_delay": load_delay,
            "convex_threshold": convex_threshold,
            "bin_z": bin_z,
            "same_object_z": same_object_z,
            "epochs": len(group),
            "success_rate": stable_count / len(group) if group else 0.0,
            "mean_accepted_items": float(accepted_counts.mean()) if len(accepted_counts) else 0.0,
            "std_accepted_items": float(accepted_counts.std()) if len(accepted_counts) else 0.0,
            "mean_action_time_ms": float(np.mean([row["mean_action_time_ms"] for row in group])) if group else 0.0,
            "mean_accepted_utilization": float(np.mean([row["accepted_utilization"] for row in group])) if group else 0.0,
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-items", type=int, default=500)
    parser.add_argument("--bin-z", type=float, default=1.8)
    parser.add_argument("--same-object-z", action="store_true", default=True)
    parser.add_argument("--variable-object-z", action="store_false", dest="same_object_z")
    parser.add_argument("--remote-api-port", type=int, default=23000)
    parser.add_argument("--engines", nargs="+", default=["bullet_2_83", "ode", "newton"])
    parser.add_argument("--heuristics", nargs="+", default=["convex_hull", "convex_hull_old"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default="stability_experiment_results.csv")
    parser.add_argument("--summary-csv", default="stability_experiment_summary.csv")
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--sample-time", type=float, default=0.1)
    parser.add_argument("--load-delay", type=float, default=0.5)
    parser.add_argument("--drop-heights", nargs="+", type=float, default=[0.005])
    parser.add_argument("--convex-thresholds", nargs="+", type=float, default=[0.3])
    parser.add_argument("--object-mass", type=float, default=0.1)
    parser.add_argument("--sim-object-xy-scale", type=float, default=1.0)
    parser.add_argument("--anchor-step", type=int, default=60)
    parser.add_argument("--simulation-time-step", type=float, default=None)
    parser.add_argument("--stepping", action="store_true")
    parser.add_argument("--position-tolerance", type=float, default=0.01)
    parser.add_argument("--rotation-tolerance", type=float, default=0.05)
    parser.add_argument("--drift-tolerance", type=float, default=0.002)
    parser.add_argument("--settle-by-velocity", action="store_true")
    parser.add_argument("--linear-velocity-threshold", type=float, default=0.01)
    parser.add_argument("--angular-velocity-threshold", type=float, default=0.05)
    parser.add_argument("--settle-stable-duration", type=float, default=0.2)
    parser.add_argument("--max-settle-wait-time", type=float, default=3.0)
    parser.add_argument("--settle-poll-interval", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()

    if args.sim_object_xy_scale <= 0 or args.sim_object_xy_scale > 1:
        raise ValueError("--sim-object-xy-scale must be in the interval (0, 1].")
    unknown_heuristics = sorted(set(args.heuristics) - set(HEURISTICS))
    if unknown_heuristics:
        raise ValueError("Unknown heuristics: {}".format(", ".join(unknown_heuristics)))

    container_size = (600, 600, int(round(args.bin_z * 1000)))
    xy_dims = np.array([120, 180, 240, 300], dtype=np.int32)
    z_dims = np.array([120], dtype=np.int32) if args.same_object_z else xy_dims
    rng = np.random.default_rng(args.seed)
    object_sequences = [
        np.column_stack([
            rng.choice(xy_dims, args.max_items),
            rng.choice(xy_dims, args.max_items),
            rng.choice(z_dims, args.max_items),
        ]).astype(np.int32)
        for _ in range(args.epochs)
    ]

    stability_kwargs = {
        "settle_time": args.settle_time,
        "sample_time": args.sample_time,
        "position_tolerance": args.position_tolerance,
        "rotation_tolerance": args.rotation_tolerance,
        "drift_tolerance": args.drift_tolerance,
    }
    settle_by_velocity_kwargs = None
    if args.settle_by_velocity:
        settle_by_velocity_kwargs = {
            "linear_velocity_threshold": args.linear_velocity_threshold,
            "angular_velocity_threshold": args.angular_velocity_threshold,
            "stable_duration": args.settle_stable_duration,
            "max_wait_time": args.max_settle_wait_time,
            "poll_interval": args.settle_poll_interval,
        }

    rows = []
    for engine in args.engines:
        for drop_height in args.drop_heights:
            env = PackingEnv(
                physics_engine=None if engine == "default" else engine,
                drop_height=drop_height,
                object_mass=args.object_mass,
                port=args.remote_api_port,
                simulation_time_step=args.simulation_time_step,
                stepping=args.stepping,
            )
            try:
                for heuristic_name in args.heuristics:
                    convex_thresholds = args.convex_thresholds if heuristic_name in CONVEX_HEURISTICS else [None]
                    for convex_threshold in convex_thresholds:
                        for epoch, objects in enumerate(object_sequences, start=1):
                            row = run_epoch(
                                objects=objects,
                                heuristic_name=heuristic_name,
                                physics_engine=engine,
                                env=env,
                                drop_height=drop_height,
                                load_delay=args.load_delay,
                                container_size=container_size,
                                same_object_z=args.same_object_z,
                                stability_kwargs=stability_kwargs,
                                settle_by_velocity_kwargs=settle_by_velocity_kwargs,
                                convex_threshold=convex_threshold,
                                sim_object_xy_scale=args.sim_object_xy_scale,
                                seed=args.seed + epoch,
                                anchor_step=args.anchor_step,
                            )
                            row["epoch"] = epoch
                            rows.append(row)
                            threshold_label = "n/a" if convex_threshold is None else "{:.3f}".format(convex_threshold)
                            print(
                                "[{}][{}][threshold={}][drop={:.4f}][delay={:.3f}][dt={}][stepping={}] epoch {}/{}: stable={}, accepted_items={}, reason={}".format(
                                    engine,
                                    heuristic_name,
                                    threshold_label,
                                    drop_height,
                                    args.load_delay,
                                    row["simulation_time_step"],
                                    row["stepping"],
                                    epoch,
                                    args.epochs,
                                    row["stable_epoch"],
                                    row["accepted_items"],
                                    row["failure_reason"],
                                )
                            )
                            if args.save_every > 0 and epoch % args.save_every == 0:
                                write_csv(args.output_csv, rows)
                                write_csv(args.summary_csv, summarize(rows))
            finally:
                env.stop()

    summary = summarize(rows)
    write_csv(args.output_csv, rows)
    write_csv(args.summary_csv, summary)
    for row in summary:
        threshold_label = "n/a" if row["convex_threshold"] == "" else "{:.3f}".format(float(row["convex_threshold"]))
        print(
            "{} / {} / threshold={} / drop={:.4f} / delay={:.3f} / dt={} / stepping={}: success_rate={:.3f}, mean_accepted_items={:.2f}, mean_action_time_ms={:.4f}".format(
                row["physics_engine"],
                row["heuristic"],
                threshold_label,
                row["drop_height"],
                row["load_delay"],
                row.get("simulation_time_step", ""),
                row.get("stepping", False),
                row["success_rate"],
                row["mean_accepted_items"],
                row["mean_action_time_ms"],
            )
        )


if __name__ == "__main__":
    main()
