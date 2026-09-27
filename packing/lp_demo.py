"""Run a deterministic LP/Three.js demo: python -m packing.lp_demo --help."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item
from packing_env.gym_env import PackingEnv
from packing_env.lbcp.load_bounds import GlobalLoadStatus
from packing_env.visualization.config import VisualConfig

from .lp_analysis import build_lp_frame
from .threejs_visualization import ThreeLiveServer, ThreeReplayRecorder


def run_demo(
    output: str | Path,
    *,
    scene: str = "bridge",
    material_density: float = 1e-6,
    force_view: str = "resultant",
    live_server: ThreeLiveServer | None = None,
):
    """Export each committed placement with LP contact and load overlays."""
    if scene == "stack":
        layout = [(0, 0, z, 100, 100, 50) for z in (0, 50, 100)]
    elif scene == "bridge":
        layout = [
            (0, 0, 0, 100, 100, 50),
            (200, 0, 0, 100, 100, 50),
            (0, 0, 50, 300, 100, 50),
        ]
    else:
        raise ValueError(f"Unknown demo scene: {scene}")
    if force_view not in {"resultant", "lp-solution", "resultant-only"}:
        raise ValueError(f"Unknown force view: {force_view}")
    env = PackingEnv(container_size=(400, 300, 300))
    env.reset(seed=41)
    recorder = ThreeReplayRecorder(output)
    config = VisualConfig(interface_force_view=force_view)
    for step, (x, y, z, dx, dy, dz) in enumerate(layout, start=1):
        env.pack(Item(FLB=Point3D(x, y, z), Dim=Orthogonal3D(dx, dy, dz)))
        result, frame = build_lp_frame(
            env, f"{scene.title()} — placement {step}",
            material_density=material_density, config=config,
        )
        if result.status is not GlobalLoadStatus.OPTIMAL:
            raise RuntimeError(f"Demo LP failed: {result.status.value}: {result.message}")
        recorder.capture(frame.title, frame)
        if live_server is not None:
            live_server.push(frame)
    return result, recorder.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stack", "bridge"), default="bridge")
    parser.add_argument("--output", type=Path, default=Path("outputs/plotly_live/lp_demo"))
    parser.add_argument("--density", type=float, default=1e-6, help="uniform material density in kg/mm^3")
    parser.add_argument("--force-view", choices=("resultant", "lp-solution", "resultant-only"), default="resultant")
    parser.add_argument("--live", action="store_true", help="serve interactive frames on localhost until Ctrl+C")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    server = ThreeLiveServer(args.output, port=args.port) if args.live else None
    try:
        result, path = run_demo(
            args.output, scene=args.scene, material_density=args.density,
            force_view=args.force_view, live_server=server,
        )
        print(f"LP: {result.status.value}; {result.interface_count} interfaces, {result.variable_count} variables")
        print(f"Open replay: {path.resolve()}")
        if server is not None:
            print(f"Live viewer: {server.start()} (Ctrl+C to stop)", flush=True)
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
