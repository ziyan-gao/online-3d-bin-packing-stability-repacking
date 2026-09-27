"""Global LP analysis and Three.js load frames for an existing packing environment."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING

from packing_env.data_type.support_vis import LoadContactPatchVis
from packing_env.lbcp.contact_patch import item_top_z
from packing_env.lbcp.load_bounds import (
    GlobalLoadBoundsResult,
    GlobalLoadStatus,
    solve_global_load_bounds,
)
from packing_env.visualization.config import VisualConfig

from .threejs_visualization import FocusLoadingOverlay, ThreeVisualizationBuilder, VisualizationFrame

if TYPE_CHECKING:
    from packing_env.gym_env import PackingEnv


def solve_packing_loads(
    env: PackingEnv,
    *,
    material_density: float,
    gravity: float = 9.81,
) -> GlobalLoadBoundsResult:
    """Recompute fixed-COM bounds using the environment's current cached LBCP.

    Geometry is in mm, density in kg/mm³, gravity in m/s², forces in N.
    This analyzes the current state; it does not modify candidate filtering.
    """
    for name, value in (("material_density", material_density), ("gravity", gravity)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    items = list(env.container.placed_items)
    polygons = env.heu_stable.lbcp_polygon_map()
    missing = {item.to_key() for item in items} - polygons.keys()
    if missing:
        raise ValueError(
            f"Missing cached LBCP for {len(missing)} placed item(s); "
            "construct the state through env.pack() so support geometry is recorded."
        )
    return solve_global_load_bounds(
        items,
        lbcp_polygons=polygons,
        material_density=material_density,
        gravity=gravity,
    )


def build_lp_frame(
    env: PackingEnv,
    title: str = "Global load LP",
    *,
    material_density: float,
    gravity: float = 9.81,
    config: VisualConfig | None = None,
) -> tuple[GlobalLoadBoundsResult, VisualizationFrame]:
    """Analyze a state and show all contacts, including ground reaction forces.

    Each interface's minimum and maximum is a separate optimization, so extrema
    shown on different interfaces need not occur in the same equilibrium state.
    """
    result = solve_packing_loads(env, material_density=material_density, gravity=gravity)
    items = {item.to_key(): item for item in env.container.placed_items}
    patches = []
    if result.status is GlobalLoadStatus.OPTIMAL:
        for key, interface in result.interfaces.items():
            bounds = result.interface_bounds[key]
            forces = result.interface_vertex_forces[key]
            lower = items.get(key.lower_key)
            patches.append(LoadContactPatchVis(
                polygon_xy=interface.polygon_xy,
                z=item_top_z(lower) if lower is not None else 0.0,
                force=bounds.maximum,
                pressure=bounds.maximum / interface.area,
                force_min=bounds.minimum,
                force_max=bounds.maximum,
                area=interface.area,
                vertex_forces_at_force_min=forces.at_force_min,
                vertex_forces_at_force_max=forces.at_force_max,
            ))
    overlay = FocusLoadingOverlay(
        focus_key=None,
        related_keys=frozenset(items),
        floor_keys=frozenset(
            key.upper_key for key in result.interfaces if key.lower_key is None
        ),
        patches=tuple(patches),
    )
    frame = ThreeVisualizationBuilder(config or VisualConfig()).build(
        env,
        f"{title} | LP: {result.status.value} | Independent interface bounds",
        focus_overlay=overlay,
        show_anchor=False,
    )
    visual_config = config or VisualConfig()
    representation = "vertex forces" if visual_config.interface_force_view == "lp-solution" else "contact resultants"
    if visual_config.interface_force_view == "resultant":
        representation += " and vertex forces"
    legend = [entry for entry in frame.legend if entry["kind"] != "feasible"]
    if patches:
        for scenario, color in (
            ("force_min", visual_config.load_force_min_arrow_color),
            ("force_max", visual_config.load_force_max_arrow_color),
        ):
            if visual_config.force_min_only and scenario == "force_max":
                continue
            legend.append({"kind": scenario,
                           "label": f"{scenario.replace('_', ' ').title()}: {representation} (independent extrema; square-root scale)",
                           "color": color})
    frame = replace(frame, legend=legend, metrics={
        **frame.metrics,
        "lp_status": result.status.value,
        "lp_message": result.message,
        "lp_interfaces": result.interface_count,
        "lp_variables": result.variable_count,
        "lp_solve_ms": result.solve_time_sec * 1000,
        "material_density_kg_per_mm3": material_density,
        "gravity_m_per_s2": gravity,
    })
    return result, frame
