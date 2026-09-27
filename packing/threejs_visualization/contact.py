"""Serialize loading contact patches and vertex forces as semantic objects."""

from __future__ import annotations

from collections.abc import Sequence
import math
from dataclasses import replace

from packing_env.data_type.support_vis import LoadContactPatchVis
from packing_env.visualization.config import VisualConfig

from .frame import SceneObject


_MIN_VISIBLE_FORCE = 1e-12


def build_contact_objects(
    patches: Sequence[LoadContactPatchVis],
    config: VisualConfig,
    *,
    bearing_capacity: float | None = None,
) -> list[SceneObject]:
    """Build renderer-neutral patch meshes and globally scaled force arrows."""
    _validate_config(config)
    _validate_physical_data(patches, bearing_capacity)
    valid_patches = [patch for patch in patches if len(patch.polygon_xy) >= 3]
    if not valid_patches:
        return []

    pressure_scale = _pressure_scale(valid_patches, config, bearing_capacity)
    objects = _patch_objects(valid_patches, config, pressure_scale)
    max_patches = [p for p in valid_patches if p.independent_interface_max]
    if config.interface_force_view == "lp-solution":
        objects.extend(_force_arrow_objects(max_patches, config))
    else:
        resultants = _interface_max_arrows(
            max_patches, replace(config, load_force_max_arrow_color="#FF0000"))
        vertices = _force_arrow_objects(
            max_patches, replace(config, load_force_max_arrow_color="#0066FF"))
        # Both layers share the same force-to-length mapping.
        max_total = max((a.geometry["force_newtons"] for a in resultants), default=0.0)
        max_vertex = max((a.geometry["force_newtons"] for a in vertices), default=0.0)
        if max_total > 0:
            for arrow in vertices:
                arrow.geometry["display_length"] *= math.sqrt(max_vertex / max_total)
        objects.extend(resultants)
        if config.interface_force_view != "resultant-only":
            objects.extend(vertices)
    ordinary = [p for p in valid_patches if not p.independent_interface_max]
    feasible = [p for p in ordinary if len(p.vertex_forces_at_feasible) == len(p.polygon_xy)]
    bounds = [p for p in ordinary if len(p.vertex_forces_at_feasible) != len(p.polygon_xy)]
    objects.extend(_force_arrow_objects(feasible, config))
    objects.extend(_bound_arrow_objects(bounds, config))
    return objects


def _bound_arrow_objects(patches, config):
    """Render fixed-COM interface extrema, preserving their force/moment witnesses."""
    if config.interface_force_view == "lp-solution":
        return _force_arrow_objects(patches, config)
    scenarios = ("force_min",) if config.force_min_only else ("force_max", "force_min")
    arrows = [arrow for scenario in scenarios
              for arrow in _interface_max_arrows(patches, config, scenario=scenario)]
    if config.interface_force_view != "resultant-only":
        arrows.extend(_force_arrow_objects(patches, config))
    # Resultants and vertex forces from both extrema share one square-root scale.
    if arrows:
        max_force = max(a.geometry["force_newtons"] for a in arrows)
        max_length = max(a.geometry["display_length"] for a in arrows)
        for arrow in arrows:
            arrow.geometry["display_length"] = max_length * math.sqrt(arrow.geometry["force_newtons"] / max_force)
    return arrows


def _pressure_scale(
    patches: Sequence[LoadContactPatchVis],
    config: VisualConfig,
    bearing_capacity: float | None,
) -> float | None:
    if not config.color_contact_pressure:
        return None
    if bearing_capacity is not None and bearing_capacity > 0.0:
        return float(bearing_capacity)
    return max(max(float(patch.pressure) for patch in patches), 1e-12)


def _patch_objects(
    patches: Sequence[LoadContactPatchVis],
    config: VisualConfig,
    pressure_scale: float | None,
) -> list[SceneObject]:
    objects: list[SceneObject] = []
    seen_ids: dict[str, int] = {}
    z_lift = float(config.load_contact_z_lift)
    thickness = float(config.load_contact_thickness)
    opacity = _clamp_unit(float(config.contact_patch_alpha))

    for patch_index, patch in enumerate(patches):
        prefix = f"patches[{patch_index}]"
        polygon = [[float(x), float(y)] for x, y in patch.polygon_xy]
        if pressure_scale is None:
            color = config.load_contact_neutral_color
            color_meaning = "geometry only (neutral)"
        else:
            color = _pressure_color(float(patch.pressure), pressure_scale)
            color_meaning = "max-equivalent average pressure"

        area = float(patch.area)
        if area <= 0.0 and patch.pressure > 0.0:
            area = float(patch.force_max / patch.pressure)
        area = _finite_number(area, f"{prefix}.tooltip.area_mm2")
        z0 = _finite_number(float(patch.z) + z_lift, f"{prefix}.z_range[0]")
        z1 = _finite_number(z0 + thickness, f"{prefix}.z_range[1]")
        patch_key = _patch_key(patch)
        objects.append(
            SceneObject(
                id=_unique_id(f"contact-patch:{patch_key}", seen_ids),
                kind="contact_patch",
                geometry={"polygon_xy": polygon, "z_range": [z0, z1],
                          "lbcp_display_z": float(patch.z) + (float(config.hull_thickness) if patch.z > 0 else 0.0)},
                style={
                    "color": color,
                    "edge_color": "rgba(0,0,0,0.9)",
                    "edge_width_px": 3,
                    "opacity": opacity,
                    **({"pattern": "diagonal_hatch"} if patch.independent_interface_max or patch.sampled_com_feasible else {}),
                },
                tooltip={
                    "type": "contact_patch",
                    "color_meaning": color_meaning,
                    "area_mm2": area,
                    "force_min_n": None if patch.independent_interface_max or patch.sampled_com_feasible else float(patch.force_min),
                    "force_max_n": None if patch.sampled_com_feasible else float(patch.force_max),
                    **({"force_n": float(patch.force),
                        "interpretation": "One globally feasible solution with fixed randomly sampled CoM; not a load bound"}
                       if patch.sampled_com_feasible else {}),
                    ("average_pressure_n_per_mm2" if patch.sampled_com_feasible else "max_equivalent_average_pressure_n_per_mm2"): float(
                        patch.pressure
                    ),
                    "pressure_diagnostic": "not local pressure",
                    **({"interpretation": "Independent interface maximum; extrema need not coexist"}
                       if patch.independent_interface_max else {}),
                },
            )
        )
    return objects


def _interface_max_arrows(patches, config, *, scenario="force_max"):
    """Moment-equivalent resultant of each CI's existing extremizing solution."""
    if not patches:
        return []
    span = max(1.0, *(max(p[axis] for patch in patches for p in patch.polygon_xy)
                      - min(p[axis] for patch in patches for p in patch.polygon_xy)
                      for axis in (0, 1)))
    records = []
    for patch in patches:
        forces = getattr(patch, f"vertex_forces_at_{scenario}")
        if len(forces) != len(patch.polygon_xy):
            raise ValueError("CI resultant requires aligned vertex forces")
        total = math.fsum(forces)
        if total <= _MIN_VISIBLE_FORCE:
            continue
        xy = [math.fsum(p[axis] * f for p, f in zip(patch.polygon_xy, forces)) / total
              for axis in (0, 1)]
        records.append((patch, total, xy))
    scale = max((total for _, total, _ in records), default=1.0)
    width = getattr(config, f"load_{scenario}_arrow_width")
    objects = []
    for index, (patch, total, xy) in enumerate(records):
        physical = [*xy, float(patch.z)]
        origin = [*xy, float(patch.z) + config.load_contact_z_lift + config.load_contact_thickness]
        objects.append(SceneObject(
            id=f"ci-resultant:{scenario}:{_patch_key(patch)}:{index}", kind="force_arrow",
            geometry={
                "scenario": scenario, "representation": "ci_resultant",
                "origin": origin, "physical_origin": physical,
                "lbcp_display_z": float(patch.z) + (config.hull_thickness if patch.z > 0 else 0.0),
                "direction": [0.0, 0.0, 1.0], "force_newtons": total,
                "display_length": span * config.load_force_arrow_max_length_ratio * math.sqrt(total / scale),
                "length_scale": "sqrt",
            },
            style={
                "color": getattr(config, f"load_{scenario}_arrow_color"),
                "shaft_width_px": width,
                "head_length_px": max(16.0, 2.0 * width),
                "head_width_px": max(12.0, 1.5 * width),
                "opacity": _clamp_unit(config.force_arrow_alpha * getattr(config, f"load_{scenario}_arrow_opacity")),
            },
            tooltip={
                "type": "CI resultant", "force_n": total,
                "physical_position_mm": physical,
                f"interface_{scenario}_n": float(getattr(patch, scenario)),
                "interpretation": f"Moment-equivalent resultant at this CI {scenario}; different CI extrema need not coexist",
            },
        ))
    return objects


def _force_arrow_objects(
    patches: Sequence[LoadContactPatchVis],
    config: VisualConfig,
) -> list[SceneObject]:
    feasible_lengths = [
        len(patch.vertex_forces_at_feasible) for patch in patches
    ]
    if all(
        length == len(patch.polygon_xy)
        for patch, length in zip(patches, feasible_lengths)
    ):
        scenarios = (
            (
                "feasible",
                "vertex_forces_at_feasible",
                (config.load_force_max_arrow_color
                 if patches and all(p.sampled_com_feasible for p in patches)
                 else config.load_feasible_arrow_color),
                config.load_feasible_arrow_width,
                config.load_feasible_arrow_opacity,
            ),
        )
    else:
        if any(feasible_lengths):
            raise ValueError("feasible vertex forces do not align with contact patches")
        scenarios = (
            (
                "force_max",
                "vertex_forces_at_force_max",
                config.load_force_max_arrow_color,
                config.load_force_max_arrow_width,
                config.load_force_max_arrow_opacity,
            ),
            (
                "force_min",
                "vertex_forces_at_force_min",
                config.load_force_min_arrow_color,
                config.load_force_min_arrow_width,
                config.load_force_min_arrow_opacity,
            ),
        )
        if patches and all(p.independent_interface_max for p in patches):
            scenarios = scenarios[:1]
        elif config.force_min_only:
            scenarios = scenarios[1:]

    visible_forces = [
        float(force)
        for patch in patches
        for _scenario, attribute, _color, _width, _opacity in scenarios
        if len(getattr(patch, attribute)) == len(patch.polygon_xy)
        for force in getattr(patch, attribute)
        if float(force) > _MIN_VISIBLE_FORCE
    ]
    if not visible_forces:
        return []

    vertices = [
        (float(x), float(y)) for patch in patches for x, y in patch.polygon_xy
    ]
    x_values = [vertex[0] for vertex in vertices]
    y_values = [vertex[1] for vertex in vertices]
    xy_span = _finite_number(
        max(
            max(x_values) - min(x_values),
            max(y_values) - min(y_values),
            1.0,
        ),
        "force_arrow.xy_span",
    )
    max_length = _finite_number(
        xy_span * float(config.load_force_arrow_max_length_ratio),
        "force_arrow.max_length",
    )
    force_scale = max(visible_forces)

    objects: list[SceneObject] = []
    seen_ids: dict[str, int] = {}
    for scenario, attribute, color, width, base_opacity in scenarios:
        width_number = float(width)
        head_length = _finite_number(
            max(16.0, 2.0 * width_number),
            f"{scenario}.head_length_px",
        )
        head_width = _finite_number(
            max(12.0, 1.5 * width_number),
            f"{scenario}.head_width_px",
        )
        shaft_width = int(width)
        opacity = _clamp_unit(
            _finite_number(
                float(base_opacity) * float(config.force_arrow_alpha),
                f"{scenario}.opacity",
            )
        )
        for patch_index, patch in enumerate(patches):
            forces = getattr(patch, attribute)
            if len(forces) != len(patch.polygon_xy):
                continue
            patch_key = _patch_key(patch)
            for vertex_index, ((vertex_x, vertex_y), raw_force) in enumerate(
                zip(patch.polygon_xy, forces)
            ):
                force = float(raw_force)
                if force <= _MIN_VISIBLE_FORCE:
                    continue
                origin = [float(vertex_x), float(vertex_y), float(patch.z)]
                arrow_key = (
                    f"force-arrow:{patch_key}:{scenario}:"
                    f"{_number(vertex_x)}:{_number(vertex_y)}:{vertex_index}"
                )
                objects.append(
                    SceneObject(
                        id=_unique_id(arrow_key, seen_ids),
                        kind="force_arrow",
                        geometry={
                            "scenario": scenario,
                            "origin": origin,
                            "direction": [0.0, 0.0, 1.0],
                            "force_newtons": force,
                            "display_length": _finite_number(
                                max_length * math.sqrt(force / force_scale),
                                (
                                    f"patches[{patch_index}].{attribute}"
                                    f"[{vertex_index}].display_length"
                                ),
                            ),
                        },
                        style={
                            "color": color,
                            "shaft_width_px": shaft_width,
                            "head_length_px": head_length,
                            "head_width_px": head_width,
                            "opacity": opacity,
                        },
                        tooltip={
                            "type": "force_arrow",
                            "length_scale": "Square-root display scaling; force_n is unchanged",
                            "scenario": scenario,
                            "vertex_mm": list(origin),
                            "force_n": force,
                            **({
                                f"interface_{scenario}_n": float(getattr(patch, scenario)),
                                "interpretation": (
                                    "Vertex force in this interface's total-load maximum solution; "
                                    "not individually maximized. Different interface extrema need not coexist"
                                ),
                            } if patch.independent_interface_max else {}),
                        },
                    )
                )
    return objects


def _pressure_color(pressure: float, scale: float) -> str:
    if scale <= 0.0:
        return "rgb(80,120,255)"
    ratio = max(0.0, min(1.0, pressure / scale))
    red = int(40 + 215 * ratio)
    green = int(120 * (1.0 - ratio))
    blue = int(255 * (1.0 - ratio))
    return f"rgb({red},{green},{blue})"


def _patch_key(patch: LoadContactPatchVis) -> str:
    vertices = ":".join(
        f"{_number(x)},{_number(y)}" for x, y in patch.polygon_xy
    )
    return f"{vertices}:z={_number(patch.z)}"


def _unique_id(base: str, seen: dict[str, int]) -> str:
    occurrence = seen.get(base, 0)
    seen[base] = occurrence + 1
    return base if occurrence == 0 else f"{base}:{occurrence}"


def _number(value: object) -> str:
    return float(value).hex()


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validate_physical_data(
    patches: Sequence[LoadContactPatchVis],
    bearing_capacity: float | None,
) -> None:
    if bearing_capacity is not None:
        _finite_number(bearing_capacity, "bearing_capacity")

    for patch_index, patch in enumerate(patches):
        prefix = f"patches[{patch_index}]"
        for field in ("z", "area", "pressure", "force", "force_min", "force_max"):
            _finite_number(getattr(patch, field), f"{prefix}.{field}")
        for vertex_index, vertex in enumerate(patch.polygon_xy):
            for coordinate_index, coordinate in enumerate(vertex):
                _finite_number(
                    coordinate,
                    f"{prefix}.polygon_xy[{vertex_index}][{coordinate_index}]",
                )
        for field in (
            "vertex_forces_at_force_min",
            "vertex_forces_at_force_max",
            "vertex_forces_at_feasible",
        ):
            for force_index, force in enumerate(getattr(patch, field)):
                _finite_number(force, f"{prefix}.{field}[{force_index}]")


def _validate_config(config: VisualConfig) -> None:
    for field in (
        "load_contact_z_lift",
        "load_contact_thickness",
        "contact_patch_alpha",
        "load_force_arrow_max_length_ratio",
        "load_force_min_arrow_opacity",
        "load_force_max_arrow_opacity",
        "load_feasible_arrow_opacity",
        "force_arrow_alpha",
        "load_force_min_arrow_width",
        "load_force_max_arrow_width",
        "load_feasible_arrow_width",
    ):
        _finite_number(getattr(config, field), f"config.{field}")


def _finite_number(value: object, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number
