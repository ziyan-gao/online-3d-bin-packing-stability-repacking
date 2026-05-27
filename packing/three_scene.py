from __future__ import annotations

from collections import Counter
from typing import Any

from packing_env.data_type.item import Item


PALETTE = (
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
)

EMS_PALETTE = (
    "#636efa",
    "#ef553b",
    "#00cc96",
    "#ab63fa",
    "#ffa15a",
    "#19d3f3",
    "#ff6692",
    "#b6e880",
    "#ff97ff",
    "#fecb52",
)


def build_three_scene(
    env,
    title: str,
    *,
    show_ems: bool = False,
    ems_mode: str = "raw",
    visual_z_max: float | None = None,
) -> dict[str, Any]:
    """Build a compact renderer-neutral payload for the Three.js visualizer."""
    container = env.container
    container_size = [
        float(container.dx),
        float(container.dy),
        float(visual_z_max if visual_z_max is not None else container.dz),
    ]
    shown_ems = _shown_ems(env, show_ems=show_ems, ems_mode=ems_mode)
    anchor_points = _anchor_points(env, shown_ems=shown_ems, ems_mode=ems_mode)
    buffer_items = list(env.buffer.items)
    holding_items = list(container.holding_list)
    selected_block = getattr(env, "visual_selected_block", None)

    return {
        "title": title,
        "container": container_size,
        "placed": [_box_payload(item, _item_color(env, item), opacity=1.0) for item in container.placed_items],
        "ems": [
            _box_payload(ems, EMS_PALETTE[index % len(EMS_PALETTE)], opacity=0.14, wire=True)
            for index, ems in enumerate(shown_ems)
        ],
        "anchors": [_point_payload(point) for point in anchor_points],
        "buffer": [
            _linear_box_payload(
                box,
                index,
                _dim_color(env, _dims(box)),
                highlighted=index in set(getattr(env.buffer, "visual_highlight_indices", []) or []),
            )
            for index, box in enumerate(buffer_items)
        ],
        "holding": [
            _linear_box_payload(box, index, _dim_color(env, _dims(box)), highlighted=False)
            for index, box in enumerate(holding_items)
        ],
        "legend": _legend(env, buffer_items, holding_items, selected_block, len(shown_ems), show_ems),
    }


def _shown_ems(env, *, show_ems: bool, ems_mode: str) -> list:
    if not show_ems:
        return []
    if ems_mode == "policy":
        return list(getattr(env, "ems_list", None) or [])
    return list(env.heu_ems.get_ems_list())


def _anchor_points(env, *, shown_ems: list, ems_mode: str) -> list:
    if ems_mode != "policy":
        return list(env.heu_ems.get_anchor_points())
    points = []
    seen = set()
    for ems in shown_ems:
        key = (ems.FLB.Gx, ems.FLB.Gy, ems.FLB.Gz)
        if key not in seen:
            points.append(ems.FLB)
            seen.add(key)
    return points


def _box_payload(box, color: str, *, opacity: float = 1.0, wire: bool = False) -> dict[str, Any]:
    flb = box.True_FLB
    dx, dy, dz = _dims(box)
    return {
        "x": float(flb.x),
        "y": float(flb.y),
        "z": float(flb.z),
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "color": color,
        "opacity": float(opacity),
        "wire": bool(wire),
    }


def _linear_box_payload(box, index: int, color: str, *, highlighted: bool) -> dict[str, Any]:
    dx, dy, dz = _dims(box)
    return {
        "index": int(index),
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "color": color,
        "highlighted": bool(highlighted),
    }


def _point_payload(point) -> dict[str, float]:
    return {"x": float(point.x), "y": float(point.y), "z": float(point.z)}


def _dims(box) -> tuple[int, int, int]:
    if hasattr(box, "Dim"):
        return int(box.Dim.dx), int(box.Dim.dy), int(box.Dim.dz)
    return int(box.dx), int(box.dy), int(box.dz)


def _item_color(env, item: Item) -> str:
    return _dim_color(env, _dims(item))


def _dim_color(env, dims: tuple[int, int, int]) -> str:
    sampler = getattr(getattr(env, "buffer", None), "data_sampler", None)
    if sampler is not None:
        try:
            color = sampler.get_color(dims)
            return _color_to_hex(color)
        except Exception:
            pass
    return PALETTE[hash(dims) % len(PALETTE)]


def _color_to_hex(color) -> str:
    values = list(color[:3])
    if not any(value > 1.0 for value in values):
        values = [round(value * 255) for value in values]
    values = [max(0, min(255, int(value))) for value in values]
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def _legend(env, buffer_items: list, holding_items: list, selected_block, shown_ems_count: int, show_ems: bool) -> list[str]:
    lines = ["Item Types:"]
    for dims, count in sorted(Counter(_dims(item) for item in buffer_items).items()):
        lines.append(f"{dims[0]}x{dims[1]}x{dims[2]}: {count} pcs")
    if holding_items:
        lines.append("Staging:")
        for dims, count in sorted(Counter(_dims(item) for item in holding_items).items()):
            lines.append(f"{dims[0]}x{dims[1]}x{dims[2]}: {count} pcs")
    if selected_block is not None:
        dx, dy, dz = _dims(selected_block)
        count = getattr(selected_block, "consumed_count", 1)
        lines.append(f"Selected block: {dx}x{dy}x{dz} ({count} boxes)")
    lines.append(f"Utilization: {env.container.utilization * 100:.1f}%")
    total_ems = len(env.heu_ems.get_all_ems())
    lines.append(f"EMS: {total_ems} total, {shown_ems_count if show_ems else 0} shown")
    return lines
