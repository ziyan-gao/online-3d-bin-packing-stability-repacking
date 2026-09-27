from __future__ import annotations

from collections.abc import Mapping, Sequence

from shapely.geometry.base import BaseGeometry

from packing_env.data_type.item import Item
from packing_env.data_type.support_vis import LoadContactPatchVis
from packing_env.lbcp.contact_patch import item_top_z
from packing_env.lbcp.load_bounds import GlobalLoadStatus, ItemKey, item_footprint
from packing_env.lbcp.load_tracking import is_on_floor
from packing_env.lbcp.support_graph import collect_support_chain
from packing_env.lbcp.validator import LBCPConfig, LBCPValidator


def uncertain_com_max_patches(placed_items, cached, *, floor_contact_only=True):
    """Reuse the selected-state solve; never replace it with fixed-COM extrema."""
    if cached is None:
        return []
    keys, result = cached
    if keys != frozenset(item.to_key() for item in placed_items):
        return []  # Initial/replayed/different state: do not display stale forces.
    if result.status is not GlobalLoadStatus.OPTIMAL:
        return []
    items = {item.to_key(): item for item in placed_items}
    patches = []
    for key, interface in result.interfaces.items():
        lower = items.get(key.lower_key)
        if floor_contact_only and lower is not None and not is_on_floor(lower):
            continue
        maximum = result.interface_bounds[key].maximum
        forces = result.interface_vertex_forces[key].at_force_max
        if len(forces) != len(interface.polygon_xy):
            raise ValueError(f"vertex forces do not align with contact interface {key}")
        patches.append(LoadContactPatchVis(
            polygon_xy=interface.polygon_xy, z=item_top_z(lower) if lower else 0.0,
            force=maximum, pressure=maximum / interface.area,
            force_max=maximum, area=interface.area, independent_interface_max=True,
            vertex_forces_at_force_max=forces))
    return patches


def related_item_keys(focus_item: Item, placed_items: list[Item]) -> set[tuple]:
    return {item.to_key() for item in collect_support_chain(focus_item, placed_items)}


def floor_item_keys(placed_items: list[Item]) -> set[tuple]:
    return {item.to_key() for item in placed_items if is_on_floor(item)}


def _compute_contact_loads(
    placed_items: list[Item],
    *,
    material_density: float,
    bearing_capacity: float | None = None,
    gravity: float = 9.81,
    validator: LBCPValidator | None = None,
    floor_contact_only: bool = True,
    lbcp_polygons: Mapping[
        ItemKey, BaseGeometry | Sequence[tuple[float, float]]
    ]
    | None = None,
) -> list[LoadContactPatchVis]:
    """Contact regions colored by conservative global LP load bounds."""
    del bearing_capacity
    expected_item_keys = {item.to_key() for item in placed_items}
    result = validator.last_result if validator is not None else None
    if (
        result is None
        or result.status is not GlobalLoadStatus.OPTIMAL
        or set(result.item_payload_bounds) != expected_item_keys
    ):
        fallback = LBCPValidator(
            LBCPConfig(
                material_density=material_density,
                payload_capacity=float("inf"),
                gravity=gravity,
            )
        )
        result = fallback.rebuild_state_from_placed(
            placed_items,
            lbcp_polygons=(
                lbcp_polygons
                if lbcp_polygons is not None
                else {
                    item.to_key(): item_footprint(item) for item in placed_items
                }
            ),
        )

    if result.status is not GlobalLoadStatus.OPTIMAL:
        return []

    item_by_key = {item.to_key(): item for item in placed_items}
    contact_patches: list[LoadContactPatchVis] = []
    for key, interface in result.interfaces.items():
        if key.lower_key is None:
            continue
        lower = item_by_key[key.lower_key]
        if floor_contact_only and not is_on_floor(lower):
            continue
        bounds = result.interface_bounds[key]
        if interface.area <= 0:
            continue
        vertex_forces = result.interface_vertex_forces.get(key)
        if vertex_forces is None:
            forces_at_min: tuple[float, ...] = ()
            forces_at_max: tuple[float, ...] = ()
        else:
            forces_at_min = vertex_forces.at_force_min
            forces_at_max = vertex_forces.at_force_max
            vertex_count = len(interface.polygon_xy)
            if (
                len(forces_at_min) != vertex_count
                or len(forces_at_max) != vertex_count
            ):
                raise ValueError(
                    "vertex forces do not align with contact interface "
                    f"{key}: expected {vertex_count}, got "
                    f"{len(forces_at_min)} and {len(forces_at_max)}"
                )
        feasible_forces = result.feasible_forces
        if feasible_forces is None:
            forces_at_feasible: tuple[float, ...] = ()
        else:
            try:
                forces_at_feasible = tuple(
                    float(feasible_forces[index])
                    for index in interface.variable_indices
                )
            except IndexError as exc:
                raise ValueError(
                    "feasible forces do not align with contact interface "
                    f"{key}"
                ) from exc
        contact_patches.append(
            LoadContactPatchVis(
                polygon_xy=interface.polygon_xy,
                z=item_top_z(lower),
                force=float(bounds.maximum),
                pressure=float(bounds.maximum / interface.area),
                force_min=float(bounds.minimum),
                force_max=float(bounds.maximum),
                area=float(interface.area),
                vertex_forces_at_force_min=forces_at_min,
                vertex_forces_at_force_max=forces_at_max,
                vertex_forces_at_feasible=forces_at_feasible,
            )
        )

    return contact_patches


def compute_all_floor_contact_loads(
    placed_items: list[Item],
    *,
    material_density: float,
    bearing_capacity: float | None = None,
    gravity: float = 9.81,
    validator: LBCPValidator | None = None,
) -> list[LoadContactPatchVis]:
    """All floor contact regions with accumulated load from region state."""
    return _compute_contact_loads(
        placed_items,
        material_density=material_density,
        bearing_capacity=bearing_capacity,
        gravity=gravity,
        validator=validator,
        floor_contact_only=True,
    )


def compute_all_contact_loads(
    placed_items: list[Item],
    *,
    material_density: float,
    bearing_capacity: float | None = None,
    gravity: float = 9.81,
    validator: LBCPValidator | None = None,
) -> list[LoadContactPatchVis]:
    """All item-to-item contact regions with accumulated load from region state."""
    return _compute_contact_loads(
        placed_items,
        material_density=material_density,
        bearing_capacity=bearing_capacity,
        gravity=gravity,
        validator=validator,
        floor_contact_only=False,
    )


def compute_focus_loading_vis(
    focus_item: Item | None,
    placed_items: list[Item],
    *,
    material_density: float,
    bearing_capacity: float | None = None,
    gravity: float = 9.81,
    validator: LBCPValidator | None = None,
    floor_contact_only: bool = True,
    lbcp_polygons: Mapping[
        ItemKey, BaseGeometry | Sequence[tuple[float, float]]
    ]
    | None = None,
) -> tuple[set[tuple] | None, tuple | None, set[tuple], list[LoadContactPatchVis]]:
    if focus_item is None:
        related_keys = None
        focus_key = None
    else:
        related_keys = related_item_keys(focus_item, placed_items)
        focus_key = focus_item.to_key()

    floor_keys = floor_item_keys(placed_items)
    contact_patches = _compute_contact_loads(
        placed_items,
        material_density=material_density,
        bearing_capacity=bearing_capacity,
        gravity=gravity,
        validator=validator,
        floor_contact_only=floor_contact_only,
        lbcp_polygons=lbcp_polygons,
    )
    return related_keys, focus_key, floor_keys, contact_patches


def compute_bearing_contact_loads(
    focus_item: Item,
    placed_items: list[Item],
    *,
    material_density: float,
    bearing_capacity: float | None = None,
    gravity: float = 9.81,
    validator: LBCPValidator | None = None,
    floor_contact_only: bool = True,
) -> list[LoadContactPatchVis]:
    del focus_item
    return _compute_contact_loads(
        placed_items,
        material_density=material_density,
        bearing_capacity=bearing_capacity,
        gravity=gravity,
        validator=validator,
        floor_contact_only=floor_contact_only,
    )


def compute_floor_contact_loads(
    focus_item: Item,
    placed_items: list[Item],
    *,
    material_density: float,
    bearing_capacity: float | None = None,
    gravity: float = 9.81,
    validator: LBCPValidator | None = None,
) -> list[LoadContactPatchVis]:
    del focus_item
    return compute_all_floor_contact_loads(
        placed_items,
        material_density=material_density,
        bearing_capacity=bearing_capacity,
        gravity=gravity,
        validator=validator,
    )
