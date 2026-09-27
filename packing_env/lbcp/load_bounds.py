from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, vstack
from shapely.geometry import Polygon, box as shapely_box
from shapely.geometry.base import BaseGeometry

from packing_env.data_type.item import Item
from packing_env.lbcp.contact_patch import (
    Z_TOLERANCE,
    contacts_vertically,
    item_bottom_z,
)


ItemKey = tuple[int, int, int, int, int, int]


class GlobalLoadStatus(str, Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True)
class SupportInterfaceKey:
    upper_key: ItemKey
    lower_key: ItemKey | None


@dataclass(frozen=True)
class LoadBounds:
    minimum: float | None
    maximum: float


@dataclass(frozen=True)
class SupportInterface:
    key: SupportInterfaceKey
    polygon_xy: tuple[tuple[float, float], ...]
    area: float
    variable_indices: tuple[int, ...]


@dataclass(frozen=True)
class InterfaceVertexForces:
    """Vertex reactions in the two interface-total extremum solutions."""

    at_force_min: tuple[float, ...]
    at_force_max: tuple[float, ...]


@dataclass
class GlobalLoadBoundsResult:
    status: GlobalLoadStatus
    interface_bounds: dict[SupportInterfaceKey, LoadBounds] = field(default_factory=dict)
    item_payload_bounds: dict[ItemKey, LoadBounds] = field(default_factory=dict)
    interfaces: dict[SupportInterfaceKey, SupportInterface] = field(default_factory=dict)
    interface_vertex_forces: dict[
        SupportInterfaceKey, InterfaceVertexForces
    ] = field(default_factory=dict)
    feasible_forces: np.ndarray | None = None
    message: str = ""
    bounds_complete: bool = True
    solve_time_sec: float = 0.0
    linprog_time_sec: float = 0.0
    linprog_calls: int = 0
    interface_count: int = 0
    variable_count: int = 0


@dataclass
class _SolveTimer:
    started_at: float = field(default_factory=lambda: time.perf_counter())
    linprog_time_sec: float = 0.0
    linprog_calls: int = 0
    interface_count: int = 0
    variable_count: int = 0

    def run_linprog(self, *args: Any, **kwargs: Any):
        started_at = time.perf_counter()
        try:
            return linprog(*args, **kwargs)
        finally:
            self.linprog_time_sec += time.perf_counter() - started_at
            self.linprog_calls += 1

    def finish(self, result: GlobalLoadBoundsResult) -> GlobalLoadBoundsResult:
        result.solve_time_sec = time.perf_counter() - self.started_at
        result.linprog_time_sec = self.linprog_time_sec
        result.linprog_calls = self.linprog_calls
        result.interface_count = self.interface_count
        result.variable_count = self.variable_count
        return result


def item_footprint(item: Item) -> Polygon:
    flb = item.True_FLB
    return shapely_box(
        float(flb.x),
        float(flb.y),
        float(flb.x + item.Dim.dx),
        float(flb.y + item.Dim.dy),
    )


def _as_geometry(value: BaseGeometry | Sequence[tuple[float, float]]) -> BaseGeometry:
    if isinstance(value, BaseGeometry):
        return value
    return Polygon(value)


def _effective_region(
    upper: Item,
    lower: Item,
    lbcp_polygons: Mapping[ItemKey, BaseGeometry | Sequence[tuple[float, float]]],
) -> Polygon | None:
    overlap = item_footprint(upper).intersection(item_footprint(lower))
    if overlap.is_empty or overlap.area <= 0:
        return None

    lower_lbcp = lbcp_polygons.get(lower.to_key())
    if lower_lbcp is None:
        if item_bottom_z(lower) > Z_TOLERANCE:
            return None
        lower_geometry = item_footprint(lower)
    else:
        lower_geometry = _as_geometry(lower_lbcp)

    region = overlap.intersection(lower_geometry)
    if region.is_empty or region.area <= 0:
        return None
    hull = region.convex_hull
    if not isinstance(hull, Polygon) or hull.area <= 0:
        return None
    return hull


def _polygon_vertices(polygon: Polygon) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in polygon.exterior.coords[:-1])


def _solver_status(status: int) -> GlobalLoadStatus:
    if status == 0:
        return GlobalLoadStatus.OPTIMAL
    if status == 2:
        return GlobalLoadStatus.INFEASIBLE
    if status == 3:
        return GlobalLoadStatus.UNBOUNDED
    return GlobalLoadStatus.NUMERICAL_FAILURE


def _clean_value(value: float, tolerance: float = 1e-8) -> float:
    value = float(value)
    return 0.0 if abs(value) <= tolerance else value


def solve_global_load_bounds(
    items: Sequence[Item],
    *,
    lbcp_polygons: Mapping[
        ItemKey,
        BaseGeometry | Sequence[tuple[float, float]],
    ],
    material_density: float,
    gravity: float = 9.81,
    payload_max_only: bool = False,
    feasibility_only: bool = False,
    interface_bounds_only: bool = False,
    uncertain_com: bool = False,
    interface_max_only: bool = False,
    fixed_com_xy: Mapping[ItemKey, tuple[float, float]] | None = None,
) -> GlobalLoadBoundsResult:
    """Solve all admissible vertical contact loads for one packing state."""
    if interface_max_only and (payload_max_only or feasibility_only):
        raise ValueError("interface_max_only cannot be combined with payload_max_only or feasibility_only")
    timer = _SolveTimer()
    items = list(items)
    if fixed_com_xy is not None:
        if uncertain_com:
            raise ValueError("fixed_com_xy and uncertain_com are mutually exclusive")
        if set(fixed_com_xy) != {item.to_key() for item in items}:
            raise ValueError("fixed_com_xy must specify every item exactly once")
        if any(len(point) != 2 or not all(np.isfinite(v) for v in point)
               for point in fixed_com_xy.values()):
            raise ValueError("fixed_com_xy must contain finite XY coordinates")
    if not items:
        return timer.finish(GlobalLoadBoundsResult(status=GlobalLoadStatus.OPTIMAL))

    item_by_key = {item.to_key(): item for item in items}
    if len(item_by_key) != len(items):
        return timer.finish(
            GlobalLoadBoundsResult(
                status=GlobalLoadStatus.NUMERICAL_FAILURE,
                message="items must have unique geometry keys",
            )
        )

    raw_interfaces: list[
        tuple[SupportInterfaceKey, Polygon, tuple[tuple[float, float], ...]]
    ] = []
    for upper in items:
        if item_bottom_z(upper) <= Z_TOLERANCE:
            ground_region = item_footprint(upper)
            raw_interfaces.append(
                (
                    SupportInterfaceKey(upper.to_key(), None),
                    ground_region,
                    _polygon_vertices(ground_region),
                )
            )

        for lower in items:
            if lower is upper or not contacts_vertically(upper, lower):
                continue
            region = _effective_region(upper, lower, lbcp_polygons)
            if region is None:
                continue
            raw_interfaces.append(
                (
                    SupportInterfaceKey(upper.to_key(), lower.to_key()),
                    region,
                    _polygon_vertices(region),
                )
            )

    variable_count = sum(len(vertices) for _key, _polygon, vertices in raw_interfaces)
    timer.interface_count = len(raw_interfaces)
    timer.variable_count = variable_count
    if variable_count == 0:
        return timer.finish(
            GlobalLoadBoundsResult(
                status=GlobalLoadStatus.INFEASIBLE,
                message="packing has no active support interfaces",
            )
        )

    item_row = {key: 3 * idx for idx, key in enumerate(item_by_key)}
    matrix = lil_matrix((3 * len(items), variable_count), dtype=float)
    rhs = np.zeros(3 * len(items), dtype=float)
    for key, item in item_by_key.items():
        row = item_row[key]
        rhs[row] = float(material_density) * float(item.Dim.Volume) * float(gravity)
        if fixed_com_xy is not None:
            cx, cy = fixed_com_xy[key]
            rhs[row + 1] = rhs[row] * (cy - (item.True_FLB.y + item.Dim.dy / 2.0))
            rhs[row + 2] = -rhs[row] * (cx - (item.True_FLB.x + item.Dim.dx / 2.0))

    interfaces: dict[SupportInterfaceKey, SupportInterface] = {}
    next_variable = 0
    for key, polygon, vertices in raw_interfaces:
        indices = tuple(range(next_variable, next_variable + len(vertices)))
        next_variable += len(vertices)
        interfaces[key] = SupportInterface(
            key=key,
            polygon_xy=vertices,
            area=float(polygon.area),
            variable_indices=indices,
        )

        upper = item_by_key[key.upper_key]
        upper_row = item_row[key.upper_key]
        upper_com_x = float(upper.True_FLB.x + upper.Dim.dx / 2.0)
        upper_com_y = float(upper.True_FLB.y + upper.Dim.dy / 2.0)
        for variable, (x, y) in zip(indices, vertices):
            matrix[upper_row, variable] += 1.0
            matrix[upper_row + 1, variable] += y - upper_com_y
            matrix[upper_row + 2, variable] += -(x - upper_com_x)

            if key.lower_key is not None:
                lower = item_by_key[key.lower_key]
                lower_row = item_row[key.lower_key]
                lower_com_x = float(lower.True_FLB.x + lower.Dim.dx / 2.0)
                lower_com_y = float(lower.True_FLB.y + lower.Dim.dy / 2.0)
                matrix[lower_row, variable] -= 1.0
                matrix[lower_row + 1, variable] -= y - lower_com_y
                matrix[lower_row + 2, variable] += x - lower_com_x

    matrix = matrix.tocsr()
    constraints = dict(A_eq=matrix, b_eq=rhs)
    if uncertain_com:
        # COM rectangle centered at geometric center: +/-10% of dx/dy.
        # Moment rows are about that fixed center (y moment, negative x moment).
        force_rows = np.arange(0, 3 * len(items), 3)
        moment_rows = np.array([r + k for r in force_rows for k in (1, 2)])
        limits = np.array([
            rhs[3 * idx] * dimension * 0.1
            for idx, item in enumerate(items)
            for dimension in (item.Dim.dy, item.Dim.dx)
        ])
        moments = matrix[moment_rows]
        constraints = dict(A_eq=matrix[force_rows], b_eq=rhs[force_rows],
                           A_ub=vstack([moments, -moments], format="csr"),
                           b_ub=np.concatenate([limits, limits]))
    variable_bounds = [(0.0, None)] * variable_count
    feasibility = timer.run_linprog(
        np.zeros(variable_count, dtype=float),
        **constraints,
        bounds=variable_bounds,
        method="highs",
    )
    status = _solver_status(feasibility.status)
    if status is not GlobalLoadStatus.OPTIMAL:
        return timer.finish(
            GlobalLoadBoundsResult(
                status=status,
                interfaces=interfaces,
                message=str(feasibility.message),
            )
        )

    if feasibility_only:
        return timer.finish(
            GlobalLoadBoundsResult(
                status=GlobalLoadStatus.OPTIMAL,
                interfaces=interfaces,
                feasible_forces=np.asarray(feasibility.x, dtype=float),
                bounds_complete=False,
            )
        )

    def objective_bounds(
        selector: np.ndarray,
    ) -> tuple[
        LoadBounds | None,
        GlobalLoadStatus,
        str,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        if not np.any(selector):
            feasible_x = np.asarray(feasibility.x, dtype=float)
            return (
                LoadBounds(0.0, 0.0),
                GlobalLoadStatus.OPTIMAL,
                "",
                feasible_x,
                feasible_x,
            )
        lower = timer.run_linprog(
            selector,
            **constraints,
            bounds=variable_bounds,
            method="highs",
        )
        lower_status = _solver_status(lower.status)
        if lower_status is not GlobalLoadStatus.OPTIMAL:
            return None, lower_status, str(lower.message), None, None
        upper = timer.run_linprog(
            -selector,
            **constraints,
            bounds=variable_bounds,
            method="highs",
        )
        upper_status = _solver_status(upper.status)
        if upper_status is not GlobalLoadStatus.OPTIMAL:
            return None, upper_status, str(upper.message), None, None
        return (
            LoadBounds(
                minimum=_clean_value(lower.fun),
                maximum=_clean_value(-upper.fun),
            ),
            GlobalLoadStatus.OPTIMAL,
            "",
            np.asarray(lower.x, dtype=float),
            np.asarray(upper.x, dtype=float),
        )

    interface_bounds: dict[SupportInterfaceKey, LoadBounds] = {}
    interface_vertex_forces: dict[
        SupportInterfaceKey, InterfaceVertexForces
    ] = {}
    if not payload_max_only:
        for key, interface in interfaces.items():
            selector = np.zeros(variable_count, dtype=float)
            selector[list(interface.variable_indices)] = 1.0
            if interface_max_only:
                maximum = timer.run_linprog(-selector, **constraints,
                                           bounds=variable_bounds, method="highs")
                status = _solver_status(maximum.status)
                if status is not GlobalLoadStatus.OPTIMAL:
                    return timer.finish(GlobalLoadBoundsResult(
                        status=status, interfaces=interfaces, message=str(maximum.message),
                        feasible_forces=np.asarray(feasibility.x, dtype=float),
                        bounds_complete=False))
                interface_bounds[key] = LoadBounds(None, _clean_value(-maximum.fun))
                # Preserve the existing maximum solution for visualization, without
                # another solve or inventing an uncomputed minimum solution.
                interface_vertex_forces[key] = InterfaceVertexForces(
                    at_force_min=(),
                    at_force_max=tuple(
                        _clean_value(maximum.x[index])
                        for index in interface.variable_indices
                    ),
                )
                continue
            bounds, bound_status, message, force_min_solution, force_max_solution = (
                objective_bounds(selector)
            )
            if bounds is None:
                return timer.finish(
                    GlobalLoadBoundsResult(
                        status=bound_status,
                        interfaces=interfaces,
                        feasible_forces=np.asarray(feasibility.x, dtype=float),
                        message=message,
                    )
                )
            interface_bounds[key] = bounds
            assert force_min_solution is not None
            assert force_max_solution is not None
            interface_vertex_forces[key] = InterfaceVertexForces(
                at_force_min=tuple(
                    _clean_value(force_min_solution[index])
                    for index in interface.variable_indices
                ),
                at_force_max=tuple(
                    _clean_value(force_max_solution[index])
                    for index in interface.variable_indices
                ),
            )

    if interface_bounds_only or interface_max_only:
        return timer.finish(
            GlobalLoadBoundsResult(
                status=GlobalLoadStatus.OPTIMAL,
                interface_bounds=interface_bounds,
                interfaces=interfaces,
                interface_vertex_forces=interface_vertex_forces,
                feasible_forces=np.asarray(feasibility.x, dtype=float),
                bounds_complete=False,
            )
        )

    item_payload_bounds: dict[ItemKey, LoadBounds] = {}
    for item_key in item_by_key:
        selector = np.zeros(variable_count, dtype=float)
        for key, interface in interfaces.items():
            if key.lower_key == item_key:
                selector[list(interface.variable_indices)] = 1.0
        if payload_max_only:
            if not np.any(selector):
                bounds = LoadBounds(minimum=None, maximum=0.0)
                bound_status = GlobalLoadStatus.OPTIMAL
                message = ""
            else:
                upper = timer.run_linprog(
                    -selector,
                    **constraints,
                    bounds=variable_bounds,
                    method="highs",
                )
                bound_status = _solver_status(upper.status)
                message = str(upper.message)
                bounds = (
                    LoadBounds(minimum=None, maximum=_clean_value(-upper.fun))
                    if bound_status is GlobalLoadStatus.OPTIMAL
                    else None
                )
        else:
            bounds, bound_status, message, _minimum_solution, _maximum_solution = (
                objective_bounds(selector)
            )
        if bounds is None:
            return timer.finish(
                GlobalLoadBoundsResult(
                    status=bound_status,
                    interfaces=interfaces,
                    interface_bounds=interface_bounds,
                    feasible_forces=np.asarray(feasibility.x, dtype=float),
                    message=message,
                )
            )
        item_payload_bounds[item_key] = bounds

    return timer.finish(
        GlobalLoadBoundsResult(
            status=GlobalLoadStatus.OPTIMAL,
            interface_bounds=interface_bounds,
            item_payload_bounds=item_payload_bounds,
            interfaces=interfaces,
            interface_vertex_forces=interface_vertex_forces,
            feasible_forces=np.asarray(feasibility.x, dtype=float),
            bounds_complete=not payload_max_only,
        )
    )
