from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


PolygonXY = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class SupportVisData:
    """Visualization payload for a placed item's support geometry."""

    support_polygon_xy: PolygonXY
    support_z0: float
    support_z1: float
    virtual_item_polygon_xy: PolygonXY



@dataclass(frozen=True)
class LoadContactPatchVis:
    """Bottom contact patch colored by transferred load."""

    polygon_xy: PolygonXY
    z: float
    force: float
    pressure: float
    force_min: float = 0.0
    force_max: float = 0.0
    area: float = 0.0
    vertex_forces_at_force_min: tuple[float, ...] = ()
    vertex_forces_at_force_max: tuple[float, ...] = ()
    vertex_forces_at_feasible: tuple[float, ...] = ()
    independent_interface_max: bool = False
    sampled_com_feasible: bool = False
