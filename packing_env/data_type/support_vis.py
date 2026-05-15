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

