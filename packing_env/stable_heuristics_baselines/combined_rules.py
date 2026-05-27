from __future__ import annotations

from typing import List

import numpy as np

from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item
from packing_env.data_type.maps import HeightMap

from .convex_hull import ConvexHullPlainBaseline


class CombinedRulesBaseline(ConvexHullPlainBaseline):
    """Area/corner support rule baseline.

    The default rule set matches the previous area-corner support experiment:
    enough support area plus enough supported corners is accepted as stable.
    """

    def __init__(
        self,
        dx: int = 600,
        dy: int = 600,
        z_tolerance: float = 1e-6,
        area_corner_rules=((0.60, 4), (0.80, 3), (0.95, 0)),
    ):
        super().__init__(dx=dx, dy=dy)
        self.z_tolerance = z_tolerance
        self.area_corner_rules = tuple(area_corner_rules)

    def is_supported(self, window: np.ndarray) -> bool:
        support_mask = np.abs(window - window.max()) <= self.z_tolerance
        support_area_ratio = support_mask.sum() / support_mask.size
        corner_count = sum(
            bool(corner)
            for corner in (
                support_mask[0, 0],
                support_mask[0, -1],
                support_mask[-1, 0],
                support_mask[-1, -1],
            )
        )
        for area_threshold, required_corners in self.area_corner_rules:
            if support_area_ratio >= area_threshold and corner_count >= required_corners:
                return True
        return False

    def __call__(
        self,
        o3d: Orthogonal3D,
        hm: HeightMap,
        candidates: np.ndarray,
        scale: float = 0.2,
    ) -> tuple[List[Point3D], np.ndarray]:
        stable_coords: list[Point3D | None] = []
        flags: list[bool] = []
        if len(candidates) == 0:
            return [], np.zeros((0,), dtype=bool)

        height_map_windows = hm.sliding_window_view(o3d)
        for x, y in candidates:
            window = height_map_windows[x][y]
            stable = self.is_supported(window)
            flags.append(stable)
            stable_coords.append(
                Point3D(int(x * hm.resolution), int(y * hm.resolution), int(window.max()))
                if stable
                else None
            )
        return stable_coords, np.array(flags, dtype=bool)

    def update(self, hm: HeightMap, box: Item) -> None:
        return None
