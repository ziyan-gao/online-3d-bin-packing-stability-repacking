from __future__ import annotations

from typing import List

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from shapely.geometry import Point, Polygon

from packing_env.data_type.geometry import Orthogonal3D, Point3D, Rectangle
from packing_env.data_type.item import Item
from packing_env.data_type.maps import HeightMap, Map
from packing_env.heu_stable import Heu_Stable


class ConvexHullBaseline(Heu_Stable):
    """Current convex-hull stability heuristic, exported with baseline naming."""


class ConvexHullOldBaseline(Map):
    """Legacy convex-hull baseline with the old feasible-map update rule."""

    def __init__(self, dx: int = 600, dy: int = 600):
        super().__init__(dx=dx, dy=dy)
        self.prev_cache: dict[Item, np.ndarray] = {}

    def reset(self) -> None:
        self.Value = self.Value * 0
        self.prev_cache = {}

    @property
    def prevCache(self):
        return self.prev_cache

    @prevCache.setter
    def prevCache(self, value):
        self.prev_cache = value

    @staticmethod
    def compute_hull_from_support(support_points: np.ndarray) -> Polygon | bool:
        if support_points.shape[0] <= 2:
            return False
        support_augment = (
            support_points.tolist()
            + (support_points + np.ones_like(support_points)).tolist()
            + (support_points + np.ones_like(support_points) * np.array([[0, 1]])).tolist()
            + (support_points + np.ones_like(support_points) * np.array([[1, 0]])).tolist()
        )
        try:
            hull = cv2.convexHull(np.array(support_augment))
            return Polygon(hull.reshape((hull.shape[0], 2)))
        except Exception:
            return False

    @staticmethod
    def get_com_bound(o3d: Orthogonal3D, scale: float = 0.2) -> Polygon:
        coord2d = scale * (o3d.coord2d.copy() - o3d.o2d.numpy() / 2) + o3d.o2d.numpy() / 2
        return Polygon(coord2d)

    def _convex_hull_validate(
        self,
        o3d: Orthogonal3D,
        point: tuple[int, int],
        hm_window: np.ndarray,
        fm_window: np.ndarray,
        scale: float = 0.2,
    ) -> tuple[bool, Point3D | None]:
        support = np.argwhere(np.abs((hm_window * (fm_window == 0)) - hm_window.max()) == 0)
        polygon = self.compute_hull_from_support(support)
        if not isinstance(polygon, Polygon):
            return False, None
        if polygon.contains(self.get_com_bound(o3d, scale)):
            return True, Point3D(
                int(point[0] * self.resolution),
                int(point[1] * self.resolution),
                int(hm_window.max()),
            )
        return False, None

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

        feasible_map_windows = self.sliding_window_view(o3d)
        height_map_windows = hm.sliding_window_view(o3d)
        for x, y in candidates:
            stable, coord = self._convex_hull_validate(
                o3d,
                (int(x), int(y)),
                height_map_windows[x][y],
                feasible_map_windows[x][y],
                scale=scale,
            )
            flags.append(stable)
            stable_coords.append(coord)
        return stable_coords, np.array(flags, dtype=bool)

    @staticmethod
    def _box_roi(box: Item) -> Rectangle:
        return Rectangle(box.True_FLB.p2d, box.Dim.o2d)

    def update(self, hm: HeightMap, box: Item) -> None:
        roi = self._box_roi(box)
        self.prev_cache[box] = self.slice(roi).copy()
        window = hm.slice(roi)
        h_max = window.max()
        value_map = h_max - window
        self.Value = self.with_updated_roi(roi, value=value_map + self.slice(roi))

    def unpack(self, box: Item) -> None:
        roi = self._box_roi(box)
        prev_value = self.prev_cache.get(box)
        if prev_value is not None:
            self.Value[roi.Hx : roi.Hx + roi.Hdx, roi.Hy : roi.Hy + roi.Hdy] = prev_value
        self.prev_cache.pop(box, None)


class ConvexHullPlainBaseline(ConvexHullOldBaseline):
    """Convex-hull support check without feasible-map history."""

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
        fm_window = np.zeros((o3d.Hdx, o3d.Hdy))
        for x, y in candidates:
            stable, coord = self._convex_hull_validate(
                o3d,
                (int(x), int(y)),
                height_map_windows[x][y],
                fm_window,
                scale=scale,
            )
            flags.append(stable)
            stable_coords.append(coord)
        return stable_coords, np.array(flags, dtype=bool)

    def update(self, hm: HeightMap, box: Item) -> None:
        return None

    def unpack(self, box: Item) -> None:
        return None

    def reset(self) -> None:
        return None
