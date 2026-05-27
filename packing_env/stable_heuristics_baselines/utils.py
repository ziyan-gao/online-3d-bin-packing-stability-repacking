from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item
from packing_env.data_type.maps import HeightMap


def build_height_bound_candidates(
    o3d: Orthogonal3D,
    hm: HeightMap,
    anchor_step: int = 60,
) -> np.ndarray:
    """Return feasible top-left height-map coordinates for an item.

    Coordinates are height-map pixels `(Hx, Hy)`. `anchor_step` is in the
    project distance unit (mm), matching the 60 mm placement grid used by the
    original convex-hull experiments.
    """

    if o3d.Hdx > hm.Hdx or o3d.Hdy > hm.Hdy:
        return np.zeros((0, 2), dtype=np.int32)

    windows = sliding_window_view(hm.Value, (o3d.Hdx, o3d.Hdy))
    height_ok = windows.max(axis=(-2, -1)) + o3d.dz <= hm.zmax
    coords = np.argwhere(height_ok)
    if len(coords) == 0:
        return coords.astype(np.int32)

    if anchor_step > 0:
        anchor_pixels = max(1, int(round(anchor_step / hm.resolution)))
        coords = coords[
            (coords[:, 0] % anchor_pixels == 0)
            & (coords[:, 1] % anchor_pixels == 0)
        ]
    return coords.astype(np.int32)


def make_item(o3d: Orthogonal3D, coord: Point3D, buffer_space: int = 0) -> Item:
    return Item(FLB=coord, Dim=o3d, buffer_space=buffer_space)
