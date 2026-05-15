from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .geometry import DZ, Orthogonal2D, Point2D, Point3D, Rectangle
from .item import Item



@dataclass
class Map(Orthogonal2D):

    def __post_init__(self):
        super().__post_init__()
        self.Value = np.zeros((self.Hdx, self.Hdy))

    def slice(self, rect: Rectangle):
        return self.Value[rect.Hx : rect.Hx + rect.Hdx, 
                          rect.Hy : rect.Hy + rect.Hdy]
    
    def with_updated_roi(self, rect:Rectangle, value):
        Value_ = self.Value.copy()
        Value_[rect.Hx : rect.Hx + rect.Hdx,
               rect.Hy : rect.Hy + rect.Hdy] = value
        return Value_
    
    def sliding_window_view(self, o2d: Orthogonal2D):
        return sliding_window_view(self.Value, (o2d.Hdx, o2d.Hdy))

    def reset(self):
        self.Value = self.Value * 0



@dataclass
class HeightMap(Map):
    zmax: float = DZ

    def __post_init__(self):
        super().__post_init__()
        self.Value = np.zeros((self.Hdx, self.Hdy))
        self.prev_cache = {}

    def compute_flb(self, obj: np.ndarray, pxy: np.ndarray):
        o2d = Orthogonal2D(*obj[:2])
        p2d = Point2D(*pxy)
        rect = Rectangle(p2d, o2d)
        hmax = self.slice(rect).max()
        flb = Point3D(*pxy, hmax)
        return rect, flb

    def update(self, box:Item):
        obj, p=box.Dim, box.True_FLB
        self.prev_cache[box] = self.Value[p.Hx: p.Hx + obj.Hdx,
                                         p.Hy: p.Hy + obj.Hdy].copy()
        self.Value[p.Hx: p.Hx + obj.Hdx,
                   p.Hy: p.Hy + obj.Hdy] = p.z + obj.dz
    
    def unpack(self, box:Item):
        prev_value = self.prev_cache[box]
        obj, p=box.Dim, box.True_FLB
        self.Value[p.Hx: p.Hx + obj.Hdx,
                   p.Hy: p.Hy + obj.Hdy] = prev_value
        self.prev_cache.pop(box, None)

    def reset(self):
        self.Value = self.Value * 0
        self.prev_cache = {}

    @property
    def prevCache(self):
        return self.prev_cache

    @prevCache.setter
    def prevCache(self, value):
        self.prev_cache = value

__all__ = ["Map", "HeightMap"]
