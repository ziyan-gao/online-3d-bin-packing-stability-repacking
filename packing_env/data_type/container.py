from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time

import numpy as np

from .geometry import Orthogonal3D, Point3D
from .item import Item

container_timing = defaultdict(list)



@dataclass
class Container(Orthogonal3D):
    empty_value: int = 0
    occupied_value: int = 1
    waste_value: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        self.state = np.ones((self.Gdx, self.Gdy, self.Gdz)) * self.empty_value
        self.state_vis = np.ones((self.Gdx, self.Gdy, self.Gdz)) * self.empty_value
        self.placed_items: list[Item] = []
        self.holding_list: list[Item] = []
        self.bound_x = Item(Point3D(), Orthogonal3D(6000, 0, 6000))
        self.bound_y = Item(Point3D(), Orthogonal3D(0, 6000, 6000))
        self.bound_z = Item(Point3D(), Orthogonal3D(6000, 6000, 0))
        self.ground = Item(Point3D(), Orthogonal3D(6000,     6000, 0))
        self.prev_cache: dict[Item, np.ndarray] = {}
    
    @property
    def number_of_placed_items(self) -> int:
        return len(self.placed_items)

    def _update(self, box: Item) -> None:
        obj = box.Virtual_Dim
        p = box.FLB
        start = time.perf_counter()
        self.state[
            p.Gx : p.Gx + obj.Gdx,
            p.Gy : p.Gy + obj.Gdy,
            p.Gz : p.Gz + obj.Gdz,
        ] = self.occupied_value
        container_timing["_update:occupied"].append(time.perf_counter() - start)
        
        start = time.perf_counter()
        self.state_vis[
            p.Gx : p.Gx + obj.Gdx,
            p.Gy : p.Gy + obj.Gdy,
            p.Gz : p.Gz + obj.Gdz,
        ] = np.random.randn()
        container_timing["_update:statevis"].append(time.perf_counter() - start)

        start = time.perf_counter()
        waste_idxs = np.where(
            self.state[p.Gx : p.Gx + obj.Gdx, p.Gy : p.Gy + obj.Gdy, : p.Gz]
            == self.empty_value
        )
        self.state[p.Gx : p.Gx + obj.Gdx, p.Gy : p.Gy + obj.Gdy, : p.Gz][
            waste_idxs
        ] = self.waste_value
        container_timing["_update:waste"].append(time.perf_counter() - start)

    def add(self, box: Item) -> None:
        obj = box.Virtual_Dim
        p = box.FLB
        if obj.Gdx + p.Gx > self.Gdx:
            raise ValueError("Exceeding the boundary along x axis")
        if obj.Gdy + p.Gy > self.Gdy:
            raise ValueError("Exceeding the boundary along y axis")
        if obj.Gdz + p.Gz > self.Gdz:
            raise ValueError("Exceeding the boundary along z axis")

        start = time.perf_counter()
        state_crop = self.state[p.Gx:p.Gx + obj.Gdx,
                                p.Gy:p.Gy + obj.Gdy,
                                p.Gz:]
        container_timing["Add:state_crop"].append(time.perf_counter() - start)

        start = time.perf_counter()
        penetration_check = (state_crop == self.empty_value).all()
        container_timing["Add:penetration_check"].append(time.perf_counter() - start)

        if not penetration_check:
            raise RuntimeError("Penetration happened.")
        else:
            start = time.perf_counter()
            if p.Gz == 0:
                self.ground.children.append(box)
            self.prev_cache[box] = self.state[
                p.Gx : p.Gx + obj.Gdx,
                p.Gy : p.Gy + obj.Gdy,
                :,
            ].copy()
            self._update(box)
            container_timing["Add:_update"].append(time.perf_counter() - start)

            start = time.perf_counter()
            for item in self.placed_items:
                if item.is_below(box):
                    item.children.append(box)
            container_timing["Add:parent_loop"].append(time.perf_counter() - start)

        start = time.perf_counter()
        self.placed_items.append(box)
        container_timing["Add:append"].append(time.perf_counter() - start)

    def get_parents(self, query_item: Item) -> list[Item]:
        parents: list[Item] = []
        for item in self.placed_items:
            if query_item in item.children:
                parents.append(item)
        return parents

    def find_matching_item(
        self,
        query: Item,
        items: list[Item] | None = None,
    ) -> Item | None:
        if items is None:
            items = self.placed_items
        for item in items:
            if item == query:
                return item
        return None
                
    def is_placeable(self, box: Item) -> bool:
        obj = box.Virtual_Dim
        p = box.FLB
        return (self.state[
            p.Gx : p.Gx + obj.Gdx,
            p.Gy : p.Gy + obj.Gdy,
            p.Gz : p.Gz + obj.Gdz,
        ] == self.empty_value).all()
        
    def unpack(self, item_unpack: Item) -> None:
        prev_state = self.prev_cache[item_unpack]
        obj = item_unpack.Virtual_Dim
        p = item_unpack.FLB
        self.placed_items.remove(item_unpack)
        self.state[p.Gx:p.Gx + obj.Gdx,
                    p.Gy:p.Gy + obj.Gdy,
                    :] = prev_state
        self.state_vis[p.Gx:p.Gx + obj.Gdx,
                        p.Gy:p.Gy + obj.Gdy,
                        p.Gz:p.Gz + obj.Gdz] = self.empty_value
        for item in self.placed_items:
            if item_unpack in item.children:
                item.children.remove(item_unpack)
        if p.Gz==0:
            self.ground.children.remove(item_unpack)
        self.prev_cache.pop(item_unpack, None)
        self.holding_list.append(item_unpack)
    
    def include(self, other: Orthogonal3D) -> bool:
        return (
            (self.Gdx >= other.Gdx)
            and (self.Gdy >= other.Gdy)
            and (self.Gdz >= other.Gdz))

    @property
    def unpackable_boxes(self) -> list[Item]:
        boxes: list[Item] = []
        for item in self.placed_items:
            if len(item.children) == 0:
                boxes.append(item)
        return boxes

    def clear(self) -> None:
        self.state = np.ones((self.Gdx, self.Gdy, self.Gdz)) * self.empty_value
        self.state_vis = np.ones_like(self.state) * self.empty_value
        self.placed_items = []
        self.holding_list = []
        self.prev_cache = {}
        self.ground = Item(Point3D(), Orthogonal3D(6000,     6000, 0))

    @property
    def utilization(self) -> float:
        return sum(self.placed_items) / self.Volume

__all__ = ["Container", "container_timing"]
