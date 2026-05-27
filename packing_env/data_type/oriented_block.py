from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Orthogonal3D, Point3D
from .item import Item, SimpleBlock


@dataclass(frozen=True)
class OrientedBlock:
    block: SimpleBlock
    source_index: int
    rotate_xy: bool

    @classmethod
    def from_block(
        cls,
        block: SimpleBlock,
        *,
        source_index: int,
        rotate_xy: bool,
    ) -> "OrientedBlock":
        return cls(
            block=block,
            source_index=int(source_index),
            rotate_xy=bool(rotate_xy),
        )

    @property
    def oriented_block(self) -> SimpleBlock:
        return self.block.rotated(self.rotate_xy)

    @property
    def Dim(self) -> Orthogonal3D:
        return self.oriented_block.Dim

    @property
    def Virtual_Dim(self) -> Orthogonal3D:
        return self.oriented_block.Virtual_Dim

    @property
    def consumed_count(self) -> int:
        return self.oriented_block.consumed_count

    @property
    def buffer_space(self) -> int:
        return self.oriented_block.buffer_space

    def to_item(self, flb: Point3D) -> Item:
        placed = Item(FLB=flb, Dim=self.Dim, buffer_space=self.buffer_space)
        placed.rot = self.rotate_xy
        return placed

    def feature_row(
        self,
        container_size: tuple[int, int, int],
        max_stack_count: int = 12,
    ) -> np.ndarray:
        container = np.asarray(container_size, dtype=np.float32)
        stack_scale = max(float(max_stack_count), 1.0)
        return np.asarray(
            [
                *(self.Dim.raw().astype(np.float32) / container),
                *(self.Virtual_Dim.raw().astype(np.float32) / container),
                self.consumed_count / stack_scale,
                float(self.rotate_xy),
            ],
            dtype=np.float32,
        )


__all__ = ["OrientedBlock"]
