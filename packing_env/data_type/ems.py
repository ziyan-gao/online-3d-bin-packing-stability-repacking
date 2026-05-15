from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import DX, DY, DZ, Orthogonal3D
from .item import Item



@dataclass
class EmptyMaximalSpace(Item):
    """Class representing an EMS in 3D space, measured in Grid unit"""

    def __post_init__(self):
        return super().__post_init__()

    @property
    def Volume(self):
        return self.dx * self.dy * self.dz

    def __lt__(self, other: EmptyMaximalSpace):
        return self.Volume < other.Volume

    def __le__(self, other: EmptyMaximalSpace):
        return self.Volume <= other.Volume

    def __eq__(self, other: EmptyMaximalSpace):
        return (
            (self.Gx == other.Gx)
            and (self.Gy == other.Gy)
            and (self.Gz == other.Gz)
            and (self.Gdx == other.Gdx)
            and (self.Gdy == other.Gdy)
            and (self.Gdz == other.Gdz)
        )
    

    def numpy(self, normalize=True):
        if not normalize:
            return np.array(
                [
                    self.FLB.x,
                    self.FLB.y,
                    self.FLB.z,
                    self.Dim.dx,
                    self.Dim.dy,
                    self.Dim.dz
                ]
            ).astype(np.int32)
        else:
            return np.array(
                [
                    self.FLB.x/DX,
                    self.FLB.y/DY,
                    self.FLB.z/DZ,
                    (self.Dim.dx)/DX,
                    (self.Dim.dy)/DY,
                    (self.Dim.dz)/DZ
                ]
            ).astype(np.int32)
   

    def include(self, other:Orthogonal3D):
        return (
            (self.Gdx >= other.Gdx)
            and (self.Gdy >= other.Gdy)
            and (self.Gdz >= other.Gdz))

__all__ = ["EmptyMaximalSpace"]
