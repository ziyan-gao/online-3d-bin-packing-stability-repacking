from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat

import numpy as np

RESOLUTION = 10
GRID_SIZE = 10
GRID_HEIGHT = 1
DX = 1200
DY = 1000
DZ = 1350
offset = 0.0



@dataclass
class Point2D:
    """Class representing a front left bottom 3D point"""

    x: int=0 #in mm
    y: int=0 #in mm
    # resolution:float = RESOLUTION

    def __post_init__(self):
        self.resolution = RESOLUTION
        checkArr = list(map(self.full_divisible, [self.x, self.y], repeat(self.resolution)))
        if (~np.array(checkArr)).any():
            raise Exception("inputs are illegal.")
        self.Hx, self.Hy = self.topix()

    def topix(self):
        Hx = self.x // self.resolution
        Hy = self.y // self.resolution
        return np.array([Hx, Hy]).astype(np.int32)

    def numpy(self):
        return np.array([self.x, self.y]).astype(np.int32)

    def full_divisible(self, big_number, scale:int|None = None):
        if scale is None:
            scale = self.resolution
        res = big_number % scale
        if res == 0:
            return True
        else:
            return False



@dataclass
class Point3D(Point2D):
    """Class representing a front left bottom 3D point"""

    z: int=0 # in mm

    def __post_init__(self):
        self.resolution = RESOLUTION
        self.grid_size = GRID_SIZE
        self.grid_height = GRID_HEIGHT
        super().__post_init__()
        checkArr = list(map(self.full_divisible, 
                            [self.x, self.y, self.z], 
                            [GRID_SIZE, GRID_SIZE, GRID_HEIGHT]))
        if (~np.array(checkArr)).any():
            raise Exception("inputs are illegal.")
        self.Hx, self.Hy = self.topix()
        self.Gx, self.Gy, self.Gz = self.togrid()
        self.p2d = Point2D(self.x, self.y)
    
    def __eq__(self,other:Point3D):
        return (self.Gx == other.Gx) & (self.Gy ==other.Gy) & (self.Gz == other.Gz)

    def __hash__(self):
        return hash((self.Gx, self.Gy, self.Gz))

    def __add__(self, other: Point3D):
        if isinstance(other, int):
            return Point3D(self.x + other, self.y + other, self.z + other)
        else:
            return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def togrid(self):
        Gx = self.x // self.grid_size
        Gy = self.y // self.grid_size
        Gz = self.z // self.grid_height
        return np.array([Gx, Gy, Gz]).astype(np.int32)

    def numpy(self):
        return np.array([self.x, self.y, self.z]).astype(np.int32)



@dataclass
class Orthogonal2D:
    dx: int=DX
    dy: int=DY

    def __post_init__(self):
        self.resolution = RESOLUTION
        self.grid_size = GRID_SIZE
        self.grid_height = GRID_HEIGHT
        self.ratio = int(self.grid_size / self.resolution)
        # self.ratio = int(self.grid_size / self.resolution)
        checkArr = list(map(self.full_divisible, [self.dx, self.dy], repeat(self.resolution)))
        if (~np.array(checkArr)).any():
            raise Exception("inputs are illegal.")
        self.Hdx, self.Hdy = self.topix()

    @property
    def coord2d(self):
        coord = np.array([[0, 0],
                          [0, self.Hdy],
                          [self.Hdx, self.Hdy], 
                          [self.Hdx, 0]]).astype(np.int32)
        return coord

    def numpy(self):
        return np.array([self.Hdx, self.Hdy])

    def topix(self):
        Hx = self.dx // self.resolution
        Hy = self.dy // self.resolution
        return np.array([Hx, Hy]).astype(np.int32)

    def full_divisible(self, big_number, scale:int|None = None):
        if scale is None:
            scale = self.resolution
        res = big_number % scale
        if res == 0:
            return True
        else:
            return False

    @property
    def Area(self):
        # in grid
        return self.Hdx * self.Hdy



@dataclass
class Orthogonal3D(Orthogonal2D):
    """Class representing a orthogonal space in 3D"""

    dz: int=DZ
    offset:int = offset

    def __hash__(self):
        return hash((self.dx, self.dy, self.dz))
    
    def __eq__(self, other):
        if not isinstance(other, Orthogonal3D):
            return False
        return (self.dx == other.dx and self.dy == other.dy and 
                self.dz == other.dz)
    
    def __repr__(self):
        return f"Box({self.dx},{self.dy},{self.dz})"

    def __post_init__(self):
        super().__post_init__()
        # if self.ratio < 1:
        #     raise Exception("grid size should be greater than resolution.")
        checkArr = list(map(self.full_divisible, [self.dx, self.dy, self.dz],
                            [self.grid_size, self.grid_size, self.grid_height]))
        if (~np.array(checkArr)).any():
            raise Exception("inputs are illegal.")
        self.Gdx, self.Gdy, self.Gdz = self.togrid()
        self.o2d = Orthogonal2D(self.dx, self.dy)
        # if self.offset:
        #     H_offset = int(self.offset / self.resolution)
        #     self.Hdx = self.Hdx - H_offset
        #     self.Hdy = self.Hdy - H_offset

    def raw(self):
        return np.array([self.dx, self.dy, self.dz]).astype(np.int32)

    def to_dim_key(self) -> tuple[int, int, int]:
        return (int(self.dx), int(self.dy), int(self.dz))
    
    def togrid(self):
        Gdx = self.dx // self.grid_size
        Gdy = self.dy // self.grid_size
        Gdz = self.dz // self.grid_height
        return np.array([Gdx, Gdy, Gdz]).astype(np.int32)


    def numpy(self):
        return np.array([self.Gdx, self.Gdy, self.Gdz]).astype(np.int32)


    @property
    def Volume(self):
        # in grid
        return self.dx * self.dy * self.dz



@dataclass
class Rectangle:
    """Class representing a cuboid with position and dimensions"""

    FL: Point2D
    O2d: Orthogonal2D

    def __post_init__(self):
        self.Hx = self.FL.Hx
        self.Hy = self.FL.Hy
        self.Hdx = self.O2d.Hdx
        self.Hdy = self.O2d.Hdy

__all__ = [
    "DX",
    "DY",
    "DZ",
    "GRID_HEIGHT",
    "GRID_SIZE",
    "RESOLUTION",
    "Point2D",
    "Point3D",
    "Orthogonal2D",
    "Orthogonal3D",
    "Rectangle",
]
