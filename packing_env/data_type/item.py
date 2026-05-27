from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import box as shapely_box

from .geometry import Orthogonal3D, Point3D



@dataclass
class Item:
    """Class representing a cuboid with position and dimensions"""

    FLB: Point3D
    Dim: Orthogonal3D
    Gcx: float = None
    Gcy: float = None
    buffer_space: int = 0

    def __post_init__(self):
        self.buffer_space = int(self.buffer_space)
        if self.buffer_space < 0:
            raise ValueError("buffer_space must be non-negative.")
        self.Gx = self.FLB.Gx
        self.Gy = self.FLB.Gy
        self.Gz = self.FLB.Gz
        self.Gdx = self.Dim.Gdx
        self.Gdy = self.Dim.Gdy
        self.Gdz = self.Dim.Gdz
        if self.Gcx is None:
            self.Gcx = self.True_FLB.Gx + self.Dim.Gdx / 2
        if self.Gcy is None:
            self.Gcy = self.True_FLB.Gy + self.Dim.Gdy / 2
        self.children = []
        self.rot = False
        # self.parents = []

    @property
    def Virtual_Dim(self) -> Orthogonal3D:
        return Orthogonal3D(
            self.Dim.dx + self.buffer_space,
            self.Dim.dy + self.buffer_space,
            self.Dim.dz,
        )

    @property
    def True_FLB(self) -> Point3D:
        return self.FLB

    def __repr__(self):
        return (
            f"Item(pos=({self.FLB.x},{self.FLB.y},{self.FLB.z}), "
            f"dim=({self.Dim.dx},{self.Dim.dy},{self.Dim.dz}))"
        )
    
    def __eq__(self, other:Item):
        return (
            (self.Gx == other.Gx)
            and (self.Gy == other.Gy)
            and (self.Gz == other.Gz)
            and (self.Gdx == other.Gdx)
            and (self.Gdy == other.Gdy)
            and (self.Gdz == other.Gdz)
            and (self.buffer_space == other.buffer_space)
        )

    def __hash__(self):
        return hash(
            (
                self.Gx,
                self.Gy,
                self.Gz,
                self.Gdx,
                self.Gdy,
                self.Gdz,
                self.buffer_space,
            )
        )

    def to_key(self) -> tuple[int, int, int, int, int, int]:
        return (
            int(self.Gx),
            int(self.Gy),
            int(self.Gz),
            int(self.Gdx),
            int(self.Gdy),
            int(self.Gdz),
        )

    def to_dim_key(self) -> tuple[int, int, int]:
        return self.Dim.to_dim_key()

    def __add__(self, other: Item):
        if isinstance(other, int):
            return self.Dim.Volume + other
        else:
            return self.Dim.Volume + other.Dim.Volume
    
    def with_offset(self, offset: Point3D | tuple) -> 'Item':
        """Create a new Item with FLB offset by the given coordinates.
        
        Args:
            offset: Point3D or tuple (x, y, z) to offset the FLB position
        
        Returns:
            A new Item with the offset FLB position
        """
        if isinstance(offset, tuple):
            offset = Point3D(*offset)
        new_flb = self.FLB + offset
        return Item(FLB=new_flb, Dim=self.Dim, Gcx=None, Gcy=None, buffer_space=self.buffer_space)

    # Define the reverse addition for compatibility with sum()
    def __radd__(self, other: Item):
        if other == 0:
            # If other is 0, return self to start the sum with the correct type
            return self.Dim.Volume
        return self.__add__(other)

    def __lt__(self, other):
        """Define less than based on volume."""
        if not isinstance(other, Item):
            return NotImplemented
        return self.Dim.Volume < other.Dim.Volume

    def is_below(self, other: Item):
        self_flb = self.True_FLB
        other_flb = other.True_FLB
        if other_flb.Gz >= self_flb.Gz + self.Dim.Gdz:
            box1 = shapely_box(other_flb.Gx,
                       other_flb.Gy,
                       other_flb.Gx + other.Dim.Gdx,
                       other_flb.Gy + other.Dim.Gdy)
            box2 = shapely_box(self_flb.Gx,
                       self_flb.Gy,
                       self_flb.Gx + self.Dim.Gdx,
                       self_flb.Gy + self.Dim.Gdy)
            if box1.intersection(box2).area > 0:
                return True
            else:
                return False
        
        else:
            return False

    def is_overlap(self, other: Item) -> bool:
        return (
            self.FLB.Gx < other.FLB.Gx + other.Virtual_Dim.Gdx
            and self.FLB.Gx + self.Virtual_Dim.Gdx > other.FLB.Gx
            and self.FLB.Gy < other.FLB.Gy + other.Virtual_Dim.Gdy
            and self.FLB.Gy + self.Virtual_Dim.Gdy > other.FLB.Gy
            and self.FLB.Gz < other.FLB.Gz + other.Virtual_Dim.Gdz
            and self.FLB.Gz + self.Virtual_Dim.Gdz > other.FLB.Gz
        )
    
    
    
    @property
    def coord3d(self):
        flb = self.True_FLB
        coords = np.array([
            [flb.Gx,              flb.Gy,              flb.Gz],
            [flb.Gx+self.Dim.Gdx, flb.Gy,              flb.Gz],
            [flb.Gx+self.Dim.Gdx, flb.Gy+self.Dim.Gdy, flb.Gz],
            [flb.Gx,              flb.Gy+self.Dim.Gdy, flb.Gz],
            [flb.Gx,              flb.Gy,              flb.Gz+self.Dim.Gdz],
            [flb.Gx+self.Dim.Gdx, flb.Gy,              flb.Gz+self.Dim.Gdz],
            [flb.Gx+self.Dim.Gdx, flb.Gy+self.Dim.Gdy, flb.Gz+self.Dim.Gdz],
            [flb.Gx,              flb.Gy+self.Dim.Gdy, flb.Gz+self.Dim.Gdz],
        ]).astype(np.int32)
        return coords

    def cm(self):
        return np.array([self.Gcx, self.Gcy])

    def gc(self):  # Projection of Geometrical center on XY plane
        flb = self.True_FLB
        return np.array([flb.Gx + self.Dim.Gdx / 2, flb.Gy + self.Dim.Gdy / 2])

class SimpleBlock:
    """Placeable block candidate generated from buffered box dimensions."""

    def __init__(
        self,
        Dim: Orthogonal3D | None = None,
        *,
        box: Orthogonal3D | None = None,
        stack_dims: tuple[int, int, int] = (1, 1, 1),
        buffer_space: int = 0,
        FLB: Point3D | None = None,
        rot: bool = False,
        preserve_order: bool = False,
    ) -> None:
        if Dim is None and box is None:
            raise ValueError("SimpleBlock requires either Dim or box.")
        if Dim is not None and box is not None:
            raise ValueError("SimpleBlock accepts only one of Dim or box.")
        self.box = Dim if Dim is not None else box
        self.stack_dims = tuple(int(value) for value in stack_dims)
        if len(self.stack_dims) != 3 or any(value <= 0 for value in self.stack_dims):
            raise ValueError("stack_dims must contain three positive integers.")
        self.buffer_space = int(buffer_space)
        if self.buffer_space < 0:
            raise ValueError("buffer_space must be non-negative.")
        self.FLB = FLB
        self.rot = bool(rot)
        self.preserve_order = bool(preserve_order)

    @property
    def Dim(self) -> Orthogonal3D:
        nx, ny, nz = self.stack_dims
        return Orthogonal3D(self.box.dx * nx, self.box.dy * ny, self.box.dz * nz)

    @property
    def no_boxes_wrt_axis(self) -> tuple[int, int, int]:
        return self.stack_dims

    @property
    def consumed_count(self) -> int:
        nx, ny, nz = self.stack_dims
        return nx * ny * nz

    @property
    def dx(self) -> int:
        return self.Dim.dx

    @property
    def dy(self) -> int:
        return self.Dim.dy

    @property
    def dz(self) -> int:
        return self.Dim.dz

    @property
    def Gdx(self) -> int:
        return self.Dim.Gdx

    @property
    def Gdy(self) -> int:
        return self.Dim.Gdy

    @property
    def Gdz(self) -> int:
        return self.Dim.Gdz

    @property
    def Hdx(self) -> int:
        return self.Dim.Hdx

    @property
    def Hdy(self) -> int:
        return self.Dim.Hdy

    @property
    def Virtual_Dim(self) -> Orthogonal3D:
        dim = self.Dim
        return Orthogonal3D(
            dim.dx + self.buffer_space,
            dim.dy + self.buffer_space,
            dim.dz,
        )

    @property
    def volume(self) -> int:
        return self.Dim.Volume

    def raw(self) -> np.ndarray:
        return self.Dim.raw()

    def place(self, flb: Point3D) -> None:
        self.FLB = flb

    def transpose(self) -> "SimpleBlock":
        nx, ny, nz = self.stack_dims
        return SimpleBlock(
            box=Orthogonal3D(self.box.dy, self.box.dx, self.box.dz),
            stack_dims=(ny, nx, nz),
            buffer_space=self.buffer_space,
            FLB=self.FLB,
            rot=not self.rot,
            preserve_order=self.preserve_order,
        )

    def rotated(self, rotate_xy: bool) -> "SimpleBlock":
        if not rotate_xy:
            return SimpleBlock(
                box=self.box,
                stack_dims=self.stack_dims,
                buffer_space=self.buffer_space,
                FLB=self.FLB,
                rot=self.rot,
                preserve_order=self.preserve_order,
            )
        return self.transpose()

    def to_item(self, flb: Point3D | None = None, rotate_xy: bool | None = None) -> Item:
        place_flb = self.FLB if flb is None else flb
        if place_flb is None:
            raise ValueError("SimpleBlock requires FLB to build an Item")

        apply_rot = self.rot if rotate_xy is None else bool(rotate_xy)
        dim = self.transpose().Dim if apply_rot else self.Dim
        placed = Item(FLB=place_flb, Dim=dim, buffer_space=self.buffer_space)
        placed.rot = bool(apply_rot)
        return placed

    def __add__(self, other: Item | "SimpleBlock" | int):
        if isinstance(other, int):
            return self.Dim.Volume + other
        return self.Dim.Volume + other.Dim.Volume

    def __radd__(self, other: int):
        if other == 0:
            return self.Dim.Volume
        return self.__add__(other)

    def __repr__(self):
        return f"SimpleBlock(box={self.box.raw()}, stack_dims={self.stack_dims})"


__all__ = ["Item", "SimpleBlock"]
