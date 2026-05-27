from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from typing import Iterable, List, Sequence

import numpy as np

from .data_type.container import Container
from .data_type.ems import EmptyMaximalSpace
from .data_type.geometry import Orthogonal3D, Point3D
from .data_type.item import Item
from .data_type.maps import HeightMap


def _to_o3d(box: Orthogonal3D | Sequence[int] | np.ndarray) -> Orthogonal3D:
    if isinstance(box, Orthogonal3D):
        return box
    return Orthogonal3D(*np.asarray(box, dtype=np.int32).tolist())


@dataclass(frozen=True, order=True)
class _SpaceKey:
    x: int
    y: int
    z: int
    dx: int
    dy: int
    dz: int

    @classmethod
    def from_ems(cls, ems: EmptyMaximalSpace) -> "_SpaceKey":
        return cls(
            ems.FLB.x,
            ems.FLB.y,
            ems.FLB.z,
            ems.Dim.dx,
            ems.Dim.dy,
            ems.Dim.dz,
        )


class EMS:
    """Difference-process EMS manager.

    This class maintains a mutable list of `EmptyMaximalSpace` objects and
    updates them after each placement.
    """

    def __init__(
        self,
        container: Container | Orthogonal3D | None = None,
        min_vol: int = 1,
        min_dim: int = 1,
        k_placement: int | None = None,
        remove_inscribed: bool = False,
    ):
        self.container = container if container is not None else Container()
        if isinstance(self.container, Container):
            dim = Orthogonal3D(self.container.dx, self.container.dy, self.container.dz)
        else:
            dim = self.container
        self._container_dim = dim
        self.min_vol = int(min_vol)
        self.min_dim = int(min_dim)
        self.k_placement = None if k_placement is None else int(k_placement)
        self.remove_inscribed = bool(remove_inscribed)
        self.debug = os.environ.get("EMS_DEBUG", "0") == "1"
        self.__ems_list: List[EmptyMaximalSpace] = []
        self._ems_by_flb: dict[Point3D, list[EmptyMaximalSpace]] = {}
        self._packed_items: List[Item] = []
        self.reset()

    def __len__(self) -> int:
        return len(self.__ems_list)

    def __iter__(self):
        return iter(self.__ems_list)

    def __getitem__(self, index: int) -> EmptyMaximalSpace:
        return self.__ems_list[index]

    def reset(self) -> None:
        self.__ems_list = [
            EmptyMaximalSpace(
                FLB=Point3D(0, 0, 0),
                Dim=Orthogonal3D(
                    self._container_dim.dx,
                    self._container_dim.dy,
                    self._container_dim.dz,
                ),
            )
        ]
        self._packed_items = []
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._ems_by_flb = {}
        for ems in self.__ems_list:
            self._ems_by_flb.setdefault(ems.FLB, []).append(ems)

    @staticmethod
    def overlapped(space_a: EmptyMaximalSpace, space_b: EmptyMaximalSpace) -> bool:
        a0 = np.array([space_a.FLB.x, space_a.FLB.y, space_a.FLB.z], dtype=np.int64)
        a1 = a0 + np.array([space_a.Dim.dx, space_a.Dim.dy, space_a.Dim.dz], dtype=np.int64)
        b0 = np.array([space_b.FLB.x, space_b.FLB.y, space_b.FLB.z], dtype=np.int64)
        b1 = b0 + np.array([space_b.Dim.dx, space_b.Dim.dy, space_b.Dim.dz], dtype=np.int64)
        overlap_dim = np.minimum(a1, b1) - np.maximum(a0, b0)
        overlap_vol = int(np.prod(np.maximum(overlap_dim, 0)))
        return overlap_vol > 0

    @staticmethod
    def inscribed(inner: EmptyMaximalSpace, outer: EmptyMaximalSpace) -> bool:
        in0 = np.array([inner.FLB.x, inner.FLB.y, inner.FLB.z], dtype=np.int64)
        in1 = in0 + np.array([inner.Dim.dx, inner.Dim.dy, inner.Dim.dz], dtype=np.int64)
        out0 = np.array([outer.FLB.x, outer.FLB.y, outer.FLB.z], dtype=np.int64)
        out1 = out0 + np.array([outer.Dim.dx, outer.Dim.dy, outer.Dim.dz], dtype=np.int64)
        return bool(np.all(out0 <= in0) and np.all(in1 <= out1))

    @staticmethod
    def _build_space(min_corner: Sequence[int], max_corner: Sequence[int]) -> EmptyMaximalSpace | None:
        min_corner = np.asarray(min_corner, dtype=np.int32)
        max_corner = np.asarray(max_corner, dtype=np.int32)
        dim = max_corner - min_corner
        if np.any(dim <= 0):
            return None
        return EmptyMaximalSpace(
            FLB=Point3D(int(min_corner[0]), int(min_corner[1]), int(min_corner[2])),
            Dim=Orthogonal3D(int(dim[0]), int(dim[1]), int(dim[2])),
        )

    @staticmethod
    def _split_three(host: EmptyMaximalSpace, placed: EmptyMaximalSpace) -> List[EmptyMaximalSpace]:
        """Three-way DP split (corner-anchored placement)."""
        x1, y1, z1 = host.FLB.x, host.FLB.y, host.FLB.z
        x2 = x1 + host.Dim.dx
        y2 = y1 + host.Dim.dy
        z2 = z1 + host.Dim.dz

        x3, y3, z3 = placed.FLB.x, placed.FLB.y, placed.FLB.z
        x4 = x3 + placed.Dim.dx
        y4 = y3 + placed.Dim.dy
        z4 = z3 + placed.Dim.dz

        candidates = [
            EMS._build_space((x4, y1, z1), (x2, y2, z2)),
            EMS._build_space((x1, y4, z1), (x2, y2, z2)),
            EMS._build_space((x1, y1, z4), (x2, y2, z2)),
        ]
        return [c for c in candidates if c is not None]

    @staticmethod
    def _split_three_with_locked_axes(
        host: EmptyMaximalSpace, placed: EmptyMaximalSpace
    ) -> List[tuple[EmptyMaximalSpace, int]]:
        x1, y1, z1 = host.FLB.x, host.FLB.y, host.FLB.z
        x2 = x1 + host.Dim.dx
        y2 = y1 + host.Dim.dy
        z2 = z1 + host.Dim.dz

        x3, y3, z3 = placed.FLB.x, placed.FLB.y, placed.FLB.z
        x4 = x3 + placed.Dim.dx
        y4 = y3 + placed.Dim.dy
        z4 = z3 + placed.Dim.dz

        candidates = [
            (EMS._build_space((x4, y1, z1), (x2, y2, z2)), 0),
            (EMS._build_space((x1, y4, z1), (x2, y2, z2)), 1),
            (EMS._build_space((x1, y1, z4), (x2, y2, z2)), 2),
        ]
        return [(space, axis) for space, axis in candidates if space is not None]

    @staticmethod
    def _split_three_from_placed_flb(
        host: EmptyMaximalSpace, placed: EmptyMaximalSpace
    ) -> List[tuple[EmptyMaximalSpace, int]]:
        x1, y1, z1 = placed.FLB.x, placed.FLB.y, placed.FLB.z
        x2 = host.FLB.x + host.Dim.dx
        y2 = host.FLB.y + host.Dim.dy
        z2 = host.FLB.z + host.Dim.dz

        x4 = x1 + placed.Dim.dx
        y4 = y1 + placed.Dim.dy
        z4 = z1 + placed.Dim.dz

        candidates = [
            (EMS._build_space((x4, y1, z1), (x2, y2, z2)), 0),
            (EMS._build_space((x1, y4, z1), (x2, y2, z2)), 1),
            (EMS._build_space((x1, y1, z4), (x2, y2, z2)), 2),
        ]
        return [(space, axis) for space, axis in candidates if space is not None]

    @staticmethod
    def _space_corners(space: EmptyMaximalSpace) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([space.FLB.x, space.FLB.y, space.FLB.z], dtype=np.int64)
        hi = lo + np.array([space.Dim.dx, space.Dim.dy, space.Dim.dz], dtype=np.int64)
        return lo, hi

    @staticmethod
    def _extend_split_with_host(
        split_space: EmptyMaximalSpace,
        host: EmptyMaximalSpace,
        locked_axis: int,
    ) -> EmptyMaximalSpace | None:
        split_lo, _ = EMS._space_corners(split_space)
        host_lo, host_hi = EMS._space_corners(host)
        extended_lo = host_lo.copy()
        extended_lo[locked_axis] = split_lo[locked_axis]
        return EMS._build_space(extended_lo, host_hi)

    @staticmethod
    def _intersection(
        space_a: EmptyMaximalSpace, space_b: EmptyMaximalSpace
    ) -> EmptyMaximalSpace | None:
        a0 = np.array([space_a.FLB.x, space_a.FLB.y, space_a.FLB.z], dtype=np.int64)
        a1 = a0 + np.array([space_a.Dim.dx, space_a.Dim.dy, space_a.Dim.dz], dtype=np.int64)
        b0 = np.array([space_b.FLB.x, space_b.FLB.y, space_b.FLB.z], dtype=np.int64)
        b1 = b0 + np.array([space_b.Dim.dx, space_b.Dim.dy, space_b.Dim.dz], dtype=np.int64)

        lo = np.maximum(a0, b0)
        hi = np.minimum(a1, b1)
        return EMS._build_space(lo, hi)

    @staticmethod
    def _subtract_overlap(
        host: EmptyMaximalSpace, cutter: EmptyMaximalSpace
    ) -> List[EmptyMaximalSpace]:
        """Return axis-aligned residual EMSs after removing overlap with cutter."""
        inter = EMS._intersection(host, cutter)
        if inter is None:
            return [host]

        x0, y0, z0 = host.FLB.x, host.FLB.y, host.FLB.z
        x1 = x0 + host.Dim.dx
        y1 = y0 + host.Dim.dy
        z1 = z0 + host.Dim.dz

        ix0, iy0, iz0 = inter.FLB.x, inter.FLB.y, inter.FLB.z
        ix1 = ix0 + inter.Dim.dx
        iy1 = iy0 + inter.Dim.dy
        iz1 = iz0 + inter.Dim.dz

        parts = [
            EMS._build_space((x0, y0, z0), (ix0, y1, z1)),
            EMS._build_space((ix1, y0, z0), (x1, y1, z1)),
            EMS._build_space((x0, y0, z0), (x1, iy0, z1)),
            EMS._build_space((x0, iy1, z0), (x1, y1, z1)),
            # EMS._build_space((x0, y0, z0), (x1, y1, iz0)),
            EMS._build_space((x0, y0, iz1), (x1, y1, z1)),
        ]
        return [p for p in parts if p is not None]

    @staticmethod
    def _is_valid_dim(space: EmptyMaximalSpace, min_dim: int, min_vol: int) -> bool:
        dims = np.array([space.Dim.dx, space.Dim.dy, space.Dim.dz], dtype=np.int64)
        if int(dims.min()) < min_dim:
            return False
        if int(np.prod(dims)) < min_vol:
            return False
        return True

    @staticmethod
    def _dedup(spaces: Iterable[EmptyMaximalSpace]) -> List[EmptyMaximalSpace]:
        out: List[EmptyMaximalSpace] = []
        seen = set()
        for space in spaces:
            key = _SpaceKey.from_ems(space)
            if key in seen:
                continue
            seen.add(key)
            out.append(space)
        return out

    @staticmethod
    def _remove_inscribed(spaces: Iterable[EmptyMaximalSpace]) -> List[EmptyMaximalSpace]:
        spaces = list(spaces)
        out: List[EmptyMaximalSpace] = []
        for i, space in enumerate(spaces):
            if any(
                i != j and EMS.inscribed(space, other)
                for j, other in enumerate(spaces)
            ):
                continue
            out.append(space)
        return out

    @staticmethod
    def _remove_same_flb_dominated(spaces: Iterable[EmptyMaximalSpace]) -> List[EmptyMaximalSpace]:
        groups: dict[tuple[int, int, int], list[EmptyMaximalSpace]] = defaultdict(list)
        for space in spaces:
            groups[(space.FLB.x, space.FLB.y, space.FLB.z)].append(space)

        out: List[EmptyMaximalSpace] = []
        for group in groups.values():
            for i, space in enumerate(group):
                dominated = False
                for j, other in enumerate(group):
                    if i == j:
                        continue
                    if (
                        other.Dim.dx >= space.Dim.dx
                        and other.Dim.dy >= space.Dim.dy
                        and other.Dim.dz >= space.Dim.dz
                        and (
                            other.Dim.dx > space.Dim.dx
                            or other.Dim.dy > space.Dim.dy
                            or other.Dim.dz > space.Dim.dz
                        )
                    ):
                        dominated = True
                        break
                if not dominated:
                    out.append(space)
        return out

    @staticmethod
    def _supported_by_heightmap(space: EmptyMaximalSpace, hm: HeightMap) -> bool:
        x0, y0 = space.FLB.Hx, space.FLB.Hy
        x1, y1 = x0 + space.Dim.Hdx, y0 + space.Dim.Hdy
        if x0 < 0 or y0 < 0 or x1 > hm.Value.shape[0] or y1 > hm.Value.shape[1]:
            return False

        footprint = hm.Value[x0:x1, y0:y1]
        if footprint.size == 0:
            return False
        return bool(np.any(footprint == space.FLB.z))

    @staticmethod
    def _remove_floating(
        spaces: Iterable[EmptyMaximalSpace],
        hm: HeightMap | None,
    ) -> List[EmptyMaximalSpace]:
        if hm is None:
            return list(spaces)
        return [space for space in spaces if EMS._supported_by_heightmap(space, hm)]

    @staticmethod
    def _fmt_space(space: EmptyMaximalSpace) -> str:
        lo, hi = EMS._space_corners(space)
        return f"{tuple(lo.tolist())}->{tuple(hi.tolist())}"

    def _resolve_selected_ems(self, placed: EmptyMaximalSpace) -> EmptyMaximalSpace:
        flb_candidates = [
            ems
            for ems in self.get_ems_by_point(placed.FLB, match_mode="flb")
            if self.inscribed(placed, ems)
        ]
        if len(flb_candidates) == 1:
            return flb_candidates[0]

        contain_candidates = [
            ems for ems in self.__ems_list if self.inscribed(placed, ems)
        ]
        if not contain_candidates:
            raise ValueError(
                "No EMS can contain the placed box "
                f"{self._fmt_space(placed)}."
            )

        contain_candidates.sort(
            key=lambda ems: (
                ems.Dim.dx * ems.Dim.dy * ems.Dim.dz,
                ems.FLB.z,
                ems.FLB.y,
                ems.FLB.x,
                ems.Dim.dz,
                ems.Dim.dy,
                ems.Dim.dx,
            )
        )
        return contain_candidates[0]

    def set_thresholds(self, min_vol: int | None = None, min_dim: int | None = None) -> None:
        """Update per-instance thresholds used to prune tiny EMSs."""
        if min_vol is not None:
            self.min_vol = int(min_vol)
        if min_dim is not None:
            self.min_dim = int(min_dim)

    def update(
        self,
        box: Item,
        selected_ems: EmptyMaximalSpace | None = None,
        hm: HeightMap | None = None,
        record_history: bool = True,
    ) -> EmptyMaximalSpace:
        """Update EMS set after placing `box` in the explicitly selected EMS."""
        if getattr(box, "FLB", None) is None:
            raise ValueError("`box.FLB` is required. Call `place(...)` first.")

        flb = box.FLB
        o3d = _to_o3d(box.Virtual_Dim)
        placed = EmptyMaximalSpace(FLB=flb, Dim=o3d)
        resolved_selected_ems = selected_ems is None
        if selected_ems is None:
            selected_ems = self._resolve_selected_ems(placed)
        elif not any(selected_ems is ems for ems in self.__ems_list):
            matches = [ems for ems in self.__ems_list if ems == selected_ems]
            if not matches:
                raise ValueError("`selected_ems` is not in the current EMS list.")
            selected_ems = matches[0]
        if not self.inscribed(placed, selected_ems):
            raise ValueError("`box.Virtual_Dim` cannot fit in the matched EMS.")

        updated: List[EmptyMaximalSpace] = []
        debug_counts = {
            "start": len(self.__ems_list),
            "split": 0,
            "split_invalid": 0,
            "extended": 0,
            "extended_invalid": 0,
            "kept": 0,
            "residual": 0,
            "residual_invalid": 0,
        }

        # 1) Replace selected EMS by a corner split when possible. Replay after
        # unpack can place a box inside an EMS whose FLB differs from box.FLB;
        # in that case use the general overlap subtractor for the selected EMS.
        if selected_ems.FLB == placed.FLB:
            for new_ems, locked_axis in self._split_three_with_locked_axes(selected_ems, placed):
                if not self._is_valid_dim(new_ems, min_dim=self.min_dim, min_vol=self.min_vol):
                    debug_counts["split_invalid"] += 1
                    continue
                updated.append(new_ems)
                debug_counts["split"] += 1
        else:
            for residual in self._subtract_overlap(selected_ems, placed):
                if self._is_valid_dim(residual, min_dim=self.min_dim, min_vol=self.min_vol):
                    updated.append(residual)
                    debug_counts["residual"] += 1
                else:
                    debug_counts["residual_invalid"] += 1
            if resolved_selected_ems:
                for new_ems, locked_axis in self._split_three_from_placed_flb(selected_ems, placed):
                    if not self._is_valid_dim(new_ems, min_dim=self.min_dim, min_vol=self.min_vol):
                        debug_counts["split_invalid"] += 1
                        continue
                    updated.append(new_ems)
                    debug_counts["split"] += 1

        # 2) For all other EMS, clip only those intersecting the placed box.
        for host in self.__ems_list:
            if host is selected_ems:
                continue
            if not self.overlapped(placed, host):
                updated.append(host)
                debug_counts["kept"] += 1
                continue

            residuals = self._subtract_overlap(host, placed)
            for residual in residuals:
                if self._is_valid_dim(residual, min_dim=self.min_dim, min_vol=self.min_vol):
                    updated.append(residual)
                    debug_counts["residual"] += 1
                else:
                    debug_counts["residual_invalid"] += 1

        before_dedup = len(updated)
        deduped = self._dedup(updated)
        after_dedup = len(deduped)
        if self.remove_inscribed:
            deduped = self._remove_inscribed(deduped)
        self.__ems_list = deduped
        self._rebuild_index()
        if record_history:
            self._packed_items.append(box)
        if self.debug:
            print(
                "[EMS] update "
                f"selected={self._fmt_space(selected_ems)} placed={self._fmt_space(placed)} "
                f"counts={debug_counts} before_dedup={before_dedup} "
                f"after_dedup={after_dedup} final={len(self.__ems_list)} "
            )

    def unpack(
        self,
        box: Item,
        hm: HeightMap | None = None,
    ) -> None:
        if getattr(box, "FLB", None) is None:
            raise ValueError("`box.FLB` is required.")

        try:
            remove_idx = self._packed_items.index(box)
        except ValueError as exc:
            raise ValueError("`box` is not in EMS packed item history.") from exc

        remaining_items = self._packed_items[:remove_idx] + self._packed_items[remove_idx + 1:]
        self.reset()
        for item in remaining_items:
            self.update(item, record_history=False)
        self._packed_items = list(remaining_items)

    @property
    def packed_items(self) -> list[Item]:
        return list(self._packed_items)

    def get_ems(self) -> List[EmptyMaximalSpace]:
        """Return EMSs in the legacy policy order, capped by k_placement.

        The previous item-wise project sorted EMS candidates by lower-corner
        position as ``[z, y, x]``. Keeping that order makes action/candidate
        presentation match the checkpoint's training environment more closely.
        """
        sorted_ems = sorted(
            self.__ems_list,
            key=lambda ems: (ems.FLB.z, ems.FLB.y, ems.FLB.x),
        )
        if self.k_placement is None:
            return sorted_ems
        if self.k_placement <= 0:
            return []
        return sorted_ems[: self.k_placement]

    def get_all_ems(self) -> List[EmptyMaximalSpace]:
        """Return all EMSs in legacy policy order, without applying k_placement."""
        return sorted(
            self.__ems_list,
            key=lambda ems: (ems.FLB.z, ems.FLB.y, ems.FLB.x),
        )

    def get_ems_list(self) -> List[EmptyMaximalSpace]:
        return self.get_ems()

    def get_anchor_points(self) -> List[Point3D]:
        """Build anchor points from EMS FLB positions.

        This method first fetches current EMSs via ``self.get_ems()`` and then
        returns one ``Point3D`` per unique EMS FLB location.
        """
        ems_list = self.get_ems()
        points: List[Point3D] = []
        seen = set()
        for ems in ems_list:
            key = (ems.FLB.Gx, ems.FLB.Gy, ems.FLB.Gz)
            if key in seen:
                continue
            seen.add(key)
            points.append(ems.FLB)
        return points

    def get_extreme_points(self) -> List[Point3D]:
        return self.get_anchor_points()

    def get_ems_map(self) -> dict[Point3D, list[EmptyMaximalSpace]]:
        """Return a shallow copy of FLB->EMS-list map."""
        return {flb: list(ems_list) for flb, ems_list in self._ems_by_flb.items()}

    @staticmethod
    def _to_xyz(point: Point3D | Sequence[int] | np.ndarray) -> tuple[int, int, int]:
        if isinstance(point, Point3D):
            return int(point.x), int(point.y), int(point.z)
        arr = np.asarray(point, dtype=np.int32).reshape(-1)
        if arr.size != 3:
            raise ValueError("Point must contain exactly 3 values: (x, y, z).")
        return int(arr[0]), int(arr[1]), int(arr[2])

    def get_ems_by_point(
        self,
        point: Point3D | Sequence[int] | np.ndarray,
        match_mode: str = "flb",
    ) -> list[EmptyMaximalSpace]:
        """Get EMSs corresponding to a 3D point.

        Args:
            point: Query point in real units (x, y, z).
            match_mode:
                - "flb": exact match with EMS lower corner (FLB).
                - "contain": point lies inside EMS cuboid
                  (inclusive lower bound, exclusive upper bound).
        """
        x, y, z = self._to_xyz(point)

        if match_mode == "flb":
            return list(self._ems_by_flb.get(Point3D(x, y, z), []))

        if match_mode == "contain":
            matches = []
            for ems in self.__ems_list:
                x0, y0, z0 = ems.FLB.x, ems.FLB.y, ems.FLB.z
                x1, y1, z1 = x0 + ems.Dim.dx, y0 + ems.Dim.dy, z0 + ems.Dim.dz
                if x0 <= x < x1 and y0 <= y < y1 and z0 <= z < z1:
                    matches.append(ems)
            return matches

        raise ValueError(f"Unsupported match_mode: {match_mode}")
