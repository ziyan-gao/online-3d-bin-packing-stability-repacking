from __future__ import annotations
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
import cv2
from typing import List
from .data_type.geometry import Orthogonal3D, Point2D, Point3D, Rectangle
from .data_type.item import Item
from .data_type.maps import HeightMap, Map
from .data_type.support_vis import SupportVisData
import copy

class Heu_Stable(Map):
    def __init__(self, dx: int = 1200, dy: int = 1000):
        super().__init__(dx=dx, dy=dy)
        self._convex_hull_vis_cache: list[tuple[Item, tuple[int, int, int], SupportVisData]] = []
        self.prev_cache = {}

    @property
    def support_vis_records(self) -> list[tuple[Item, tuple[int, int, int], SupportVisData]]:
        """Read-only view for visualization consumers."""
        return list(self._convex_hull_vis_cache)

    def get_com_bound(self, o3d: Orthogonal3D, scale: float = 0.1) -> Polygon:
        """
        Get center-of-mass boundary polygon.
        
        Creates a scaled polygon representing the COM region of the box.
        Scale < 1.0 shrinks the polygon toward the center.
        
        Args:
            o3d: Box dimensions
            scale: Scaling factor (0.0 to 1.0)
        
        Returns:
            Polygon representing the COM boundary
        """
        coord2d = scale * (o3d.coord2d.copy() - o3d.o2d.numpy() / 2) + o3d.o2d.numpy() / 2
        return Polygon(coord2d)
    
    def reset(self) -> None:
        """Clear feasible map and convex hull visualization cache."""
        self._convex_hull_vis_cache = []
        self.prev_cache = {}
        self.Value = self.Value * 0

    @property
    def prevCache(self):
        return self.prev_cache

    @prevCache.setter
    def prevCache(self, value):
        self.prev_cache = value

    def __call__(self, o3d:Orthogonal3D, hm:HeightMap, candidates, scale=0.2) -> tuple[List[Point3D], np.ndarray]:
        """
        Check stability of candidates by verifying if COM is within support polygon.
        
        Returns:
            Tuple of (stable coordinates, stability status array)
        """
        stable_coords: List[Point3D] = []
        is_stable_flags: List[bool] = []
        
        # Get sliding window views for feasible map and height map
        feasible_map_windows = self.sliding_window_view(o3d)
        height_map_windows = hm.sliding_window_view(o3d)
        
        self._check_orthogonal3d_stability(
            o3d, feasible_map_windows, height_map_windows,
            candidates, scale, stable_coords, is_stable_flags
        )
        
        return stable_coords, np.array(is_stable_flags)
    
    def _check_orthogonal3d_stability(self, o3d: Orthogonal3D, feasible_map_windows, height_map_windows,
                                       candidates, scale: float, stable_coords: List[Point3D],
                                       is_stable_flags: List[bool]) -> None:
        """Check stability for Orthogonal3D box."""
        for x, y in candidates:
            hm_window = height_map_windows[x][y]
            fm_window = feasible_map_windows[x][y]
            
            is_stable, coord = self._convex_hull_validate(
                o3d, (x, y), hm_window, fm_window, scale=scale
            )
            
            is_stable_flags.append(is_stable)
            stable_coords.append(coord)
    
    @staticmethod
    def compute_hull_from_support(support_points: np.ndarray) -> Polygon | bool:
        """
        Compute convex hull from support point coordinates.
        
        Augments support points with nearby points to ensure a proper polygon
        (handles cases with 2 or fewer points).
        
        Args:
            support_points: Nx2 array of support point coordinates
        
        Returns:
            Polygon object if hull computed successfully, False otherwise
        """
        if support_points.shape[0] <= 2:
            return False
        
        # Augment support points to ensure valid polygon
        support_augment = (
            support_points.tolist() +
            (support_points + np.ones_like(support_points)).tolist() +
            (support_points + np.ones_like(support_points) * np.array([[0, 1]])).tolist() +
            (support_points + np.ones_like(support_points) * np.array([[1, 0]])).tolist()
        )
        
        try:
            hull = cv2.convexHull(np.array(support_augment))
            return Polygon(hull.reshape((hull.shape[0], 2)))
        except Exception:
            return False
    
    @staticmethod
    def coordset(stables: List[Point3D]) -> set:
        """Convert list of Point3D to set of coordinate tuples."""
        return set([(ele.Gx, ele.Gy, ele.Gz) for ele in stables])

    def _convex_hull_validate(self, o3d: Orthogonal3D, point: tuple[int, int], hm_window: np.ndarray, 
                              fm_window: np.ndarray, scale: float = 0.2) -> tuple[bool, Point3D | None]:
        """
        Check if COM of box is within the convex hull of support points.
        
        A box is stable if its center-of-mass polygon is completely contained 
        within the convex hull formed by the highest support points.
        
        Args:
            o3d: Box dimension object
            point: Grid position (x, y)
            hm_window: Height map window at this position
            fm_window: Feasible map window at this position
            scale: Scale factor for COM polygon (0.0-1.0, where 1.0 is full box size)
        
        Returns:
            Tuple of (is_stable, stable_coordinate)
            - is_stable: True if COM is within support hull
            - stable_coordinate: Point3D of support region if stable, None otherwise
        """
        # Find highest support points (where height equals max AND feasible)
        support = np.argwhere(np.abs((hm_window * (fm_window == 0)) - hm_window.max()) == 0)
        polygon = self.compute_hull_from_support(support)
        
        # Get COM boundary polygon (scaled down version of box)
        com_bound = self.get_com_bound(o3d, scale)
        
        if isinstance(polygon, Polygon):
            if polygon.contains(com_bound):
                stable_coord = Point3D(
                    *np.array([*np.array(point) * self.resolution, hm_window.max()])
                )
                return True, stable_coord
            else:
                return False, None
        else:
            return False, None

    def update(self, hm: HeightMap, box: Item) -> None:
        """Update feasible map based on box placement."""
        roi = self._box_roi(box)
        self.prev_cache[box] = self.slice(roi).copy()
        self.Value = self._compute_updated_feasible_map(box, hm)

    def unpack(self, box: Item) -> None:
        """Restore feasible map and visual support cache for an unpacked box."""
        roi = self._box_roi(box)
        prev_value = self.prev_cache.get(box)
        if prev_value is not None:
            self.Value[roi.Hx : roi.Hx + roi.Hdx, roi.Hy : roi.Hy + roi.Hdy] = prev_value
        self.prev_cache.pop(box, None)
        self._remove_convex_hull_vis_record(box)

    @staticmethod
    def _same_box(left: Item, right: Item) -> bool:
        return (
            left.FLB == right.FLB
            and left.Dim.dx == right.Dim.dx
            and left.Dim.dy == right.Dim.dy
            and left.Dim.dz == right.Dim.dz
            and left.buffer_space == right.buffer_space
        )

    @staticmethod
    def _box_roi(box: Item) -> Rectangle:
        return Rectangle(box.True_FLB.p2d, box.Dim.o2d)

    def _remove_convex_hull_vis_record(self, box: Item) -> None:
        for idx in range(len(self._convex_hull_vis_cache) - 1, -1, -1):
            cached_box, _, _ = self._convex_hull_vis_cache[idx]
            if self._same_box(cached_box, box):
                self._convex_hull_vis_cache.pop(idx)
                return
    
    def _compute_updated_feasible_map(self, box: Item, hm: HeightMap) -> np.ndarray:
        """
        Compute updated feasible map for a box.
        Updates feasible regions based on convex hull of support points.
        """
        box_dim = box.Dim.raw()
        box_real = Item(FLB=copy.deepcopy(box.True_FLB), Dim=Orthogonal3D(*box_dim))
        o3d = box_real.Dim
        pxy = box_real.FLB.p2d
        roi = self._box_roi(box_real)
        
        # Get height map window for this box placement
        cH_window = hm.slice(roi)
        h_max = cH_window.max()
        value_map = h_max - cH_window
        
        # Get feasible map window
        cH_fsb_window = self.slice(roi)
        
        # Compute convex hull from support points
        support = np.argwhere(np.abs((cH_window * (cH_fsb_window == 0)) - cH_window.max()) == 0)
        cH = self.compute_hull_from_support(support)
        
        # Cache visualization data as stable append-only records.
        # The support hull is computed from the true footprint, while the box
        # outline is the virtual occupied space used for collision clearance.
        coords = np.array(list(cH.exterior.coords[:-1])) + np.array([roi.Hx, roi.Hy])
        box_dims = tuple(int(v) for v in box.Dim.raw().tolist())
        virtual_dim = box.Virtual_Dim
        virtual_pxy = box.FLB.p2d
        support_polygon = [tuple(map(float, xy)) for xy in (coords * self.resolution).tolist()]
        virtual_polygon = [
            tuple(map(float, xy))
            for xy in (virtual_dim.coord2d * self.resolution + virtual_pxy.numpy()).tolist()
        ]
        self._convex_hull_vis_cache.append(
            (
                copy.deepcopy(box),
                box_dims,
                SupportVisData(
                    support_polygon_xy=support_polygon,
                    support_z0=float(h_max),
                    support_z1=float(h_max + o3d.dz),
                    virtual_item_polygon_xy=virtual_polygon,
                ),
            )
        )
        
        # Mark infeasible regions (regions inside hull but outside support)
        pointset = np.argwhere(value_map > 0)
        for point in pointset:
            if Point((point[0], point[1])).within(cH):
                value_map[(point[0], point[1])] = -cH_fsb_window[(point[0], point[1])]
        
        # Return updated feasible map
        return self.with_updated_roi(roi, value=value_map + self.slice(roi))
