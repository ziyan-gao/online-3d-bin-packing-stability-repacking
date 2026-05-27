from __future__ import annotations

import copy
from itertools import combinations
from typing import List

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from packing_env.data_type.geometry import GRID_SIZE, Orthogonal3D, Point3D
from packing_env.data_type.item import Item
from packing_env.data_type.maps import HeightMap


class Node:
    def __init__(
        self,
        dim: np.ndarray,
        p: np.ndarray,
        name: str = "node",
        mass: float = 0.1,
        z_tolerance: float = 1e-6,
        support_tolerance: float = 0.0,
    ):
        self.mass = mass
        self.dim = np.asarray(dim, dtype=np.float64)
        self.p = np.asarray(p, dtype=np.float64)
        self.children: list[Node] = []
        self.extern_masses: dict[Node, tuple[float, np.ndarray]] = {}
        self.node_name = name
        self.z_tolerance = z_tolerance
        self.support_tolerance = support_tolerance

    @property
    def adaptive_com(self):
        if len(self.extern_masses) == 0:
            return self.com
        dist_masses = []
        ps_exert = []
        for _, (dist_mass, p_exert) in self.extern_masses.items():
            dist_masses.append(dist_mass)
            ps_exert.append(p_exert)
        dist_masses = np.array(dist_masses)
        ps_exert = np.array(ps_exert)
        return ((ps_exert * dist_masses[:, None]).sum(0) + self.com * self.mass) / (
            self.mass + dist_masses.sum()
        )

    @property
    def accumulative_mass(self):
        if len(self.extern_masses) == 0:
            return self.mass
        return sum(dist_mass for dist_mass, _ in self.extern_masses.values()) + self.mass

    @property
    def com(self):
        return (self.p + self.dim / 2)[:2]

    @property
    def coord2D(self):
        return np.array(
            [
                [0, 0],
                [0, self.dim[1]],
                [self.dim[0], self.dim[1]],
                [self.dim[0], 0],
            ]
        ) + self.p[None, :2]

    @property
    def z_bottom(self):
        return self.p[-1]

    @property
    def z_surface(self):
        return self.dim[2] + self.p[-1]

    def add_child(self, node: "Node"):
        self.children.append(node)

    def obtain_children(self, nodes):
        for node in nodes:
            if abs(self.z_bottom - node.z_surface) <= self.z_tolerance:
                res = self.get_support_region(node)
                if isinstance(res, Polygon) and res.area > 0:
                    self.add_child(node)

    @property
    def stable(self):
        if len(self.children) == 0 and self.z_bottom > self.z_tolerance:
            return False
        convex_hull = self.get_convexHull()
        com_region = self.get_com_region()
        return convex_hull.covers(com_region)

    def get_com_region(self):
        com = self.adaptive_com
        if self.support_tolerance <= 0:
            return Point(com)
        tol = self.support_tolerance
        return Polygon(
            [
                [com[0] - tol, com[1] - tol],
                [com[0] - tol, com[1] + tol],
                [com[0] + tol, com[1] + tol],
                [com[0] + tol, com[1] - tol],
            ]
        )

    def get_convexHull(self):
        support_point_set = []
        if len(self.children) != 0:
            for child in self.children:
                intersected_region = np.array(self.get_support_region(child).exterior.coords)
                support_point_set.extend(intersected_region)
            hull = cv2.convexHull(np.array(support_point_set).astype(np.float32)).squeeze()
            return Polygon(hull.reshape((-1, 2)))
        return Polygon(self.coord2D)

    def masses_to_children(self):
        if len(self.children) == 0:
            return None
        if len(self.children) == 1:
            contact_center = self.get_centroids_of_contacts_with_children()[0]
            self.children[0].extern_masses[self] = (self.accumulative_mass, contact_center)
            return None

        index = np.arange(len(self.children))
        contact_centers = self.get_centroids_of_contacts_with_children()
        comb = list(combinations(list(zip(index, contact_centers)), 2))
        a = np.zeros((len(comb) + 1, len(self.children)))
        b = np.zeros((len(comb) + 1))
        for i, ele in enumerate(comb):
            if np.linalg.norm(self.adaptive_com[:2] - ele[1][-1]) != 0:
                a[i, ele[0][0]] = np.linalg.norm(
                    self.adaptive_com[:2] - ele[0][-1]
                ) / np.linalg.norm(self.adaptive_com[:2] - ele[1][-1])
                a[i, ele[1][0]] = -1
            else:
                a[i, ele[0][0]] = 0
                a[i, ele[1][0]] = 1
                b[i] = self.accumulative_mass
        a[-1] = np.ones((len(contact_centers)))
        b[-1] = self.accumulative_mass
        distributed_masses = self.project_mass_distribution(np.linalg.pinv(a).dot(b))
        for i, child in enumerate(self.children):
            child.extern_masses[self] = (distributed_masses[i], contact_centers[i])
        return None

    def project_mass_distribution(self, distributed_masses):
        distributed_masses = np.maximum(distributed_masses, 0)
        mass_sum = distributed_masses.sum()
        if mass_sum <= 0:
            return np.ones(len(self.children)) * self.accumulative_mass / len(self.children)
        return distributed_masses * self.accumulative_mass / mass_sum

    def get_centroids_of_contacts_with_children(self):
        if len(self.children) == 0:
            return np.array([self.com])
        return np.array(
            [
                np.array(self.get_support_region(child).centroid.coords).squeeze()
                for child in self.children
            ]
        )

    def get_support_region(self, node: "Node"):
        return Polygon(self.coord2D).intersection(Polygon(node.coord2D))


class AdaptiveTreeBaseline:
    def __init__(
        self,
        dx: int = 600,
        dy: int = 600,
        z_tolerance: float = 1e-6,
        support_tolerance: float = 5.0,
        item_mass: float = 0.1,
    ):
        self.dx = dx
        self.dy = dy
        self.nodes: list[Node] = []
        self.z_tolerance = z_tolerance
        self.support_tolerance = support_tolerance / GRID_SIZE
        self.item_mass = item_mass

    @staticmethod
    def _node_from_item(box: Item, name: str, item_mass: float, z_tolerance: float, support_tolerance: float):
        return Node(
            dim=np.array([box.Dim.Gdx, box.Dim.Gdy, box.Dim.Gdz], dtype=np.float64),
            p=np.array([box.FLB.Gx, box.FLB.Gy, box.FLB.Gz], dtype=np.float64),
            name=name,
            mass=item_mass,
            z_tolerance=z_tolerance,
            support_tolerance=support_tolerance,
        )

    def reset(self):
        self.nodes = []

    def stableCheck(self, new_node: Node):
        new_node.obtain_children(self.nodes)
        count = 1
        if not new_node.stable:
            return False, count
        count += 1
        if len(new_node.children) == 0:
            return True, count

        children_list = []
        node_list = [new_node]
        while True:
            nodes_next_iter = []
            for node in node_list:
                count += 1
                node.masses_to_children()
                children_list.extend(node.children)
                nodes_next_iter.extend(node.children)
            node_list = nodes_next_iter
            if len(nodes_next_iter) == 0:
                break
        check_array = np.array([ele.stable for ele in children_list])
        return bool(np.sum(check_array) == len(check_array)), count

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
        for i, (x, y) in enumerate(candidates):
            z = int(height_map_windows[x][y].max())
            coord = Point3D(int(x * hm.resolution), int(y * hm.resolution), z)
            candidate_box = Item(FLB=coord, Dim=o3d)
            node_new = self._node_from_item(
                candidate_box,
                name=f"candidate_{i}",
                item_mass=self.item_mass,
                z_tolerance=self.z_tolerance,
                support_tolerance=self.support_tolerance,
            )
            tree_tmp = copy.deepcopy(self)
            stable, _ = tree_tmp.stableCheck(node_new)
            flags.append(stable)
            stable_coords.append(coord if stable else None)
        return stable_coords, np.array(flags, dtype=bool)

    def update(self, hm: HeightMap, box: Item) -> None:
        node_new = self._node_from_item(
            box,
            name=f"node_{len(self.nodes) + 1}",
            item_mass=self.item_mass,
            z_tolerance=self.z_tolerance,
            support_tolerance=self.support_tolerance,
        )
        tree_tmp = copy.deepcopy(self)
        stable, _ = tree_tmp.stableCheck(node_new)
        if not stable:
            raise ValueError("AdaptiveTreeBaseline rejected unstable appended item.")
        tree_tmp.nodes.append(node_new)
        self.nodes = tree_tmp.nodes

    def unpack(self, box: Item) -> None:
        # Baseline replay support is intentionally simple: callers that need
        # unpacking should reset and replay placements in order.
        raise NotImplementedError("AdaptiveTreeBaseline does not support incremental unpack.")
