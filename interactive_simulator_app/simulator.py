from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packing_env.data_type.geometry import Orthogonal2D, Orthogonal3D, Point2D, Point3D, Rectangle
from packing_env.data_type.item import Item
from packing_env.gym_env import PackingEnv

GRID_CANDIDATE_STEP = 30
FIXED_ITEM_HEIGHT = 120


def item_dims(item) -> tuple[int, int, int]:
    return int(item.dx), int(item.dy), int(item.dz)


def placed_item_payload(item: Item, idx: int) -> dict:
    return {
        "id": idx,
        "x": int(item.FLB.x),
        "y": int(item.FLB.y),
        "z": int(item.FLB.z),
        "dx": int(item.Dim.dx),
        "dy": int(item.Dim.dy),
        "dz": int(item.Dim.dz),
    }


class InteractivePackingSimulator:
    def __init__(
        self,
        seed: int = 0,
        ds_name: str = "random",
        buffer_capacity: int = 12,
        container_size: tuple[int, int, int] = (600, 600, 600),
        k_placement: int = 80,
        buffer_space: int = 0,
        remove_inscribed_ems: bool = True,
        same_item_height: bool = False,
    ) -> None:
        self.seed = int(seed)
        self.ds_name = ds_name
        self.buffer_capacity = int(buffer_capacity)
        self.container_size = tuple(int(v) for v in container_size)
        self.k_placement = int(k_placement)
        self.buffer_space = int(buffer_space)
        self.remove_inscribed_ems = bool(remove_inscribed_ems)
        self.same_item_height = bool(same_item_height)
        self.env = self._make_env()
        self.env.reset(seed=self.seed)
        self._apply_item_height_mode()
        self.selected_rotation = 0
        self.message = "Click an anchor point to choose a loading position."

    def _make_env(self) -> PackingEnv:
        return PackingEnv(
            k_placement=self.k_placement,
            ds_name=self.ds_name,
            buffer_capacity=self.buffer_capacity,
            container_size=self.container_size,
            item_buffer_space=self.buffer_space,
            remove_inscribed_ems=self.remove_inscribed_ems,
        )

    def reset(self) -> dict:
        self.env.reset(seed=self.seed)
        self._apply_item_height_mode()
        self.selected_rotation = 0
        self.message = "Reset complete."
        return self.state()

    def set_same_item_height(self, enabled: bool) -> dict:
        self.same_item_height = bool(enabled)
        self.env.reset(seed=self.seed)
        self._apply_item_height_mode()
        self.selected_rotation = 0
        if self.same_item_height:
            self.message = f"Same item height enabled; item height fixed to {FIXED_ITEM_HEIGHT} mm."
        else:
            self.message = "Same item height disabled; original sampled heights restored."
        return self.state()

    def resize_container(self, dx: int, dy: int, dz: int) -> dict:
        size = tuple(self._clean_container_dim(v) for v in (dx, dy, dz))
        self.container_size = size
        self.env = self._make_env()
        self.env.reset(seed=self.seed)
        self._apply_item_height_mode()
        self.selected_rotation = 0
        self.message = f"Container resized to {size[0]}x{size[1]}x{size[2]} and reset."
        return self.state()

    def _clean_container_dim(self, value: int | float | str) -> int:
        dim = int(round(float(value) / 10.0) * 10)
        if dim < 120:
            raise ValueError("Container dimensions must be at least 120 mm.")
        return dim

    def set_rotation(self, rotation: int) -> dict:
        self.selected_rotation = 1 if int(rotation) else 0
        return self.state()

    def state(self) -> dict:
        self._apply_item_height_mode()
        current_item = self._current_item()
        actions = self._candidate_actions(current_item)
        return {
            "container": {
                "dx": int(self.env.container.dx),
                "dy": int(self.env.container.dy),
                "dz": int(self.env.container.dz),
            },
            "currentItem": self._item_payload(current_item),
            "buffer": [self._item_payload(item) for item in self.env.buffer.items[1:]],
            "placed": [
                placed_item_payload(item, idx)
                for idx, item in enumerate(self.env.container.placed_items, start=1)
            ],
            "support": self._support_payload(),
            "actions": actions,
            "gridCandidates": self._grid_candidates(current_item),
            "rotation": self.selected_rotation,
            "utilization": float(self.env.container.utilization),
            "sameItemHeight": self.same_item_height,
            "fixedItemHeight": FIXED_ITEM_HEIGHT,
            "message": self.message,
        }

    def place(self, x: float, y: float, rotation: int | None = None) -> dict:
        self._apply_item_height_mode()
        current_item = self._current_item()
        if current_item is None:
            self.message = "Buffer is empty."
            return self.state()

        if rotation is not None:
            self.selected_rotation = 1 if int(rotation) else 0

        actions = [
            action
            for action in self._candidate_actions(current_item)
            if action["rotation"] == self.selected_rotation
        ]
        if not actions:
            self.message = "No feasible loading position for this rotation."
            return self.state()

        target_x = float(x)
        target_y = float(y)
        chosen = min(
            actions,
            key=lambda action: (action["x"] - target_x) ** 2 + (action["y"] - target_y) ** 2,
        )
        self._place_action(current_item, chosen)
        self.message = (
            "Placed item "
            f"{chosen['dx']}x{chosen['dy']}x{chosen['dz']} at "
            f"({chosen['x']}, {chosen['y']}, {chosen['z']})."
        )
        return self.state()

    def place_grid(self, x: float, y: float, rotation: int | None = None) -> dict:
        self._apply_item_height_mode()
        current_item = self._current_item()
        if current_item is None:
            self.message = "Buffer is empty."
            return self.state()

        if rotation is not None:
            self.selected_rotation = 1 if int(rotation) else 0

        snapped_x = int(round(float(x) / GRID_CANDIDATE_STEP) * GRID_CANDIDATE_STEP)
        snapped_y = int(round(float(y) / GRID_CANDIDATE_STEP) * GRID_CANDIDATE_STEP)
        candidate = self._grid_candidate(current_item, snapped_x, snapped_y, self.selected_rotation)
        if candidate is None:
            self.message = f"({snapped_x}, {snapped_y}) is outside the container."
            return self.state()
        if not candidate["placeable"]:
            self.message = (
                f"({snapped_x}, {snapped_y}, {candidate['z']}) is blocked or outside the container."
            )
            return self.state()
        if not candidate["stable"]:
            self.message = (
                f"({snapped_x}, {snapped_y}, {candidate['z']}) is placeable but not stable."
            )
            return self.state()

        try:
            self._place_grid_action(current_item, candidate)
        except ValueError as exc:
            self.message = f"Cannot place at grid position: {exc}"
            return self.state()

        self.message = (
            "Placed item "
            f"{candidate['dx']}x{candidate['dy']}x{candidate['dz']} at "
            f"({candidate['x']}, {candidate['y']}, {candidate['z']})."
        )
        return self.state()

    def _current_item(self):
        if not self.env.buffer.has_items:
            return None
        return self.env.buffer.sample_item()

    def _item_payload(self, item) -> dict | None:
        if item is None:
            return None
        dx, dy, dz = item_dims(item)
        return {"dx": dx, "dy": dy, "dz": dz}

    def _apply_item_height_mode(self) -> None:
        if not self.same_item_height:
            return
        self.env.buffer.buffer = [
            Orthogonal3D(int(item.dx), int(item.dy), FIXED_ITEM_HEIGHT)
            for item in self.env.buffer.items
        ]

    def _support_payload(self) -> list[dict]:
        records = getattr(self.env.heu_stable, "support_vis_records", [])
        payload = []
        for idx, (item, dims, vis_data) in enumerate(records, start=1):
            dx, dy, dz = self._dims_payload(dims)
            payload.append(
                {
                    "id": idx,
                    "itemId": self._placed_item_id(item),
                    "itemDims": {"dx": dx, "dy": dy, "dz": dz},
                    "supportPolygon": self._polygon_payload(vis_data.support_polygon_xy),
                    "virtualPolygon": self._polygon_payload(vis_data.virtual_item_polygon_xy),
                    "z0": float(vis_data.support_z0),
                    "z1": float(vis_data.support_z1),
                }
            )
        return payload

    def _dims_payload(self, dims) -> tuple[int, int, int]:
        if hasattr(dims, "dx"):
            return int(dims.dx), int(dims.dy), int(dims.dz)
        return int(dims[0]), int(dims[1]), int(dims[2])

    def _placed_item_id(self, target_item) -> int | None:
        for idx, item in enumerate(self.env.container.placed_items, start=1):
            if item is target_item:
                return idx
            if item == target_item:
                return idx
        return None

    def _polygon_payload(self, polygon) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in polygon]

    def _candidate_actions(self, current_item) -> list[dict]:
        if current_item is None:
            return []

        obs = self.env.get_pack_data(current_item)
        mask = np.asarray(obs["action_mask"], dtype=bool).reshape(1, 2, self.env.k_placement)[0]
        actions = []
        base_dims = item_dims(current_item)
        for rotation in (0, 1):
            dims = base_dims if rotation == 0 else (base_dims[1], base_dims[0], base_dims[2])
            for idx, is_valid in enumerate(mask[rotation]):
                if not is_valid or idx >= len(self.env.ems_list):
                    continue
                ems = self.env.ems_list[idx]
                actions.append(
                    {
                        "index": int(idx),
                        "rotation": int(rotation),
                        "x": int(ems.FLB.x),
                        "y": int(ems.FLB.y),
                        "z": int(ems.FLB.z),
                        "dx": int(dims[0]),
                        "dy": int(dims[1]),
                        "dz": int(dims[2]),
                    }
                )
        return actions

    def _grid_candidates(self, current_item) -> list[dict]:
        if current_item is None:
            return []

        base_dims = item_dims(current_item)
        if self.selected_rotation:
            dx, dy, dz = base_dims[1], base_dims[0], base_dims[2]
        else:
            dx, dy, dz = base_dims

        candidates = []
        max_x = int(self.env.container.dx - dx)
        max_y = int(self.env.container.dy - dy)
        if max_x < 0 or max_y < 0:
            return []

        for x in range(0, max_x + 1, GRID_CANDIDATE_STEP):
            for y in range(0, max_y + 1, GRID_CANDIDATE_STEP):
                candidate = self._grid_candidate(current_item, x, y, self.selected_rotation)
                if candidate is not None:
                    candidates.append(candidate)
        return candidates

    def _grid_candidate(self, current_item, x: int, y: int, rotation: int) -> dict | None:
        base_dims = item_dims(current_item)
        if rotation:
            dx, dy, dz = base_dims[1], base_dims[0], base_dims[2]
        else:
            dx, dy, dz = base_dims

        if x < 0 or y < 0 or x + dx > self.env.container.dx or y + dy > self.env.container.dy:
            return None

        rect = Rectangle(Point2D(x, y), Orthogonal2D(dx, dy))
        z = int(self.env.hm.slice(rect).max())
        placed_box = Item(
            FLB=Point3D(x, y, z),
            Dim=Orthogonal3D(dx, dy, dz),
            buffer_space=self.env.item_buffer_space,
        )
        placed_box.rot = bool(rotation)
        virtual_dim = placed_box.Virtual_Dim

        within_container = (
            x + virtual_dim.dx <= self.env.container.dx
            and y + virtual_dim.dy <= self.env.container.dy
            and z + virtual_dim.dz <= self.env.container.dz
        )
        placeable = bool(within_container and self.env.container.is_placeable(placed_box))
        stable = False
        if placeable:
            hm_window = self.env.hm.slice(rect)
            feasible_window = self.env.heu_stable.slice(rect)
            stable, _ = self.env.heu_stable._convex_hull_validate(
                placed_box.Dim,
                (x // 10, y // 10),
                hm_window,
                feasible_window,
                scale=0.2,
            )

        return {
            "x": int(x),
            "y": int(y),
            "z": int(z),
            "dx": int(dx),
            "dy": int(dy),
            "dz": int(dz),
            "rotation": int(rotation),
            "placeable": bool(placeable),
            "stable": bool(stable),
            "valid": bool(placeable and stable),
        }

    def _place_action(self, source_item, action: dict) -> None:
        placed_box = Item(
            FLB=Point3D(action["x"], action["y"], action["z"]),
            Dim=Orthogonal3D(action["dx"], action["dy"], action["dz"]),
            buffer_space=self.env.item_buffer_space,
        )
        placed_box.rot = bool(action["rotation"])
        selected_ems = self.env.ems_list[action["index"]]
        self.env.pack(deepcopy(placed_box), selected_ems=selected_ems)
        self.env.buffer.update(source_item)
        self._apply_item_height_mode()
        self.env.validate_packing_state()

    def _place_grid_action(self, source_item, candidate: dict) -> None:
        placed_box = Item(
            FLB=Point3D(candidate["x"], candidate["y"], candidate["z"]),
            Dim=Orthogonal3D(candidate["dx"], candidate["dy"], candidate["dz"]),
            buffer_space=self.env.item_buffer_space,
        )
        placed_box.rot = bool(candidate["rotation"])
        self.env.pack(deepcopy(placed_box))
        self.env.buffer.update(source_item)
        self._apply_item_height_mode()
        self.env.validate_packing_state()
