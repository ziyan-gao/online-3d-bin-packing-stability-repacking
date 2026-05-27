from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .data_type.buffer import Buffer
from .data_type.container import Container
from .data_type.data_sampler import DataSampler
from .data_type.ems import EmptyMaximalSpace
from .data_type.geometry import Orthogonal3D, Point3D
from .data_type.item import Item, SimpleBlock
from .data_type.maps import HeightMap
from .data_type.oriented_block import OrientedBlock
from .heu_stable import Heu_Stable
from .heu_ems import EMS

class PackingEnv(gym.Env):
    def __init__(
        self,
        k_placement: int = 80,
        ds_name: str = 'random',
        buffer_capacity: int = 12,
        container_size: tuple[int, int, int] = (600, 600, 600),
        item_buffer_space: int = 0,
        remove_inscribed_ems: bool = False,
        stack_only: bool = False,
        use_simple_blocks: bool = False,
        policy_mode: str = "largest_block_baseline",
    ) -> None:
        """Initialize the packing environment.
        
        Args:
            k_placement: Maximum number of placement candidates
            ds_name: Dataset/box distribution name
            buffer_capacity: Buffer size for storing items
        """
        dx, dy, dz = container_size
        self.container = Container(dx=dx, dy=dy, dz=dz)
        self.hm = HeightMap(dx=dx, dy=dy, zmax=dz)
        self.heu_stable = Heu_Stable(dx=dx, dy=dy)
        self.heu_ems = EMS(
            container=self.container,
            min_vol=1,
            min_dim=1,
            k_placement=k_placement,
            remove_inscribed=remove_inscribed_ems,
        )
        self.k_placement = k_placement
        self.item_buffer_space = int(item_buffer_space)
        self.remove_inscribed_ems = bool(remove_inscribed_ems)
        self.policy_mode = str(policy_mode)
        if self.policy_mode not in {
            "largest_block_baseline",
            "cascaded_block_selector",
        }:
            raise ValueError(f"Unknown policy_mode: {self.policy_mode!r}")
        self.stack_only = bool(stack_only)
        self.use_simple_blocks = bool(use_simple_blocks or stack_only)
        if self.policy_mode == "cascaded_block_selector":
            self.stack_only = True
            self.use_simple_blocks = True
        if self.item_buffer_space < 0:
            raise ValueError("item_buffer_space must be non-negative.")
        if self.item_buffer_space % self.hm.resolution != 0:
            raise ValueError("item_buffer_space must be divisible by the map resolution.")
        self.img_counter = 0
        self.bin_size = np.array(
            [self.container.dx, self.container.dy, self.container.dz],
            dtype=np.float32,
        )
        
        # Initialize buffer with data sampler
        data_sampler = DataSampler(data_name=ds_name)
        self.buffer = Buffer(
            capacity=buffer_capacity,
            data_sampler=data_sampler,
            stack_only=self.stack_only,
            container_size=container_size,
            buffer_space=self.item_buffer_space,
        )
        self.max_oriented_blocks = 2 * self.buffer.capacity
        self._render_visualizer = None
        self._set_space()


    def _set_space(self) -> None:
        if self.policy_mode == "cascaded_block_selector":
            self.action_space = spaces.Discrete(self.max_oriented_blocks * self.k_placement)
            self.observation_space = spaces.Dict(
                {
                    "oriented_blocks": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.max_oriented_blocks, 8),
                        dtype=np.float32,
                    ),
                    "block_mask": spaces.MultiBinary(self.max_oriented_blocks),
                    "ems": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.k_placement, 6),
                        dtype=np.float32,
                    ),
                    "loading_mask": spaces.MultiBinary(
                        (self.max_oriented_blocks, self.k_placement)
                    ),
                    "action_mask": spaces.MultiBinary(
                        (self.max_oriented_blocks, self.k_placement)
                    ),
                    "placable": spaces.Discrete(2),
                }
            )
            return

        self.action_space = spaces.Discrete(2 * self.k_placement)
        self.observation_space = spaces.Dict(
            {
                "new_item": spaces.Box(low=0,
                                       high=1,
                                       shape=(3,), 
                                       dtype=np.float32),

                "buffer_space": spaces.Box(low=0,
                                  high=max(self.container.dx, self.container.dy),
                                  shape=(1,),
                                  dtype=np.int32),
                
                "ems": spaces.Box(low=0, 
                                  high=1,
                                  shape=(int(self.k_placement*6),),  
                                  dtype=np.float32),
                
                "action_mask": spaces.MultiBinary((2, self.k_placement)),
            }
        )

    def to_key(self) -> tuple:
        return tuple(sorted(item.to_key() for item in self.container.placed_items))

    def find_placed_item(self, query: Item) -> Item:
        item = self.container.find_matching_item(query)
        if item is None:
            raise AssertionError(f"MCTS requested unpack of a box that is not placed: {query}")
        return item

    def validate_packing_state(self) -> None:
        placed = self.container.placed_items
        for idx, item in enumerate(placed):
            assert item.FLB.Gx >= 0 and item.FLB.Gy >= 0 and item.FLB.Gz >= 0
            assert item.FLB.Gx + item.Virtual_Dim.Gdx <= self.container.Gdx
            assert item.FLB.Gy + item.Virtual_Dim.Gdy <= self.container.Gdy
            assert item.FLB.Gz + item.Virtual_Dim.Gdz <= self.container.Gdz
            for other in placed[idx + 1 :]:
                assert not item.is_overlap(other), f"overlap detected: {item} vs {other}"

    def get_stable_lps_mask(
        self,
        item: Orthogonal3D | SimpleBlock,
        placements: np.ndarray,
        ems_list: List[EmptyMaximalSpace],
        buffer_space: Optional[int] = None,
    ) -> np.ndarray:
        """Get stability mask for valid placement positions.
        
        Determines which EMS placements can contain the item and are stable.
        
        Args:
            item: Item with 3D dimensions to place
            placements: Array of placement positions
            ems_list: List of empty maximal spaces
        
        Returns:
            Boolean mask of length k_placement indicating stable placements
        """
        buffer_space = self.item_buffer_space if buffer_space is None else int(buffer_space)
        item = item.Dim if isinstance(item, SimpleBlock) else item
        mask = np.zeros((self.k_placement,), dtype=bool)
        # Check which EMS can contain the item
        virtual_item = Orthogonal3D(
            item.dx + buffer_space,
            item.dy + buffer_space,
            item.dz,
        )
        can_contain = np.array([ems.include(virtual_item) for ems in ems_list], dtype=bool)
        
        candidate_placements = placements[can_contain]
        candidate_indices = np.arange(len(ems_list))[can_contain]
        
        # Check stability of each candidate
        stable_placements, is_stable = self.heu_stable(
            o3d=item,
            hm=self.hm,
            candidates=candidate_placements
        )
        
        # Validate stability matches EMS
        for i, ems_idx in enumerate(candidate_indices):
            expected_flb = ems_list[ems_idx].FLB
            if stable_placements[i] is None:
                is_stable[i] = False
            elif expected_flb != stable_placements[i] or not is_stable[i]:
                is_stable[i] = False
        
        mask[:len(ems_list)][can_contain] = is_stable
        return mask

    def get_vectorized_ems(self, ems_list:List[EmptyMaximalSpace]):
        candidates = np.zeros((self.k_placement, 6))
        if len(ems_list) != 0:
            if len(ems_list)<=self.k_placement:
                candidates[:len(ems_list)] = np.c_[
                    [ele.numpy(normalize=False) for ele in ems_list]
                ]
            else:
                candidates[:self.k_placement] = np.c_[
                    [ele.numpy(normalize=False) for ele in ems_list[:self.k_placement]]
                ]
        return np.round(candidates, 4)

    def _ems_can_fit_item(self, ems: EmptyMaximalSpace, item: Orthogonal3D | SimpleBlock) -> bool:
        dim = item.Dim if isinstance(item, SimpleBlock) else item
        buffer_space = item.buffer_space if isinstance(item, SimpleBlock) else self.item_buffer_space
        virtual_dim = Orthogonal3D(
            dim.dx + buffer_space,
            dim.dy + buffer_space,
            dim.dz,
        )
        if ems.include(virtual_dim):
            return True
        rotated_virtual_dim = Orthogonal3D(
            dim.dy + buffer_space,
            dim.dx + buffer_space,
            dim.dz,
        )
        return ems.include(rotated_virtual_dim)

    def _item_orientations(
        self,
        item: Orthogonal3D | SimpleBlock,
    ) -> list[tuple[Orthogonal3D, Orthogonal3D]]:
        dim = item.Dim if isinstance(item, SimpleBlock) else item
        buffer_space = item.buffer_space if isinstance(item, SimpleBlock) else self.item_buffer_space
        normal_virtual = Orthogonal3D(dim.dx + buffer_space, dim.dy + buffer_space, dim.dz)
        rotated_dim = Orthogonal3D(dim.dy, dim.dx, dim.dz)
        rotated_virtual = Orthogonal3D(dim.dy + buffer_space, dim.dx + buffer_space, dim.dz)
        return [(dim, normal_virtual), (rotated_dim, rotated_virtual)]

    def _ems_has_stable_fit(
        self,
        ems: EmptyMaximalSpace,
        items: list[Orthogonal3D | SimpleBlock],
    ) -> bool:
        candidates = ems.FLB.topix()[None, :]
        for item in items:
            for dim, virtual_dim in self._item_orientations(item):
                if not ems.include(virtual_dim):
                    continue
                stable_placements, is_stable = self.heu_stable(
                    o3d=dim,
                    hm=self.hm,
                    candidates=candidates,
                )
                if (
                    len(is_stable) > 0
                    and stable_placements[0] is not None
                    and stable_placements[0] == ems.FLB
                    and bool(is_stable[0])
                ):
                    return True
        return False

    def _rank_fit_ems(
        self,
        ems_list: list[EmptyMaximalSpace],
        items: list[Orthogonal3D | SimpleBlock],
        prefer_stable: bool,
    ) -> list[EmptyMaximalSpace]:
        stable_cache = {}
        if prefer_stable:
            stable_cache = {
                id(ems): self._ems_has_stable_fit(ems, items)
                for ems in ems_list
            }
        return sorted(
            ems_list,
            key=lambda ems: (
                0 if stable_cache.get(id(ems), False) else 1,
                ems.FLB.z,
                ems.FLB.y,
                ems.FLB.x,
                -ems.Volume,
            ),
        )

    def _collapse_same_flb_ems(
        self,
        ems_list: list[EmptyMaximalSpace],
    ) -> list[EmptyMaximalSpace]:
        best_by_flb: dict[tuple[int, int, int], EmptyMaximalSpace] = {}
        for ems in ems_list:
            key = (ems.FLB.x, ems.FLB.y, ems.FLB.z)
            current = best_by_flb.get(key)
            if current is None or ems.Volume < current.Volume:
                best_by_flb[key] = ems
        return list(best_by_flb.values())

    def _get_item_fit_ems_list(
        self,
        items=None,
        *,
        cap: bool = True,
        prefer_stable: bool = False,
    ) -> List[EmptyMaximalSpace]:
        ems_list = self.heu_ems.get_all_ems()
        if items is None:
            items = self.buffer.all_blocks if self.use_simple_blocks else self.buffer.items
        if isinstance(items, (Orthogonal3D, SimpleBlock)):
            items = [items]
        elif isinstance(items, np.ndarray):
            items_arr = np.asarray(items)
            if items_arr.ndim == 1:
                items_arr = items_arr.reshape(1, -1)
            items = [Orthogonal3D(*map(int, item)) for item in items_arr]
        else:
            normalized_items = []
            for item in items:
                if isinstance(item, (Orthogonal3D, SimpleBlock)):
                    normalized_items.append(item)
                    continue
                item_arr = np.asarray(item).reshape(-1)
                if item_arr.shape[0] != 3:
                    raise ValueError("each item must contain exactly 3 dimensions")
                normalized_items.append(Orthogonal3D(*map(int, item_arr)))
            items = normalized_items
        if not items:
            return []

        if len(items) == 1 and not prefer_stable:
            return self._get_single_item_fit_ems_list(items[0], ems_list, cap=cap)

        filtered = [
            ems
            for ems in ems_list
            if any(self._ems_can_fit_item(ems, item) for item in items)
        ]
        if cap:
            filtered = self._collapse_same_flb_ems(filtered)
        filtered = self._rank_fit_ems(filtered, items, prefer_stable=prefer_stable)
        if self.k_placement <= 0:
            return []
        return filtered[: self.k_placement] if cap else filtered

    def _get_single_item_fit_ems_list(
        self,
        item: Orthogonal3D | SimpleBlock,
        ems_list: list[EmptyMaximalSpace],
        *,
        cap: bool,
    ) -> list[EmptyMaximalSpace]:
        if not ems_list or self.k_placement <= 0:
            return []

        dim = item.Dim if isinstance(item, SimpleBlock) else item
        buffer_space = item.buffer_space if isinstance(item, SimpleBlock) else self.item_buffer_space
        flb = np.array([[ems.FLB.x, ems.FLB.y, ems.FLB.z] for ems in ems_list], dtype=np.int64)
        dims = np.array([[ems.Dim.dx, ems.Dim.dy, ems.Dim.dz] for ems in ems_list], dtype=np.int64)
        volume = dims[:, 0] * dims[:, 1] * dims[:, 2]
        normal = np.array([dim.dx + buffer_space, dim.dy + buffer_space, dim.dz], dtype=np.int64)
        rotated = np.array([dim.dy + buffer_space, dim.dx + buffer_space, dim.dz], dtype=np.int64)
        fit_mask = np.all(dims >= normal, axis=1) | np.all(dims >= rotated, axis=1)
        fit_indices = np.flatnonzero(fit_mask)
        if fit_indices.size == 0:
            return []

        if cap:
            best_by_flb: dict[tuple[int, int, int], int] = {}
            for idx in fit_indices.tolist():
                key = tuple(int(value) for value in flb[idx])
                current = best_by_flb.get(key)
                if current is None or volume[idx] < volume[current]:
                    best_by_flb[key] = idx
            fit_indices = np.array(list(best_by_flb.values()), dtype=np.int64)

        order = np.lexsort(
            (
                -volume[fit_indices],
                flb[fit_indices, 0],
                flb[fit_indices, 1],
                flb[fit_indices, 2],
            )
        )
        ranked_indices = fit_indices[order]
        if cap:
            ranked_indices = ranked_indices[: self.k_placement]
        return [ems_list[int(idx)] for idx in ranked_indices]

    def select_largest_policy_block(self) -> SimpleBlock | None:
        """Select the largest block that is usable in its policy-visible EMS set."""
        ranked_blocks = sorted(
            self.buffer.all_blocks,
            key=lambda block: (
                block.volume,
                block.consumed_count,
                block.Dim.dz,
                block.Dim.dy,
                block.Dim.dx,
            ),
            reverse=True,
        )
        for block in ranked_blocks:
            ems_list = self._get_item_fit_ems_list([block])
            if self.buffer._is_block_usable(block, ems_list, self.heu_stable, self.hm):
                self.buffer.simple_blocks = {block.box: [block]}
                return block
        self.buffer.simple_blocks = {}
        return None

    def get_cascaded_block_candidates(self):
        oriented_candidates = []
        mask_rows = []
        ems_list = self._get_item_fit_ems_list(
            self.buffer.all_blocks,
            prefer_stable=True,
        )

        if not ems_list:
            return (
                oriented_candidates,
                ems_list,
                np.zeros((0, self.k_placement), dtype=bool),
            )

        lps = np.array([ems.FLB.topix() for ems in ems_list])
        for source_index, block in enumerate(self.buffer.all_blocks):
            for rotate_xy in (False, True):
                oriented = OrientedBlock.from_block(
                    block,
                    source_index=source_index,
                    rotate_xy=rotate_xy,
                )
                fit_mask = np.array(
                    [ems.include(oriented.Virtual_Dim) for ems in ems_list],
                    dtype=bool,
                )
                if not fit_mask.any():
                    continue

                stable_placements, is_stable = self.heu_stable(
                    o3d=oriented.Dim,
                    hm=self.hm,
                    candidates=lps[fit_mask],
                )
                row = np.zeros((self.k_placement,), dtype=bool)
                fit_indices = np.flatnonzero(fit_mask)
                for local_index, ems_index in enumerate(fit_indices):
                    stable_placement = stable_placements[local_index]
                    if (
                        bool(is_stable[local_index])
                        and stable_placement is not None
                        and stable_placement == ems_list[int(ems_index)].FLB
                    ):
                        row[int(ems_index)] = True
                if row.any():
                    oriented_candidates.append(oriented)
                    mask_rows.append(row)

        if not mask_rows:
            return (
                oriented_candidates,
                ems_list,
                np.zeros((0, self.k_placement), dtype=bool),
            )
        return oriented_candidates, ems_list, np.asarray(mask_rows, dtype=bool)

    def _coerce_items(self, items) -> list[Orthogonal3D]:
        """Convert raw item dimensions or Orthogonal3D objects to a list."""
        if isinstance(items, Orthogonal3D):
            return [items]
        if isinstance(items, SimpleBlock):
            return [items.Dim]

        if isinstance(items, np.ndarray):
            items = np.asarray(items)
            if items.ndim == 1:
                items = items.reshape(1, -1)
            if items.ndim != 2 or items.shape[1] != 3:
                raise ValueError("items must have shape (3,) or (N, 3)")
            return [Orthogonal3D(*map(int, item)) for item in items]

        if not items:
            return []

        if isinstance(items[0], SimpleBlock):
            return [item.Dim for item in items]

        if not isinstance(items[0], Orthogonal3D):
            items_arr = np.asarray(items)
            if items_arr.ndim == 1:
                items_arr = items_arr.reshape(1, -1)
            if items_arr.ndim == 2 and items_arr.shape[1] == 3:
                return [Orthogonal3D(*map(int, item)) for item in items_arr]

        coerced_items = []
        for item in items:
            if isinstance(item, Orthogonal3D):
                coerced_items.append(item)
                continue
            if isinstance(item, SimpleBlock):
                coerced_items.append(item.Dim)
                continue

            item_arr = np.asarray(item).reshape(-1)
            if item_arr.shape[0] != 3:
                raise ValueError("each item must contain exactly 3 dimensions")
            coerced_items.append(Orthogonal3D(*map(int, item_arr)))
        return coerced_items

    def get_feasible_mask(
        self,
        items,
        ems_list: Optional[List[EmptyMaximalSpace]] = None,
    ) -> np.ndarray:
        """Build placement masks for each item and its x/y rotation."""
        items = self._coerce_items(items)
        if ems_list is None:
            ems_list = self.heu_ems.get_ems_list()

        masks = []
        if len(ems_list) == 0:
            return np.zeros((len(items), 2, self.k_placement), dtype=bool)

        lps = np.array([ele.FLB.topix() for ele in ems_list])
        for item in items:
            mask = self.get_stable_lps_mask(item, placements=lps, ems_list=ems_list)
            item_rot = Orthogonal3D(item.dy, item.dx, item.dz)
            mask_rot = self.get_stable_lps_mask(item_rot, placements=lps, ems_list=ems_list)
            masks.append(np.c_[mask, mask_rot].T)

        return np.array(masks).astype(bool).reshape(len(items), 2, self.k_placement)

    def get_pack_data(
        self,
        items=None,
        vectorized_ems: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> dict:
        """Create a batch of pack observations for candidate items.

        Args:
            items: Candidate item dimensions as Orthogonal3D, a list of
                Orthogonal3D, a raw (3,) array, or a raw (N, 3) array. When
                omitted, all currently buffered items are evaluated.
            vectorized_ems: Optional precomputed EMS array with shape
                (k_placement, 6).
            mask: Optional precomputed feasibility mask with shape
                (N, 2, k_placement).
        """
        if items is None:
            if self.use_simple_blocks:
                self.select_largest_policy_block()
                items = self.buffer.all_blocks
            else:
                items = self.buffer.items
        raw_items = items
        items = self._coerce_items(items)
        num_items = len(items)

        ems_list = self._get_item_fit_ems_list(raw_items)
        self.ems_list = ems_list
        if vectorized_ems is None:
            vectorized_ems = self.get_vectorized_ems(ems_list=ems_list)

        self.candidates = vectorized_ems.copy()

        if mask is None:
            mask = self.get_feasible_mask(items, ems_list=ems_list)
        mask = np.asarray(mask).astype(bool).reshape(num_items, 2, self.k_placement)

        if num_items == 0:
            item_raw = np.zeros((0, 3), dtype=np.float32)
        else:
            item_raw = np.array([item.raw() for item in items]).astype(np.float32).reshape(num_items, 3)
        ems_normalized = vectorized_ems.copy().astype(np.float32)
        ems_normalized[:, :3] = ems_normalized[:, :3] / self.bin_size
        ems_normalized[:, 3:] = ems_normalized[:, 3:] / self.bin_size
        done = np.ones((num_items,), dtype=bool)
        if num_items > 0:
            done = ~mask.reshape(num_items, -1).any(axis=1)

        data = {
            "new_item": item_raw / self.bin_size[None, :],
            "new_item_unnorm": item_raw,
            "buffer_space": np.full((num_items,), self.item_buffer_space, dtype=np.int32),
            "ems": np.tile(ems_normalized.reshape(1, self.k_placement, 6), [num_items, 1, 1]),
            "action_mask": mask,
            "ems_unnorm": np.tile(
                vectorized_ems.reshape(1, self.k_placement, 6).astype(np.float32),
                [num_items, 1, 1],
            ),
            "done": done,
        }
        data["item"] = data["new_item"]
        data["item_raw"] = item_raw
        data["mask"] = data["action_mask"]
        data["placable"] = bool(mask.any())
        return data

    def get_cascaded_observation(self) -> dict:
        oriented_candidates, ems_list, loading_rows = self.get_cascaded_block_candidates()
        self.oriented_block_candidates = oriented_candidates
        self.ems_list = ems_list

        vectorized_ems = self.get_vectorized_ems(ems_list=ems_list)
        normalized_ems = vectorized_ems.copy().astype(np.float32)
        normalized_ems[:, :3] = normalized_ems[:, :3] / self.bin_size
        normalized_ems[:, 3:] = normalized_ems[:, 3:] / self.bin_size
        self.candidates = vectorized_ems.copy()

        oriented_blocks = np.zeros((self.max_oriented_blocks, 8), dtype=np.float32)
        block_mask = np.zeros((self.max_oriented_blocks,), dtype=bool)
        loading_mask = np.zeros(
            (self.max_oriented_blocks, self.k_placement),
            dtype=bool,
        )

        visible_count = min(len(oriented_candidates), self.max_oriented_blocks)
        for idx, oriented in enumerate(oriented_candidates[:visible_count]):
            oriented_blocks[idx] = oriented.feature_row(
                container_size=(
                    self.container.dx,
                    self.container.dy,
                    self.container.dz,
                ),
                max_stack_count=self.buffer.capacity,
            )
            block_mask[idx] = True
        if visible_count > 0:
            loading_mask[:visible_count] = loading_rows[:visible_count]

        action_mask = loading_mask.copy()
        placable = bool(action_mask.any())
        self.done = not placable
        self.current_obs = {
            "oriented_blocks": oriented_blocks,
            "block_mask": block_mask,
            "ems": normalized_ems.astype(np.float32),
            "loading_mask": loading_mask,
            "action_mask": action_mask,
            "placable": placable,
        }
        self.selected_item = None
        return self.current_obs


    def get_next_observation(self):
        if self.policy_mode == "cascaded_block_selector":
            return self.get_cascaded_observation()

        if self.use_simple_blocks:
            self.select_largest_policy_block()
        else:
            ems_list = self._get_item_fit_ems_list()
            self.ems_list = ems_list
        if not self.buffer.has_items:
            self.done = True
            self.current_obs = {
                'new_item': np.zeros((3,), dtype=np.float32),
                'buffer_space': np.array([self.item_buffer_space], dtype=np.int32),
                'ems': np.zeros((self.k_placement*6,), dtype=np.float32),
                'action_mask': np.zeros((2, self.k_placement), dtype=bool)
            }
            self.selected_item = None
            return self.current_obs
        if self.use_simple_blocks and len(self.buffer.all_blocks) == 0:
            self.done = True
            self.current_obs = {
                'new_item': np.zeros((3,), dtype=np.float32),
                'buffer_space': np.array([self.item_buffer_space], dtype=np.int32),
                'ems': np.zeros((self.k_placement*6,), dtype=np.float32),
                'action_mask': np.zeros((2, self.k_placement), dtype=bool)
            }
            self.selected_item = None
            return self.current_obs

        if self.use_simple_blocks:
            item = self.buffer.sample_blocks(deterministic=True)
        else:
            item = self.buffer.sample_item()
        ems_list = self._get_item_fit_ems_list([item])
        self.ems_list = ems_list
        vectorized_ems = self.get_vectorized_ems(ems_list=ems_list)
        ems = vectorized_ems.copy()
        ems[:,:3] = ems[:,:3]/self.bin_size
        ems[:,3:] = ems[:,3:]/self.bin_size
        self.candidates = vectorized_ems.copy()
        self.new_item = item.raw()
        lps = np.array(
            [ele.FLB.topix() for ele in ems_list]
        )
        mask = self.get_stable_lps_mask(item, placements=lps, ems_list=ems_list)
        item_rot = Orthogonal3D(item.dy, item.dx, item.dz)
        mask_rot = self.get_stable_lps_mask(item_rot, placements=lps, ems_list=ems_list)
        masks = np.c_[mask, mask_rot].T
        self.done = np.sum(masks.astype(bool).reshape(2, self.k_placement))==0
        self.current_obs = {
            'new_item': (self.new_item.astype(np.float32).reshape(-1)/self.bin_size).astype(np.float32),
            'buffer_space': np.array([self.item_buffer_space], dtype=np.int32),
            'ems':ems.reshape(-1,).astype(np.float32),
            'action_mask':masks.astype(bool).reshape(2, self.k_placement),
            # 'ems_unnorm':vectorized_ems.copy(),
        }
        self.selected_item = item
        return self.current_obs

    def _step(
        self,
        source_item: Orthogonal3D | SimpleBlock,
        placed_item: Item,
        selected_ems: EmptyMaximalSpace,
    ) -> None:
        # self.heu_stable.update(o3d=box.Dim, pxy=box.FLB.p2d, hm=self.hm)
        self.heu_stable.update(box=placed_item, hm=self.hm)
        self.container.add(box=placed_item)
        self.hm.update(box=placed_item)
        self.buffer.update(source_item)
        self.heu_ems.update(box=placed_item, selected_ems=selected_ems, hm=self.hm)

    def pack(
        self,
        box: Item,
        selected_ems: Optional[EmptyMaximalSpace] = None,
    ) -> None:
        """Place an item and update placement state.

        Args:
            box: Item to place.
            selected_ems: EMS selected for the placement. If omitted, the EMS
                manager resolves the containing EMS from the current EMS list.
        """
        self.heu_stable.update(box=box, hm=self.hm)
        self.container.add(box=box)
        self.hm.update(box=box)
        self.heu_ems.update(box=box, selected_ems=selected_ems, hm=self.hm)

    def repack(
        self,
        box: Item,
        pos: np.ndarray,
        rot: bool,
        selected_ems: Optional[EmptyMaximalSpace] = None,
    ) -> Item:
        """Move a placed item to a new position and optional x/y rotation."""
        self.unpack(box)
        self.remove_holding_item(box)
        pack_item = box.Dim.raw()
        if rot:
            pack_item = pack_item[[1, 0, 2]]
        repacked_box = Item(
            FLB=Point3D(*np.asarray(pos, dtype=np.int32).reshape(3)),
            Dim=Orthogonal3D(*map(int, pack_item)),
            buffer_space=box.buffer_space,
        )
        repacked_box.rot = bool(rot)
        if selected_ems is not None:
            current_matches = [ems for ems in self.heu_ems.get_ems_list() if ems == selected_ems]
            selected_ems = current_matches[0] if current_matches else None
        self.pack(repacked_box, selected_ems=selected_ems)
        return repacked_box

    def unpack(self, box: Item) -> None:
        """Remove a placed item and restore container, height, stability, and EMS state."""
        self.container.unpack(box)
        self.hm.unpack(box)
        self.heu_stable.unpack(box)
        self.heu_ems.unpack(box, hm=self.hm)

    def remove_holding_item(self, item_or_index: Item | int | None) -> None:
        """Remove one item from the holding list by index or object identity."""
        if item_or_index is None:
            return
        if isinstance(item_or_index, int):
            self.container.holding_list.pop(item_or_index)
            return
        for idx, holding_item in enumerate(self.container.holding_list):
            if holding_item is item_or_index:
                self.container.holding_list.pop(idx)
                return
        self.container.holding_list.remove(item_or_index)
                       
    def idx2pos(self, idx: int) -> tuple:
        if idx > self.k_placement-1:
            idx = idx - self.k_placement
            rot = True
        else:
            rot = False
        pos = self.candidates[idx][:3]
        selected_ems = self.ems_list[idx]
        return pos, rot, selected_ems

    def decode_cascaded_action(self, action: int) -> tuple[int, int, OrientedBlock, EmptyMaximalSpace]:
        action = int(action)
        if action < 0 or action >= self.max_oriented_blocks * self.k_placement:
            raise ValueError(f"cascaded action out of range: {action}")
        if not hasattr(self, "current_obs"):
            raise ValueError("cascaded action requires a current observation")

        oriented_index, ems_index = divmod(action, self.k_placement)
        action_mask = np.asarray(self.current_obs["action_mask"], dtype=bool)
        if oriented_index >= action_mask.shape[0] or ems_index >= action_mask.shape[1]:
            raise ValueError(f"cascaded action out of mask range: {action}")
        if not bool(action_mask[oriented_index, ems_index]):
            raise ValueError(
                f"cascaded action selects invalid block/EMS pair: "
                f"block={oriented_index}, ems={ems_index}"
            )
        if oriented_index >= len(self.oriented_block_candidates):
            raise ValueError(f"cascaded block index unavailable: {oriented_index}")
        if ems_index >= len(self.ems_list):
            raise ValueError(f"cascaded EMS index unavailable: {ems_index}")

        oriented = self.oriented_block_candidates[oriented_index]
        selected_ems = self.ems_list[ems_index]
        return oriented_index, ems_index, oriented, selected_ems

    def step_cascaded(self, action: int) -> tuple:
        _, _, oriented, selected_ems = self.decode_cascaded_action(action)
        placed_item = oriented.to_item(selected_ems.FLB)

        self.last_placed_source = oriented
        self.img_counter += 1
        self._step(oriented.block, placed_item, selected_ems)

        reward = placed_item.Dim.Volume / self.container.Volume
        obs_next = self.get_next_observation()
        done = self.done
        info = {
            "packed_objects": len(self.container.placed_items),
            "utilization_ratio": self.container.utilization,
            "selected_stack_height": oriented.block.no_boxes_wrt_axis[2],
        }
        return obs_next, reward, done, False, info

    def step(self, action: int) -> tuple:
        """Execute one step of the environment.
        
        Places an item at the selected position/orientation and updates state.
        
        Args:
            action: Index of placement position (0 to k_placement-1)
        
        Returns:
            Tuple of (next_obs, reward, done, truncated, info)
        """
        if self.policy_mode == "cascaded_block_selector":
            return self.step_cascaded(action)

        selected_block = self.selected_item
        
        # Decode action to position and rotation
        pos, rot, selected_ems = self.idx2pos(action)
        
        # Place item with appropriate rotation
        placed_item = selected_block.to_item(
            flb=Point3D(*pos),
            rotate_xy=rot,
        )
        
        self.img_counter += 1
        
        # Calculate stability reward (feasible area reduction)
        # feasible_area_before = np.sum(self.heu_stable.Value == 0)
        try:
            self._step(selected_block, placed_item, selected_ems)
        except Exception as e:
            print(pos)
            print(f"Error in placement step: {e}")
        
        # feasible_area_after = np.sum(self.heu_stable.Value == 0)
        # reduced_feasible = (feasible_area_after - feasible_area_before) / (
        #     block_t.Dim.Hdx * block_t.Dim.Hdy * 100
        # )
        
        # Reward = volume contribution + stability contribution
        # reward = block_t.Dim.Volume / self.container.Volume + reduced_feasible
        reward = placed_item.Dim.Volume / self.container.Volume
        
        obs_next = self.get_next_observation()
        done = self.done
        
        # Return Gymnasium format
        info = {
            'packed_objects': len(self.container.placed_items),
            'utilization_ratio': self.container.utilization
        }
        return obs_next, reward, done, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        """Reset the packing environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional environment-specific options
        
        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        if self.buffer.data_sampler.is_random_distribution:
            self.buffer.data_sampler.resample_distribution()
        self.buffer.reset()
        self.container.clear()
        self.hm.reset()
        self.heu_stable.reset()
        self.heu_ems.reset()
        self.done = False
        
        obs = self.get_next_observation()
        
        self.img_counter = 0
        return obs, {}
    
    def render(self) -> None:
        """Render current packing state to file.
        
        Saves visualization showing container with packed items and support regions.
        """
        import json

        from packing.interactive_replay import REPLAY_TEMPLATE
        from packing.three_scene import build_three_scene

        title = "Buffer + Stability Support"
        scene = build_three_scene(self, title, show_ems=True)
        html = REPLAY_TEMPLATE.replace(
            "__FRAMES__",
            json.dumps([{"title": title, "scene": scene}]),
        )
        html = html.replace("__INTERVAL_MS__", "700")
        with open("buffer_support_live.html", "w", encoding="utf-8") as f:
            f.write(html)

    def seed(self, seed_value: Optional[int] = None) -> None:
        """Set random seed for reproducibility.
        
        Args:
            seed_value: Random seed value
        """
        np.random.seed(seed_value)

if __name__=='__main__':
    pass
