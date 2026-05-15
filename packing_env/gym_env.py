from typing import List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .data_type.buffer import Buffer
from .data_type.container import Container
from .data_type.data_sampler import DataSampler
from .data_type.ems import EmptyMaximalSpace
from .data_type.geometry import Orthogonal3D, Point3D
from .data_type.item import Item
from .data_type.maps import HeightMap
from .heu_stable import Heu_Stable
from .heu_ems import EMS
from .visualization import PackVisualizer

class PackingEnv(gym.Env):
    def __init__(
        self,
        k_placement: int = 80,
        ds_name: str = 'random',
        buffer_capacity: int = 12,
        container_size: tuple[int, int, int] = (600, 600, 600),
        item_buffer_space: int = 0,
        remove_inscribed_ems: bool = False,
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
        self.buffer = Buffer(capacity=buffer_capacity, data_sampler=data_sampler)
        self._render_visualizer: PackVisualizer | None = None
        self._set_space()


    def _set_space(self) -> None:
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
        item: Orthogonal3D,
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

    def _coerce_items(self, items) -> list[Orthogonal3D]:
        """Convert raw item dimensions or Orthogonal3D objects to a list."""
        if isinstance(items, Orthogonal3D):
            return [items]

        if isinstance(items, np.ndarray):
            items = np.asarray(items)
            if items.ndim == 1:
                items = items.reshape(1, -1)
            if items.ndim != 2 or items.shape[1] != 3:
                raise ValueError("items must have shape (3,) or (N, 3)")
            return [Orthogonal3D(*map(int, item)) for item in items]

        if not items:
            return []

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
            items = self.buffer.items
        items = self._coerce_items(items)
        num_items = len(items)

        ems_list = self.heu_ems.get_ems_list()
        self.ems_list = ems_list
        if vectorized_ems is None:
            vectorized_ems = self.get_vectorized_ems(ems_list=ems_list)

        self.candidates = vectorized_ems.copy()

        if mask is None:
            mask = self.get_feasible_mask(items, ems_list=ems_list)
        mask = np.asarray(mask).astype(bool).reshape(num_items, 2, self.k_placement)

        item_raw = np.array([item.raw() for item in items]).astype(np.float32).reshape(num_items, 3)
        ems_normalized = vectorized_ems.copy().astype(np.float32)
        ems_normalized[:, :3] = ems_normalized[:, :3] / self.bin_size
        ems_normalized[:, 3:] = ems_normalized[:, 3:] / self.bin_size

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
            "done": ~mask.reshape(num_items, -1).any(axis=1),
        }
        data["item"] = data["new_item"]
        data["item_raw"] = item_raw
        data["mask"] = data["action_mask"]
        data["placable"] = bool(mask.any())
        return data


    def get_next_observation(self):
        ems_list = self.heu_ems.get_ems_list()
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

        vectorized_ems = self.get_vectorized_ems(ems_list=ems_list)
        ems = vectorized_ems.copy()
        ems[:,:3] = ems[:,:3]/self.bin_size
        ems[:,3:] = ems[:,3:]/self.bin_size

        self.candidates = vectorized_ems.copy()
        item = self.buffer.sample_item()
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
        item_dim: Orthogonal3D,
        placed_item: Item,
        selected_ems: EmptyMaximalSpace,
    ) -> None:
        # self.heu_stable.update(o3d=box.Dim, pxy=box.FLB.p2d, hm=self.hm)
        self.heu_stable.update(box=placed_item, hm=self.hm)
        self.container.add(box=placed_item)
        self.hm.update(box=placed_item)
        self.buffer.update(item_dim)
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

    def step(self, action: int) -> tuple:
        """Execute one step of the environment.
        
        Places an item at the selected position/orientation and updates state.
        
        Args:
            action: Index of placement position (0 to k_placement-1)
        
        Returns:
            Tuple of (next_obs, reward, done, truncated, info)
        """
        item_dim = self.selected_item
        
        # Decode action to position and rotation
        pos, rot, selected_ems = self.idx2pos(action)
        
        # Place item with appropriate rotation
        if rot:
            placed_dim = Orthogonal3D(item_dim.dy, item_dim.dx, item_dim.dz)
        else:
            placed_dim = item_dim
        placed_item = Item(
            FLB=Point3D(*pos),
            Dim=placed_dim,
            buffer_space=self.item_buffer_space,
        )
        
        self.img_counter += 1
        
        # Calculate stability reward (feasible area reduction)
        # feasible_area_before = np.sum(self.heu_stable.Value == 0)
        try:
            self._step(item_dim, placed_item, selected_ems)
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
        if self._render_visualizer is None:
            self._render_visualizer = PackVisualizer(
                self,
                title="Buffer + Stability Support",
            )
        else:
            self._render_visualizer.env = self
            self._render_visualizer.title = "Buffer + Stability Support"
        fig, _ = self._render_visualizer.refresh()
        if fig is not None:
            fig.write_html("buffer_support_live.html", auto_open=False, include_plotlyjs="cdn")

    def seed(self, seed_value: Optional[int] = None) -> None:
        """Set random seed for reproducibility.
        
        Args:
            seed_value: Random seed value
        """
        np.random.seed(seed_value)

if __name__=='__main__':
    pass
