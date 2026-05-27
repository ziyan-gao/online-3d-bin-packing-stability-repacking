from __future__ import annotations

import numpy as np

from .data_sampler import DataSampler
from .geometry import Orthogonal3D
from .item import SimpleBlock

class Buffer:
    """Buffer for managing sampled item dimensions."""
    
    def __init__(
        self,
        capacity: int,
        data_sampler: 'DataSampler',
        stack_only: bool = False,
        container_size: tuple[int, int, int] | None = None,
        buffer_space: int = 0,
    ):
        self.capacity = int(capacity)
        self.data_sampler = data_sampler
        self.stack_only = bool(stack_only)
        self.container_size = container_size
        self.buffer_space = int(buffer_space)
        self.buffer = []
        self.simple_blocks: dict[Orthogonal3D, list[SimpleBlock]] = {}
        self.fill_buffer()

    def fill_buffer(self):
        """Fill buffer with sampled boxes up to capacity."""
        req_length = self.capacity - len(self.buffer)
        if req_length > 0:
            self.buffer.extend(self.data_sampler.sample(req_length))
        self.simple_blocks = self.generate_all_blocks_from_buffer()

    def reset(self):
        """Reset buffer and refill sampled items."""
        self.buffer = []
        self.simple_blocks = {}
        self.fill_buffer()

    @property
    def items(self) -> list['SimpleBlock']:
        return [SimpleBlock(Dim=item, buffer_space=self.buffer_space) for item in self.buffer]

    @property
    def all_blocks(self) -> list['SimpleBlock']:
        blocks = []
        for blocks_for_type in self.simple_blocks.values():
            blocks.extend(blocks_for_type)
        return blocks

    def dims(self) -> list[tuple[int, int, int]]:
        return [(int(item.dx), int(item.dy), int(item.dz)) for item in self.buffer]

    @property
    def has_items(self) -> bool:
        return len(self.buffer) > 0

    def sample_item(self) -> 'SimpleBlock':
        """Return the next buffered block without changing sequence order."""
        if not self.buffer:
            raise ValueError("No items available to sample from")
        return SimpleBlock(
            Dim=self.buffer[0],
            buffer_space=self.buffer_space,
            preserve_order=True,
        )

    def _block_fits_container(self, block: SimpleBlock) -> bool:
        if self.container_size is None:
            return True
        dx, dy, dz = self.container_size
        return (
            block.Virtual_Dim.dx <= dx
            and block.Virtual_Dim.dy <= dy
            and block.Virtual_Dim.dz <= dz
        )

    def generate_all_blocks_from_buffer(self) -> dict['Orthogonal3D', list['SimpleBlock']]:
        """Generate stack-only block candidates from the current raw buffer."""
        blocks_by_type: dict[Orthogonal3D, list[SimpleBlock]] = {}
        for box_type, count in self.summary.items():
            blocks_for_type = []
            max_count = count if self.stack_only else 1
            for stack_height in range(1, max_count + 1):
                block = SimpleBlock(
                    box=box_type,
                    stack_dims=(1, 1, stack_height),
                    buffer_space=self.buffer_space,
                )
                if self._block_fits_container(block):
                    blocks_for_type.append(block)
            blocks_by_type[box_type] = blocks_for_type
        return blocks_by_type

    def _is_block_usable(self, block: SimpleBlock, ems_list, heu_stable, hm) -> bool:
        for candidate in (block, block.transpose()):
            fit_ems = [
                ems
                for ems in ems_list
                if ems.include(candidate.Virtual_Dim)
                and self._height_matches_ems(candidate.Dim, ems, hm)
            ]
            if not fit_ems:
                continue
            candidates = np.array([ems.FLB.topix() for ems in fit_ems])
            stable_placements, is_stable = heu_stable(
                o3d=candidate.Dim,
                hm=hm,
                candidates=candidates,
            )
            for ems, stable_placement, stable in zip(fit_ems, stable_placements, is_stable):
                if stable_placement is not None and stable_placement == ems.FLB and bool(stable):
                    return True
        return False

    @staticmethod
    def _height_matches_ems(dim, ems, hm) -> bool:
        if hm is None:
            return True
        x, y = ems.FLB.topix()
        x1 = x + dim.Hdx
        y1 = y + dim.Hdy
        if x < 0 or y < 0 or x1 > hm.Value.shape[0] or y1 > hm.Value.shape[1]:
            return False
        return hm.Value[x:x1, y:y1].max() == ems.FLB.z

    def update_usable(self, ems_list, heu_stable, hm) -> None:
        """Filter generated blocks to those that can fit and pass stability."""
        for box_type, blocks_for_type in list(self.simple_blocks.items()):
            self.simple_blocks[box_type] = [
                block
                for block in blocks_for_type
                if self._is_block_usable(block, ems_list, heu_stable, hm)
            ]

    def select_largest_usable(self, ems_list, heu_stable, hm) -> SimpleBlock | None:
        """Keep only the largest usable block candidate and return it."""
        ranked_blocks = sorted(
            self.all_blocks,
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
            if self._is_block_usable(block, ems_list, heu_stable, hm):
                self.simple_blocks = {block.box: [block]}
                return block
        self.simple_blocks = {}
        return None

    def sample_blocks(
        self,
        num_samples: int = 1,
        deterministic: bool = True,
        random_sample: bool = False,
    ) -> 'SimpleBlock | list[SimpleBlock]':
        """Sample generated block candidates, preferring larger blocks by default."""
        flat_blocks = self.all_blocks
        if not flat_blocks:
            raise ValueError("No blocks available to sample from")

        sample_count = min(int(num_samples), len(flat_blocks))
        if random_sample:
            selected_indices = np.random.choice(len(flat_blocks), size=sample_count, replace=False)
        else:
            volumes = np.array([block.volume for block in flat_blocks])
            if deterministic:
                selected_indices = np.argsort(-volumes)[:sample_count]
            else:
                probabilities = volumes / volumes.sum()
                selected_indices = np.random.choice(
                    len(flat_blocks),
                    size=sample_count,
                    replace=False,
                    p=probabilities,
                )

        selected = [flat_blocks[int(index)] for index in selected_indices]
        return selected[0] if num_samples == 1 else selected
    
    def remove(self, index_list):
        """Remove items at specified indices."""
        for index in sorted(index_list, reverse=True):
            self.buffer.pop(index)
    
    def dequeue(self, length: int):
        """Remove and return last n items from buffer."""
        if length > len(self.buffer):
            raise ValueError("Requested length exceeds buffer size.")
        dequeued_items = self.buffer[-length:]
        self.buffer = self.buffer[:-length]
        return dequeued_items

    @property
    def summary(self) -> dict:
        """Get summary of box types and counts in buffer."""
        item_set = set(self.buffer)
        summary_dict = {item: self.buffer.count(item) for item in item_set}
        return summary_dict
    
    def update(self, sampled_item: 'Orthogonal3D | SimpleBlock') -> None:
        """Consume sampled raw boxes and refill the buffer."""
        if not self.buffer:
            raise ValueError("Cannot update an empty buffer")

        preserve_order = not isinstance(sampled_item, SimpleBlock)
        if isinstance(sampled_item, SimpleBlock):
            sampled_dim = sampled_item.box
            consume_count = sampled_item.consumed_count
            preserve_order = sampled_item.preserve_order
        else:
            sampled_dim = sampled_item
            consume_count = 1

        if preserve_order:
            next_item = self.buffer[0]
            if next_item != sampled_dim:
                raise ValueError(
                    f"Buffer sequence mismatch: expected {next_item.raw()}, got {sampled_dim.raw()}"
                )
            self.buffer.pop(0)
        else:
            for _ in range(consume_count):
                try:
                    self.buffer.remove(sampled_dim)
                except ValueError as exc:
                    raise ValueError(
                        f"Buffer does not contain enough boxes of type {sampled_dim.raw()}"
                    ) from exc

        self.fill_buffer()

__all__ = ["Buffer"]
