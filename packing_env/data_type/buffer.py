from __future__ import annotations

from .data_sampler import DataSampler
from .geometry import Orthogonal3D

class Buffer:
    """Buffer for managing sampled item dimensions."""
    
    def __init__(self, capacity: int, data_sampler: 'DataSampler'):
        self.capacity = capacity
        self.data_sampler = data_sampler
        self.buffer = []
        self.fill_buffer()

    def fill_buffer(self):
        """Fill buffer with sampled boxes up to capacity."""
        req_length = self.capacity - len(self.buffer)
        if req_length > 0:
            self.buffer.extend(self.data_sampler.sample(req_length))
        else:
            return

    def reset(self):
        """Reset buffer and refill sampled items."""
        self.buffer = []
        self.fill_buffer()

    @property
    def items(self) -> list['Orthogonal3D']:
        return self.buffer

    def dims(self) -> list[tuple[int, int, int]]:
        return [(int(item.dx), int(item.dy), int(item.dz)) for item in self.buffer]

    @property
    def has_items(self) -> bool:
        return len(self.buffer) > 0

    def sample_item(self) -> 'Orthogonal3D':
        """Return the next item dimensions without changing sequence order."""
        if not self.buffer:
            raise ValueError("No items available to sample from")
        return self.buffer[0]
    
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
    
    def update(self, sampled_item: 'Orthogonal3D') -> None:
        """Consume the next buffered item and refill.

        `sampled_item` is checked against the head item so callers cannot skip
        ahead when duplicate box dimensions are present.
        """
        if not self.buffer:
            raise ValueError("Cannot update an empty buffer")
        next_item = self.buffer[0]
        if next_item != sampled_item:
            raise ValueError(
                f"Buffer sequence mismatch: expected {next_item.raw()}, got {sampled_item.raw()}"
            )
        self.buffer.pop(0)
        self.fill_buffer()

__all__ = ["Buffer"]
