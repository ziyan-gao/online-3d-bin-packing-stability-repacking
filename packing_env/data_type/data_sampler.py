from __future__ import annotations

import os

import numpy as np

from .geometry import Orthogonal3D

class DataSampler:
    """Samples box dimensions from predefined distributions."""
    LEGACY_DATASET_SCALE = 60
    
    CJ_COLORS = [
        (0.8, 0.2, 0.2),  # Red
        (0.2, 0.8, 0.2),  # Green
        (0.2, 0.2, 0.8),  # Blue
        (0.8, 0.8, 0.2),  # Yellow
        (0.8, 0.2, 0.8),  # Magenta
    ]

    CJ_BOXES = [
        (200, 180, 134),
        (250, 180, 140),
        (250, 220, 158),
        (320, 240, 210),
        (320, 280, 257),
    ]

    BOX_DISTRIBUTIONS = {
        'random': {
            'boxes': [(120 + i * 60, 120 + j * 60, 120 + k * 60)
                    for i in range(4) for j in range(4) for k in range(4)],
            'distribution': None,  # Uniform by default
            'colors': None
        },
        'CJ': {
            'boxes': CJ_BOXES,
            'distribution': np.array([0.606, 0.126, 0.135, 0.095, 0.038]),
            'colors': CJ_COLORS
        },
        'CJ_uniform': {
            'boxes': CJ_BOXES,
            'distribution': None,  # Uniform over CJ boxes
            'colors': CJ_COLORS
        },
        'CJ_big': {
            'boxes': CJ_BOXES,
            'distribution': np.array([0.038, 0.095, 0.126, 0.135, 0.606]),
            'colors': CJ_COLORS
        },
        'CJ_random': {
            'boxes': CJ_BOXES,
            'distribution': 'random',  # Randomized in DataSampler
            'colors': CJ_COLORS
        }
    }
    
    def __init__(self, data_name: str, episode_index: int = 0):
        self.data_name = data_name
        self.episode_index = episode_index
        self.item_index = 0
        self.dataset_episodes = None

        if data_name not in self.BOX_DISTRIBUTIONS:
            dataset_path = os.path.expanduser(data_name)
            if not os.path.exists(dataset_path):
                choices = ", ".join(sorted(self.BOX_DISTRIBUTIONS))
                raise ValueError(
                    f"Unknown distribution or dataset path: {data_name}. "
                    f"Known distributions: {choices}"
                )
            self._init_dataset(dataset_path, episode_index)
            return

        config = self.BOX_DISTRIBUTIONS[data_name]
        self.box_set = config['boxes']
        self._dist_cfg = config['distribution']
        self.resample_distribution()
        self.colors = config['colors']

        # Auto-generate colors if not provided
        if self.colors is None:
            import matplotlib.pyplot as plt
            cmap = plt.cm.Set3(np.linspace(0, 1, len(self.box_set)))
            self.colors = [tuple(c) for c in cmap]
        
        # Create mapping from box dimensions to colors
        self.box_to_color = {tuple(box): color for box, color in zip(self.box_set, self.colors)}
        self.box_canonical_to_color = {}
        self.box_xy_to_color = {}
        for box, color in zip(self.box_set, self.colors):
            dx, dy, dz = (int(box[0]), int(box[1]), int(box[2]))
            key_3d = (*sorted((dx, dy)), dz)
            if key_3d not in self.box_canonical_to_color:
                self.box_canonical_to_color[key_3d] = color
            key_xy = tuple(sorted((dx, dy)))
            if key_xy not in self.box_xy_to_color:
                self.box_xy_to_color[key_xy] = color

    def _init_dataset(self, dataset_path: str, episode_index: int) -> None:
        import torch
        import matplotlib.pyplot as plt

        episodes = torch.load(dataset_path, map_location="cpu")
        if len(episodes) == 0:
            raise ValueError(f"Dataset is empty: {dataset_path}")

        self.dataset_episodes = episodes
        self.episode_index = episode_index % len(episodes)
        self.item_index = 0
        self._dist_cfg = None
        self.distribution = None

        unique_boxes = []
        seen = set()
        for episode in episodes:
            for raw_box in episode:
                box = tuple(int(v) * self.LEGACY_DATASET_SCALE for v in raw_box)
                if box not in seen:
                    seen.add(box)
                    unique_boxes.append(box)
        sentinel_box = (
            10 * self.LEGACY_DATASET_SCALE,
            10 * self.LEGACY_DATASET_SCALE,
            10 * self.LEGACY_DATASET_SCALE,
        )
        if sentinel_box not in seen:
            unique_boxes.append(sentinel_box)

        self.box_set = unique_boxes
        cmap = plt.cm.Set3(np.linspace(0, 1, len(self.box_set)))
        self.colors = [tuple(c) for c in cmap]
        self.box_to_color = {tuple(box): color for box, color in zip(self.box_set, self.colors)}
        self.box_canonical_to_color = {}
        self.box_xy_to_color = {}
        for box, color in zip(self.box_set, self.colors):
            dx, dy, dz = (int(box[0]), int(box[1]), int(box[2]))
            key_3d = (*sorted((dx, dy)), dz)
            if key_3d not in self.box_canonical_to_color:
                self.box_canonical_to_color[key_3d] = color
            key_xy = tuple(sorted((dx, dy)))
            if key_xy not in self.box_xy_to_color:
                self.box_xy_to_color[key_xy] = color

    @property
    def is_random_distribution(self) -> bool:
        return isinstance(self._dist_cfg, str) and self._dist_cfg == 'random'

    def resample_distribution(self) -> np.ndarray:
        """Refresh active probability vector.

        For '*_random' configs this draws a new Dirichlet distribution.
        For fixed/uniform configs this is a no-op refresh to canonical values.
        """
        if self.is_random_distribution:
            # Draw a fresh probability vector over all configured box types.
            self.distribution = np.random.dirichlet(np.ones(len(self.box_set)))
        elif self._dist_cfg is None:
            self.distribution = np.ones(len(self.box_set)) / len(self.box_set)
        else:
            self.distribution = np.asarray(self._dist_cfg, dtype=float)
        return self.distribution
    
    def get_color(self, box_dims: tuple) -> tuple:
        """Get color for a box by its first two dimensions (dx, dy).

        x/y swapped orientations map to the same color.
        """
        dims = tuple(int(v) for v in box_dims)
        if len(dims) < 2:
            return (0.5, 0.5, 0.5)

        dx, dy = dims[0], dims[1]
        if len(dims) >= 3:
            dz = dims[2]
            color = self.box_canonical_to_color.get((*sorted((dx, dy)), dz))
            if color is not None:
                return color

        color = self.box_xy_to_color.get(tuple(sorted((dx, dy))))
        if color is not None:
            return color

        return (0.5, 0.5, 0.5)
            
    def sample(self, length: int) -> list['Orthogonal3D']:
        """Sample boxes from distribution."""
        if self.dataset_episodes is not None:
            episode = self.dataset_episodes[self.episode_index]
            boxes = []
            for _ in range(length):
                if self.item_index < len(episode):
                    raw_box = episode[self.item_index]
                    self.item_index += 1
                else:
                    raw_box = (10, 10, 10)
                    self.item_index += 1
                boxes.append(
                    Orthogonal3D(
                        *(int(v) * self.LEGACY_DATASET_SCALE for v in raw_box)
                    )
                )
            return boxes

        idxs = np.random.choice(len(self.box_set), size=length, p=self.distribution)
        return [Orthogonal3D(*self.box_set[idx]) for idx in idxs]

__all__ = ["DataSampler"]
