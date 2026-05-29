import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packing_env.data_type import (
    EmptyMaximalSpace,
    OrientedBlock,
    Orthogonal3D,
    Point3D,
    SimpleBlock,
)
from packing_env.data_type.buffer import Buffer
from packing_env.gym_env import PackingEnv


class FakeSampler:
    is_random_distribution = False

    def __init__(self, items):
        self.items = list(items)
        self.cursor = 0

    def sample(self, n):
        sampled = []
        for _ in range(n):
            sampled.append(self.items[self.cursor % len(self.items)])
            self.cursor += 1
        return sampled


def test_oriented_block_preserves_normal_dimensions_and_metadata():
    block = SimpleBlock(
        box=Orthogonal3D(100, 200, 100),
        stack_dims=(1, 1, 1),
        buffer_space=10,
    )

    candidate = OrientedBlock.from_block(block, source_index=2, rotate_xy=False)

    assert candidate.Dim.raw().tolist() == [100, 200, 100]
    assert candidate.Virtual_Dim.raw().tolist() == [110, 210, 100]
    assert candidate.consumed_count == 1
    assert candidate.source_index == 2
    assert candidate.rotate_xy is False
    assert candidate.feature_row(container_size=(600, 600, 600)).shape == (8,)


def test_oriented_block_to_item_applies_transpose_once():
    block = SimpleBlock(
        box=Orthogonal3D(100, 200, 100),
        stack_dims=(1, 1, 1),
        buffer_space=10,
    )

    placed = OrientedBlock.from_block(
        block,
        source_index=0,
        rotate_xy=True,
    ).to_item(Point3D(10, 20, 3))

    np.testing.assert_array_equal(placed.Dim.raw(), np.array([200, 100, 100]))
    np.testing.assert_array_equal(placed.Virtual_Dim.raw(), np.array([210, 110, 100]))
    assert placed.rot is True


def test_cascaded_candidates_expose_only_stable_oriented_blocks():
    box = Orthogonal3D(300, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=2,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.reset(seed=1)
    env.buffer = Buffer(
        capacity=2,
        data_sampler=FakeSampler([box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )

    candidates, ems_list, loading_mask = env.get_cascaded_block_candidates()

    assert len(candidates) > 0
    assert loading_mask.shape == (len(candidates), env.k_placement)
    assert loading_mask.any(axis=1).all()
    assert all(candidate.consumed_count in (1, 2) for candidate in candidates)


def test_cascaded_vectorized_fit_mask_matches_scalar_include_checks():
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=2,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    blocks = [
        SimpleBlock(
            box=Orthogonal3D(300, 100, 50),
            stack_dims=(1, 1, 1),
            buffer_space=10,
        ),
        SimpleBlock(
            box=Orthogonal3D(160, 250, 80),
            stack_dims=(1, 1, 1),
            buffer_space=10,
        ),
    ]
    oriented = [
        OrientedBlock.from_block(block, source_index=index, rotate_xy=rotate_xy)
        for index, block in enumerate(blocks)
        for rotate_xy in (False, True)
    ]
    ems_list = [
        EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(320, 120, 100)),
        EmptyMaximalSpace(Point3D(0, 150, 0), Orthogonal3D(120, 320, 100)),
        EmptyMaximalSpace(Point3D(200, 0, 0), Orthogonal3D(170, 260, 100)),
        EmptyMaximalSpace(Point3D(0, 0, 100), Orthogonal3D(90, 90, 100)),
    ]

    expected = np.array(
        [[ems.include(candidate.Virtual_Dim) for ems in ems_list] for candidate in oriented],
        dtype=bool,
    )

    np.testing.assert_array_equal(
        env._build_cascaded_fit_mask(oriented, ems_list),
        expected,
    )


def test_cascaded_stability_cache_reuses_repeated_dim_ems_checks(monkeypatch):
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=2,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.reset(seed=1)
    ems = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(400, 400, 100))
    dim = Orthogonal3D(100, 100, 50)
    taller_stack_dim = Orthogonal3D(100, 100, 150)
    call_count = 0
    original_heu_stable = env.heu_stable

    def counting_heu_stable(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_heu_stable(*args, **kwargs)

    monkeypatch.setattr(env, "heu_stable", counting_heu_stable)
    cache = {}

    first = env._get_cached_stability(dim, [ems], cache)
    second = env._get_cached_stability(dim, [ems], cache)
    third = env._get_cached_stability(taller_stack_dim, [ems], cache)

    assert call_count == 1
    assert first == second
    assert first == third


def test_cascaded_observation_contains_blocks_and_loading_mask():
    box = Orthogonal3D(300, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=2,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.reset(seed=1)
    env.buffer = Buffer(
        capacity=2,
        data_sampler=FakeSampler([box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )

    obs = env.get_next_observation()

    assert obs["oriented_blocks"].shape == (env.max_oriented_blocks, 8)
    assert obs["block_mask"].shape == (env.max_oriented_blocks,)
    assert obs["ems"].shape == (env.k_placement, 6)
    assert obs["loading_mask"].shape == (env.max_oriented_blocks, env.k_placement)
    assert obs["action_mask"].shape == (env.max_oriented_blocks, env.k_placement)
    assert obs["block_mask"].any()

    valid_rows = obs["block_mask"].astype(bool)
    assert obs["loading_mask"][valid_rows].any(axis=1).all()
    assert np.array_equal(obs["action_mask"], obs["loading_mask"])


def test_cascaded_observation_space_contains_large_buffer_stack_features():
    box = Orthogonal3D(100, 100, 100)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=13,
        container_size=(2000, 2000, 2000),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.reset(seed=1)
    env.buffer = Buffer(
        capacity=13,
        data_sampler=FakeSampler([box]),
        stack_only=True,
        container_size=(2000, 2000, 2000),
    )

    obs = env.get_next_observation()

    assert obs["oriented_blocks"].max() <= 1.0
    assert env.observation_space.contains(obs)


def test_cascaded_action_decoding_applies_orientation_once():
    box = Orthogonal3D(300, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=2,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.reset(seed=1)
    env.buffer = Buffer(
        capacity=2,
        data_sampler=FakeSampler([box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    obs = env.get_next_observation()
    oriented_index, ems_index = np.argwhere(obs["action_mask"])[0]
    assert obs["action_mask"][oriented_index, ems_index]

    flat_action = int(oriented_index * env.k_placement + ems_index)
    _, reward, _, _, _ = env.step(flat_action)

    assert reward > 0
    assert len(env.container.placed_items) == 1
    placed = env.container.placed_items[0]
    np.testing.assert_array_equal(placed.Dim.raw(), env.last_placed_source.Dim.raw())
