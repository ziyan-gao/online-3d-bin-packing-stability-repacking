import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packing_env.data_type.buffer import Buffer
from packing_env.data_type.ems import EmptyMaximalSpace
from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item, SimpleBlock
from packing_env.gym_env import PackingEnv
from packing import test_utils
from packing.mcts import rollout


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


class FakeEms:
    def __init__(self, dim, flb=None):
        self.Dim = dim
        self.FLB = flb or Point3D(0, 0, 0)

    def include(self, other):
        return (
            self.Dim.Gdx >= other.Gdx
            and self.Dim.Gdy >= other.Gdy
            and self.Dim.Gdz >= other.Gdz
        )


class AlwaysStable:
    def __call__(self, o3d, hm, candidates):
        return [Point3D(0, 0, 0) for _ in candidates], np.ones(len(candidates), dtype=bool)


class CountingStable:
    def __init__(self, stable=True):
        self.stable = bool(stable)
        self.calls = []

    def __call__(self, o3d, hm, candidates):
        self.calls.append(o3d.to_dim_key())
        if self.stable:
            return [Point3D(0, 0, 0) for _ in candidates], np.ones(len(candidates), dtype=bool)
        return [None for _ in candidates], np.zeros(len(candidates), dtype=bool)


def test_unique_item_orientations_deduplicates_repeated_buffer_types():
    from packing_env.heu_ems import EMS

    orientations = EMS._unique_item_orientations(
        [
            Orthogonal3D(100, 100, 50),
            Orthogonal3D(100, 100, 50),
            Orthogonal3D(100, 200, 50),
            Orthogonal3D(100, 200, 50),
        ]
    )

    assert [orientation.to_dim_key() for orientation in orientations] == [
        (100, 100, 50),
        (100, 200, 50),
        (200, 100, 50),
    ]


def test_prune_unstable_removes_ems_when_stability_fails():
    from packing_env.heu_ems import EMS

    ems_manager = EMS(container=Orthogonal3D(300, 300, 300), remove_inscribed=False)
    unstable = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(200, 200, 100))
    ems_manager._EMS__ems_list = [unstable]
    ems_manager._rebuild_index()

    hm = PackingEnv(container_size=(300, 300, 300)).hm
    ems_manager.prune_unstable(
        hm=hm,
        feasibility_map=CountingStable(stable=False),
        item_types=[Orthogonal3D(100, 100, 50)],
    )

    assert ems_manager.get_all_ems() == []


def test_simpleblock_dimensions_transpose_and_to_item():
    box = Orthogonal3D(100, 200, 50)
    block = SimpleBlock(box=box, stack_dims=(1, 1, 3), buffer_space=10)

    assert block.no_boxes_wrt_axis == (1, 1, 3)
    assert block.raw().tolist() == [100, 200, 150]
    assert block.Virtual_Dim.raw().tolist() == [110, 210, 150]

    transposed = block.transpose()
    assert transposed.box.raw().tolist() == [200, 100, 50]
    assert transposed.no_boxes_wrt_axis == (1, 1, 3)
    assert transposed.raw().tolist() == [200, 100, 150]

    placed = block.to_item(flb=Point3D(0, 0, 0), rotate_xy=True)
    assert isinstance(placed, Item)
    assert placed.Dim.raw().tolist() == [200, 100, 150]
    assert placed.buffer_space == 10
    assert placed.rot is True


def test_buffer_generates_stack_only_blocks_and_refills_after_update():
    box_a = Orthogonal3D(100, 100, 50)
    box_b = Orthogonal3D(200, 100, 50)
    buffer = Buffer(
        capacity=3,
        data_sampler=FakeSampler([box_a, box_a, box_a, box_b, box_b]),
        stack_only=True,
        container_size=(600, 600, 600),
    )

    assert buffer.summary == {box_a: 3}
    assert sorted(block.no_boxes_wrt_axis for block in buffer.all_blocks) == [
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 3),
    ]

    two_box_stack = next(
        block for block in buffer.all_blocks if block.no_boxes_wrt_axis == (1, 1, 2)
    )
    buffer.update(two_box_stack)

    assert len(buffer.buffer) == 3
    assert buffer.summary == {box_a: 1, box_b: 2}
    assert sorted(block.no_boxes_wrt_axis for block in buffer.simple_blocks[box_b]) == [
        (1, 1, 1),
        (1, 1, 2),
    ]


def test_sample_item_keeps_single_box_fifo_update_contract():
    box_a = Orthogonal3D(100, 100, 50)
    box_b = Orthogonal3D(200, 100, 50)
    buffer = Buffer(capacity=2, data_sampler=FakeSampler([box_a, box_b, box_b]))

    sampled = buffer.sample_item()
    assert sampled.box == box_a
    assert sampled.consumed_count == 1

    buffer.update(sampled)
    assert buffer.dims() == [(200, 100, 50), (200, 100, 50)]


def test_generated_single_box_block_consumes_matching_type_not_fifo():
    box_a = Orthogonal3D(100, 100, 50)
    box_b = Orthogonal3D(200, 100, 50)
    buffer = Buffer(
        capacity=2,
        data_sampler=FakeSampler([box_a, box_b, box_a]),
        stack_only=True,
        container_size=(600, 600, 600),
    )

    selected = next(block for block in buffer.all_blocks if block.box == box_b)
    assert selected.consumed_count == 1

    buffer.update(selected)

    assert buffer.dims() == [(100, 100, 50), (100, 100, 50)]


def test_update_usable_keeps_block_when_transposed_orientation_fits():
    box = Orthogonal3D(300, 100, 50)
    buffer = Buffer(capacity=1, data_sampler=FakeSampler([box]), stack_only=True)
    ems = FakeEms(Orthogonal3D(100, 300, 50))

    buffer.update_usable([ems], AlwaysStable(), hm=None)

    assert len(buffer.all_blocks) == 1
    assert buffer.all_blocks[0].box == box


def test_packing_env_stack_only_mode_places_simpleblock_as_item():
    box = Orthogonal3D(100, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=3,
        container_size=(600, 600, 600),
        stack_only=True,
    )
    env.buffer = Buffer(
        capacity=3,
        data_sampler=FakeSampler([box, box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )

    obs = env.get_next_observation()
    valid_actions = np.flatnonzero(obs["action_mask"].reshape(-1))
    assert len(valid_actions) > 0

    _, reward, done, _, _ = env.step(int(valid_actions[0]))

    assert reward > 0
    assert len(env.container.placed_items) == 1
    assert isinstance(env.container.placed_items[0], Item)
    assert len(env.buffer.buffer) == env.buffer.capacity
    assert done in (True, False)


def test_step_prunes_ems_after_buffer_update(monkeypatch):
    first = Orthogonal3D(100, 100, 50)
    second = Orthogonal3D(200, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=1,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
    )
    env.buffer = Buffer(
        capacity=1,
        data_sampler=FakeSampler([first, second, second]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    env.selected_item = env.buffer.sample_blocks()
    env.candidates = np.array([[0, 0, 0, 600, 600, 600]], dtype=np.int32)
    env.ems_list = env.heu_ems.get_all_ems()
    captured = {}

    def capture_prune(hm, feasibility_map, item_types):
        captured["item_types"] = list(item_types)

    monkeypatch.setattr(env.heu_ems, "prune_unstable", capture_prune)
    box = env.selected_item.to_item(Point3D(0, 0, 0))

    env._step(env.selected_item, box, env.ems_list[0])

    assert captured["item_types"] == [second]
    assert env.buffer.summary == {second: 1}


def test_pack_prunes_ems_with_current_buffer_item_types(monkeypatch):
    buffer_type = Orthogonal3D(100, 100, 50)
    placed_type = Orthogonal3D(200, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=2,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
    )
    env.buffer = Buffer(
        capacity=2,
        data_sampler=FakeSampler([buffer_type, buffer_type]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    captured = {}

    def capture_prune(hm, feasibility_map, item_types):
        captured["item_types"] = list(item_types)

    monkeypatch.setattr(env.heu_ems, "prune_unstable", capture_prune)
    selected_ems = env.heu_ems.get_all_ems()[0]
    env.pack(Item(Point3D(0, 0, 0), placed_type), selected_ems=selected_ems)

    assert captured["item_types"] == [buffer_type, placed_type]
    assert env.buffer.summary == {buffer_type: 2}


def test_pack_prunes_ems_with_holding_item_types(monkeypatch):
    buffer_type = Orthogonal3D(100, 100, 50)
    holding_type = Orthogonal3D(300, 100, 50)
    placed_type = Orthogonal3D(200, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=1,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
    )
    env.buffer = Buffer(
        capacity=1,
        data_sampler=FakeSampler([buffer_type]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    env.container.holding_list.append(Item(Point3D(0, 0, 0), holding_type))
    captured = {}

    def capture_prune(hm, feasibility_map, item_types):
        captured["item_types"] = list(item_types)

    monkeypatch.setattr(env.heu_ems, "prune_unstable", capture_prune)
    selected_ems = env.heu_ems.get_all_ems()[0]
    env.pack(Item(Point3D(0, 0, 0), placed_type), selected_ems=selected_ems)

    assert captured["item_types"] == [buffer_type, holding_type, placed_type]


def test_largest_block_baseline_observation_keeps_single_largest_stack():
    box = Orthogonal3D(100, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=3,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="largest_block_baseline",
    )
    env.reset(seed=101)
    env.buffer = Buffer(
        capacity=3,
        data_sampler=FakeSampler([box, box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )

    env.select_largest_policy_block()
    obs = env.get_pack_data(env.buffer.sample_blocks(deterministic=True))

    assert obs["item_raw"].shape[0] == 1
    assert obs["item_raw"][0].tolist() == [100.0, 100.0, 150.0]


class NoopVisualizer:
    def push(self, env, title):
        pass


class ShapeCheckingAgent:
    def __init__(self):
        self.observed_item_count = None

    def predict(self, obs, value_deterministic=True, logits_deterministic=True):
        self.observed_item_count = len(obs["item_raw"])
        return DeterministicPlacementAgent().predict(obs, value_deterministic, logits_deterministic)


class DeterministicPlacementAgent:
    def predict(self, obs, value_deterministic=True, logits_deterministic=True):
        mask = np.asarray(obs["mask"], dtype=bool)
        item_raw = np.asarray(obs["item_raw"], dtype=np.int32)
        ems = np.asarray(obs["ems_unnorm"], dtype=np.int32)

        placable_items = np.where(mask.reshape(mask.shape[0], -1).any(axis=1))[0]
        if len(placable_items) == 0:
            raise ValueError("No placable item is available in the observation mask.")

        item_idx = int(placable_items[0])
        rot_idx, placement_idx = np.argwhere(mask[item_idx])[0]
        act_idx = int(placement_idx + rot_idx * mask.shape[2])
        dims = item_raw[item_idx].copy()
        if rot_idx:
            dims = dims[[1, 0, 2]]
        buffer_space = int(
            np.asarray(
                obs.get("buffer_space", np.zeros(len(item_raw), dtype=np.int32))
            ).reshape(-1)[item_idx]
        )

        box = Item(
            FLB=Point3D(*map(int, ems[item_idx, placement_idx, :3])),
            Dim=Orthogonal3D(*map(int, dims)),
            buffer_space=buffer_space,
        )
        return box, (item_idx, act_idx, float(np.prod(item_raw[item_idx])))


def test_pack_until_blocked_uses_largest_usable_simple_block():
    box = Orthogonal3D(100, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=3,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
    )
    env.buffer = Buffer(
        capacity=3,
        data_sampler=FakeSampler([box, box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    config = test_utils.TestConfig(
        max_steps=1,
        target_util=0.001,
        stack_only=True,
        use_simple_blocks=True,
    )

    agent = ShapeCheckingAgent()

    _, _, reached, episode_reward = test_utils.pack_until_blocked(
        config,
        env,
        agent,
        seed=101,
        visualizer=NoopVisualizer(),
    )

    assert reached is True
    assert episode_reward == pytest.approx(env.container.utilization)
    assert agent.observed_item_count == 1
    assert len(env.container.placed_items) == 1
    assert env.container.placed_items[0].Dim.raw().tolist() == [100, 100, 150]
    assert test_utils.contained_item_count(env) == 3
    assert len(env.buffer.buffer) == env.buffer.capacity


def test_pack_until_blocked_prunes_with_post_update_buffer_types(monkeypatch):
    first = Orthogonal3D(100, 100, 50)
    second = Orthogonal3D(200, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=1,
        container_size=(600, 600, 600),
    )
    env.buffer = Buffer(capacity=1, data_sampler=FakeSampler([first, second]))
    config = test_utils.TestConfig(max_steps=1, target_util=0.001)
    captured = {}
    original_pack = env.pack

    def capture_pack(box, selected_ems=None, pruning_item_types=None):
        captured["item_types"] = list(pruning_item_types)
        return original_pack(
            box,
            selected_ems=selected_ems,
            pruning_item_types=pruning_item_types,
        )

    monkeypatch.setattr(env, "pack", capture_pack)

    test_utils.pack_until_blocked(
        config,
        env,
        DeterministicPlacementAgent(),
        seed=101,
        visualizer=NoopVisualizer(),
    )

    assert captured["item_types"] == [second]
    assert env.buffer.summary == {second: 1}


def test_mcts_rollout_prunes_with_post_pack_candidate_dims(monkeypatch):
    incoming = Orthogonal3D(100, 100, 50)
    holding = Orthogonal3D(200, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=1,
        container_size=(600, 600, 600),
    )
    env.container.holding_list.append(Item(Point3D(0, 0, 0), holding))
    captured = {"item_types": []}
    original_pack = env.pack

    def capture_pack(box, selected_ems=None, pruning_item_types=None):
        captured["item_types"].append(list(pruning_item_types))
        return original_pack(
            box,
            selected_ems=selected_ems,
            pruning_item_types=pruning_item_types,
        )

    monkeypatch.setattr(env, "pack", capture_pack)

    rollout(
        env,
        DeterministicPlacementAgent(),
        incoming,
        Uti_requirement=0.0,
    )

    assert [incoming.to_dim_key()] in captured["item_types"]


def test_pack_until_blocked_handles_no_usable_simple_blocks():
    box = Orthogonal3D(700, 700, 700)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=1,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
    )
    env.buffer = Buffer(
        capacity=1,
        data_sampler=FakeSampler([box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    config = test_utils.TestConfig(
        max_steps=1,
        stack_only=True,
        use_simple_blocks=True,
    )

    blocked_item, pack_history, reached, episode_reward = test_utils.pack_until_blocked(
        config,
        env,
        DeterministicPlacementAgent(),
        seed=101,
        visualizer=NoopVisualizer(),
    )

    assert blocked_item is None
    assert pack_history == []
    assert reached is False
    assert episode_reward == 0.0


def test_get_pack_data_accepts_empty_items():
    env = PackingEnv(k_placement=4, container_size=(600, 600, 600))

    obs = env.get_pack_data([])

    assert obs["new_item"].shape == (0, 3)
    assert obs["action_mask"].shape == (0, 2, 4)
    assert obs["done"].shape == (0,)
    assert obs["placable"] is False


def test_get_pack_data_filters_squeezed_ems_before_k_cap():
    env = PackingEnv(k_placement=1, container_size=(600, 600, 600))
    squeezed = EmptyMaximalSpace(
        FLB=Point3D(0, 0, 0),
        Dim=Orthogonal3D(90, 90, 600),
    )
    usable = EmptyMaximalSpace(
        FLB=Point3D(100, 0, 0),
        Dim=Orthogonal3D(120, 100, 100),
    )
    env.heu_ems._EMS__ems_list = [squeezed, usable]
    env.heu_ems._rebuild_index()

    env.get_pack_data(Orthogonal3D(100, 100, 100))

    assert env.ems_list == [usable]
    assert env.candidates[0, :3].tolist() == [100, 0, 0]


def test_same_flb_dominated_ems_are_pruned():
    from packing_env.heu_ems import EMS

    flb = Point3D(0, 0, 0)
    large = EmptyMaximalSpace(FLB=flb, Dim=Orthogonal3D(300, 300, 300))
    small = EmptyMaximalSpace(FLB=flb, Dim=Orthogonal3D(200, 300, 300))
    different_anchor = EmptyMaximalSpace(
        FLB=Point3D(100, 0, 0),
        Dim=Orthogonal3D(200, 300, 300),
    )

    pruned = EMS._remove_same_flb_dominated([small, large, different_anchor])

    assert pruned == [large, different_anchor]


def test_prune_unstable_removes_unsupported_ems_by_heightmap():
    from packing_env.heu_ems import EMS

    ems_manager = EMS(container=Orthogonal3D(300, 300, 300), remove_inscribed=False)
    supported = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(100, 100, 100))
    floating = EmptyMaximalSpace(Point3D(100, 0, 50), Orthogonal3D(100, 100, 100))
    ems_manager._EMS__ems_list = [supported, floating]
    ems_manager._rebuild_index()

    hm = PackingEnv(container_size=(300, 300, 300)).hm
    ems_manager.prune_unstable(
        hm=hm,
        feasibility_map=AlwaysStable(),
        item_types=[Orthogonal3D(50, 50, 50)],
    )

    assert ems_manager.get_all_ems() == [supported]


def test_prune_unstable_keeps_supported_stable_fitting_ems():
    from packing_env.heu_ems import EMS

    ems_manager = EMS(container=Orthogonal3D(300, 300, 300), remove_inscribed=False)
    usable = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(100, 100, 100))
    too_small = EmptyMaximalSpace(Point3D(100, 0, 0), Orthogonal3D(40, 40, 100))
    ems_manager._EMS__ems_list = [usable, too_small]
    ems_manager._rebuild_index()

    hm = PackingEnv(container_size=(300, 300, 300)).hm
    ems_manager.prune_unstable(
        hm=hm,
        feasibility_map=AlwaysStable(),
        item_types=[Orthogonal3D(50, 50, 50)],
    )

    assert ems_manager.get_all_ems() == [usable]


def test_prune_unstable_tests_rotated_item_orientation():
    from packing_env.heu_ems import EMS

    ems_manager = EMS(container=Orthogonal3D(300, 300, 300), remove_inscribed=False)
    rotated_fit = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(100, 200, 100))
    ems_manager._EMS__ems_list = [rotated_fit]
    ems_manager._rebuild_index()

    hm = PackingEnv(container_size=(300, 300, 300)).hm
    ems_manager.prune_unstable(
        hm=hm,
        feasibility_map=AlwaysStable(),
        item_types=[Orthogonal3D(200, 100, 50)],
    )

    assert ems_manager.get_all_ems() == [rotated_fit]


def test_item_fit_ems_ranking_prefers_low_z_then_large_volume():
    env = PackingEnv(k_placement=2, container_size=(600, 600, 600))
    high_large = EmptyMaximalSpace(
        FLB=Point3D(0, 0, 100),
        Dim=Orthogonal3D(500, 500, 500),
    )
    low_small = EmptyMaximalSpace(
        FLB=Point3D(100, 0, 0),
        Dim=Orthogonal3D(120, 120, 120),
    )
    low_large = EmptyMaximalSpace(
        FLB=Point3D(0, 0, 0),
        Dim=Orthogonal3D(300, 300, 300),
    )
    env.heu_ems._EMS__ems_list = [high_large, low_small, low_large]
    env.heu_ems._rebuild_index()

    ranked = env._get_item_fit_ems_list(Orthogonal3D(100, 100, 100), prefer_stable=False)

    assert ranked == [low_large, low_small]
