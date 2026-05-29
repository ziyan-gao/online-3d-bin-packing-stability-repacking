import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packing_env.data_type import EmptyMaximalSpace, Orthogonal3D, Point3D, SimpleBlock
from packing_env.gym_env import PackingEnv


def make_layered_env(**kwargs):
    defaults = {
        "k_placement": 8,
        "buffer_capacity": 3,
        "container_size": (300, 300, 300),
        "stack_only": True,
        "use_simple_blocks": True,
        "policy_mode": "largest_block_baseline",
        "layered_achievability": True,
        "layered_num_chunks": 3,
    }
    defaults.update(kwargs)
    return PackingEnv(**defaults)


def ems_tuple(ems):
    return (
        int(ems.FLB.x),
        int(ems.FLB.y),
        int(ems.FLB.z),
        int(ems.Dim.dx),
        int(ems.Dim.dy),
        int(ems.Dim.dz),
    )


def test_layered_stage_windows_for_three_chunks():
    env = make_layered_env()

    assert env._layered_stage_window(1) == (0, 100)
    assert env._layered_stage_window(2) == (0, 200)
    assert env._layered_stage_window(3) == (100, 300)


def test_layered_stage_window_rejects_out_of_range_stage():
    env = make_layered_env()

    with pytest.raises(ValueError, match="layered stage"):
        env._layered_stage_window(0)
    with pytest.raises(ValueError, match="layered stage"):
        env._layered_stage_window(4)


def test_clip_ems_truncates_discards_and_preserves_spaces():
    env = make_layered_env()
    raw = [
        EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(300, 300, 300)),
        EmptyMaximalSpace(Point3D(0, 0, 220), Orthogonal3D(300, 300, 50)),
        EmptyMaximalSpace(Point3D(100, 0, 120), Orthogonal3D(100, 100, 30)),
    ]

    clipped = env._clip_ems_to_layer_window(raw, stage=2)

    assert [ems_tuple(ems) for ems in clipped] == [
        (0, 0, 0, 300, 300, 200),
        (100, 0, 120, 100, 100, 30),
    ]
    assert env.resolve_policy_ems_source(clipped[0]) is raw[0]
    assert env.resolve_policy_ems_source(clipped[1]) is raw[2]


def test_clip_ems_preserves_sources_for_duplicate_clipped_geometry():
    env = make_layered_env()
    raw = [
        EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(300, 300, 300)),
        EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(300, 300, 250)),
    ]

    clipped = env._clip_ems_to_layer_window(raw, stage=2)

    assert [ems_tuple(ems) for ems in clipped] == [
        (0, 0, 0, 300, 300, 200),
        (0, 0, 0, 300, 300, 200),
    ]
    assert env.resolve_policy_ems_source(clipped[0]) is raw[0]
    assert env.resolve_policy_ems_source(clipped[1]) is raw[1]
    assert env._policy_ems_source_by_id

    old_policy_ids = set(env._policy_ems_source_by_id)
    env.reset()

    assert env._policy_ems_source_by_id
    assert not old_policy_ids & set(env._policy_ems_source_by_id)
    reset_sources = list(env._policy_ems_source_by_id.values())
    reset_raw_ems = env.heu_ems.get_all_ems()
    assert len(reset_sources) == len(reset_raw_ems)
    assert all(source is raw for source, raw in zip(reset_sources, reset_raw_ems))


def test_largest_policy_block_uses_current_layer_clipped_ems(monkeypatch):
    env = make_layered_env(container_size=(300, 300, 300), layered_num_chunks=3)
    env.layered_stage = 1
    low_ems = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(300, 300, 300))
    monkeypatch.setattr(env.heu_ems, "get_all_ems", lambda: [low_ems])

    small = SimpleBlock(box=Orthogonal3D(100, 100, 80), stack_dims=(1, 1, 1))
    tall = SimpleBlock(box=Orthogonal3D(100, 100, 150), stack_dims=(1, 1, 1))
    env.buffer.simple_blocks = {
        small.box: [small],
        tall.box: [tall],
    }

    selected = env.select_largest_policy_block()

    assert selected is small
    assert env.buffer.all_blocks == [small]
    assert len(env.ems_list) == 1
    assert env.ems_list[0].Dim.dz == 100


def test_pack_resolves_layered_policy_ems_to_raw_source(monkeypatch):
    env = make_layered_env(container_size=(300, 300, 300), layered_num_chunks=3)
    env.layered_stage = 1
    block = SimpleBlock(box=Orthogonal3D(100, 100, 80), stack_dims=(1, 1, 1))
    raw_ems = env.heu_ems.get_all_ems()[0]

    env.get_pack_data(items=[block])
    policy_ems = env.ems_list[0]
    assert policy_ems is not raw_ems
    assert policy_ems.Dim.dz == 100

    captured = {}
    original_update = env.heu_ems.update

    def capture_update(*, box, selected_ems=None, hm=None, record_history=True):
        captured["selected_ems"] = selected_ems
        return original_update(
            box=box,
            selected_ems=selected_ems,
            hm=hm,
            record_history=record_history,
        )

    monkeypatch.setattr(env.heu_ems, "update", capture_update)

    env.pack(block.to_item(flb=policy_ems.FLB), selected_ems=policy_ems)

    assert captured["selected_ems"] is raw_ems
