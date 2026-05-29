import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packing_env.data_type import EmptyMaximalSpace, Orthogonal3D, Point3D
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
