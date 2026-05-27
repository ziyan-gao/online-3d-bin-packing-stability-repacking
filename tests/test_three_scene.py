from packing.three_scene import build_three_scene
from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item
from packing_env.gym_env import PackingEnv


def test_legend_keeps_selected_item_row_when_nothing_selected():
    env = PackingEnv(container_size=(600, 600, 600))
    env.reset(seed=123)

    scene = build_three_scene(env, "empty")

    assert "Selected: 0 items" in scene["legend"]


def test_legend_can_hold_previous_selected_item_count():
    env = PackingEnv(container_size=(600, 600, 600))
    env.reset(seed=123)
    env.visual_selected_count = 7

    scene = build_three_scene(env, "empty")

    assert "Selected: 7 items" in scene["legend"]


def test_legend_counts_source_items_inside_container():
    env = PackingEnv(container_size=(600, 600, 600))
    env.reset(seed=123)
    item = Item(FLB=Point3D(0, 0, 0), Dim=Orthogonal3D(100, 100, 150))
    item.source_item_count = 3
    env.pack(item)

    scene = build_three_scene(env, "packed")

    assert "In container: 3 items" in scene["legend"]
