from __future__ import annotations

from typing import TYPE_CHECKING

from .config import VisualConfig
from .drawer import Drawer

if TYPE_CHECKING:
    from packing_env.data_type.buffer import Buffer
    from packing_env.data_type.ems import EmptyMaximalSpace
    from packing_env.data_type.geometry import Point3D
    from packing_env.data_type.item import Item
    from packing_env.heu_stable import Heu_Stable


def draw_left_plot(
    drawer: Drawer,
    buffer: Buffer,
    heu_stable: Heu_Stable,
    ems_list: list[EmptyMaximalSpace] | None,
    anchor_points: list[Point3D] | None,
    highlighted_items: list[Item] | None,
    repacked_items: list[Item] | None,
    repack_spaces: list[Item] | None,
    config: VisualConfig,
) -> None:
    for cached_item, item_dims, vis_data in heu_stable.support_vis_records:
        box_color = buffer.data_sampler.get_color(item_dims)
        base_color = drawer._to_rgb(box_color)
        has_buffer = getattr(cached_item, "buffer_space", 0) > 0
        drawer.draw_support(vis_data, base_color=base_color, has_buffer=has_buffer)

    if ems_list is not None:
        for ems_idx, ems in enumerate(ems_list):
            drawer.draw_ems(ems, ems_idx)

    drawer.draw_anchor_points(anchor_points)

    if highlighted_items is not None:
        for item in highlighted_items:
            drawer.draw_item(
                item,
                color=config.placed_item_color,
                opacity=0.5,
                name="Just Unpacked",
                edge_color=config.placed_item_edge_color,
            )
    if repack_spaces is not None:
        for item in repack_spaces:
            drawer.draw_item(
                item,
                color=config.repack_space_color,
                opacity=0.35,
                name="Repack Target Space",
                edge_color=config.repack_space_edge_color,
                inflate=6.0,
            )
    if repacked_items is not None:
        for item in repacked_items:
            drawer.draw_item(
                item,
                color=config.repacked_item_color,
                opacity=0.5,
                name="Repack Source Item",
                edge_color=config.repacked_item_edge_color,
                inflate=2.0,
            )
