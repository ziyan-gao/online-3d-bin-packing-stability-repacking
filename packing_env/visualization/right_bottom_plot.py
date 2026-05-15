from __future__ import annotations

from typing import TYPE_CHECKING

from .drawer import Drawer

if TYPE_CHECKING:
    from packing_env.data_type.buffer import Buffer
    from packing_env.data_type.item import Item

def draw_right_bottom_plot(
    drawer: Drawer,
    holding_items: list[Item],
    buffer: Buffer,
    buffer_scene: dict[str, object],
) -> tuple[dict[tuple[int, int, int], list[int]], dict[str, object]]:
    holding_item_types, holding_scene = drawer.draw_linear_items_scene(
        items=holding_items,
        color_lookup=buffer.data_sampler.get_color,
        row=2,
        col=2,
        fixed_x_range=buffer_scene["xaxis"]["range"][1],
        fixed_y_range=tuple(buffer_scene["yaxis"]["range"]),
        fixed_z_range=tuple(buffer_scene["zaxis"]["range"]),
    )
    holding_scene["aspectratio"] = dict(buffer_scene["aspectratio"])
    holding_scene["aspectratio"]["x"] = buffer_scene["aspectratio"]["x"] / 2.0
    drawer.draw_empty_staging_area(holding_scene, row=2, col=2)
    return holding_item_types, holding_scene
