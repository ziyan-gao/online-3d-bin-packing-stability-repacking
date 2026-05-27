from __future__ import annotations

from typing import TYPE_CHECKING

from .drawer import Drawer

if TYPE_CHECKING:
    from packing_env.data_type.buffer import Buffer


def draw_right_up_plot(
    drawer: Drawer, buffer: Buffer
) -> tuple[dict[tuple[int, int, int], list[int]], dict[str, object]]:
    return drawer.draw_linear_items_scene(
        items=buffer.items,
        color_lookup=buffer.data_sampler.get_color,
        highlight_indices=getattr(buffer, "visual_highlight_indices", None),
        row=1,
        col=2,
    )
