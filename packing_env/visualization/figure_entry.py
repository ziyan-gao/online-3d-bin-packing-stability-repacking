from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .config import DEFAULT_VISUAL_CONFIG, VisualConfig
from .left_plot import draw_left_plot
from .right_bottom_plot import draw_right_bottom_plot
from .right_up_plot import draw_right_up_plot
from .drawer import Drawer

if TYPE_CHECKING:
    from packing_env.gym_env import PackingEnv
    from packing_env.data_type.item import Item


class PackVisualizer:
    def __init__(
        self,
        env: PackingEnv,
        config: VisualConfig = DEFAULT_VISUAL_CONFIG,
        title: str = "Buffer + Support",
        show_anchor: bool = True,
        show_ems: bool = False,
    ) -> None:
        self.env = env
        self.config = config
        self.title = title
        self.show_anchor = show_anchor
        self.show_ems = show_ems
        self._previous_placed_items = list(env.container.placed_items)
        self.highlighted_items: list[Item] | None = None
        self.repacked_items: list[Item] | None = None
        self.repack_spaces: list[Item] | None = None

    def refresh(self) -> tuple[object, tuple[object, object]]:
        self._refresh_overlays()
        result = self._build_with_current_overlays()
        self._previous_placed_items = list(self.env.container.placed_items)
        return result

    def reset_history(self) -> None:
        self._previous_placed_items = list(self.env.container.placed_items)

    def _build_with_current_overlays(self) -> tuple[object, tuple[object, object]]:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return self._build_fallback_figure()

        fig = self._make_figure(make_subplots)
        drawer = Drawer(go=go, fig=fig, config=self.config)
        self._draw_left_plot(drawer)
        buffer_item_types, buffer_scene = draw_right_up_plot(drawer, self.env.buffer)
        holding_items = self._holding_items()
        holding_item_types, holding_scene = draw_right_bottom_plot(
            drawer,
            holding_items=holding_items,
            buffer=self.env.buffer,
            buffer_scene=buffer_scene,
        )
        self._apply_layout(fig, buffer_scene, holding_scene, len(holding_items))
        self._add_legend(fig, buffer_item_types, holding_item_types)
        return fig, (None, None)

    def _refresh_overlays(self) -> None:
        previous_by_key = {item.to_key(): item for item in self._previous_placed_items}
        current_by_key = {item.to_key(): item for item in self.env.container.placed_items}
        removed_items = [
            previous_item
            for key, previous_item in previous_by_key.items()
            if key not in current_by_key
        ]
        added_items = [
            current_item
            for key, current_item in current_by_key.items()
            if key not in previous_by_key
        ]

        if removed_items and added_items:
            self.highlighted_items = None
            self.repacked_items = removed_items
            self.repack_spaces = added_items
        elif removed_items:
            self.highlighted_items = removed_items
            self.repacked_items = None
            self.repack_spaces = None
        else:
            self.highlighted_items = None
            self.repacked_items = None
            self.repack_spaces = None

    def _build_fallback_figure(self) -> tuple[object, tuple[object, object]]:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=self.config.fallback_size)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Plotly is not installed", ha="center", va="center")
        ax.axis("off")
        return fig, (ax, None)

    def _make_figure(self, make_subplots):
        return make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene", "rowspan": 2}, {"type": "scene"}],
                [None, {"type": "scene"}],
            ],
            subplot_titles=self.config.subplot_titles,
            horizontal_spacing=self.config.horizontal_spacing,
            vertical_spacing=self.config.vertical_spacing,
            column_widths=list(self.config.column_widths),
            row_heights=list(self.config.row_heights),
        )

    def _draw_left_plot(self, drawer: Drawer) -> None:
        anchor_points = self.env.heu_ems.get_anchor_points() if self.show_anchor else None
        ems_list = self.env.heu_ems.get_ems_list() if self.show_ems else None
        draw_left_plot(
            drawer,
            buffer=self.env.buffer,
            heu_stable=self.env.heu_stable,
            ems_list=ems_list,
            anchor_points=anchor_points,
            highlighted_items=self.highlighted_items,
            repacked_items=self.repacked_items,
            repack_spaces=self.repack_spaces,
            config=self.config,
        )

    def _apply_layout(
        self,
        fig,
        buffer_scene: dict[str, object],
        holding_scene: dict[str, object],
        holding_count: int,
    ) -> None:
        fig.update_layout(
            title=self._figure_title(len(self.env.buffer.items), holding_count),
            margin=dict(l=10, r=10, t=70, b=95),
            scene=self._left_scene_layout(),
            scene2=buffer_scene,
            scene3=holding_scene,
        )

    def _left_scene_layout(self) -> dict[str, object]:
        container = self.env.container
        if container:
            container_dx = container.dx
            container_dy = container.dy
            container_dz = container.dz
        else:
            container_dx, container_dy, container_dz = self.config.default_container_size

        angle_deg = self._left_camera_angle_deg()
        angle_rad = np.deg2rad(angle_deg)
        scene_eye = dict(
            x=float(self.config.left_camera_radius * np.cos(angle_rad)),
            y=float(self.config.left_camera_radius * np.sin(angle_rad)),
            z=self.config.left_camera_z,
        )
        pad = self.config.plot_edge_padding
        scene_x_range = [-pad, float(container_dx) + pad]
        scene_y_range = [-pad, float(container_dy) + pad]
        scene_z_range = [-pad, float(container_dz) + pad]
        scene_x_span = scene_x_range[1] - scene_x_range[0]
        scene_y_span = scene_y_range[1] - scene_y_range[0]
        scene_z_span = scene_z_range[1] - scene_z_range[0]
        scene_max_span = max(scene_x_span, scene_y_span, scene_z_span)

        return dict(
            xaxis_title=self.config.axis_title_x,
            yaxis_title=self.config.axis_title_y,
            zaxis_title=self.config.axis_title_z,
            xaxis=dict(range=scene_x_range),
            yaxis=dict(range=scene_y_range),
            zaxis=dict(range=scene_z_range),
            aspectmode="manual",
            aspectratio=dict(
                x=scene_x_span / scene_max_span,
                y=scene_y_span / scene_max_span,
                z=scene_z_span / scene_max_span,
            ),
            camera=dict(eye=scene_eye),
        )

    def _left_camera_angle_deg(self) -> float:
        if self.config.left_camera_angle_deg is not None:
            return float(self.config.left_camera_angle_deg)
        if self.config.auto_rotate_left:
            return (
                self.config.auto_rotate_base_deg
                + self.config.left_camera_rotation_step_deg * len(self.env.container.placed_items)
            )
        return self.config.auto_rotate_base_deg

    def _add_legend(
        self,
        fig,
        buffer_item_types: dict[tuple[int, int, int], list[int]],
        holding_item_types: dict[tuple[int, int, int], list[int]],
    ) -> None:
        utilization_rate = self.env.container.utilization * 100
        legend_lines = ["Item Types:"]
        for (dx, dy, dz), indices in sorted(buffer_item_types.items()):
            legend_lines.append(f"{dx}x{dy}x{dz}: {len(indices)} pcs")
        if holding_item_types:
            legend_lines.append("Staging:")
            for (dx, dy, dz), indices in sorted(holding_item_types.items()):
                legend_lines.append(f"{dx}x{dy}x{dz}: {len(indices)} pcs")
        legend_lines.append(f"Utilization: {utilization_rate:.1f}%")
        fig.add_annotation(
            text="<br>".join(legend_lines),
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.08,
            showarrow=False,
            align="left",
            bordercolor="black",
            borderwidth=1,
            bgcolor=self.config.legend_bgcolor,
        )

    def _holding_items(self) -> list[Item]:
        return list(self.env.container.holding_list)

    def _figure_title(self, buffer_count: int, holding_count: int) -> str:
        return f"{self.title} (Buffer: {buffer_count}, Staging: {holding_count})"
