from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from packing_env.data_type.geometry import Point3D
from packing_env.data_type.support_vis import SupportVisData

if TYPE_CHECKING:
    from packing_env.data_type.ems import EmptyMaximalSpace
    from packing_env.data_type.item import Item
    import plotly.graph_objects as go_types

from .config import VisualConfig

class Drawer:
    def __init__(self, go: ModuleType, fig: go_types.Figure, config: VisualConfig) -> None:
        self.go = go
        self.fig = fig
        self.config = config

    @staticmethod
    def box_dims(box: Item | EmptyMaximalSpace | object) -> tuple[int, int, int]:
        if hasattr(box, "Dim"):
            return int(box.Dim.dx), int(box.Dim.dy), int(box.Dim.dz)
        return int(box.dx), int(box.dy), int(box.dz)

    def _to_rgb(self, color: tuple[float, ...] | list[float]) -> str:
        vals = list(color[:3])
        if any(v > 1.0 for v in vals):
            vals = [int(max(0, min(255, v))) for v in vals]
        else:
            vals = [int(max(0, min(255, round(v * 255)))) for v in vals]
        return f"rgb({vals[0]},{vals[1]},{vals[2]})"

    def _ems_color(self, idx: int, alpha: float = 1.0) -> str:
        r, g, b = self.config.ems_palette[idx % len(self.config.ems_palette)]
        a = max(0.0, min(1.0, float(alpha)))
        return f"rgba({r},{g},{b},{a:.2f})"

    def _polygon_prism_mesh(
        self, coords_xy: Sequence[tuple[float, float]] | None, z0: float, z1: float
    ) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int]] | None:
        if coords_xy is None or len(coords_xy) < 3:
            return None
        coords = [(float(p[0]), float(p[1])) for p in coords_xy]
        n = len(coords)
        x = [p[0] for p in coords] + [p[0] for p in coords]
        y = [p[1] for p in coords] + [p[1] for p in coords]
        z = [float(z0)] * n + [float(z1)] * n
        i_idx, j_idx, k_idx = [], [], []
        for t in range(1, n - 1):
            i_idx.append(0)
            j_idx.append(t)
            k_idx.append(t + 1)
        top0 = n
        for t in range(1, n - 1):
            i_idx.append(top0)
            j_idx.append(top0 + t + 1)
            k_idx.append(top0 + t)
        for t in range(n):
            t_next = (t + 1) % n
            i_idx.extend([t, t_next])
            j_idx.extend([t_next, top0 + t_next])
            k_idx.extend([top0 + t, top0 + t])
        return x, y, z, i_idx, j_idx, k_idx

    def _add_filled_prism(
        self,
        coords_xy: Sequence[tuple[float, float]] | None,
        z0: float,
        z1: float,
        fill_color: str,
        opacity: float = 0.45,
        row: int = 1,
        col: int = 1,
    ) -> None:
        mesh = self._polygon_prism_mesh(coords_xy, z0, z1)
        if mesh is None:
            return
        x, y, z, i_idx, j_idx, k_idx = mesh
        self.fig.add_trace(
            self.go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i_idx,
                j=j_idx,
                k=k_idx,
                color=fill_color,
                opacity=opacity,
                flatshading=True,
                showscale=False,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def _add_wire_prism(
        self,
        coords_xy: Sequence[tuple[float, float]] | None,
        z0: float,
        z1: float,
        color: str = "rgba(30,144,255,0.55)",
        width: int = 3,
        row: int = 1,
        col: int = 1,
    ) -> None:
        if coords_xy is None or len(coords_xy) < 3:
            return
        xs = [float(p[0]) for p in coords_xy]
        ys = [float(p[1]) for p in coords_xy]
        n = len(xs)
        for i in range(n):
            j = (i + 1) % n
            self.fig.add_trace(
                self.go.Scatter3d(
                    x=[xs[i], xs[j]],
                    y=[ys[i], ys[j]],
                    z=[float(z0), float(z0)],
                    mode="lines",
                    line=dict(color=color, width=width),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            self.fig.add_trace(
                self.go.Scatter3d(
                    x=[xs[i], xs[j]],
                    y=[ys[i], ys[j]],
                    z=[float(z1), float(z1)],
                    mode="lines",
                    line=dict(color=color, width=width),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            self.fig.add_trace(
                self.go.Scatter3d(
                    x=[xs[i], xs[i]],
                    y=[ys[i], ys[i]],
                    z=[float(z0), float(z1)],
                    mode="lines",
                    line=dict(color=color, width=max(1, width - 1)),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    def _box_mesh(
        self, x0: float, y0: float, z0: float, dx: float, dy: float, dz: float
    ) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int]]:
        x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
        x = [x0, x1, x1, x0, x0, x1, x1, x0]
        y = [y0, y0, y1, y1, y0, y0, y1, y1]
        z = [z0, z0, z0, z0, z1, z1, z1, z1]
        i = [0, 0, 4, 4, 0, 2, 1, 3, 0, 1, 2, 3]
        j = [1, 2, 5, 6, 4, 6, 5, 7, 1, 2, 3, 0]
        k = [2, 3, 6, 7, 5, 7, 4, 4, 5, 6, 7, 4]
        return x, y, z, i, j, k

    def _add_box_mesh(
        self,
        x0: float,
        y0: float,
        z0: float,
        dx: float,
        dy: float,
        dz: float,
        color: str,
        opacity: float = 1.0,
        name: str = "",
        row: int = 1,
        col: int = 1,
    ) -> None:
        x, y, z, i_idx, j_idx, k_idx = self._box_mesh(x0, y0, z0, dx, dy, dz)
        self.fig.add_trace(
            self.go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i_idx,
                j=j_idx,
                k=k_idx,
                color=color,
                opacity=opacity,
                flatshading=True,
                showscale=False,
                name=name,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def _add_box_overlay(
        self,
        items: list[Item | EmptyMaximalSpace] | None,
        color: str,
        opacity: float,
        name: str,
        edge_color: str,
        inflate: float = 0.0,
        row: int = 1,
        col: int = 1,
    ) -> None:
        if items is None:
            return
        for item in items:
            flb = item.True_FLB
            x0 = float(flb.x) - inflate
            y0 = float(flb.y) - inflate
            z0 = float(flb.z) - inflate
            dx = float(item.Dim.dx) + 2.0 * inflate
            dy = float(item.Dim.dy) + 2.0 * inflate
            dz = float(item.Dim.dz) + 2.0 * inflate
            self._add_box_mesh(
                x0, y0, z0, dx, dy, dz, color=color, opacity=opacity, name=name, row=row, col=col
            )
            self._add_wire_prism(
                [(x0, y0), (x0 + dx, y0), (x0 + dx, y0 + dy), (x0, y0 + dy)],
                z0,
                z0 + dz,
                color=edge_color,
                width=7,
                row=row,
                col=col,
            )

    def draw_item(
        self,
        item: Item | EmptyMaximalSpace,
        color: str,
        opacity: float = 1.0,
        name: str = "",
        edge_color: str | None = None,
        inflate: float = 0.0,
        row: int = 1,
        col: int = 1,
    ) -> None:
        flb = item.True_FLB
        x0 = float(flb.x) - inflate
        y0 = float(flb.y) - inflate
        z0 = float(flb.z) - inflate
        dx = float(item.Dim.dx) + 2.0 * inflate
        dy = float(item.Dim.dy) + 2.0 * inflate
        dz = float(item.Dim.dz) + 2.0 * inflate
        self._add_box_mesh(x0, y0, z0, dx, dy, dz, color=color, opacity=opacity, name=name, row=row, col=col)
        if edge_color is not None:
            self._add_wire_prism(
                [(x0, y0), (x0 + dx, y0), (x0 + dx, y0 + dy), (x0, y0 + dy)],
                z0,
                z0 + dz,
                color=edge_color,
                width=7,
                row=row,
                col=col,
            )

    def draw_ems(self, ems: EmptyMaximalSpace, ems_idx: int, row: int = 1, col: int = 1) -> None:
        face_color = self._ems_color(ems_idx, alpha=1.0)
        edge_color = self._ems_color(ems_idx, alpha=0.55)
        x0 = float(ems.FLB.x)
        y0 = float(ems.FLB.y)
        z0 = float(ems.FLB.z)
        dx = float(ems.Dim.dx)
        dy = float(ems.Dim.dy)
        dz = float(ems.Dim.dz)
        self._add_box_mesh(x0, y0, z0, dx, dy, dz, color=face_color, opacity=0.18, name="EMS", row=row, col=col)
        self._add_wire_prism(
            [(x0, y0), (x0 + dx, y0), (x0 + dx, y0 + dy), (x0, y0 + dy)],
            z0,
            z0 + dz,
            color=edge_color,
            width=5,
            row=row,
            col=col,
        )

    def draw_anchor_points(self, anchor_points: list[Point3D] | None, row: int = 1, col: int = 1) -> None:
        if anchor_points is None:
            return
        self.fig.add_trace(
            self.go.Scatter3d(
                x=[float(pt.x) for pt in anchor_points],
                y=[float(pt.y) for pt in anchor_points],
                z=[float(pt.z) for pt in anchor_points],
                mode="markers",
                marker=dict(
                    size=self.config.anchor_marker_size,
                    color=self.config.anchor_marker_color,
                    symbol=self.config.anchor_marker_symbol,
                ),
                showlegend=False,
                name="Anchor Points",
            ),
            row=row,
            col=col,
        )

    def draw_support(
        self, vis_data: SupportVisData, base_color: str, has_buffer: bool, row: int = 1, col: int = 1
    ) -> None:
        self._add_filled_prism(
            vis_data.virtual_item_polygon_xy,
            vis_data.support_z0,
            vis_data.support_z1,
            fill_color=base_color,
            opacity=0.95 if has_buffer else 1.0,
            row=row,
            col=col,
        )
        self._add_wire_prism(
            vis_data.virtual_item_polygon_xy,
            vis_data.support_z0,
            vis_data.support_z1,
            color=self.config.support_edge_color,
            width=8,
            row=row,
            col=col,
        )
        if has_buffer:
            support_z0 = vis_data.support_z0
            support_z1 = vis_data.support_z1 + self.config.hull_thickness
        else:
            support_z0 = vis_data.support_z1
            support_z1 = vis_data.support_z1 + self.config.hull_thickness
        self._add_filled_prism(
            vis_data.support_polygon_xy,
            support_z0,
            support_z1,
            fill_color=self.config.support_color,
            opacity=1.0,
            row=row,
            col=col,
        )
        self._add_wire_prism(
            vis_data.support_polygon_xy,
            support_z0,
            support_z1,
            color=self.config.support_edge_color,
            width=2,
            row=row,
            col=col,
        )

    def draw_linear_items_scene(
        self,
        items: list[Item | EmptyMaximalSpace | object],
        color_lookup,
        row: int,
        col: int,
        highlight_indices: list[int] | set[int] | None = None,
        fixed_x_range: float | None = None,
        fixed_y_range: tuple[float, float] | None = None,
        fixed_z_range: tuple[float, float] | None = None,
    ) -> tuple[dict[tuple[int, int, int], list[int]], dict[str, object]]:
        item_types: dict[tuple[int, int, int], list[int]] = {}
        for i, box in enumerate(items):
            key = self.box_dims(box)
            item_types.setdefault(key, []).append(i)

        x_offset = 0.0
        max_x = 0.0
        max_y = 0.0
        max_z = 0.0
        highlighted = set(highlight_indices or [])
        for index, box in enumerate(items):
            dx, dy, dz = self.box_dims(box)
            color = self._to_rgb(color_lookup((dx, dy, dz)))
            self._add_box_mesh(
                x_offset,
                0.0,
                0.0,
                float(dx),
                float(dy),
                float(dz),
                color=color,
                opacity=1.0,
                name=f"{dx}x{dy}x{dz}",
                row=row,
                col=col,
            )
            if index in highlighted:
                self._add_wire_prism(
                    [
                        (x_offset, 0.0),
                        (x_offset + float(dx), 0.0),
                        (x_offset + float(dx), float(dy)),
                        (x_offset, float(dy)),
                    ],
                    0.0,
                    float(dz),
                    color=self.config.moving_item_edge_color,
                    width=9,
                    row=row,
                    col=col,
                )
            x_offset += float(dx) + 50.0
            max_x = max(max_x, x_offset)
            max_y = max(max_y, float(dy))
            max_z = max(max_z, float(dz))

        y_padding = max(max_y * 0.2, 50.0)
        z_padding = max(max_z * 0.2, 50.0)
        x_range = max(1.0, max_x)
        y_axis_range = [-y_padding, max_y + y_padding]
        z_axis_range = [-z_padding, max_z + z_padding]
        if fixed_x_range is not None:
            x_range = max(1.0, float(fixed_x_range))
        if fixed_y_range is not None:
            y_axis_range = [float(fixed_y_range[0]), float(fixed_y_range[1])]
        if fixed_z_range is not None:
            z_axis_range = [float(fixed_z_range[0]), float(fixed_z_range[1])]

        y_range = max(1.0, y_axis_range[1] - y_axis_range[0])
        z_range = max(1.0, z_axis_range[1] - z_axis_range[0])
        max_range = max(x_range, y_range, z_range)
        right_camera_angle_rad = np.deg2rad(self.config.right_camera_angle_deg)
        right_camera_radius = self.config.right_camera_radius
        right_camera_eye = dict(
            x=float(right_camera_radius * np.sin(right_camera_angle_rad)),
            y=float(right_camera_radius * np.cos(right_camera_angle_rad)),
            z=0.5,
        )
        scene_layout: dict[str, object] = dict(
            xaxis_title=self.config.axis_title_x,
            yaxis_title=self.config.axis_title_y,
            zaxis_title=self.config.axis_title_z,
            xaxis=dict(range=[0, x_range], ticks="", showticklabels=False),
            yaxis=dict(range=y_axis_range, ticks="", showticklabels=False),
            zaxis=dict(range=z_axis_range, ticks="", showticklabels=False),
            aspectmode="manual",
            aspectratio=dict(
                x=x_range / max_range,
                y=y_range / max_range,
                z=z_range / max_range,
            ),
            camera=dict(eye=right_camera_eye),
        )
        return item_types, scene_layout

    def draw_empty_staging_area(self, scene_layout: dict[str, object], row: int, col: int) -> None:
        x0, x1 = scene_layout["xaxis"]["range"]
        y0, y1 = scene_layout["yaxis"]["range"]
        z1 = max(1.0, float(scene_layout["zaxis"]["range"][1]))
        floor_z = 0.0
        self.fig.add_trace(
            self.go.Mesh3d(
                x=[x0, x1, x1, x0],
                y=[y0, y0, y1, y1],
                z=[floor_z, floor_z, floor_z, floor_z],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=self.config.empty_staging_color,
                opacity=0.12,
                flatshading=True,
                showscale=False,
                showlegend=False,
                name="Empty Staging Area",
            ),
            row=row,
            col=col,
        )
        self._add_wire_prism(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            floor_z,
            z1,
            color=self.config.empty_staging_edge_color,
            width=3,
            row=row,
            col=col,
        )
