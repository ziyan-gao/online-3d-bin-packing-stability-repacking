"""Build renderer-neutral semantic scenes for the Three.js visualizer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Iterable, Sequence

from packing_env.data_type.support_vis import LoadContactPatchVis
from packing_env.visualization.config import VisualConfig

from .contact import build_contact_objects
from .frame import SceneFrame, SceneObject, VisualizationFrame

if TYPE_CHECKING:
    from packing_env.data_type.item import Item
    from packing_env.gym_env import PackingEnv


@dataclass(frozen=True)
class FocusLoadingOverlay:
    """Loading context carried by a frame while force rendering is added later."""

    focus_key: tuple | None
    related_keys: frozenset[tuple] = field(default_factory=frozenset)
    floor_keys: frozenset[tuple] = field(default_factory=frozenset)
    patches: tuple[LoadContactPatchVis, ...] = ()
    bearing_capacity: float | None = None
    sampled_com_status: str | None = None
    sampled_com_positions: dict | None = None


class ThreeVisualizationBuilder:
    """Translate a packing environment into three semantic scene records."""

    def __init__(self, config: VisualConfig) -> None:
        self.config = config

    def build(
        self,
        env: PackingEnv,
        title: str,
        *,
        focus_overlay: FocusLoadingOverlay | None = None,
        failed_candidate: Item | None = None,
        highlighted_items: list[Item] | None = None,
        repacked_items: list[Item] | None = None,
        repack_spaces: list[Item] | None = None,
        show_anchor: bool = True,
        show_ems: bool = False,
        virtual_boxes: bool = False,
    ) -> VisualizationFrame:
        container = env.container
        container_bounds = [
            0.0,
            0.0,
            0.0,
            float(container.dx),
            float(container.dy),
            float(container.dz),
        ]
        container_objects = [
            self._container_wire(container_bounds),
            self._floor_lbcp(container_bounds),
        ]
        container_objects.extend(
            self._placed_item_objects(env, focus_overlay=focus_overlay, virtual_boxes=virtual_boxes)
        )
        # LBCP is the complete cached top support polygon, not the CI intersection.
        # Keep both independent layers present when showing contact loads.
        container_objects.extend(self._support_objects(env))
        if focus_overlay is not None and focus_overlay.patches:
            container_objects.extend(
                build_contact_objects(
                    focus_overlay.patches,
                    self.config,
                    bearing_capacity=focus_overlay.bearing_capacity,
                )
            )

        visible_ems = list(env.heu_ems.get_ems_list()) if (show_anchor or show_ems) else []
        if show_ems:
            container_objects.extend(self._ems_objects(visible_ems))
        if show_anchor:
            container_objects.extend(self._anchor_objects(visible_ems))

        container_objects.extend(
            self._overlay_objects(
                [failed_candidate] if failed_candidate is not None else None,
                role="lp-infeasible-candidate",
                kind="highlighted_item",
                color=self.config.highlighted_item_color,
                edge_color=self.config.highlighted_item_edge_color,
                opacity=0.5,
            )
        )
        container_objects.extend(
            self._overlay_objects(
                highlighted_items,
                role="highlighted",
                kind="highlighted_item",
                color=self.config.placed_item_color,
                edge_color=self.config.placed_item_edge_color,
                opacity=0.5,
            )
        )
        container_objects.extend(
            self._overlay_objects(
                repack_spaces,
                role="repack-space",
                kind="repack_space",
                color=self.config.repack_space_color,
                edge_color=self.config.repack_space_edge_color,
                opacity=0.35,
                inflate=6.0,
            )
        )
        container_objects.extend(
            self._overlay_objects(
                repacked_items,
                role="repacked",
                kind="repacked_item",
                color=self.config.repacked_item_color,
                edge_color=self.config.repacked_item_edge_color,
                opacity=0.5,
                inflate=2.0,
            )
        )

        buffer_objects, buffer_bounds, buffer_camera = self._linear_scene(
            reversed(env.buffer.items), env.buffer.data_sampler, role="buffer",
            virtual_boxes=virtual_boxes, clearance=env.item_buffer_space,
        )
        holding_objects, _, _ = self._linear_scene(
            reversed(container.holding_list), env.buffer.data_sampler, role="holding",
            virtual_boxes=virtual_boxes, clearance=env.item_buffer_space,
        )
        holding_camera = dict(buffer_camera)
        holding_camera["aspectratio"] = dict(buffer_camera["aspectratio"])
        holding_camera["aspectratio"]["x"] = buffer_camera["aspectratio"]["x"] / 2.0

        scenes = {
            "container": SceneFrame(
                bounds=container_bounds,
                objects=container_objects,
                camera=self._container_camera(env),
            ),
            "buffer": SceneFrame(
                bounds=buffer_bounds,
                objects=buffer_objects,
                camera=buffer_camera,
            ),
            "holding": SceneFrame(
                bounds=list(buffer_bounds),
                objects=holding_objects,
                camera=holding_camera,
            ),
        }
        legend = self._legend()
        if focus_overlay and focus_overlay.sampled_com_status is not None:
            title += f" | Sampled CoM (±10%): feasibility LP {focus_overlay.sampled_com_status}"
            legend[-1] = {"kind": "feasible", "label": "One global solution with fixed sampled CoM",
                          "color": self.config.load_force_max_arrow_color}
            for obj in container_objects:
                if obj.kind == "item":
                    key = tuple(obj.tooltip.get("item_key", ()))
                    point = (focus_overlay.sampled_com_positions or {}).get(key)
                    if point is not None:
                        obj.tooltip["sampled_com_xy_mm"] = list(point)
        if focus_overlay and any(p.independent_interface_max for p in focus_overlay.patches):
            title += " | Uncertain COM: independent interface maxima (not simultaneous)"
            label = ("Vertex forces at each CI maximum (independent scenarios)"
                     if self.config.interface_force_view == "lp-solution"
                     else "CI maximum resultants at centers of pressure (independent scenarios)")
            legend[-1] = {"kind": "force_max", "label": label,
                          "color": self.config.load_force_max_arrow_color}
            if self.config.interface_force_view in {"resultant", "resultant-only"}:
                legend[-1]["color"] = "#FF0000"
            if self.config.interface_force_view == "resultant":
                legend.append({"kind": "force_max",
                               "label": "Vertex forces from the same CI-max solutions",
                               "color": "#0066FF"})
        for entry in legend:
            if entry.get("kind") in {"force_max", "force_min", "feasible"}:
                entry["label"] += " (arrow length: square-root scale)"
        return VisualizationFrame(
            title=f"{title} | Virtual boxes" if virtual_boxes else title,
            scenes=scenes,
            legend=legend,
            metrics={
                "utilization": float(container.utilization),
                "virtual_utilization": float(sum(item.Virtual_Dim.Volume for item in container.placed_items) / container.Volume),
                "holding_count": len(container.holding_list),
            },
        )

    def _container_wire(self, bounds: list[float]) -> SceneObject:
        size = [bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2]]
        return SceneObject(
            id="container:wire",
            kind="container_wire",
            geometry={"origin": bounds[:3], "size": size},
            style={"color": "rgba(60,60,60,0.9)", "line_width": 2.0},
            tooltip={"type": "container", "dimensions_mm": list(size)},
        )

    def _floor_lbcp(self, bounds: list[float]) -> SceneObject:
        """Display the full container floor without changing physical LP inputs."""
        x0, y0, z0, x1, y1, _ = bounds
        return SceneObject(
            id="container:floor-lbcp",
            kind="support_patch",
            geometry={
                "polygon_xy": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "z_range": [z0, z0],
            },
            style={
                "color": self.config.support_color,
                "edge_color": self.config.support_edge_color,
                "opacity": 1.0,
            },
            tooltip={
                "type": "support_patch",
                "role": "floor-lbcp",
                "interpretation": "Full container floor LBCP (visualization only)",
            },
        )

    def _placed_item_objects(
        self,
        env: PackingEnv,
        *,
        focus_overlay: FocusLoadingOverlay | None,
        virtual_boxes: bool = False,
    ) -> list[SceneObject]:
        result = []
        seen: dict[tuple, int] = {}
        for item in env.container.placed_items:
            key = tuple(item.to_key())
            opacity = 0.95 if getattr(item, "buffer_space", 0) > 0 else 1.0
            if focus_overlay is not None:
                if key == focus_overlay.focus_key:
                    opacity = self.config.focus_alpha
                elif key in focus_overlay.floor_keys:
                    opacity = self.config.floor_alpha
                else:
                    opacity = self.config.placed_alpha
            dims = self._dims(item)
            result.append(
                SceneObject(
                    id=self._unique_key_id("container-item", key, seen),
                    kind="item",
                    geometry=self._item_geometry(item, virtual_boxes=virtual_boxes),
                    style={
                        "color": self._rgb(env.buffer.data_sampler.get_color(dims)),
                        "edge_color": self.config.support_edge_color,
                        "opacity": float(opacity),
                    },
                    tooltip={
                        **self._item_tooltip(item, role="placed"),
                        "virtual_dimensions_mm": list(self._display_dims(item, virtual_boxes=True)),
                        "displayed_box": "virtual" if virtual_boxes else "real",
                    },
                )
            )
        return result

    def _support_objects(self, env: PackingEnv) -> list[SceneObject]:
        result = []
        seen: dict[tuple, int] = {}
        for item, _item_dims, vis_data in env.heu_stable.support_vis_records:
            polygon = [[float(x), float(y)] for x, y in vis_data.support_polygon_xy]
            if len(polygon) < 3:
                continue
            if getattr(item, "buffer_space", 0) > 0:
                z0 = float(vis_data.support_z0)
            else:
                z0 = float(vis_data.support_z1)
            z1 = float(vis_data.support_z1) + float(self.config.hull_thickness)
            key = tuple(item.to_key())
            result.append(
                SceneObject(
                    id=self._unique_key_id("support", key, seen),
                    kind="support_patch",
                    geometry={"polygon_xy": polygon, "z_range": [z0, z1]},
                    style={
                        "color": self.config.support_color,
                        "edge_color": self.config.support_edge_color,
                        "opacity": 1.0,
                    },
                    tooltip={
                        "type": "support_patch",
                        "interpretation": "Full LBCP on item top, not the contact-interface intersection",
                        "item_key": self._key_values(key),
                    },
                )
            )
        return result

    def _ems_objects(self, ems_list: Sequence[object]) -> list[SceneObject]:
        result = []
        seen: dict[tuple, int] = {}
        for index, ems in enumerate(ems_list):
            key = self._spatial_key(ems)
            r, g, b = self.config.ems_palette[index % len(self.config.ems_palette)]
            result.append(
                SceneObject(
                    id=self._unique_key_id("ems", key, seen),
                    kind="ems",
                    geometry={
                        "origin": self._origin(ems),
                        "size": [
                            float(ems.Dim.dx),
                            float(ems.Dim.dy),
                            max(1.0, float(self.config.hull_thickness)),
                        ],
                    },
                    style={
                        "color": f"rgb({r},{g},{b})",
                        "edge_color": f"rgba({r},{g},{b},0.75)",
                        "opacity": 0.22,
                    },
                    tooltip={"type": "ems", "key": self._key_values(key)},
                )
            )
        return result

    def _anchor_objects(self, ems_list: Sequence[object]) -> list[SceneObject]:
        result = []
        seen_points: set[tuple[float, float, float]] = set()
        for ems in ems_list:
            point = tuple(self._origin(ems))
            if point in seen_points:
                continue
            seen_points.add(point)
            result.append(
                SceneObject(
                    id="anchor:" + self._id_fragment(point),
                    kind="anchor",
                    geometry={"position": list(point)},
                    style={
                        "color": self.config.anchor_marker_color,
                        "size": float(self.config.anchor_marker_size),
                        "symbol": self.config.anchor_marker_symbol,
                    },
                    tooltip={"type": "anchor", "position_mm": list(point)},
                )
            )
        return result

    def _overlay_objects(
        self,
        items: list[Item] | None,
        *,
        role: str,
        kind: str,
        color: str,
        edge_color: str,
        opacity: float,
        inflate: float = 0.0,
    ) -> list[SceneObject]:
        result = []
        seen: dict[tuple, int] = {}
        for item in items or []:
            key = tuple(item.to_key())
            geometry = self._item_geometry(item, inflate=inflate)
            result.append(
                SceneObject(
                    id=self._unique_key_id(role, key, seen),
                    kind=kind,
                    geometry=geometry,
                    style={
                        "color": color,
                        "edge_color": edge_color,
                        "opacity": float(opacity),
                    },
                    tooltip=self._item_tooltip(item, role=role),
                )
            )
        return result

    def _linear_scene(
        self,
        items: Iterable[object],
        data_sampler: object,
        *,
        role: str,
        virtual_boxes: bool = False,
        clearance: int = 0,
    ) -> tuple[list[SceneObject], list[float], dict[str, object]]:
        objects = []
        x_offset = 0.0
        max_x = max_y = max_z = 0.0
        seen: dict[tuple, int] = {}
        for box in items:
            dims = self._dims(box)
            display_dims = self._display_dims(box, virtual_boxes=virtual_boxes, clearance=clearance)
            if hasattr(box, "to_key"):
                key = tuple(box.to_key())
            else:
                key = dims
            geometry = {
                "origin": [float(x_offset), 0.0, 0.0],
                "size": [float(value) for value in display_dims],
            }
            objects.append(
                SceneObject(
                    id=self._unique_key_id(f"{role}-item", key, seen),
                    kind="item",
                    geometry=geometry,
                    style={
                        "color": self._rgb(data_sampler.get_color(dims)),
                        "opacity": 1.0,
                    },
                    tooltip={
                        "type": "item",
                        "role": role,
                        "dimensions_mm": [float(value) for value in dims],
                        "virtual_dimensions_mm": list(self._display_dims(box, virtual_boxes=True, clearance=clearance)),
                        "displayed_box": "virtual" if virtual_boxes else "real",
                    },
                )
            )
            x_offset += float(display_dims[0]) + 50.0
            max_x = max(max_x, x_offset)
            max_y = max(max_y, float(display_dims[1]))
            max_z = max(max_z, float(display_dims[2]))

        x_range = max(1.0, max_x)
        y_padding = max(max_y * 0.2, 50.0)
        z_padding = max(max_z * 0.2, 50.0)
        bounds = [0.0, -y_padding, -z_padding, x_range, max_y + y_padding, max_z + z_padding]
        return objects, bounds, self._right_camera(bounds)

    def _container_camera(self, env: PackingEnv) -> dict[str, object]:
        angle = self.config.left_camera_angle_deg
        if angle is None:
            angle = self.config.auto_rotate_base_deg
            if self.config.auto_rotate_left:
                angle += self.config.left_camera_rotation_step_deg * len(env.container.placed_items)
        radians = math.radians(float(angle))
        eye = {
            "x": float(self.config.left_camera_radius * math.cos(radians)),
            "y": float(self.config.left_camera_radius * math.sin(radians)),
            "z": float(self.config.left_camera_z),
        }
        pad = float(self.config.plot_edge_padding)
        ranges = {
            "x": [-pad, float(env.container.dx) + pad],
            "y": [-pad, float(env.container.dy) + pad],
            "z": [-pad, float(env.container.dz) + pad],
        }
        spans = {axis: values[1] - values[0] for axis, values in ranges.items()}
        max_span = max(spans.values())
        return {
            "eye": eye,
            "ranges": ranges,
            "aspectratio": {axis: span / max_span for axis, span in spans.items()},
            "axis_titles": {
                "x": self.config.axis_title_x,
                "y": self.config.axis_title_y,
                "z": self.config.axis_title_z,
            },
        }

    def _right_camera(self, bounds: list[float]) -> dict[str, object]:
        angle = math.radians(float(self.config.right_camera_angle_deg))
        eye = {
            "x": float(self.config.right_camera_radius * math.sin(angle)),
            "y": float(self.config.right_camera_radius * math.cos(angle)),
            "z": float(self.config.right_camera_z),
        }
        spans = {
            "x": bounds[3] - bounds[0],
            "y": bounds[4] - bounds[1],
            "z": bounds[5] - bounds[2],
        }
        max_span = max(1.0, *spans.values())
        return {
            "eye": eye,
            "ranges": {
                "x": [bounds[0], bounds[3]],
                "y": [bounds[1], bounds[4]],
                "z": [bounds[2], bounds[5]],
            },
            "aspectratio": {axis: span / max_span for axis, span in spans.items()},
        }

    def _legend(self) -> list[dict[str, object]]:
        return [
            {"kind": "item", "label": "Item"},
            {
                "kind": "contact_patch",
                "label": "Effective contact patch",
                "color": self.config.load_contact_neutral_color,
            },
            {
                "kind": "feasible",
                "label": "Globally feasible vertex force",
                "color": self.config.load_feasible_arrow_color,
            },
        ]

    @staticmethod
    def _dims(box: object) -> tuple[int, int, int]:
        source = box.Dim if hasattr(box, "Dim") else box
        return int(source.dx), int(source.dy), int(source.dz)

    @staticmethod
    def _origin(box: object) -> list[float]:
        point = box.True_FLB if hasattr(box, "True_FLB") else box.FLB
        return [float(point.x), float(point.y), float(point.z)]

    def _display_dims(self, box: object, *, virtual_boxes: bool, clearance: int = 0) -> tuple[int, int, int]:
        if not virtual_boxes:
            return self._dims(box)
        if hasattr(box, "Virtual_Dim"):
            return self._dims(box.Virtual_Dim)
        from packing_env.data_type.item import Item

        dx, dy, dz = self._dims(box)
        return (
            Item.round_buffered_dim(dx, clearance, box.resolution),
            Item.round_buffered_dim(dy, clearance, box.resolution),
            dz,
        )

    def _item_geometry(self, item: Item, *, inflate: float = 0.0, virtual_boxes: bool = False) -> dict[str, list[float]]:
        point = item.FLB if virtual_boxes else item.True_FLB
        origin = [float(value) - float(inflate) for value in (point.x, point.y, point.z)]
        size = [float(value) + 2.0 * float(inflate) for value in self._display_dims(item, virtual_boxes=virtual_boxes)]
        return {"origin": origin, "size": size}

    def _item_tooltip(self, item: Item, *, role: str) -> dict[str, object]:
        return {
            "type": "item",
            "role": role,
            "item_key": self._key_values(tuple(item.to_key())),
            "position_mm": self._origin(item),
            "dimensions_mm": [float(value) for value in self._dims(item)],
        }

    @classmethod
    def _spatial_key(cls, box: object) -> tuple[float, ...]:
        return (*cls._origin(box), *(float(value) for value in cls._dims(box)))

    @staticmethod
    def _rgb(color: Sequence[float]) -> str:
        values = list(color[:3])
        if any(value > 1.0 for value in values):
            channels = [int(max(0, min(255, value))) for value in values]
        else:
            channels = [int(max(0, min(255, round(value * 255)))) for value in values]
        return f"rgb({channels[0]},{channels[1]},{channels[2]})"

    @classmethod
    def _unique_key_id(cls, role: str, key: tuple, seen: dict[tuple, int]) -> str:
        occurrence = seen.get(key, 0)
        seen[key] = occurrence + 1
        suffix = "" if occurrence == 0 else f":{occurrence}"
        return f"{role}:{cls._id_fragment(key)}{suffix}"

    @staticmethod
    def _id_fragment(values: Sequence[object]) -> str:
        return ":".join(str(value) for value in values)

    @staticmethod
    def _key_values(key: tuple) -> list[object]:
        return [value.item() if hasattr(value, "item") else value for value in key]
