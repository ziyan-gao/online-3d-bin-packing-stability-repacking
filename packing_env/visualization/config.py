from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VisualConfig:
    fallback_size: tuple[int, int] = (12, 4)
    subplot_titles: tuple[str, str, str] = (
        "Support Regions",
        "Buffer Items - Linear Array",
        "Staging Area - Holding List",
    )
    horizontal_spacing: float = 0.03
    vertical_spacing: float = 0.08
    column_widths: tuple[float, float] = (0.5, 0.5)
    row_heights: tuple[float, float] = (0.5, 0.5)
    axis_title_x: str = "X (mm)"
    axis_title_y: str = "Y (mm)"
    axis_title_z: str = "Z (mm)"
    default_container_size: tuple[int, int, int] = (1200, 1000, 1350)
    plot_edge_padding: float = 10.0
    hull_thickness: float = 2.0
    support_color: str = "rgb(255,215,0)"
    support_edge_color: str = "rgba(0,0,0,0.9)"
    anchor_marker_size: int = 6
    anchor_marker_color: str = "rgba(255,100,0,0.95)"
    anchor_marker_symbol: str = "diamond"
    moving_item_color: str = "rgb(40,140,255)"
    moving_item_edge_color: str = "rgba(0,80,180,0.95)"
    placed_item_color: str = "rgb(128,128,128)"
    placed_item_edge_color: str = "rgba(60,60,60,0.9)"
    highlighted_item_color: str = "rgb(220,40,40)"
    highlighted_item_edge_color: str = "rgba(170,0,0,0.9)"
    repack_space_color: str = "rgb(220,40,40)"
    repack_space_edge_color: str = "rgba(170,0,0,0.9)"
    repacked_item_color: str = "rgb(0,180,80)"
    repacked_item_edge_color: str = "rgba(0,120,50,0.95)"
    empty_staging_color: str = "rgb(150,150,150)"
    empty_staging_edge_color: str = "rgba(90,90,90,0.45)"
    right_camera_angle_deg: float = -150.0
    right_camera_radius: float = 0.75
    left_camera_radius: float = 2.1213203435596424
    left_camera_z: float = 0.8
    left_camera_angle_deg: float | None = None
    auto_rotate_left: bool = False
    auto_rotate_base_deg: float = 45.0
    left_camera_rotation_step_deg: float = 4.0
    legend_bgcolor: str = "rgba(255,248,220,0.9)"
    ems_palette: tuple[tuple[int, int, int], ...] = field(
        default_factory=lambda: (
            (99, 110, 250),
            (239, 85, 59),
            (0, 204, 150),
            (171, 99, 250),
            (255, 161, 90),
            (25, 211, 243),
            (255, 102, 146),
            (182, 232, 128),
            (255, 151, 255),
            (254, 203, 82),
        )
    )


DEFAULT_VISUAL_CONFIG = VisualConfig()
