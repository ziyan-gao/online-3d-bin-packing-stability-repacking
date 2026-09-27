from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import box as shapely_box

from packing_env.data_type.item import Item

Z_TOLERANCE = 0.5  # mm


@dataclass(frozen=True)
class ContactPatch:
    cx: float
    cy: float
    area: float  # mm²


def _xy_box(item: Item):
    flb = item.True_FLB
    return shapely_box(
        flb.x,
        flb.y,
        flb.x + item.Dim.dx,
        flb.y + item.Dim.dy,
    )


def _polygon_xy_from_intersection(inter) -> list[tuple[float, float]] | None:
    if inter.is_empty or inter.area <= 0:
        return None
    if inter.geom_type == "Polygon":
        geom = inter
    elif inter.geom_type == "MultiPolygon":
        geom = max(inter.geoms, key=lambda piece: piece.area)
    else:
        return None
    return [(float(x), float(y)) for x, y in geom.exterior.coords[:-1]]


def item_top_z(item: Item) -> float:
    return float(item.True_FLB.z + item.Dim.dz)


def item_bottom_z(item: Item) -> float:
    return float(item.True_FLB.z)


def contacts_vertically(upper: Item, lower: Item) -> bool:
    return abs(item_bottom_z(upper) - item_top_z(lower)) <= Z_TOLERANCE


def contact_patch_between(upper: Item, lower: Item) -> ContactPatch | None:
    geometry = contact_patch_geometry_between(upper, lower)
    if geometry is None:
        return None
    patch, _polygon = geometry
    return patch


def contact_patch_geometry_between(
    upper: Item,
    lower: Item,
) -> tuple[ContactPatch, list[tuple[float, float]]] | None:
    if not contacts_vertically(upper, lower):
        return None
    inter = _xy_box(upper).intersection(_xy_box(lower))
    polygon_xy = _polygon_xy_from_intersection(inter)
    if polygon_xy is None:
        return None
    centroid = inter.centroid
    patch = ContactPatch(cx=float(centroid.x), cy=float(centroid.y), area=float(inter.area))
    return patch, polygon_xy
