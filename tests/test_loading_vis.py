import pytest

from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item
from packing_env.lbcp.load_bounds import InterfaceVertexForces, item_footprint
from packing_env.lbcp.loading_vis import (
    compute_all_contact_loads,
    compute_focus_loading_vis,
    related_item_keys,
)
from packing_env.lbcp.validator import LBCPConfig, LBCPValidator


def _item(x, y, z, dx, dy, dz):
    return Item(FLB=Point3D(x, y, z), Dim=Orthogonal3D(dx, dy, dz))


def test_uncertain_max_overlay_reuses_all_interfaces_including_floor():
    from packing_env.lbcp.loading_vis import uncertain_com_max_patches
    from packing_env.lbcp.load_bounds import solve_global_load_bounds
    from packing.threejs_visualization.contact import build_contact_objects
    from packing_env.visualization.config import VisualConfig
    items = [_item(0, 0, 0, 50, 50, 50), _item(0, 0, 50, 50, 50, 50)]
    result = solve_global_load_bounds(items,
        lbcp_polygons={i.to_key(): item_footprint(i) for i in items},
        material_density=1e-6, uncertain_com=True, interface_max_only=True)
    cached = (frozenset(i.to_key() for i in items), result)
    patches = uncertain_com_max_patches(items, cached, floor_contact_only=False)
    assert len(patches) == result.interface_count == 2
    assert all(p.independent_interface_max for p in patches)
    objects = build_contact_objects(patches, VisualConfig())
    arrows = [o for o in objects if o.kind == 'force_arrow']
    for patch in patches:
        assert len(patch.vertex_forces_at_force_max) == len(patch.polygon_xy)
        assert sum(patch.vertex_forces_at_force_max) == pytest.approx(patch.force_max)
        actual = [a for a in arrows if a.geometry.get("representation") == "ci_resultant"
                  and a.geometry["physical_origin"][2] == patch.z]
        assert len(actual) == 1
        arrow = actual[0]
        assert arrow.geometry["force_newtons"] == pytest.approx(patch.force_max)
        for axis in (0, 1):
            moment = sum(xy[axis] * force for xy, force in
                         zip(patch.polygon_xy, patch.vertex_forces_at_force_max))
            assert arrow.geometry["origin"][axis] * patch.force_max == pytest.approx(moment)
        assert arrow.geometry["origin"][2] == pytest.approx(patch.z + 6)
        assert arrow.geometry["scenario"] == "force_max"
        assert arrow.style["color"] == "#FF0000"
        vertex_arrows = [a for a in arrows if a.geometry.get("representation") != "ci_resultant"
                         and a.geometry["origin"][2] == patch.z]
        assert sum(a.geometry["force_newtons"] for a in vertex_arrows) == pytest.approx(patch.force_max)
        assert all(a.style["color"] == "#0066FF" for a in vertex_arrows)
    assert all(o.style["pattern"] == "diagonal_hatch" for o in objects if o.kind == "contact_patch")
    assert all(o.tooltip['force_min_n'] is None for o in objects if o.kind == 'contact_patch')
    assert uncertain_com_max_patches(items[:1], cached) == []
    only = build_contact_objects(patches, VisualConfig(interface_force_view="resultant-only"))
    only_arrows = [o for o in only if o.kind == "force_arrow"]
    assert len(only_arrows) == len(patches)
    assert all(o.geometry.get("representation") == "ci_resultant" for o in only_arrows)
    assert all(o.style["color"] == "#FF0000" for o in only_arrows)
    vertex_objects = build_contact_objects(patches, VisualConfig(interface_force_view="lp-solution"))
    vertices = [o for o in vertex_objects if o.kind == "force_arrow"]
    assert all("physical_origin" not in o.geometry for o in vertices)
    for patch in patches:
        shown = [o for o in vertices if o.geometry["origin"][2] == patch.z]
        expected = [(list(xy) + [patch.z], force) for xy, force in
                    zip(patch.polygon_xy, patch.vertex_forces_at_force_max) if force > 1e-12]
        assert len(shown) == len(expected)
        for obj, (origin, force) in zip(shown, expected):
            assert obj.geometry["origin"] == origin
            assert obj.geometry["force_newtons"] == pytest.approx(force)
        assert sum(o.geometry["force_newtons"] for o in shown) == pytest.approx(patch.force_max)


def test_related_item_keys_for_stack():
    top = _item(0, 0, 100, 50, 50, 50)
    mid = _item(0, 0, 50, 50, 50, 50)
    bottom = _item(0, 0, 0, 50, 50, 50)
    unrelated = _item(200, 0, 0, 40, 40, 40)
    placed = [mid, bottom, unrelated]

    related = related_item_keys(top, placed)
    assert top.to_key() in related
    assert mid.to_key() in related
    assert bottom.to_key() in related
    assert unrelated.to_key() not in related


def test_compute_focus_loading_vis_returns_contact_patches():
    top = _item(0, 0, 100, 50, 50, 50)
    mid = _item(0, 0, 50, 50, 50, 50)
    bottom = _item(0, 0, 0, 50, 50, 50)
    placed = [bottom, mid]

    related_keys, focus_key, floor_keys, patches = compute_focus_loading_vis(
        top,
        placed,
        material_density=1e-6,
    )
    assert focus_key == top.to_key()
    assert mid.to_key() in related_keys
    assert bottom.to_key() in floor_keys
    assert len(patches) >= 1
    assert patches[0].pressure > 0
    assert patches[0].force == pytest.approx(patches[0].force_max)
    assert 0.0 <= patches[0].force_min <= patches[0].force_max
    assert len(patches[0].vertex_forces_at_force_min) == len(
        patches[0].polygon_xy
    )
    assert len(patches[0].vertex_forces_at_force_max) == len(
        patches[0].polygon_xy
    )
    assert sum(patches[0].vertex_forces_at_force_min) == pytest.approx(
        patches[0].force_min
    )
    assert sum(patches[0].vertex_forces_at_force_max) == pytest.approx(
        patches[0].force_max
    )
    assert len(patches[0].vertex_forces_at_feasible) == len(
        patches[0].polygon_xy
    )


def test_all_contact_regions_include_mid_level_patch():
    top = _item(0, 0, 100, 50, 50, 50)
    mid = _item(0, 0, 50, 50, 50, 50)
    bottom = _item(0, 0, 0, 50, 50, 50)
    placed = [bottom, mid, top]

    floor_only = compute_focus_loading_vis(
        top,
        placed,
        material_density=1e-6,
        floor_contact_only=True,
    )[3]
    all_contacts = compute_focus_loading_vis(
        top,
        placed,
        material_density=1e-6,
        floor_contact_only=False,
    )[3]

    assert len(all_contacts) > len(floor_only)
    assert any(abs(patch.z - 100.0) < 1e-6 for patch in all_contacts)
    assert any(abs(patch.z - 50.0) < 1e-6 for patch in all_contacts)


def test_visualized_vertex_forces_share_one_globally_feasible_solution():
    placed = [
        _item(120, 260, 0, 240, 180, 120),
        _item(400, 0, 0, 180, 300, 120),
        _item(60, 0, 0, 300, 240, 120),
        _item(200, 20, 120, 300, 120, 120),
    ]

    patches = compute_all_contact_loads(placed, material_density=1e-6)
    supporting_patches = [
        patch for patch in patches if patch.z == pytest.approx(120.0)
    ]
    visualized_force = sum(
        sum(patch.vertex_forces_at_feasible) for patch in supporting_patches
    )
    supported_item = placed[-1]
    expected_weight = 1e-6 * supported_item.Dim.Volume * 9.81

    assert len(supporting_patches) == 2
    assert visualized_force == pytest.approx(expected_weight)
    assert all(
        len(patch.vertex_forces_at_feasible) == len(patch.polygon_xy)
        for patch in supporting_patches
    )


def test_contact_load_visualization_rejects_misaligned_vertex_forces():
    bottom = _item(0, 0, 0, 50, 50, 50)
    top = _item(0, 0, 50, 50, 50, 50)
    placed = [bottom, top]
    validator = LBCPValidator(
        LBCPConfig(material_density=1e-6, payload_capacity=float("inf"))
    )
    result = validator.rebuild_state_from_placed(
        placed,
        lbcp_polygons={item.to_key(): item_footprint(item) for item in placed},
    )
    interface_key = next(key for key in result.interfaces if key.lower_key is not None)
    original = result.interface_vertex_forces[interface_key]
    result.interface_vertex_forces[interface_key] = InterfaceVertexForces(
        at_force_min=original.at_force_min[:-1],
        at_force_max=original.at_force_max,
    )

    with pytest.raises(ValueError, match="vertex forces.*interface"):
        compute_all_contact_loads(
            placed,
            material_density=1e-6,
            validator=validator,
        )
