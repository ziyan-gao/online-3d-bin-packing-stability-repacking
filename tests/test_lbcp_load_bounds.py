import pytest
from shapely.geometry import box as shapely_box

import packing_env.lbcp.load_bounds as load_bounds_module
from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item
from packing_env.lbcp.load_bounds import (
    GlobalLoadStatus,
    SupportInterfaceKey,
    solve_global_load_bounds,
)


def _item(x, y, z, dx, dy, dz):
    return Item(Point3D(x, y, z), Orthogonal3D(dx, dy, dz))


def _full_top_lbcp(item):
    flb = item.True_FLB
    return shapely_box(
        flb.x,
        flb.y,
        flb.x + item.Dim.dx,
        flb.y + item.Dim.dy,
    )


def test_uncertain_com_increases_possible_payload_on_bridge_supports():
    left = _item(0, 0, 0, 20, 20, 10)
    right = _item(80, 0, 0, 20, 20, 10)
    bridge = _item(0, 0, 10, 100, 20, 10)
    kwargs = dict(lbcp_polygons={i.to_key(): _full_top_lbcp(i) for i in (left, right)},
                  material_density=1e-3, gravity=1, payload_max_only=True)
    fixed = solve_global_load_bounds([left, right, bridge], **kwargs)
    uncertain = solve_global_load_bounds([left, right, bridge], uncertain_com=True, **kwargs)
    weight = bridge.Dim.Volume * 1e-3
    assert fixed.status is uncertain.status is GlobalLoadStatus.OPTIMAL
    for support in (left, right):
        assert fixed.item_payload_bounds[support.to_key()].maximum == pytest.approx(.625 * weight)
        assert uncertain.item_payload_bounds[support.to_key()].maximum == pytest.approx(.75 * weight)
        assert uncertain.item_payload_bounds[support.to_key()].minimum is None
    assert uncertain.item_payload_bounds[bridge.to_key()].maximum == 0
    assert uncertain.linprog_calls == 3  # feasibility + two nonzero objectives
    assert not uncertain.interface_bounds

    interface_max = solve_global_load_bounds(
        [left, right, bridge], lbcp_polygons=kwargs['lbcp_polygons'],
        material_density=1e-3, gravity=1, uncertain_com=True, interface_max_only=True)
    assert interface_max.status is GlobalLoadStatus.OPTIMAL
    assert interface_max.linprog_calls == 1 + interface_max.interface_count
    assert not interface_max.item_payload_bounds
    assert len(interface_max.interface_vertex_forces) == interface_max.interface_count
    for key, forces in interface_max.interface_vertex_forces.items():
        assert forces.at_force_min == ()
        assert sum(forces.at_force_max) == pytest.approx(interface_max.interface_bounds[key].maximum)
    assert all(bound.minimum is None for bound in interface_max.interface_bounds.values())
    for support in (left, right):
        key = SupportInterfaceKey(bridge.to_key(), support.to_key())
        assert interface_max.interface_bounds[key].maximum == pytest.approx(.75 * weight)
    fixed_max = solve_global_load_bounds(
        [left, right, bridge], lbcp_polygons=kwargs['lbcp_polygons'],
        material_density=1e-3, gravity=1, interface_max_only=True)
    assert fixed_max.linprog_calls == interface_max.linprog_calls
    assert fixed_max.status is GlobalLoadStatus.OPTIMAL
    assert not fixed_max.item_payload_bounds
    for support in (left, right):
        key = SupportInterfaceKey(bridge.to_key(), support.to_key())
        assert fixed_max.interface_bounds[key].maximum == pytest.approx(.625 * weight)


def test_single_grounded_item_has_zero_payload_and_weight_ground_reaction():
    item = _item(0, 0, 0, 50, 50, 40)
    result = solve_global_load_bounds(
        [item],
        lbcp_polygons={item.to_key(): _full_top_lbcp(item)},
        material_density=1e-3,
    )

    assert result.status is GlobalLoadStatus.OPTIMAL
    expected_weight = 1e-3 * item.Dim.Volume * 9.81
    ground_key = SupportInterfaceKey(item.to_key(), None)
    assert result.interface_bounds[ground_key].minimum == pytest.approx(expected_weight)
    assert result.interface_bounds[ground_key].maximum == pytest.approx(expected_weight)
    assert result.item_payload_bounds[item.to_key()].minimum == pytest.approx(0.0)
    assert result.item_payload_bounds[item.to_key()].maximum == pytest.approx(0.0)


def test_two_item_column_payload_excludes_lower_self_weight():
    lower = _item(0, 0, 0, 50, 50, 40)
    upper = _item(0, 0, 40, 50, 50, 20)
    result = solve_global_load_bounds(
        [lower, upper],
        lbcp_polygons={lower.to_key(): _full_top_lbcp(lower)},
        material_density=1e-3,
    )

    assert result.status is GlobalLoadStatus.OPTIMAL
    upper_weight = 1e-3 * upper.Dim.Volume * 9.81
    lower_weight = 1e-3 * lower.Dim.Volume * 9.81
    contact_key = SupportInterfaceKey(upper.to_key(), lower.to_key())
    ground_key = SupportInterfaceKey(lower.to_key(), None)
    assert result.interface_bounds[contact_key].minimum == pytest.approx(upper_weight)
    assert result.interface_bounds[contact_key].maximum == pytest.approx(upper_weight)
    assert result.item_payload_bounds[lower.to_key()].maximum == pytest.approx(upper_weight)
    assert result.interface_bounds[ground_key].minimum == pytest.approx(
        upper_weight + lower_weight
    )


def test_missing_support_makes_global_equilibrium_infeasible():
    floating = _item(0, 0, 40, 50, 50, 20)
    result = solve_global_load_bounds(
        [floating],
        lbcp_polygons={},
        material_density=1e-3,
    )

    assert result.status is GlobalLoadStatus.INFEASIBLE


def test_lbcp_overlap_intersection_is_the_only_effective_support_region():
    base = _item(0, 0, 0, 100, 100, 40)
    lower = _item(0, 0, 40, 100, 100, 40)
    upper = _item(0, 0, 80, 100, 100, 20)
    left_only_lbcp = shapely_box(0, 0, 40, 100)

    result = solve_global_load_bounds(
        [base, lower, upper],
        lbcp_polygons={
            base.to_key(): _full_top_lbcp(base),
            lower.to_key(): left_only_lbcp,
        },
        material_density=1e-3,
    )

    assert result.status is GlobalLoadStatus.INFEASIBLE


def test_bridge_interface_bounds_retain_global_load_indeterminacy():
    left = _item(0, 0, 0, 40, 100, 40)
    right = _item(60, 0, 0, 40, 100, 40)
    bridge = _item(0, 0, 40, 100, 100, 20)
    result = solve_global_load_bounds(
        [left, right, bridge],
        lbcp_polygons={
            left.to_key(): _full_top_lbcp(left),
            right.to_key(): _full_top_lbcp(right),
        },
        material_density=1e-3,
    )

    assert result.status is GlobalLoadStatus.OPTIMAL
    bridge_weight = 1e-3 * bridge.Dim.Volume * 9.81
    left_key = SupportInterfaceKey(bridge.to_key(), left.to_key())
    right_key = SupportInterfaceKey(bridge.to_key(), right.to_key())
    left_bounds = result.interface_bounds[left_key]
    right_bounds = result.interface_bounds[right_key]
    assert 0.0 < left_bounds.minimum < left_bounds.maximum < bridge_weight
    assert 0.0 < right_bounds.minimum < right_bounds.maximum < bridge_weight
    assert left_bounds.minimum == pytest.approx(right_bounds.minimum)
    assert left_bounds.maximum == pytest.approx(right_bounds.maximum)

    for key, bounds in ((left_key, left_bounds), (right_key, right_bounds)):
        scenario = result.interface_vertex_forces[key]
        vertex_count = len(result.interfaces[key].polygon_xy)
        assert len(scenario.at_force_min) == vertex_count
        assert len(scenario.at_force_max) == vertex_count
        assert all(force >= 0.0 for force in scenario.at_force_min)
        assert all(force >= 0.0 for force in scenario.at_force_max)
        assert sum(scenario.at_force_min) == pytest.approx(bounds.minimum)
        assert sum(scenario.at_force_max) == pytest.approx(bounds.maximum)
        assert scenario.at_force_min != pytest.approx(scenario.at_force_max)


def test_payload_max_only_mode_skips_unneeded_interface_and_lower_bounds():
    lower = _item(0, 0, 0, 50, 50, 40)
    upper = _item(0, 0, 40, 50, 50, 20)
    result = solve_global_load_bounds(
        [lower, upper],
        lbcp_polygons={lower.to_key(): _full_top_lbcp(lower)},
        material_density=1e-3,
        payload_max_only=True,
    )

    assert result.status is GlobalLoadStatus.OPTIMAL
    assert result.bounds_complete is False
    assert result.interface_bounds == {}
    assert result.item_payload_bounds[lower.to_key()].minimum is None
    assert result.item_payload_bounds[lower.to_key()].maximum > 0


def test_feasibility_only_mode_skips_every_bound_objective():
    item = _item(0, 0, 0, 50, 50, 40)
    result = solve_global_load_bounds(
        [item],
        lbcp_polygons={item.to_key(): _full_top_lbcp(item)},
        material_density=1e-3,
        feasibility_only=True,
    )

    assert result.status is GlobalLoadStatus.OPTIMAL
    assert result.bounds_complete is False
    assert result.interface_bounds == {}
    assert result.item_payload_bounds == {}


def test_interface_bounds_only_skips_item_payload_objectives():
    lower = _item(0, 0, 0, 50, 50, 40)
    upper = _item(0, 0, 40, 50, 50, 20)

    result = solve_global_load_bounds(
        [lower, upper],
        lbcp_polygons={lower.to_key(): _full_top_lbcp(lower)},
        material_density=1e-3,
        interface_bounds_only=True,
    )

    assert result.status is GlobalLoadStatus.OPTIMAL
    assert result.interface_count == 2
    assert result.linprog_calls == 1 + 2 * result.interface_count
    assert len(result.interface_bounds) == result.interface_count
    assert len(result.interface_vertex_forces) == result.interface_count
    assert result.item_payload_bounds == {}
    assert result.bounds_complete is False


def test_full_solve_reports_total_and_accumulated_highs_time(monkeypatch):
    clock = iter([0.0, 1.0, 3.0, 4.0, 6.0, 7.0, 9.0, 12.0])
    monkeypatch.setattr(load_bounds_module.time, "perf_counter", lambda: next(clock))
    item = _item(0, 0, 0, 50, 50, 40)

    result = solve_global_load_bounds(
        [item],
        lbcp_polygons={item.to_key(): _full_top_lbcp(item)},
        material_density=1e-3,
    )

    assert result.status is GlobalLoadStatus.OPTIMAL
    assert result.linprog_calls == 3
    assert result.linprog_time_sec == pytest.approx(6.0)
    assert result.solve_time_sec == pytest.approx(12.0)
    assert result.interface_count == 1
    assert result.variable_count == 4


def test_empty_solve_reports_zero_sized_timing_metadata(monkeypatch):
    clock = iter([10.0, 10.25])
    monkeypatch.setattr(load_bounds_module.time, "perf_counter", lambda: next(clock))

    result = solve_global_load_bounds([], lbcp_polygons={}, material_density=1e-3)

    assert result.status is GlobalLoadStatus.OPTIMAL
    assert result.solve_time_sec == pytest.approx(0.25)
    assert result.linprog_time_sec == 0.0
    assert result.linprog_calls == 0
    assert result.interface_count == 0
    assert result.variable_count == 0


def test_unsupported_floating_item_early_return_is_timed(monkeypatch):
    clock = iter([20.0, 20.5])
    monkeypatch.setattr(load_bounds_module.time, "perf_counter", lambda: next(clock))
    floating = _item(0, 0, 40, 50, 50, 20)

    result = solve_global_load_bounds(
        [floating], lbcp_polygons={}, material_density=1e-3
    )

    assert result.status is GlobalLoadStatus.INFEASIBLE
    assert result.solve_time_sec == pytest.approx(0.5)
    assert result.linprog_calls == 0
    assert result.interface_count == 0
    assert result.variable_count == 0
