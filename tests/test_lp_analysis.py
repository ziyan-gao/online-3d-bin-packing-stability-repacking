import json
import math

import pytest
from shapely.geometry import Polygon

from packing.lp_analysis import build_lp_frame, solve_packing_loads
from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item
from packing_env.gym_env import PackingEnv
from packing_env.lbcp.load_bounds import GlobalLoadStatus, SupportInterfaceKey
from packing.threejs_visualization import ThreeReplayRecorder


def stack():
    env = PackingEnv(container_size=(300, 300, 300))
    env.reset(seed=41)
    boxes = [Item(FLB=Point3D(0, 0, z), Dim=Orthogonal3D(100, 100, 50)) for z in (0, 50)]
    for box in boxes:
        env.pack(box)
    return env, boxes


def test_cache_exposes_world_coordinates_and_tracks_unpack_reset():
    env, boxes = stack()
    polygons = env.heu_stable.lbcp_polygon_map()
    assert set(polygons) == {box.to_key() for box in boxes}
    assert polygons[boxes[0].to_key()].bounds == (0, 0, 100, 100)
    env.unpack(boxes[-1])
    assert set(env.heu_stable.lbcp_polygon_map()) == {boxes[0].to_key()}
    env.reset(seed=41)
    assert env.heu_stable.lbcp_polygon_map() == {}


def test_stack_reactions_and_payload_match_weight_in_newtons():
    env, (lower, upper) = stack()
    result = solve_packing_loads(env, material_density=1e-6)
    assert result.status is GlobalLoadStatus.OPTIMAL
    weight = 0.5 * 9.81
    ground = result.interface_bounds[SupportInterfaceKey(lower.to_key(), None)]
    assert ground.minimum == pytest.approx(2 * weight)
    assert ground.maximum == pytest.approx(2 * weight)
    assert result.item_payload_bounds[lower.to_key()].maximum == pytest.approx(weight)


def test_adapter_uses_restricted_cached_lbcp_and_displays_infeasible_status():
    env, (lower, upper) = stack()
    # A support patch entirely to the left of the upper item's COM cannot balance it.
    from dataclasses import replace
    cache = env.heu_stable._convex_hull_vis_cache
    item, dims, vis = cache[0]
    cache[0] = (item, dims, replace(vis, support_polygon_xy=((0, 0), (10, 0), (10, 100), (0, 100))))
    result, frame = build_lp_frame(env, material_density=1e-6)
    assert result.status is GlobalLoadStatus.INFEASIBLE
    assert frame.metrics['lp_status'] == 'infeasible'
    assert 'infeasible' in frame.title.lower()
    assert not any(obj.kind == 'force_arrow' for obj in frame.scenes['container'].objects)


def test_missing_support_cache_is_not_silently_replaced_with_full_footprint():
    env, _ = stack()
    env.heu_stable._convex_hull_vis_cache.clear()
    with pytest.raises(ValueError, match='LBCP'):
        solve_packing_loads(env, material_density=1e-6)


@pytest.mark.parametrize('value', [0, -1, math.inf, math.nan])
def test_adapter_rejects_nonpositive_or_nonfinite_density_and_gravity(value):
    env, _ = stack()
    with pytest.raises(ValueError, match='material_density'):
        solve_packing_loads(env, material_density=value)
    with pytest.raises(ValueError, match='gravity'):
        solve_packing_loads(env, material_density=1e-6, gravity=value)


def test_lp_frame_and_offline_replay_include_ground_contacts_and_force_witnesses(tmp_path):
    env, _ = stack()
    result, frame = build_lp_frame(env, material_density=1e-6)
    assert result.status is GlobalLoadStatus.OPTIMAL
    objects = frame.scenes['container'].objects
    patches = [obj for obj in objects if obj.kind == 'contact_patch']
    assert len(patches) == result.interface_count == 2
    assert any(obj.kind == 'force_arrow' for obj in objects)
    assert any(obj.kind == 'support_patch' for obj in objects)
    assert frame.metrics['lp_status'] == 'optimal'
    for forces, bounds in zip(result.interface_vertex_forces.values(), result.interface_bounds.values()):
        assert sum(forces.at_force_min) == pytest.approx(bounds.minimum)
        assert sum(forces.at_force_max) == pytest.approx(bounds.maximum)
    json.dumps(frame.to_dict(), allow_nan=False)
    recorder = ThreeReplayRecorder(tmp_path / 'lp')
    recorder.capture(frame.title, frame)
    output = recorder.save()
    assert output.is_file()
    assert (output.parent / 'static' / 'renderer.bundle.js').is_file()


def test_demo_exports_actual_packing_sequence(tmp_path):
    from packing.lp_demo import run_demo
    result, output = run_demo(tmp_path / 'bridge', scene='bridge')
    assert result.status is GlobalLoadStatus.OPTIMAL
    assert output.is_file()
    data = json.loads((output.parent / 'frames.js').read_text().removeprefix('window.PACKING_THREE_REPLAY = ').removesuffix(';\n'))
    assert len(data['frames']) == 3
    assert all(record['frame']['metrics']['lp_status'] == 'optimal' for record in data['frames'])


@pytest.mark.parametrize('view', ['resultant', 'lp-solution', 'resultant-only'])
def test_force_views_render_bound_witnesses_with_correct_resultant_moments(view):
    from packing_env.visualization.config import VisualConfig
    env, _ = stack()
    result, frame = build_lp_frame(env, material_density=1e-6, config=VisualConfig(interface_force_view=view))
    arrows = [obj for obj in frame.scenes['container'].objects if obj.kind == 'force_arrow']
    assert {a.geometry['scenario'] for a in arrows} == {'force_min', 'force_max'}
    resultants = [a for a in arrows if a.geometry.get('representation') == 'ci_resultant']
    vertices = [a for a in arrows if a.geometry.get('representation') != 'ci_resultant']
    assert bool(resultants) == (view != 'lp-solution')
    assert bool(vertices) == (view != 'resultant-only')
    assert {'force_min', 'force_max'} <= {entry['kind'] for entry in frame.legend}
    assert not any(entry['kind'] == 'feasible' for entry in frame.legend)
    for arrow in resultants:
        z = arrow.geometry['physical_origin'][2]
        key = next(k for k in result.interfaces if (0 if k.lower_key is None else k.lower_key[2] + k.lower_key[5]) == z)
        interface = result.interfaces[key]
        scenario = arrow.geometry['scenario']
        forces = getattr(result.interface_vertex_forces[key], 'at_' + scenario)
        assert arrow.geometry['force_newtons'] == pytest.approx(sum(forces))
        for axis in (0, 1):
            moment = sum(xy[axis] * f for xy, f in zip(interface.polygon_xy, forces))
            assert arrow.geometry['physical_origin'][axis] * sum(forces) == pytest.approx(moment)
