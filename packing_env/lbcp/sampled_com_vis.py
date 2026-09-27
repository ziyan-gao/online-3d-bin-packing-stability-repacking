"""Independent seeded CoM sampling for visualization; never modifies experiment data."""
import hashlib
import random
from .load_bounds import solve_global_load_bounds, GlobalLoadStatus
from .contact_patch import item_top_z
from .load_tracking import is_on_floor
from ..data_type.support_vis import LoadContactPatchVis


class SampledCOMView:
    def __init__(self, seed):
        self.seed = seed
        self.com_xy = {}
        self._state = None
        self._result = None

    def solve(self, items, *, material_density, lbcp_polygons, gravity=9.81):
        state = (frozenset(i.to_key() for i in items), material_density, gravity,
                 tuple(sorted((str(k), getattr(p, "wkt", repr(p))) for k, p in lbcp_polygons.items())))
        if state == self._state:
            return self._result
        for item in items:
            key = item.to_key()
            if key not in self.com_xy:
                digest = hashlib.sha256(repr((self.seed, key, "visual-com")).encode()).digest()
                rng = random.Random(int.from_bytes(digest, "big"))
                self.com_xy[key] = (
                    item.True_FLB.x + item.Dim.dx * (0.5 + rng.uniform(-0.1, 0.1)),
                    item.True_FLB.y + item.Dim.dy * (0.5 + rng.uniform(-0.1, 0.1)),
                )
        result = solve_global_load_bounds(
            items, lbcp_polygons=lbcp_polygons, material_density=material_density,
            gravity=gravity, feasibility_only=True,
            fixed_com_xy={i.to_key(): self.com_xy[i.to_key()] for i in items})
        self._state, self._result = state, result
        return result

    def patches(self, items, result, *, floor_contact_only=True):
        if result.status is not GlobalLoadStatus.OPTIMAL:
            return []
        by_key = {item.to_key(): item for item in items}
        patches = []
        for key, interface in result.interfaces.items():
            lower = by_key.get(key.lower_key)
            if floor_contact_only and lower is not None and not is_on_floor(lower):
                continue
            forces = tuple(float(result.feasible_forces[v]) for v in interface.variable_indices)
            total = sum(forces)
            patches.append(LoadContactPatchVis(
                polygon_xy=interface.polygon_xy, z=item_top_z(lower) if lower else 0.0,
                force=total, pressure=total/interface.area, area=interface.area,
                vertex_forces_at_feasible=forces, sampled_com_feasible=True))
        return patches
