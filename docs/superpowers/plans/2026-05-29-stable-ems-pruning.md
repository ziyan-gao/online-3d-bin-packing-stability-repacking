# Stable EMS Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically prune EMSs that cannot stably support any unique current buffer item type, while keeping `remove_inscribed_ems: false` useful.

**Architecture:** `EMS` owns pruning its internal list through a new `prune_unstable(...)` method. `PackingEnv` passes the updated height map, stability map, and unique current buffer item types after each pack/update.

**Tech Stack:** Python, NumPy, Gymnasium environment code, pytest.

---

## File Structure

- Modify `packing_env/heu_ems.py`: add item-aware unstable EMS pruning helpers and mutate the private EMS list after filtering.
- Modify `packing_env/gym_env.py`: call EMS pruning after `_step(...)` and `pack(...)` EMS geometry updates.
- Modify `tests/test_simpleblock_buffer.py`: add focused pruning and integration tests using existing fake sampler/stability helpers.

---

### Task 1: Add EMS-Level Unstable Pruning

**Files:**
- Modify: `packing_env/heu_ems.py`
- Test: `tests/test_simpleblock_buffer.py`

- [ ] **Step 1: Write failing EMS pruning tests**

Add these tests near the existing EMS-related tests in `tests/test_simpleblock_buffer.py`:

```python
def test_prune_unstable_removes_unsupported_ems_by_heightmap():
    from packing_env.heu_ems import EMS

    ems_manager = EMS(container=Orthogonal3D(300, 300, 300), remove_inscribed=False)
    supported = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(100, 100, 100))
    floating = EmptyMaximalSpace(Point3D(100, 0, 50), Orthogonal3D(100, 100, 100))
    ems_manager._EMS__ems_list = [supported, floating]
    ems_manager._rebuild_index()

    hm = PackingEnv(container_size=(300, 300, 300)).hm
    ems_manager.prune_unstable(
        hm=hm,
        feasibility_map=AlwaysStable(),
        item_types=[Orthogonal3D(50, 50, 50)],
    )

    assert ems_manager.get_all_ems() == [supported]


def test_prune_unstable_keeps_supported_stable_fitting_ems():
    from packing_env.heu_ems import EMS

    ems_manager = EMS(container=Orthogonal3D(300, 300, 300), remove_inscribed=False)
    usable = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(100, 100, 100))
    too_small = EmptyMaximalSpace(Point3D(100, 0, 0), Orthogonal3D(40, 40, 100))
    ems_manager._EMS__ems_list = [usable, too_small]
    ems_manager._rebuild_index()

    hm = PackingEnv(container_size=(300, 300, 300)).hm
    ems_manager.prune_unstable(
        hm=hm,
        feasibility_map=AlwaysStable(),
        item_types=[Orthogonal3D(50, 50, 50)],
    )

    assert ems_manager.get_all_ems() == [usable]


def test_prune_unstable_tests_rotated_item_orientation():
    from packing_env.heu_ems import EMS

    ems_manager = EMS(container=Orthogonal3D(300, 300, 300), remove_inscribed=False)
    rotated_fit = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(100, 200, 100))
    ems_manager._EMS__ems_list = [rotated_fit]
    ems_manager._rebuild_index()

    hm = PackingEnv(container_size=(300, 300, 300)).hm
    ems_manager.prune_unstable(
        hm=hm,
        feasibility_map=AlwaysStable(),
        item_types=[Orthogonal3D(200, 100, 50)],
    )

    assert ems_manager.get_all_ems() == [rotated_fit]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_simpleblock_buffer.py::test_prune_unstable_removes_unsupported_ems_by_heightmap tests/test_simpleblock_buffer.py::test_prune_unstable_keeps_supported_stable_fitting_ems tests/test_simpleblock_buffer.py::test_prune_unstable_tests_rotated_item_orientation -v
```

Expected: FAIL with `AttributeError: 'EMS' object has no attribute 'prune_unstable'`.

- [ ] **Step 3: Add EMS pruning implementation**

Add these methods inside `EMS`, near `_remove_floating(...)`:

```python
    @staticmethod
    def _unique_item_orientations(
        item_types: Iterable[Orthogonal3D],
    ) -> list[Orthogonal3D]:
        orientations: list[Orthogonal3D] = []
        seen: set[tuple[int, int, int]] = set()
        for item_type in item_types:
            item = _to_o3d(item_type)
            candidates = [item]
            if item.dx != item.dy:
                candidates.append(Orthogonal3D(item.dy, item.dx, item.dz))
            for candidate in candidates:
                key = candidate.to_dim_key()
                if key in seen:
                    continue
                seen.add(key)
                orientations.append(candidate)
        return orientations

    @staticmethod
    def _has_stable_item_fit(
        space: EmptyMaximalSpace,
        hm: HeightMap,
        feasibility_map,
        item_orientations: Iterable[Orthogonal3D],
    ) -> bool:
        for item_dim in item_orientations:
            if not space.include(item_dim):
                continue
            stable_placements, is_stable = feasibility_map(
                o3d=item_dim,
                hm=hm,
                candidates=np.array([space.FLB.topix()]),
            )
            if (
                len(stable_placements) > 0
                and stable_placements[0] is not None
                and stable_placements[0] == space.FLB
                and bool(is_stable[0])
            ):
                return True
        return False

    def prune_unstable(
        self,
        hm: HeightMap,
        feasibility_map,
        item_types: Iterable[Orthogonal3D],
    ) -> None:
        item_orientations = self._unique_item_orientations(item_types)
        if not item_orientations:
            self.__ems_list = []
            self._rebuild_index()
            return

        supported = self._remove_floating(self.__ems_list, hm)
        self.__ems_list = [
            space
            for space in supported
            if self._has_stable_item_fit(space, hm, feasibility_map, item_orientations)
        ]
        self._rebuild_index()
```

- [ ] **Step 4: Run EMS pruning tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_simpleblock_buffer.py::test_prune_unstable_removes_unsupported_ems_by_heightmap tests/test_simpleblock_buffer.py::test_prune_unstable_keeps_supported_stable_fitting_ems tests/test_simpleblock_buffer.py::test_prune_unstable_tests_rotated_item_orientation -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packing_env/heu_ems.py tests/test_simpleblock_buffer.py
git commit -m "Add stable EMS pruning"
```

---

### Task 2: Verify Unique Item Types And Stability Failure Behavior

**Files:**
- Modify: `tests/test_simpleblock_buffer.py`

- [ ] **Step 1: Add focused tests for deduplication and unstable pruning**

Add these helper/test definitions to `tests/test_simpleblock_buffer.py`:

```python
class CountingStable:
    def __init__(self, stable=True):
        self.stable = bool(stable)
        self.calls = []

    def __call__(self, o3d, hm, candidates):
        self.calls.append(o3d.to_dim_key())
        if self.stable:
            return [Point3D(0, 0, 0) for _ in candidates], np.ones(len(candidates), dtype=bool)
        return [None for _ in candidates], np.zeros(len(candidates), dtype=bool)


def test_unique_item_orientations_deduplicates_repeated_buffer_types():
    from packing_env.heu_ems import EMS

    orientations = EMS._unique_item_orientations(
        [
            Orthogonal3D(100, 100, 50),
            Orthogonal3D(100, 100, 50),
            Orthogonal3D(100, 200, 50),
            Orthogonal3D(100, 200, 50),
        ]
    )

    assert [orientation.to_dim_key() for orientation in orientations] == [
        (100, 100, 50),
        (100, 200, 50),
        (200, 100, 50),
    ]


def test_prune_unstable_removes_ems_when_stability_fails():
    from packing_env.heu_ems import EMS

    ems_manager = EMS(container=Orthogonal3D(300, 300, 300), remove_inscribed=False)
    unstable = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(200, 200, 100))
    ems_manager._EMS__ems_list = [unstable]
    ems_manager._rebuild_index()

    hm = PackingEnv(container_size=(300, 300, 300)).hm
    ems_manager.prune_unstable(
        hm=hm,
        feasibility_map=CountingStable(stable=False),
        item_types=[Orthogonal3D(100, 100, 50)],
    )

    assert ems_manager.get_all_ems() == []
```

- [ ] **Step 2: Run tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_simpleblock_buffer.py::test_unique_item_orientations_deduplicates_repeated_buffer_types tests/test_simpleblock_buffer.py::test_prune_unstable_removes_ems_when_stability_fails -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_simpleblock_buffer.py
git commit -m "Cover stable EMS pruning edge cases"
```

---

### Task 3: Integrate Pruning Into PackingEnv Update Paths

**Files:**
- Modify: `packing_env/gym_env.py`
- Modify: `tests/test_simpleblock_buffer.py`

- [ ] **Step 1: Write failing integration tests**

Add these tests to `tests/test_simpleblock_buffer.py`:

```python
def test_step_prunes_ems_after_buffer_update(monkeypatch):
    first = Orthogonal3D(100, 100, 50)
    second = Orthogonal3D(200, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=1,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
    )
    env.buffer = Buffer(
        capacity=1,
        data_sampler=FakeSampler([first, second, second]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    env.selected_item = env.buffer.sample_blocks()
    env.candidates = np.array([[0, 0, 0, 600, 600, 600]], dtype=np.int32)
    env.ems_list = env.heu_ems.get_all_ems()
    captured = {}

    def capture_prune(hm, feasibility_map, item_types):
        captured["item_types"] = list(item_types)

    monkeypatch.setattr(env.heu_ems, "prune_unstable", capture_prune)
    box = env.selected_item.to_item(Point3D(0, 0, 0))

    env._step(env.selected_item, box, env.ems_list[0])

    assert captured["item_types"] == [second]
    assert env.buffer.summary == {second: 1}


def test_pack_prunes_ems_with_current_buffer_item_types(monkeypatch):
    box_type = Orthogonal3D(100, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=2,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
    )
    env.buffer = Buffer(
        capacity=2,
        data_sampler=FakeSampler([box_type, box_type]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    captured = {}

    def capture_prune(hm, feasibility_map, item_types):
        captured["item_types"] = list(item_types)

    monkeypatch.setattr(env.heu_ems, "prune_unstable", capture_prune)
    selected_ems = env.heu_ems.get_all_ems()[0]
    env.pack(Item(Point3D(0, 0, 0), box_type), selected_ems=selected_ems)

    assert captured["item_types"] == [box_type]
    assert env.buffer.summary == {box_type: 2}
```

- [ ] **Step 2: Run integration tests to verify they fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_simpleblock_buffer.py::test_step_prunes_ems_after_buffer_update tests/test_simpleblock_buffer.py::test_pack_prunes_ems_with_current_buffer_item_types -v
```

Expected: FAIL because `PackingEnv` does not call `prune_unstable(...)` yet.

- [ ] **Step 3: Add PackingEnv helper**

In `packing_env/gym_env.py`, add this method near `_step(...)`:

```python
    def _prune_unstable_ems(self) -> None:
        self.heu_ems.prune_unstable(
            hm=self.hm,
            feasibility_map=self.heu_stable,
            item_types=self.buffer.summary.keys(),
        )
```

- [ ] **Step 4: Call pruning after EMS updates**

In `_step(...)`, after:

```python
        self.heu_ems.update(box=placed_item, selected_ems=selected_ems, hm=self.hm)
```

add:

```python
        self._prune_unstable_ems()
```

In `pack(...)`, after:

```python
        self.heu_ems.update(box=box, selected_ems=selected_ems, hm=self.hm)
```

add:

```python
        self._prune_unstable_ems()
```

- [ ] **Step 5: Run integration tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_simpleblock_buffer.py::test_step_prunes_ems_after_buffer_update tests/test_simpleblock_buffer.py::test_pack_prunes_ems_with_current_buffer_item_types -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packing_env/gym_env.py tests/test_simpleblock_buffer.py
git commit -m "Prune unstable EMS after packing"
```

---

### Task 4: Verify Inscribed EMS Preservation

**Files:**
- Modify: `tests/test_simpleblock_buffer.py`

- [ ] **Step 1: Add inscribed EMS regression test**

Add this test to `tests/test_simpleblock_buffer.py`:

```python
def test_prune_unstable_preserves_stable_inscribed_ems_when_not_removing_inscribed():
    from packing_env.heu_ems import EMS

    ems_manager = EMS(container=Orthogonal3D(300, 300, 300), remove_inscribed=False)
    outer = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(300, 300, 300))
    inner = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(100, 100, 100))
    ems_manager._EMS__ems_list = [outer, inner]
    ems_manager._rebuild_index()

    hm = PackingEnv(container_size=(300, 300, 300)).hm
    ems_manager.prune_unstable(
        hm=hm,
        feasibility_map=AlwaysStable(),
        item_types=[Orthogonal3D(80, 80, 50)],
    )

    assert ems_manager.get_all_ems() == [outer, inner]
```

- [ ] **Step 2: Run test**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_simpleblock_buffer.py::test_prune_unstable_preserves_stable_inscribed_ems_when_not_removing_inscribed -v
```

Expected: PASS.

- [ ] **Step 3: Run all relevant tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_simpleblock_buffer.py tests/test_cascaded_block_candidates.py tests/test_cascaded_policy.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_simpleblock_buffer.py
git commit -m "Preserve stable inscribed EMS during pruning"
```

---

### Task 5: Final Verification

**Files:**
- No expected code changes unless verification exposes a defect.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_simpleblock_buffer.py tests/test_cascaded_block_candidates.py tests/test_cascaded_policy.py -v
```

Expected: PASS.

- [ ] **Step 2: Run a small packing smoke script**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python - <<'PY'
from packing_env.gym_env import PackingEnv

env = PackingEnv(
    k_placement=8,
    ds_name="random",
    container_size=(600, 600, 600),
    stack_only=True,
    use_simple_blocks=True,
    remove_inscribed_ems=False,
)
env.reset(seed=1)
before = len(env.heu_ems.get_all_ems())
obs = env.get_next_observation()
valid = obs["action_mask"].reshape(-1).nonzero()[0]
if len(valid):
    env.step(int(valid[0]))
after = len(env.heu_ems.get_all_ems())
print("ems_before", before)
print("ems_after", after)
print("placed", len(env.container.placed_items))
PY
```

Expected: exits 0 and prints EMS counts plus placed item count.

- [ ] **Step 3: Inspect status and recent commits**

Run:

```bash
git status --short --branch
git log --oneline -6
```

Expected: clean worktree on the current branch.
