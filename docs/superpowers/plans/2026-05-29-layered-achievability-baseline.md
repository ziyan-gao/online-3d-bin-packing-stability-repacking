# Layered Achievability Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable z-layered EMS view for the largest-block simple-block baseline so block selection and placement only use the current robot-achievable vertical band.

**Architecture:** Keep `self.heu_ems` as the true geometric EMS manager. Add temporary policy-visible clipped EMSs inside `PackingEnv`, maintain a clipped-to-source EMS mapping, and advance the 1-based layer stage by at most one when the current stage has no usable simpleblock.

**Tech Stack:** Python, Gymnasium environment code, OmegaConf YAML configs, pytest.

---

## File Structure

- Modify `packing_env/gym_env.py`: owns runtime flags, layer stage state, EMS clipping, stage progression, baseline simple-block selection, and clipped-to-source EMS resolution.
- Modify `packing/train_utils.py`: adds train config fields and passes them into `PackingEnv`.
- Modify `packing/test_utils.py`: adds test config fields, passes them into `PackingEnv`, and resolves selected clipped EMSs when using the baseline direct `env.pack(...)` validation path.
- Modify `configs/test_cj_default.yaml`: enables experiment-friendly fields for the CJ baseline test config.
- Modify `configs/train_cj_default.yaml`: includes explicit default fields so training config accepts the new options.
- Modify `tests/test_cascaded_policy.py`: verifies config loading and validation.
- Create `tests/test_layered_achievability.py`: focused unit tests for layer windows, clipping, stage progression, selection, and EMS source mapping.

---

### Task 1: Add Config Fields And Constructor Validation

**Files:**
- Modify: `packing_env/gym_env.py`
- Modify: `packing/train_utils.py`
- Modify: `packing/test_utils.py`
- Modify: `configs/test_cj_default.yaml`
- Modify: `configs/train_cj_default.yaml`
- Test: `tests/test_cascaded_policy.py`

- [ ] **Step 1: Write failing config tests**

Add these tests to `tests/test_cascaded_policy.py`:

```python
def test_train_config_accepts_layered_achievability_fields():
    config = TrainConfig(
        use_simple_blocks=True,
        stack_only=True,
        layered_achievability=True,
        layered_num_chunks=4,
    )

    assert config.layered_achievability is True
    assert config.layered_num_chunks == 4


def test_test_config_accepts_layered_achievability_fields():
    config = test_utils.TestConfig(
        use_simple_blocks=True,
        stack_only=True,
        layered_achievability=True,
        layered_num_chunks=5,
    )

    assert config.layered_achievability is True
    assert config.layered_num_chunks == 5


def test_layered_achievability_requires_simple_block_baseline():
    with pytest.raises(ValueError, match="largest_block_baseline"):
        PackingEnv(
            layered_achievability=True,
            use_simple_blocks=True,
            policy_mode="cascaded_block_selector",
        )

    with pytest.raises(ValueError, match="use_simple_blocks"):
        PackingEnv(
            layered_achievability=True,
            use_simple_blocks=False,
            policy_mode="largest_block_baseline",
        )


def test_layered_num_chunks_must_be_positive():
    with pytest.raises(ValueError, match="layered_num_chunks"):
        PackingEnv(
            layered_achievability=True,
            layered_num_chunks=0,
            use_simple_blocks=True,
            policy_mode="largest_block_baseline",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py -v
```

Expected: FAIL because `layered_achievability` and `layered_num_chunks` are not accepted yet.

- [ ] **Step 3: Add environment constructor fields**

In `packing_env/gym_env.py`, extend `PackingEnv.__init__`:

```python
        layered_achievability: bool = False,
        layered_num_chunks: int = 3,
```

After policy-mode/simple-block normalization, add:

```python
        self.layered_achievability = bool(layered_achievability)
        self.layered_num_chunks = int(layered_num_chunks)
        if self.layered_num_chunks <= 0:
            raise ValueError("layered_num_chunks must be a positive integer.")
        if self.layered_achievability:
            if self.policy_mode != "largest_block_baseline":
                raise ValueError(
                    "layered_achievability is supported only with "
                    "policy_mode='largest_block_baseline'."
                )
            if not self.use_simple_blocks:
                raise ValueError(
                    "layered_achievability requires use_simple_blocks=True."
                )
        self.layered_stage = 1
        self._policy_ems_source_by_key: dict[
            tuple[int, int, int, int, int, int],
            EmptyMaximalSpace,
        ] = {}
```

- [ ] **Step 4: Add train config fields and env plumbing**

In `packing/train_utils.py`, add fields to `TrainConfig` after `policy_mode`:

```python
    layered_achievability: bool = False
    layered_num_chunks: int = 3
```

In `TrainConfig.__post_init__`, add:

```python
        if int(self.layered_num_chunks) <= 0:
            raise ValueError("layered_num_chunks must be a positive integer.")
```

Pass these fields into all `gym.make(...)` and `PackingEnv` creation calls in `make_envs(...)` and `make_single_env(...)`:

```python
                layered_achievability=config.layered_achievability,
                layered_num_chunks=config.layered_num_chunks,
```

- [ ] **Step 5: Add test config fields and env plumbing**

In `packing/test_utils.py`, add fields to `TestConfig` after `policy_mode`:

```python
    layered_achievability: bool = False
    layered_num_chunks: int = 3
```

In `build_env(...)`, pass:

```python
        layered_achievability=config.layered_achievability,
        layered_num_chunks=config.layered_num_chunks,
```

- [ ] **Step 6: Add explicit YAML fields**

Add to `configs/test_cj_default.yaml` near `policy_mode`:

```yaml
layered_achievability: false
layered_num_chunks: 3
```

Add to `configs/train_cj_default.yaml` near `policy_mode`:

```yaml
layered_achievability: false
layered_num_chunks: 3
```

- [ ] **Step 7: Run config tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add packing_env/gym_env.py packing/train_utils.py packing/test_utils.py configs/test_cj_default.yaml configs/train_cj_default.yaml tests/test_cascaded_policy.py
git commit -m "Add layered achievability config"
```

---

### Task 2: Add Stage Window And EMS Clipping Helpers

**Files:**
- Modify: `packing_env/gym_env.py`
- Create: `tests/test_layered_achievability.py`

- [ ] **Step 1: Write failing tests for layer windows and clipping**

Create `tests/test_layered_achievability.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packing_env.data_type import EmptyMaximalSpace, Orthogonal3D, Point3D
from packing_env.gym_env import PackingEnv


def make_layered_env(**kwargs):
    defaults = {
        "k_placement": 8,
        "buffer_capacity": 3,
        "container_size": (300, 300, 300),
        "stack_only": True,
        "use_simple_blocks": True,
        "policy_mode": "largest_block_baseline",
        "layered_achievability": True,
        "layered_num_chunks": 3,
    }
    defaults.update(kwargs)
    return PackingEnv(**defaults)


def ems_tuple(ems):
    return (
        int(ems.FLB.x),
        int(ems.FLB.y),
        int(ems.FLB.z),
        int(ems.Dim.dx),
        int(ems.Dim.dy),
        int(ems.Dim.dz),
    )


def test_layered_stage_windows_for_three_chunks():
    env = make_layered_env()

    assert env._layered_stage_window(1) == (0, 100)
    assert env._layered_stage_window(2) == (0, 200)
    assert env._layered_stage_window(3) == (100, 300)


def test_layered_stage_window_rejects_out_of_range_stage():
    env = make_layered_env()

    with pytest.raises(ValueError, match="layered stage"):
        env._layered_stage_window(0)
    with pytest.raises(ValueError, match="layered stage"):
        env._layered_stage_window(4)


def test_clip_ems_truncates_discards_and_preserves_spaces():
    env = make_layered_env()
    raw = [
        EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(300, 300, 300)),
        EmptyMaximalSpace(Point3D(0, 0, 220), Orthogonal3D(300, 300, 50)),
        EmptyMaximalSpace(Point3D(100, 0, 120), Orthogonal3D(100, 100, 30)),
    ]

    clipped = env._clip_ems_to_layer_window(raw, stage=2)

    assert [ems_tuple(ems) for ems in clipped] == [
        (0, 0, 0, 300, 300, 200),
        (100, 0, 120, 100, 100, 30),
    ]
    assert env.resolve_policy_ems_source(clipped[0]) is raw[0]
    assert env.resolve_policy_ems_source(clipped[1]) is raw[2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py -v
```

Expected: FAIL because helper methods are missing.

- [ ] **Step 3: Add helper methods**

In `packing_env/gym_env.py`, add these methods inside `PackingEnv` near EMS-list helpers:

```python
    @staticmethod
    def _ems_key(ems: EmptyMaximalSpace) -> tuple[int, int, int, int, int, int]:
        return (
            int(ems.FLB.x),
            int(ems.FLB.y),
            int(ems.FLB.z),
            int(ems.Dim.dx),
            int(ems.Dim.dy),
            int(ems.Dim.dz),
        )

    def _layered_stage_window(self, stage: int | None = None) -> tuple[int, int]:
        stage = self.layered_stage if stage is None else int(stage)
        if stage < 1 or stage > self.layered_num_chunks:
            raise ValueError(
                f"layered stage must be in [1, {self.layered_num_chunks}], got {stage}"
            )
        height = int(self.container.dz)
        z_min = 0 if stage == 1 else (stage - 2) * height // self.layered_num_chunks
        z_max = stage * height // self.layered_num_chunks
        if stage == self.layered_num_chunks:
            z_max = height
        return int(z_min), int(z_max)

    def _clip_ems_to_layer_window(
        self,
        ems_list: list[EmptyMaximalSpace],
        stage: int | None = None,
    ) -> list[EmptyMaximalSpace]:
        z_min, z_max = self._layered_stage_window(stage)
        clipped: list[EmptyMaximalSpace] = []
        source_by_key: dict[tuple[int, int, int, int, int, int], EmptyMaximalSpace] = {}
        for raw in ems_list:
            raw_z0 = int(raw.FLB.z)
            raw_z1 = int(raw.FLB.z + raw.Dim.dz)
            new_z0 = max(raw_z0, z_min)
            new_z1 = min(raw_z1, z_max)
            if new_z1 <= new_z0:
                continue
            policy_ems = EmptyMaximalSpace(
                FLB=Point3D(int(raw.FLB.x), int(raw.FLB.y), int(new_z0)),
                Dim=Orthogonal3D(
                    int(raw.Dim.dx),
                    int(raw.Dim.dy),
                    int(new_z1 - new_z0),
                ),
            )
            clipped.append(policy_ems)
            source_by_key[self._ems_key(policy_ems)] = raw
        self._policy_ems_source_by_key = source_by_key
        return clipped

    def resolve_policy_ems_source(
        self,
        selected_ems: EmptyMaximalSpace | None,
    ) -> EmptyMaximalSpace | None:
        if selected_ems is None:
            return None
        return self._policy_ems_source_by_key.get(
            self._ems_key(selected_ems),
            selected_ems,
        )
```

- [ ] **Step 4: Run clipping tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packing_env/gym_env.py tests/test_layered_achievability.py
git commit -m "Add layered EMS clipping helpers"
```

---

### Task 3: Use Layered EMSs For Baseline Selection

**Files:**
- Modify: `packing_env/gym_env.py`
- Modify: `tests/test_layered_achievability.py`

- [ ] **Step 1: Write failing selection test**

Append to `tests/test_layered_achievability.py`:

```python
from packing_env.data_type import SimpleBlock


def test_largest_policy_block_uses_current_layer_clipped_ems(monkeypatch):
    env = make_layered_env(container_size=(300, 300, 300), layered_num_chunks=3)
    env.layered_stage = 1
    low_ems = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(300, 300, 300))
    monkeypatch.setattr(env.heu_ems, "get_all_ems", lambda: [low_ems])

    small = SimpleBlock(box=Orthogonal3D(100, 100, 80), stack_dims=(1, 1, 1))
    tall = SimpleBlock(box=Orthogonal3D(100, 100, 150), stack_dims=(1, 1, 1))
    env.buffer.simple_blocks = {
        small.box: [small],
        tall.box: [tall],
    }

    selected = env.select_largest_policy_block()

    assert selected is small
    assert env.buffer.all_blocks == [small]
    assert len(env.ems_list) == 1
    assert env.ems_list[0].Dim.dz == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py::test_largest_policy_block_uses_current_layer_clipped_ems -v
```

Expected: FAIL because selection still uses raw EMSs.

- [ ] **Step 3: Add policy EMS view helper**

In `packing_env/gym_env.py`, add:

```python
    def _get_policy_ems_list(
        self,
        *,
        stage: int | None = None,
    ) -> list[EmptyMaximalSpace]:
        raw_ems = self.heu_ems.get_all_ems()
        if not self.layered_achievability:
            self._policy_ems_source_by_key = {
                self._ems_key(ems): ems for ems in raw_ems
            }
            return raw_ems
        return self._clip_ems_to_layer_window(raw_ems, stage=stage)
```

Change the first line of `_get_item_fit_ems_list(...)` from:

```python
        ems_list = self.heu_ems.get_all_ems()
```

to:

```python
        ems_list = self._get_policy_ems_list()
```

- [ ] **Step 4: Ensure selected EMS list is visible during selection**

Update `select_largest_policy_block(...)` so each candidate uses the layered EMS view and the winning EMS list is stored:

```python
        for block in ranked_blocks:
            ems_list = self._get_item_fit_ems_list([block])
            if self.buffer._is_block_usable(block, ems_list, self.heu_stable, self.hm):
                self.buffer.simple_blocks = {block.box: [block]}
                self.ems_list = ems_list
                return block
```

- [ ] **Step 5: Run selection test**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py::test_largest_policy_block_uses_current_layer_clipped_ems -v
```

Expected: PASS.

- [ ] **Step 6: Run full layered tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add packing_env/gym_env.py tests/test_layered_achievability.py
git commit -m "Select baseline blocks from layered EMS view"
```

---

### Task 4: Add One-Stage Progression

**Files:**
- Modify: `packing_env/gym_env.py`
- Modify: `tests/test_layered_achievability.py`

- [ ] **Step 1: Write failing stage progression tests**

Append to `tests/test_layered_achievability.py`:

```python
def test_layered_selection_advances_exactly_one_stage(monkeypatch):
    env = make_layered_env(container_size=(300, 300, 300), layered_num_chunks=3)
    raw_ems = EmptyMaximalSpace(Point3D(0, 0, 100), Orthogonal3D(300, 300, 100))
    monkeypatch.setattr(env.heu_ems, "get_all_ems", lambda: [raw_ems])

    block = SimpleBlock(box=Orthogonal3D(100, 100, 80), stack_dims=(1, 1, 1))
    env.buffer.simple_blocks = {block.box: [block]}
    env.layered_stage = 1

    selected = env.select_largest_policy_block()

    assert selected is block
    assert env.layered_stage == 2


def test_layered_selection_does_not_skip_multiple_stages(monkeypatch):
    env = make_layered_env(
        container_size=(300, 300, 400),
        layered_num_chunks=4,
    )
    raw_ems = EmptyMaximalSpace(Point3D(0, 0, 300), Orthogonal3D(300, 300, 100))
    monkeypatch.setattr(env.heu_ems, "get_all_ems", lambda: [raw_ems])

    block = SimpleBlock(box=Orthogonal3D(100, 100, 80), stack_dims=(1, 1, 1))
    env.buffer.simple_blocks = {block.box: [block]}
    env.layered_stage = 1

    selected = env.select_largest_policy_block()

    assert selected is None
    assert env.layered_stage == 1
    assert env.buffer.all_blocks == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py::test_layered_selection_advances_exactly_one_stage tests/test_layered_achievability.py::test_layered_selection_does_not_skip_multiple_stages -v
```

Expected: FAIL because `select_largest_policy_block()` does not test the next stage.

- [ ] **Step 3: Add stage-aware selection helper**

In `packing_env/gym_env.py`, add:

```python
    def _select_largest_policy_block_for_stage(
        self,
        stage: int,
    ) -> tuple[SimpleBlock | None, list[EmptyMaximalSpace]]:
        ranked_blocks = sorted(
            self.buffer.all_blocks,
            key=lambda block: (
                block.volume,
                block.consumed_count,
                block.Dim.dz,
                block.Dim.dy,
                block.Dim.dx,
            ),
            reverse=True,
        )
        original_stage = self.layered_stage
        try:
            self.layered_stage = int(stage)
            for block in ranked_blocks:
                ems_list = self._get_item_fit_ems_list([block])
                if self.buffer._is_block_usable(block, ems_list, self.heu_stable, self.hm):
                    return block, ems_list
            return None, []
        finally:
            self.layered_stage = original_stage
```

- [ ] **Step 4: Rewrite `select_largest_policy_block` to use current and next stage**

Replace the body of `select_largest_policy_block(...)` with:

```python
        if not self.layered_achievability:
            block, ems_list = self._select_largest_policy_block_for_stage(self.layered_stage)
            if block is None:
                self.buffer.simple_blocks = {}
                self.ems_list = []
                return None
            self.buffer.simple_blocks = {block.box: [block]}
            self.ems_list = ems_list
            return block

        block, ems_list = self._select_largest_policy_block_for_stage(self.layered_stage)
        if block is None and self.layered_stage < self.layered_num_chunks:
            next_stage = self.layered_stage + 1
            next_block, next_ems_list = self._select_largest_policy_block_for_stage(next_stage)
            if next_block is not None:
                self.layered_stage = next_stage
                block, ems_list = next_block, next_ems_list

        if block is None:
            self.buffer.simple_blocks = {}
            self.ems_list = []
            return None

        self.buffer.simple_blocks = {block.box: [block]}
        self.ems_list = ems_list
        return block
```

- [ ] **Step 5: Run stage progression tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py::test_layered_selection_advances_exactly_one_stage tests/test_layered_achievability.py::test_layered_selection_does_not_skip_multiple_stages -v
```

Expected: PASS.

- [ ] **Step 6: Run full layered tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add packing_env/gym_env.py tests/test_layered_achievability.py
git commit -m "Advance layered baseline stage conservatively"
```

---

### Task 5: Resolve Source EMS During Packing

**Files:**
- Modify: `packing_env/gym_env.py`
- Modify: `packing/test_utils.py`
- Modify: `tests/test_layered_achievability.py`

- [ ] **Step 1: Write failing source mapping test**

Append to `tests/test_layered_achievability.py`:

```python
def test_layered_step_updates_source_ems_for_clipped_selection(monkeypatch):
    env = make_layered_env(container_size=(300, 300, 300), layered_num_chunks=3)
    raw_ems = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(300, 300, 300))
    monkeypatch.setattr(env.heu_ems, "get_all_ems", lambda: [raw_ems])

    block = SimpleBlock(box=Orthogonal3D(100, 100, 80), stack_dims=(1, 1, 1))
    env.selected_item = block
    env.candidates = env.get_vectorized_ems(env._clip_ems_to_layer_window([raw_ems], stage=1))
    env.ems_list = env._clip_ems_to_layer_window([raw_ems], stage=1)

    captured = {}

    def fake_step(source_item, placed_item, selected_ems):
        captured["source_item"] = source_item
        captured["placed_item"] = placed_item
        captured["selected_ems"] = selected_ems

    monkeypatch.setattr(env, "_step", fake_step)
    env.step(0)

    assert captured["source_item"] is block
    assert captured["placed_item"].FLB.z == 0
    assert captured["selected_ems"] is raw_ems
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py::test_layered_step_updates_source_ems_for_clipped_selection -v
```

Expected: FAIL because `step()` passes the clipped EMS directly.

- [ ] **Step 3: Update `idx2pos` to return source EMS**

In `packing_env/gym_env.py`, change `idx2pos(...)`:

```python
        selected_ems = self.ems_list[idx]
        source_ems = self.resolve_policy_ems_source(selected_ems)
        return pos, rot, source_ems
```

Keep the variable name in callers as `selected_ems`, because the returned EMS is now the source EMS for update.

- [ ] **Step 4: Update `pack` to resolve clipped EMSs**

In `packing_env/gym_env.py`, update the beginning of `pack(...)`:

```python
        selected_ems = self.resolve_policy_ems_source(selected_ems)
```

before calling `self.heu_ems.update(...)`.

- [ ] **Step 5: Update direct validation path**

In `packing/test_utils.py`, replace:

```python
        env.pack(box, selected_ems=env.ems_list[action_idx % env.k_placement])
```

with:

```python
        selected_ems = env.resolve_policy_ems_source(
            env.ems_list[action_idx % env.k_placement]
        )
        env.pack(box, selected_ems=selected_ems)
```

- [ ] **Step 6: Run source mapping test**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py::test_layered_step_updates_source_ems_for_clipped_selection -v
```

Expected: PASS.

- [ ] **Step 7: Run layered and baseline policy tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py tests/test_cascaded_policy.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add packing_env/gym_env.py packing/test_utils.py tests/test_layered_achievability.py
git commit -m "Resolve source EMS for layered placements"
```

---

### Task 6: Wire Observation Flow And Config Defaults End-To-End

**Files:**
- Modify: `packing_env/gym_env.py`
- Modify: `configs/test_cj_default.yaml`
- Modify: `tests/test_layered_achievability.py`

- [ ] **Step 1: Write failing observation-flow test**

Append to `tests/test_layered_achievability.py`:

```python
def test_get_next_observation_uses_layered_selection(monkeypatch):
    env = make_layered_env(container_size=(300, 300, 300), layered_num_chunks=3)
    raw_ems = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(300, 300, 300))
    monkeypatch.setattr(env.heu_ems, "get_all_ems", lambda: [raw_ems])

    small = SimpleBlock(box=Orthogonal3D(100, 100, 80), stack_dims=(1, 1, 1))
    tall = SimpleBlock(box=Orthogonal3D(100, 100, 150), stack_dims=(1, 1, 1))
    env.buffer.simple_blocks = {
        small.box: [small],
        tall.box: [tall],
    }

    obs = env.get_next_observation()

    assert env.selected_item is small
    assert env.ems_list[0].Dim.dz == 100
    assert obs["placable"] if "placable" in obs else obs["action_mask"].any()
```

- [ ] **Step 2: Run observation test**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py::test_get_next_observation_uses_layered_selection -v
```

Expected: PASS after Task 3 because `get_next_observation()` reaches `_get_item_fit_ems_list(...)`, which now derives the layered policy EMS view.

- [ ] **Step 3: Ensure baseline observation reuses selected layered EMS list**

In `packing_env/gym_env.py`, in `get_next_observation()`, keep:

```python
        if self.use_simple_blocks:
            self.select_largest_policy_block()
```

Then ensure the later simple-block branch does not replace the selected EMS list with an unlayered list. It should call:

```python
        ems_list = self._get_item_fit_ems_list([item])
```

where `_get_item_fit_ems_list` uses `_get_policy_ems_list()` from Task 3.

- [ ] **Step 4: Keep layered mode opt-in in the CJ baseline test config**

Leave `layered_achievability: false` as the committed default in `configs/test_cj_default.yaml`. To run the experiment, edit locally or create a separate config with:

```yaml
layered_achievability: true
layered_num_chunks: 3
```

Do not change the default test config to true in this task, because this feature is experimental and should be opt-in.

- [ ] **Step 5: Run full relevant tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py tests/test_cascaded_policy.py tests/test_cascaded_block_candidates.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packing_env/gym_env.py configs/test_cj_default.yaml tests/test_layered_achievability.py
git commit -m "Wire layered baseline observation flow"
```

---

### Task 7: Final Verification

**Files:**
- No expected code changes unless verification exposes a defect.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py tests/test_cascaded_policy.py tests/test_cascaded_block_candidates.py -v
```

Expected: PASS.

- [ ] **Step 2: Run a smoke observation script**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python - <<'PY'
from packing_env.gym_env import PackingEnv

env = PackingEnv(
    ds_name="random",
    container_size=(600, 600, 600),
    stack_only=True,
    use_simple_blocks=True,
    policy_mode="largest_block_baseline",
    layered_achievability=True,
    layered_num_chunks=3,
)
env.reset(seed=1)
obs = env.get_next_observation()
print("stage", env.layered_stage)
print("ems", len(env.ems_list))
print("placable", bool(obs["action_mask"].any()))
PY
```

Expected: exits 0 and prints a stage in `1..3`, an EMS count, and a boolean placable value.

- [ ] **Step 3: Inspect final diff**

Run:

```bash
git status --short
git diff --stat HEAD~6..HEAD
```

Expected: clean worktree after all task commits, and changes limited to planned files.

- [ ] **Step 4: Push feature branch**

```bash
git push -u origin layered-achievability-baseline
```

Expected: branch pushed successfully.
