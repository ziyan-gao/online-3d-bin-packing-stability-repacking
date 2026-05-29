# Layered Cascaded Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `layered_achievability` to work with `policy_mode: cascaded_block_selector`.

**Architecture:** Reuse the existing policy EMS clipping and source-resolution machinery. Add a stage-aware cascaded candidate helper that tries the current stage first, optionally advances exactly one stage, and returns clipped EMSs plus oriented block masks.

**Tech Stack:** Python, Gymnasium environment code, pytest.

---

### Task 1: Allow Cascaded Layered Construction

**Files:**
- Modify: `packing_env/gym_env.py`
- Test: `tests/test_cascaded_policy.py`

- [ ] **Step 1: Update the constructor test**

In `tests/test_cascaded_policy.py`, replace the expectation that cascaded layered mode raises an error with an assertion that it initializes:

```python
def test_layered_achievability_allows_cascaded_policy():
    env = PackingEnv(
        layered_achievability=True,
        layered_num_chunks=4,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )

    assert env.layered_achievability is True
    assert env.layered_num_chunks == 4
    assert env.policy_mode == "cascaded_block_selector"
    assert env.stack_only is True
    assert env.use_simple_blocks is True
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py::test_layered_achievability_allows_cascaded_policy -v
```

Expected: fail because `PackingEnv` rejects layered cascaded mode.

- [ ] **Step 3: Relax constructor validation**

In `packing_env/gym_env.py`, change the `layered_achievability` validation to allow `largest_block_baseline` and `cascaded_block_selector`:

```python
        if self.layered_achievability:
            if self.policy_mode not in {
                "largest_block_baseline",
                "cascaded_block_selector",
            }:
                raise ValueError(
                    "layered_achievability is supported only with "
                    "policy_mode='largest_block_baseline' or "
                    "policy_mode='cascaded_block_selector'."
                )
            if not self.use_simple_blocks:
                raise ValueError(
                    "layered_achievability requires use_simple_blocks=True."
                )
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py::test_layered_achievability_allows_cascaded_policy -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packing_env/gym_env.py tests/test_cascaded_policy.py
git commit -m "Allow layered cascaded policy construction"
```

### Task 2: Add Stage-Aware Cascaded Candidate Generation

**Files:**
- Modify: `packing_env/gym_env.py`
- Test: `tests/test_layered_achievability.py`

- [ ] **Step 1: Add cascaded layered tests**

Append these tests to `tests/test_layered_achievability.py`:

```python
def make_layered_cascaded_env(**kwargs):
    return make_layered_env(policy_mode="cascaded_block_selector", **kwargs)


def test_cascaded_candidates_use_current_layer_clipped_ems(monkeypatch):
    env = make_layered_cascaded_env(container_size=(300, 300, 300), layered_num_chunks=3)
    raw_ems = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(300, 300, 300))
    monkeypatch.setattr(env.heu_ems, "get_all_ems", lambda: [raw_ems])

    small = SimpleBlock(box=Orthogonal3D(100, 100, 80), stack_dims=(1, 1, 1))
    tall = SimpleBlock(box=Orthogonal3D(100, 100, 150), stack_dims=(1, 1, 1))
    env.buffer.simple_blocks = {
        small.box: [small],
        tall.box: [tall],
    }

    oriented, ems_list, rows = env.get_cascaded_block_candidates()

    assert oriented
    assert rows.any()
    assert ems_list[0].Dim.dz == 100
    assert all(candidate.Dim.dz <= 100 for candidate in oriented)


def test_cascaded_layered_candidates_advance_exactly_one_stage(monkeypatch):
    env = make_layered_cascaded_env(container_size=(300, 300, 300), layered_num_chunks=3)
    raw_ems = EmptyMaximalSpace(Point3D(0, 0, 100), Orthogonal3D(300, 300, 100))
    monkeypatch.setattr(env.heu_ems, "get_all_ems", lambda: [raw_ems])
    env.buffer.simple_blocks = {
        Orthogonal3D(100, 100, 80): [
            SimpleBlock(box=Orthogonal3D(100, 100, 80), stack_dims=(1, 1, 1))
        ]
    }
    env.layered_stage = 1

    oriented, ems_list, rows = env.get_cascaded_block_candidates()

    assert oriented
    assert rows.any()
    assert env.layered_stage == 2
    assert ems_list[0].FLB.z == 100


def test_cascaded_layered_candidates_do_not_skip_multiple_stages(monkeypatch):
    env = make_layered_cascaded_env(
        container_size=(300, 300, 400),
        layered_num_chunks=4,
    )
    raw_ems = EmptyMaximalSpace(Point3D(0, 0, 300), Orthogonal3D(300, 300, 100))
    monkeypatch.setattr(env.heu_ems, "get_all_ems", lambda: [raw_ems])
    env.buffer.simple_blocks = {
        Orthogonal3D(100, 100, 80): [
            SimpleBlock(box=Orthogonal3D(100, 100, 80), stack_dims=(1, 1, 1))
        ]
    }
    env.layered_stage = 1

    oriented, ems_list, rows = env.get_cascaded_block_candidates()

    assert oriented == []
    assert ems_list == []
    assert not rows.any()
    assert env.layered_stage == 1
```

- [ ] **Step 2: Run the layered tests and verify the new stage tests fail**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py -v
```

Expected: at least one new cascaded stage advancement test fails.

- [ ] **Step 3: Extract a stage-specific cascaded helper**

In `packing_env/gym_env.py`, split the existing `get_cascaded_block_candidates` body into a helper:

```python
    def _get_cascaded_block_candidates_for_stage(
        self,
        stage: int,
    ) -> tuple[list[OrientedBlock], list[EmptyMaximalSpace], np.ndarray]:
        original_stage = self.layered_stage
        try:
            self.layered_stage = int(stage)
            return self._build_cascaded_block_candidates()
        finally:
            self.layered_stage = original_stage
```

Move the current body of `get_cascaded_block_candidates` into `_build_cascaded_block_candidates`.

- [ ] **Step 4: Add one-stage advancement in `get_cascaded_block_candidates`**

Use:

```python
    def get_cascaded_block_candidates(self):
        if not self.layered_achievability:
            return self._build_cascaded_block_candidates()

        original_stage = self.layered_stage
        result = self._get_cascaded_block_candidates_for_stage(original_stage)
        if result[0]:
            return result

        if original_stage < self.layered_num_chunks:
            next_stage = original_stage + 1
            next_result = self._get_cascaded_block_candidates_for_stage(next_stage)
            if next_result[0]:
                self.layered_stage = next_stage
                return next_result

        self.layered_stage = original_stage
        return result
```

- [ ] **Step 5: Run layered tests and verify they pass**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add packing_env/gym_env.py tests/test_layered_achievability.py
git commit -m "Add layered cascaded candidate advancement"
```

### Task 3: Verify Step Resolution and Configs

**Files:**
- Modify: `tests/test_layered_achievability.py`
- May keep user edits in `configs/test_cj_cascade.yaml` and `configs/test_cj_default.yaml`

- [ ] **Step 1: Add cascaded step source-resolution test**

Append this test:

```python
def test_cascaded_step_resolves_layered_policy_ems_to_raw_source(monkeypatch):
    env = make_layered_cascaded_env(container_size=(300, 300, 300), layered_num_chunks=3)
    raw_ems = EmptyMaximalSpace(Point3D(0, 0, 0), Orthogonal3D(300, 300, 300))
    monkeypatch.setattr(env.heu_ems, "get_all_ems", lambda: [raw_ems])

    block = SimpleBlock(box=Orthogonal3D(100, 100, 80), stack_dims=(1, 1, 1))
    env.buffer.simple_blocks = {block.box: [block]}
    obs = env.get_cascaded_observation()
    assert obs["action_mask"].any()

    captured = {}

    def fake_step(source_item, placed_item, selected_ems):
        captured["source_item"] = source_item
        captured["placed_item"] = placed_item
        captured["selected_ems"] = selected_ems

    monkeypatch.setattr(env, "_step", fake_step)
    action = int(np.argwhere(obs["action_mask"])[0][1])

    env.step(action)

    assert captured["selected_ems"] is raw_ems
    assert captured["source_item"] is block
```

- [ ] **Step 2: Add `numpy` import if needed**

At the top of `tests/test_layered_achievability.py`, ensure:

```python
import numpy as np
```

- [ ] **Step 3: Run focused test**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py::test_cascaded_step_resolves_layered_policy_ems_to_raw_source -v
```

Expected: pass using existing `step(...)` EMS source resolution.

- [ ] **Step 4: Run full targeted verification**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py tests/test_cascaded_policy.py tests/test_simpleblock_buffer.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_layered_achievability.py configs/test_cj_cascade.yaml configs/test_cj_default.yaml
git commit -m "Verify layered cascaded policy behavior"
```

### Task 4: Final Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Search for stale restriction text**

Run:

```bash
rg -n "layered_achievability is supported only|largest_block_baseline" packing_env tests docs/superpowers/specs/2026-05-29-layered-cascaded-policy-design.md
```

Expected: no stale constructor restriction that excludes cascaded mode.

- [ ] **Step 2: Run targeted tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_layered_achievability.py tests/test_cascaded_policy.py tests/test_simpleblock_buffer.py -v
```

Expected: pass.

- [ ] **Step 3: Check git state**

Run:

```bash
git status --short --branch
```

Expected: branch is ahead of origin with no unstaged changes unless the user wants additional config edits.
