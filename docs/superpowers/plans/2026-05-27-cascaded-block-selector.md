# Cascaded Block Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental cascaded DRL policy mode that chooses among all feasible stable oriented vertical block candidates, then chooses the EMS anchor for the selected oriented block.

**Architecture:** Keep the current largest-usable-block policy path unchanged as the baseline. Add a separate `cascaded_block_selector` mode with new observation fields, a flat external action encoding, a cascaded actor/distribution, and checkpoint metadata checks so the baseline and experimental models can be trained and evaluated side by side.

**Tech Stack:** Python 3.12, Gymnasium, NumPy, PyTorch, Tianshou PPO, Pytest.

---

## File Structure

- Create `packing_env/data_type/oriented_block.py`
  - Owns the oriented vertical block candidate data structure and conversion helpers.
- Modify `packing_env/data_type/__init__.py`
  - Exports the new oriented block candidate type.
- Modify `packing_env/gym_env.py`
  - Adds `policy_mode`, cascaded observation construction, flat action decoding, and mode-specific stepping.
- Create `model/cascaded_actor.py`
  - Adds the cascaded actor and its output object.
- Create `model/cascaded_critic.py`
  - Adds a state-value critic for cascaded observations.
- Create `model/cascaded_policy.py`
  - Adds the Tianshou-compatible factorized distribution for flat actions.
- Modify `packing/policy_loader.py`
  - Builds the correct model family by policy mode and supports deterministic cascaded action decoding.
- Modify `packing/agents.py`
  - Allows inference with the cascaded policy while preserving the current agent behavior.
- Modify `packing/train_utils.py`
  - Adds config/checkpoint support for `policy_mode`, model construction, and output metadata.
- Modify `train.py`
  - Adds CLI flag for `--policy-mode`.
- Modify `configs/train_default.yaml`, `configs/test_default.yaml`, and `configs/test_cj_default.yaml`
  - Adds explicit `policy_mode`.
- Create `tests/test_cascaded_block_candidates.py`
  - Tests oriented block generation and masks.
- Create `tests/test_cascaded_policy.py`
  - Tests cascaded actor/distribution action decoding and log-prob behavior.
- Modify `tests/test_simpleblock_buffer.py`
  - Adds regression checks that baseline largest-block mode remains unchanged.

## Task 1: Add Policy Mode Configuration Contract

**Files:**
- Modify: `packing/train_utils.py`
- Modify: `train.py`
- Modify: `configs/train_default.yaml`
- Modify: `configs/test_default.yaml`
- Modify: `configs/test_cj_default.yaml`
- Test: `tests/test_cascaded_policy.py`

- [ ] **Step 1: Write failing config tests**

Add this file:

```python
# tests/test_cascaded_policy.py
import pytest

from packing.train_utils import TrainConfig, load_training_checkpoint


def test_train_config_accepts_cascaded_policy_mode():
    config = TrainConfig(policy_mode="cascaded_block_selector")

    assert config.policy_mode == "cascaded_block_selector"


def test_train_config_rejects_unknown_policy_mode():
    with pytest.raises(ValueError, match="policy_mode"):
        TrainConfig(policy_mode="not_a_policy")


def test_checkpoint_policy_mode_mismatch_is_rejected(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch = pytest.importorskip("torch")
    torch.save(
        {
            "model": {},
            "optim": {},
            "epoch": 1,
            "env_step": 2,
            "gradient_step": 3,
            "data_name": "random",
            "container_size": (600, 600, 600),
            "buffer_size": 12,
            "k_placement": 80,
            "remove_inscribed_ems": False,
            "stack_only": True,
            "use_simple_blocks": True,
            "policy_mode": "largest_block_baseline",
        },
        checkpoint_path,
    )

    class DummyPolicy:
        def load_state_dict(self, state):
            pass

    class DummyOptim:
        def load_state_dict(self, state):
            pass

    config = TrainConfig(
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    with pytest.raises(ValueError, match="policy_mode"):
        load_training_checkpoint(
            str(checkpoint_path),
            DummyPolicy(),
            DummyOptim(),
            config,
            "cpu",
        )
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py -v
```

Expected: FAIL because `TrainConfig` does not accept `policy_mode`.

- [ ] **Step 3: Add policy mode field and validation**

Modify `packing/train_utils.py`:

```python
VALID_POLICY_MODES = {"largest_block_baseline", "cascaded_block_selector"}


@dataclass(frozen=True)
class TrainConfig:
    data_name: str = "random"
    buffer_size: int = 12
    container_dx: int = 600
    container_dy: int = 600
    container_dz: int = 600
    k_placement: int = 80
    remove_inscribed_ems: bool = False
    stack_only: bool = False
    use_simple_blocks: bool = False
    policy_mode: str = "largest_block_baseline"
    train_env_num: int = 64
    test_env_num: int = 32
    train_env_seed: int = 5
    max_epoch: int = 1000
    step_per_epoch: int = 800 * 40
    step_per_collect: int = 4000
    episode_per_test: int = 128
    batch_size: int = 256
    learning_rate: float = 7e-5
    output_root: str = os.path.join(PROJECT_ROOT, "outputs", "train_outputs")
    output_name: str | None = None
    tb_log_dir: str | None = None
    resume_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.policy_mode not in VALID_POLICY_MODES:
            raise ValueError(
                f"policy_mode must be one of {sorted(VALID_POLICY_MODES)}, "
                f"got {self.policy_mode!r}"
            )
        if self.policy_mode == "cascaded_block_selector" and not self.use_simple_blocks:
            object.__setattr__(self, "use_simple_blocks", True)
        if self.policy_mode == "cascaded_block_selector" and not self.stack_only:
            object.__setattr__(self, "stack_only", True)
```

Update checkpoint matching in `load_training_checkpoint()`:

```python
expected = {
    "data_name": config.data_name,
    "container_size": (config.container_dx, config.container_dy, config.container_dz),
    "buffer_size": config.buffer_size,
    "k_placement": config.k_placement,
    "remove_inscribed_ems": config.remove_inscribed_ems,
    "stack_only": config.stack_only,
    "use_simple_blocks": config.use_simple_blocks,
    "policy_mode": config.policy_mode,
}
checkpoint_defaults = {
    "stack_only": False,
    "use_simple_blocks": False,
    "policy_mode": "largest_block_baseline",
}
```

Update checkpoint save payload in `make_training_callbacks()`:

```python
"stack_only": config.stack_only,
"use_simple_blocks": config.use_simple_blocks,
"policy_mode": config.policy_mode,
```

- [ ] **Step 4: Add CLI and config YAML fields**

Modify `train.py`:

```python
parser.add_argument(
    "--policy-mode",
    choices=("largest_block_baseline", "cascaded_block_selector"),
    help="Policy architecture to train.",
)
```

Add to the overrides map:

```python
"policy_mode": args.policy_mode,
```

Add to each config file:

```yaml
policy_mode: largest_block_baseline
```

For `configs/test_cj_default.yaml`, keep `stack_only: true` and `use_simple_blocks: true`; still set:

```yaml
policy_mode: largest_block_baseline
```

- [ ] **Step 5: Run tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packing/train_utils.py train.py configs/train_default.yaml configs/test_default.yaml configs/test_cj_default.yaml tests/test_cascaded_policy.py
git commit -m "Add policy mode configuration"
```

## Task 2: Add Oriented Block Candidate Type

**Files:**
- Create: `packing_env/data_type/oriented_block.py`
- Modify: `packing_env/data_type/__init__.py`
- Test: `tests/test_cascaded_block_candidates.py`

- [ ] **Step 1: Write failing oriented block tests**

Create `tests/test_cascaded_block_candidates.py`:

```python
import numpy as np

from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item, SimpleBlock
from packing_env.data_type.oriented_block import OrientedBlock


def test_oriented_block_preserves_normal_orientation():
    block = SimpleBlock(
        box=Orthogonal3D(100, 200, 50),
        stack_dims=(1, 1, 3),
        buffer_space=10,
    )

    oriented = OrientedBlock.from_block(block, source_index=2, rotate_xy=False)

    assert oriented.source_index == 2
    assert oriented.rotate_xy is False
    assert oriented.Dim.raw().tolist() == [100, 200, 150]
    assert oriented.Virtual_Dim.raw().tolist() == [110, 210, 150]
    assert oriented.consumed_count == 3
    assert oriented.feature_row(container_size=(600, 600, 600)).shape == (8,)


def test_oriented_block_applies_transpose_once_to_item():
    block = SimpleBlock(
        box=Orthogonal3D(100, 200, 50),
        stack_dims=(1, 1, 2),
        buffer_space=10,
    )

    oriented = OrientedBlock.from_block(block, source_index=0, rotate_xy=True)
    placed = oriented.to_item(Point3D(1, 2, 3))

    assert isinstance(placed, Item)
    assert placed.Dim.raw().tolist() == [200, 100, 100]
    assert placed.Virtual_Dim.raw().tolist() == [210, 110, 100]
    assert placed.rot is True
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_block_candidates.py -v
```

Expected: FAIL because `packing_env.data_type.oriented_block` does not exist.

- [ ] **Step 3: Implement oriented block type**

Create `packing_env/data_type/oriented_block.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Orthogonal3D, Point3D
from .item import Item, SimpleBlock


@dataclass(frozen=True)
class OrientedBlock:
    block: SimpleBlock
    source_index: int
    rotate_xy: bool

    @classmethod
    def from_block(
        cls,
        block: SimpleBlock,
        source_index: int,
        rotate_xy: bool,
    ) -> "OrientedBlock":
        return cls(block=block, source_index=int(source_index), rotate_xy=bool(rotate_xy))

    @property
    def oriented_block(self) -> SimpleBlock:
        return self.block.rotated(self.rotate_xy)

    @property
    def Dim(self) -> Orthogonal3D:
        return self.oriented_block.Dim

    @property
    def Virtual_Dim(self) -> Orthogonal3D:
        return self.oriented_block.Virtual_Dim

    @property
    def consumed_count(self) -> int:
        return self.block.consumed_count

    @property
    def buffer_space(self) -> int:
        return self.block.buffer_space

    def to_item(self, flb: Point3D) -> Item:
        placed = Item(
            FLB=flb,
            Dim=self.Dim,
            buffer_space=self.buffer_space,
        )
        placed.rot = self.rotate_xy
        return placed

    def feature_row(self, container_size: tuple[int, int, int]) -> np.ndarray:
        container = np.asarray(container_size, dtype=np.float32)
        dim = self.Dim.raw().astype(np.float32)
        virtual_dim = self.Virtual_Dim.raw().astype(np.float32)
        return np.array(
            [
                dim[0] / container[0],
                dim[1] / container[1],
                dim[2] / container[2],
                virtual_dim[0] / container[0],
                virtual_dim[1] / container[1],
                virtual_dim[2] / container[2],
                float(self.consumed_count) / 12.0,
                float(self.rotate_xy),
            ],
            dtype=np.float32,
        )
```

Modify `packing_env/data_type/__init__.py`:

```python
from .oriented_block import OrientedBlock
```

Add `"OrientedBlock"` to `__all__`.

- [ ] **Step 4: Run tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_block_candidates.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packing_env/data_type/oriented_block.py packing_env/data_type/__init__.py tests/test_cascaded_block_candidates.py
git commit -m "Add oriented block candidate type"
```

## Task 3: Generate Feasible Stable Oriented Blocks

**Files:**
- Modify: `packing_env/gym_env.py`
- Test: `tests/test_cascaded_block_candidates.py`

- [ ] **Step 1: Add failing tests for feasible oriented candidates**

Append to `tests/test_cascaded_block_candidates.py`:

```python
from packing_env.data_type.buffer import Buffer
from packing_env.gym_env import PackingEnv


class FakeSampler:
    is_random_distribution = False

    def __init__(self, items):
        self.items = list(items)
        self.cursor = 0

    def sample(self, n):
        sampled = []
        for _ in range(n):
            sampled.append(self.items[self.cursor % len(self.items)])
            self.cursor += 1
        return sampled


def test_cascaded_candidates_expose_only_stable_oriented_blocks():
    box = Orthogonal3D(300, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=2,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.buffer = Buffer(
        capacity=2,
        data_sampler=FakeSampler([box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    env.reset(seed=1)

    candidates, ems_list, loading_mask = env.get_cascaded_block_candidates()

    assert len(candidates) > 0
    assert loading_mask.shape == (len(candidates), env.k_placement)
    assert loading_mask.any(axis=1).all()
    assert all(candidate.consumed_count in (1, 2) for candidate in candidates)
```

- [ ] **Step 2: Run failing test**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_block_candidates.py::test_cascaded_candidates_expose_only_stable_oriented_blocks -v
```

Expected: FAIL because `policy_mode` and `get_cascaded_block_candidates()` do not exist.

- [ ] **Step 3: Add environment policy mode initialization**

Modify `packing_env/gym_env.py` constructor signature:

```python
policy_mode: str = "largest_block_baseline",
```

Store and validate:

```python
self.policy_mode = str(policy_mode)
if self.policy_mode not in {"largest_block_baseline", "cascaded_block_selector"}:
    raise ValueError(f"Unknown policy_mode: {self.policy_mode!r}")
if self.policy_mode == "cascaded_block_selector":
    self.stack_only = True
    self.use_simple_blocks = True
```

Pass `policy_mode` through all `gym.make(...)` calls in `packing/train_utils.py` after Task 6 adds it to env creation. For this task, direct construction is enough for the failing test.

- [ ] **Step 4: Add cascaded candidate builder**

Add imports to `packing_env/gym_env.py`:

```python
from .data_type.oriented_block import OrientedBlock
```

Add this method:

```python
def get_cascaded_block_candidates(self):
    oriented_candidates = []
    mask_rows = []
    ems_list = self._get_item_fit_ems_list(self.buffer.all_blocks, prefer_stable=True)

    if not ems_list:
        return oriented_candidates, ems_list, np.zeros((0, self.k_placement), dtype=bool)

    lps = np.array([ems.FLB.topix() for ems in ems_list])
    for source_index, block in enumerate(self.buffer.all_blocks):
        for rotate_xy in (False, True):
            oriented = OrientedBlock.from_block(block, source_index, rotate_xy)
            fit_mask = np.array(
                [ems.include(oriented.Virtual_Dim) for ems in ems_list],
                dtype=bool,
            )
            if not fit_mask.any():
                continue
            stable_placements, is_stable = self.heu_stable(
                o3d=oriented.Dim,
                hm=self.hm,
                candidates=lps[fit_mask],
            )
            row = np.zeros((self.k_placement,), dtype=bool)
            fit_indices = np.flatnonzero(fit_mask)
            for local_index, ems_index in enumerate(fit_indices):
                stable = bool(is_stable[local_index])
                stable_placement = stable_placements[local_index]
                if (
                    stable
                    and stable_placement is not None
                    and stable_placement == ems_list[int(ems_index)].FLB
                ):
                    row[int(ems_index)] = True
            if row.any():
                oriented_candidates.append(oriented)
                mask_rows.append(row)

    if not mask_rows:
        return oriented_candidates, ems_list, np.zeros((0, self.k_placement), dtype=bool)
    return oriented_candidates, ems_list, np.asarray(mask_rows, dtype=bool)
```

- [ ] **Step 5: Run tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_block_candidates.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packing_env/gym_env.py tests/test_cascaded_block_candidates.py
git commit -m "Expose feasible oriented block candidates"
```

## Task 4: Build Cascaded Observation And Decode Flat Actions

**Files:**
- Modify: `packing_env/gym_env.py`
- Test: `tests/test_cascaded_block_candidates.py`

- [ ] **Step 1: Add failing observation/action tests**

Append:

```python
def test_cascaded_observation_contains_blocks_and_loading_mask():
    box = Orthogonal3D(100, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=3,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.buffer = Buffer(
        capacity=3,
        data_sampler=FakeSampler([box, box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    env.reset(seed=1)

    obs = env.get_next_observation()

    assert obs["oriented_blocks"].shape == (env.max_oriented_blocks, 8)
    assert obs["block_mask"].shape == (env.max_oriented_blocks,)
    assert obs["loading_mask"].shape == (env.max_oriented_blocks, env.k_placement)
    assert obs["action_mask"].shape == (env.max_oriented_blocks, env.k_placement)
    assert obs["block_mask"].any()
    assert obs["loading_mask"][obs["block_mask"]].any(axis=1).all()


def test_cascaded_action_decoding_applies_orientation_once():
    box = Orthogonal3D(100, 200, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=2,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.buffer = Buffer(
        capacity=2,
        data_sampler=FakeSampler([box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    env.reset(seed=1)
    obs = env.get_next_observation()

    valid = np.argwhere(obs["action_mask"])
    oriented_index, ems_index = map(int, valid[0])
    _, reward, _, _, _ = env.step(oriented_index * env.k_placement + ems_index)

    assert reward > 0
    assert len(env.container.placed_items) == 1
    placed = env.container.placed_items[0]
    expected_dim = env.last_placed_source.Dim.raw().tolist()
    assert placed.Dim.raw().tolist() == expected_dim
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_block_candidates.py::test_cascaded_observation_contains_blocks_and_loading_mask tests/test_cascaded_block_candidates.py::test_cascaded_action_decoding_applies_orientation_once -v
```

Expected: FAIL because cascaded observation fields and `last_placed_source` do not exist.

- [ ] **Step 3: Add cascaded observation space**

In `PackingEnv.__init__`, add:

```python
self.max_oriented_blocks = 2 * self.buffer.capacity
```

In `_set_space()`, when `self.policy_mode == "cascaded_block_selector"`, use:

```python
self.action_space = spaces.Discrete(self.max_oriented_blocks * self.k_placement)
self.observation_space = spaces.Dict(
    {
        "oriented_blocks": spaces.Box(
            low=0,
            high=1,
            shape=(self.max_oriented_blocks, 8),
            dtype=np.float32,
        ),
        "block_mask": spaces.MultiBinary((self.max_oriented_blocks,)),
        "ems": spaces.Box(
            low=0,
            high=1,
            shape=(self.k_placement, 6),
            dtype=np.float32,
        ),
        "loading_mask": spaces.MultiBinary((self.max_oriented_blocks, self.k_placement)),
        "action_mask": spaces.MultiBinary((self.max_oriented_blocks, self.k_placement)),
    }
)
return
```

- [ ] **Step 4: Add cascaded observation builder**

Add:

```python
def get_cascaded_observation(self):
    candidates, ems_list, loading_mask = self.get_cascaded_block_candidates()
    self.oriented_block_candidates = candidates
    self.ems_list = ems_list

    block_features = np.zeros((self.max_oriented_blocks, 8), dtype=np.float32)
    block_mask = np.zeros((self.max_oriented_blocks,), dtype=bool)
    action_mask = np.zeros((self.max_oriented_blocks, self.k_placement), dtype=bool)

    for idx, candidate in enumerate(candidates[: self.max_oriented_blocks]):
        block_features[idx] = candidate.feature_row(
            (self.container.dx, self.container.dy, self.container.dz)
        )
        block_mask[idx] = True
        action_mask[idx] = loading_mask[idx]

    vectorized_ems = self.get_vectorized_ems(ems_list)
    ems_normalized = vectorized_ems.astype(np.float32)
    ems_normalized[:, :3] = ems_normalized[:, :3] / self.bin_size
    ems_normalized[:, 3:] = ems_normalized[:, 3:] / self.bin_size
    self.candidates = vectorized_ems.copy()
    self.done = not action_mask.any()

    return {
        "oriented_blocks": block_features,
        "block_mask": block_mask,
        "ems": ems_normalized,
        "loading_mask": action_mask.copy(),
        "action_mask": action_mask,
        "placable": bool(action_mask.any()),
    }
```

Change `get_next_observation()`:

```python
if self.policy_mode == "cascaded_block_selector":
    self.current_obs = self.get_cascaded_observation()
    return self.current_obs
```

- [ ] **Step 5: Add cascaded action decoding and step path**

Add:

```python
def decode_cascaded_action(self, action: int):
    oriented_index = int(action) // self.k_placement
    ems_index = int(action) % self.k_placement
    if oriented_index >= len(self.oriented_block_candidates):
        raise ValueError(f"Invalid oriented block index: {oriented_index}")
    if ems_index >= len(self.ems_list):
        raise ValueError(f"Invalid EMS index: {ems_index}")
    if not self.current_obs["action_mask"][oriented_index, ems_index]:
        raise ValueError(
            f"Invalid cascaded action: oriented_block={oriented_index}, ems={ems_index}"
        )
    return (
        self.oriented_block_candidates[oriented_index],
        self.ems_list[ems_index],
    )


def step_cascaded(self, action: int) -> tuple:
    oriented, selected_ems = self.decode_cascaded_action(action)
    placed_item = oriented.to_item(flb=selected_ems.FLB)
    self.last_placed_source = oriented
    self.img_counter += 1
    self._step(oriented.block, placed_item, selected_ems)
    reward = placed_item.Dim.Volume / self.container.Volume
    obs_next = self.get_next_observation()
    info = {
        "packed_objects": len(self.container.placed_items),
        "utilization_ratio": self.container.utilization,
        "selected_stack_height": oriented.consumed_count,
    }
    return obs_next, reward, self.done, False, info
```

At the top of `step()`:

```python
if self.policy_mode == "cascaded_block_selector":
    return self.step_cascaded(action)
```

- [ ] **Step 6: Run tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_block_candidates.py tests/test_simpleblock_buffer.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add packing_env/gym_env.py tests/test_cascaded_block_candidates.py
git commit -m "Add cascaded block observation and action decoding"
```

## Task 5: Add Cascaded Actor, Critic, And Distribution

**Files:**
- Create: `model/cascaded_actor.py`
- Create: `model/cascaded_critic.py`
- Create: `model/cascaded_policy.py`
- Test: `tests/test_cascaded_policy.py`

- [ ] **Step 1: Add failing model/distribution tests**

Append to `tests/test_cascaded_policy.py`:

```python
from types import SimpleNamespace

import torch

from model.cascaded_actor import CascadedActor
from model.cascaded_critic import CascadedCritic
from model.cascaded_policy import CascadedCategoricalMasked


def make_cascaded_obs():
    return SimpleNamespace(
        oriented_blocks=torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3, 0.11, 0.21, 0.3, 0.25, 0.0],
                    [0.2, 0.1, 0.3, 0.21, 0.11, 0.3, 0.25, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
        ems=torch.zeros((1, 4, 6), dtype=torch.float32),
        block_mask=torch.tensor([[True, True, False, False]]),
        loading_mask=torch.tensor(
            [[[True, False, False, False], [False, True, False, False], [False, False, False, False], [False, False, False, False]]]
        ),
        action_mask=torch.tensor(
            [[[True, False, False, False], [False, True, False, False], [False, False, False, False], [False, False, False, False]]]
        ),
    )


def test_cascaded_actor_outputs_flat_logits_and_masks():
    actor = CascadedActor(block_feature_dim=8, ems_feature_dim=6, embed_size=32, device=torch.device("cpu"))
    obs = make_cascaded_obs()

    out, _ = actor(obs)

    assert out.logits.shape == (1, 16)
    assert out.action_mask.shape == (1, 4, 4)
    assert out.block_logits.shape == (1, 4)


def test_cascaded_distribution_masks_invalid_flat_actions():
    actor = CascadedActor(block_feature_dim=8, ems_feature_dim=6, embed_size=32, device=torch.device("cpu"))
    out, _ = actor(make_cascaded_obs())
    dist = CascadedCategoricalMasked(out)

    assert dist.probs.shape == (1, 16)
    assert torch.isclose(dist.probs[0, 0] + dist.probs[0, 5], torch.tensor(1.0), atol=1e-5)
    assert torch.all(dist.probs[0, [1, 2, 3, 4, 6, 7]] == 0)


def test_cascaded_critic_returns_batch_value():
    critic = CascadedCritic(block_feature_dim=8, ems_feature_dim=6, embed_size=32, device=torch.device("cpu"))

    value = critic(make_cascaded_obs())

    assert value.shape == (1,)
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py::test_cascaded_actor_outputs_flat_logits_and_masks tests/test_cascaded_policy.py::test_cascaded_distribution_masks_invalid_flat_actions tests/test_cascaded_policy.py::test_cascaded_critic_returns_batch_value -v
```

Expected: FAIL because the new model modules do not exist.

- [ ] **Step 3: Implement cascaded actor**

Create `model/cascaded_actor.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class CascadedActorOutput:
    logits: torch.Tensor
    action_mask: torch.Tensor
    block_logits: torch.Tensor
    ems_logits: torch.Tensor
    block_mask: torch.Tensor
    loading_mask: torch.Tensor


class CascadedActor(nn.Module):
    def __init__(
        self,
        block_feature_dim: int = 8,
        ems_feature_dim: int = 6,
        embed_size: int = 128,
        device: torch.device | str = torch.device("cpu"),
        dtype=torch.float32,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.block_encoder = nn.Sequential(
            nn.Linear(block_feature_dim, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
        )
        self.ems_encoder = nn.Sequential(
            nn.Linear(ems_feature_dim, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
        )
        self.block_head = nn.Linear(embed_size, 1)
        self.loading_head = nn.Sequential(
            nn.Linear(embed_size * 3, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, 1),
        )
        self.to(self.device)

    @property
    def num_param(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, obs: Any, state: Any = None, info={}):
        blocks = torch.as_tensor(obs.oriented_blocks, dtype=self.dtype, device=self.device)
        ems = torch.as_tensor(obs.ems, dtype=self.dtype, device=self.device)
        block_mask = torch.as_tensor(obs.block_mask, dtype=torch.bool, device=self.device)
        loading_mask = torch.as_tensor(obs.loading_mask, dtype=torch.bool, device=self.device)

        block_embed = self.block_encoder(blocks)
        ems_embed = self.ems_encoder(ems)
        block_logits = self.block_head(block_embed).squeeze(-1)

        block_context = (
            block_embed * block_mask.unsqueeze(-1).float()
        ).sum(dim=1, keepdim=True) / block_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        block_context = block_context.expand(-1, block_embed.shape[1], -1)

        b = block_embed.shape[1]
        e = ems_embed.shape[1]
        block_pair = block_embed[:, :, None, :].expand(-1, b, e, -1)
        ems_pair = ems_embed[:, None, :, :].expand(-1, b, e, -1)
        context_pair = block_context[:, :, None, :].expand(-1, b, e, -1)
        pair_features = torch.cat([block_pair, ems_pair, context_pair], dim=-1)
        ems_logits = self.loading_head(pair_features).squeeze(-1)

        joint_logits = block_logits[:, :, None] + ems_logits
        action_mask = loading_mask & block_mask[:, :, None]
        flat_logits = joint_logits.reshape(joint_logits.shape[0], -1)
        return (
            CascadedActorOutput(
                logits=flat_logits,
                action_mask=action_mask,
                block_logits=block_logits,
                ems_logits=ems_logits,
                block_mask=block_mask,
                loading_mask=loading_mask,
            ),
            state,
        )
```

- [ ] **Step 4: Implement cascaded critic**

Create `model/cascaded_critic.py`:

```python
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CascadedCritic(nn.Module):
    def __init__(
        self,
        block_feature_dim: int = 8,
        ems_feature_dim: int = 6,
        embed_size: int = 128,
        device: torch.device | str = torch.device("cpu"),
        dtype=torch.float32,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.block_encoder = nn.Sequential(
            nn.Linear(block_feature_dim, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
        )
        self.ems_encoder = nn.Sequential(
            nn.Linear(ems_feature_dim, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, 1),
        )
        self.to(self.device)

    @property
    def num_param(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, obs: Any, state: Any = None, info={}) -> torch.Tensor:
        blocks = torch.as_tensor(obs.oriented_blocks, dtype=self.dtype, device=self.device)
        ems = torch.as_tensor(obs.ems, dtype=self.dtype, device=self.device)
        block_mask = torch.as_tensor(obs.block_mask, dtype=torch.bool, device=self.device)
        loading_mask = torch.as_tensor(obs.loading_mask, dtype=torch.bool, device=self.device)
        ems_mask = loading_mask.any(dim=1)

        block_embed = self.block_encoder(blocks)
        ems_embed = self.ems_encoder(ems)
        block_pool = (
            block_embed * block_mask.unsqueeze(-1).float()
        ).sum(dim=1) / block_mask.sum(dim=1, keepdim=True).clamp(min=1)
        ems_pool = (
            ems_embed * ems_mask.unsqueeze(-1).float()
        ).sum(dim=1) / ems_mask.sum(dim=1, keepdim=True).clamp(min=1)
        value = self.value_head(torch.cat([block_pool, ems_pool], dim=-1)).squeeze(-1)
        return value
```

- [ ] **Step 5: Implement cascaded distribution**

Create `model/cascaded_policy.py`:

```python
from __future__ import annotations

import torch


class CascadedCategoricalMasked(torch.distributions.Categorical):
    def __init__(self, actor_output):
        self.device = actor_output.logits.device
        self.masks = actor_output.action_mask.to(self.device).bool()
        self.batch_size = actor_output.logits.shape[0]
        flat_mask = self.masks.reshape(self.batch_size, -1)
        logits = actor_output.logits.clone().float()
        logits = logits.masked_fill(~flat_mask, -torch.inf)
        all_masked = ~flat_mask.any(dim=1)
        if all_masked.any():
            logits[all_masked, 0] = 0.0
            flat_mask[all_masked, 0] = True
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs * flat_mask.float()
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
        super().__init__(probs=probs, validate_args=False)
        self.reshaped_masks = flat_mask

    def entropy(self):
        p_log_p = self.probs * torch.log(self.probs.clamp(min=1e-10))
        p_log_p = torch.where(
            self.reshaped_masks,
            p_log_p,
            torch.tensor(0.0, device=self.device),
        )
        return -p_log_p.sum(-1)
```

- [ ] **Step 6: Run tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add model/cascaded_actor.py model/cascaded_critic.py model/cascaded_policy.py tests/test_cascaded_policy.py
git commit -m "Add cascaded actor critic policy"
```

## Task 6: Wire Training Model Construction And Environment Mode

**Files:**
- Modify: `packing/policy_loader.py`
- Modify: `packing/train_utils.py`
- Test: `tests/test_cascaded_policy.py`

- [ ] **Step 1: Add failing construction tests**

Append:

```python
from packing.policy_loader import build_net
from model.cascaded_actor import CascadedActor
from model.cascaded_critic import CascadedCritic


def test_build_net_returns_cascaded_models_for_cascaded_mode():
    actor, critic = build_net(device="cpu", policy_mode="cascaded_block_selector")

    assert isinstance(actor, CascadedActor)
    assert isinstance(critic, CascadedCritic)
```

- [ ] **Step 2: Run failing test**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py::test_build_net_returns_cascaded_models_for_cascaded_mode -v
```

Expected: FAIL because `build_net()` has no `policy_mode` parameter.

- [ ] **Step 3: Update model factory**

Modify `packing/policy_loader.py`:

```python
def build_net(device: str = "cuda", policy_mode: str = "largest_block_baseline"):
    runtime_device = torch.device(device)
    if policy_mode == "cascaded_block_selector":
        from model.cascaded_actor import CascadedActor
        from model.cascaded_critic import CascadedCritic

        return (
            CascadedActor(device=runtime_device),
            CascadedCritic(device=runtime_device),
        )
    if policy_mode != "largest_block_baseline":
        raise ValueError(f"Unknown policy_mode: {policy_mode!r}")

    from model.actor import Actor
    from model.critic import Critic
    from model.packing_transformer import PackingTransformer
    from model.space_embed import SpaceEmbed

    space_embed = SpaceEmbed(embed_dim=128)
    pack_transform = PackingTransformer(
        embed_dim=128,
        ffn_expansion_factor=2,
        num_heads=4,
        num_layers=3,
    )
    actor = Actor(space_embed, pack_transform, embed_size=128, device=runtime_device)
    critic = Critic(space_embed, pack_transform, embed_size=128, device=runtime_device)
    return actor, critic
```

Update `build_policy()`:

```python
def build_policy(
    checkpoint_path: str,
    device: str | None = None,
    k_placement: int = 80,
    policy_mode: str = "largest_block_baseline",
):
    runtime_device = resolve_runtime_device(device)
    actor, critic = build_net(device=str(runtime_device), policy_mode=policy_mode)
    load_policy_weights(actor, critic, checkpoint_path, runtime_device)
    return actor, critic
```

- [ ] **Step 4: Update training policy builder**

Modify `packing/train_utils.py`:

```python
from packing.policy_loader import CategoricalMasked, build_net
from model.cascaded_policy import CascadedCategoricalMasked

actor, critic = build_net(device=device, policy_mode=config.policy_mode)
dist_fn = (
    CascadedCategoricalMasked
    if config.policy_mode == "cascaded_block_selector"
    else CategoricalMasked
)
```

Use `dist_fn=dist_fn` in `PPOPolicy(...)`.

- [ ] **Step 5: Pass policy mode into environments**

In every `gym.make("OnlinePack-v2", ...)` call in `packing/train_utils.py`, add:

```python
policy_mode=config.policy_mode,
```

- [ ] **Step 6: Run tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py tests/test_cascaded_block_candidates.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add packing/policy_loader.py packing/train_utils.py tests/test_cascaded_policy.py
git commit -m "Wire cascaded model construction"
```

## Task 7: Add Cascaded Inference Agent Path

**Files:**
- Modify: `packing/agents.py`
- Modify: `packing/test_utils.py`
- Test: `tests/test_cascaded_policy.py`

- [ ] **Step 1: Add failing agent decoding test**

Append:

```python
import numpy as np

from packing_env.data_type.geometry import Orthogonal3D
from packing_env.data_type.buffer import Buffer
from packing_env.gym_env import PackingEnv


class LocalFakeSampler:
    is_random_distribution = False

    def __init__(self, items):
        self.items = list(items)
        self.cursor = 0

    def sample(self, n):
        sampled = []
        for _ in range(n):
            sampled.append(self.items[self.cursor % len(self.items)])
            self.cursor += 1
        return sampled


def test_cascaded_env_step_can_use_flat_policy_action():
    box = Orthogonal3D(100, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=3,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.buffer = Buffer(
        capacity=3,
        data_sampler=LocalFakeSampler([box, box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    env.reset(seed=1)
    obs = env.get_next_observation()
    action = int(np.flatnonzero(obs["action_mask"].reshape(-1))[0])

    _, reward, _, _, info = env.step(action)

    assert reward > 0
    assert info["selected_stack_height"] >= 1
```

- [ ] **Step 2: Run test**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py::test_cascaded_env_step_can_use_flat_policy_action -v
```

Expected: PASS if Task 4 is complete. Keep this test as regression coverage.

- [ ] **Step 3: Update `PackingAgent` constructor**

Modify `packing/agents.py`:

```python
def __init__(
    self,
    device: str | None = None,
    k_placement: int = 80,
    checkpoint_path: str | None = None,
    policy_mode: str = "largest_block_baseline",
):
    self.device = resolve_runtime_device(device)
    self.policy_mode = policy_mode
    self.actor, self.critic = build_net(device=str(self.device), policy_mode=policy_mode)
    self.k_placement = k_placement
    ...
```

- [ ] **Step 4: Add cascaded predict branch**

In `PackingAgent.predict()`:

```python
if self.policy_mode == "cascaded_block_selector":
    return self.predict_cascaded(obs, logits_deterministic=logits_deterministic)
```

Add:

```python
def predict_cascaded(self, obs, logits_deterministic=True):
    data = SimpleNamespace(
        oriented_blocks=obs["oriented_blocks"],
        ems=obs["ems"],
        block_mask=obs["block_mask"],
        loading_mask=obs["loading_mask"],
        action_mask=obs["action_mask"],
    )
    with torch.no_grad():
        act_out, _ = self.actor(data)
        _ = self.critic(data).detach().cpu().numpy().reshape(-1)

    mask = torch.as_tensor(act_out.action_mask, dtype=torch.bool, device=self.device)
    logits = act_out.logits.clone()
    flat_mask = mask.reshape(logits.shape[0], -1)
    logits = logits.masked_fill(~flat_mask, -torch.inf)
    if logits_deterministic:
        action = int(logits.argmax(dim=-1)[0].detach().cpu().item())
    else:
        from model.cascaded_policy import CascadedCategoricalMasked

        action = int(CascadedCategoricalMasked(act_out).sample()[0].detach().cpu().item())
    return action
```

- [ ] **Step 5: Update test packing loop for cascaded mode**

Modify `packing/test_utils.py` in `pack_until_blocked()`:

```python
if getattr(config, "policy_mode", "largest_block_baseline") == "cascaded_block_selector":
    obs = env.get_next_observation()
    if not obs["placable"]:
        ...
    action = agent.predict(obs)
    selected_block, selected_ems = env.decode_cascaded_action(action)
    buffer_before = env.buffer.dims()
    visualizer.push(env, pack_title)
    box = selected_block.to_item(selected_ems.FLB)
    annotate_source_item_count(box, selected_block.block)
    env.step(action)
    pack_history.append(record_pack_step(box, selected_block.block, buffer_before, env))
    ...
```

Use the same blocked/target-reached print logic as the existing branch.

- [ ] **Step 6: Run focused tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py tests/test_simpleblock_buffer.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add packing/agents.py packing/test_utils.py tests/test_cascaded_policy.py
git commit -m "Add cascaded inference path"
```

## Task 8: Add Config, Checkpoint, And CLI Completion

**Files:**
- Modify: `packing/test_utils.py`
- Modify: `test.py`
- Modify: `configs/train_default.yaml`
- Modify: `configs/test_default.yaml`
- Modify: `configs/test_cj_default.yaml`
- Test: `tests/test_cascaded_policy.py`

- [ ] **Step 1: Add policy mode to test config**

Modify `packing/test_utils.py`:

```python
policy_mode: str = "largest_block_baseline"
```

In `build_env()` pass:

```python
policy_mode=config.policy_mode,
```

In `build_agent()` pass:

```python
agent = PackingAgent(
    device=config.device,
    checkpoint_path=config.checkpoint,
    policy_mode=config.policy_mode,
)
```

- [ ] **Step 2: Add test CLI flag**

Modify `test.py`:

```python
parser.add_argument(
    "--policy-mode",
    choices=("largest_block_baseline", "cascaded_block_selector"),
)
```

Add to overrides:

```python
"policy_mode": args.policy_mode,
```

- [ ] **Step 3: Add YAML fields**

Add to all test/train YAML files:

```yaml
policy_mode: largest_block_baseline
```

- [ ] **Step 4: Add checkpoint save metadata**

Ensure `make_training_callbacks()` saves:

```python
"stack_only": config.stack_only,
"use_simple_blocks": config.use_simple_blocks,
"policy_mode": config.policy_mode,
```

- [ ] **Step 5: Add smoke test for config loading**

Append to `tests/test_cascaded_policy.py`:

```python
from packing.test_utils import TestConfig


def test_test_config_accepts_policy_mode():
    config = TestConfig(policy_mode="cascaded_block_selector")

    assert config.policy_mode == "cascaded_block_selector"
```

- [ ] **Step 6: Run tests**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_cascaded_policy.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add packing/test_utils.py test.py configs/train_default.yaml configs/test_default.yaml configs/test_cj_default.yaml tests/test_cascaded_policy.py
git commit -m "Complete cascaded policy configuration"
```

## Task 9: End-To-End Regression And Baseline Preservation

**Files:**
- Modify: `tests/test_simpleblock_buffer.py`
- Test: `tests/test_simpleblock_buffer.py`
- Test: `tests/test_cascaded_block_candidates.py`
- Test: `tests/test_cascaded_policy.py`

- [ ] **Step 1: Add baseline unchanged regression**

Append to `tests/test_simpleblock_buffer.py`:

```python
def test_largest_block_baseline_policy_mode_keeps_single_candidate_for_agent():
    box = Orthogonal3D(100, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=3,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="largest_block_baseline",
    )
    env.buffer = Buffer(
        capacity=3,
        data_sampler=FakeSampler([box, box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    env.reset(seed=1)
    env.select_largest_policy_block()
    obs = env.get_pack_data(env.buffer.sample_blocks(deterministic=True))

    assert obs["item_raw"].shape[0] == 1
    assert obs["item_raw"][0].tolist() == [100.0, 100.0, 150.0]
```

- [ ] **Step 2: Run focused regressions**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_simpleblock_buffer.py tests/test_cascaded_block_candidates.py tests/test_cascaded_policy.py -v
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

Expected: PASS with the existing `PytestCollectionWarning` for `TestConfig`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_simpleblock_buffer.py
git commit -m "Protect largest block baseline behavior"
```

## Task 10: Documentation And Experiment Command Notes

**Files:**
- Modify: `docs/superpowers/specs/2026-05-27-cascaded-block-selector-design.md`
- Create: `docs/cascaded_block_selector_experiment.md`

- [ ] **Step 1: Add experiment doc**

Create `docs/cascaded_block_selector_experiment.md`:

```markdown
# Cascaded Block Selector Experiment

## Baseline

The baseline uses vertical block candidates, filters usable blocks, selects the
largest usable block, and lets the policy choose placement.

```bash
python train.py \
  --policy-mode largest_block_baseline \
  --stack-only \
  --use-simple-blocks \
  --output-name largest-block-baseline
```

## Cascaded Selector

The cascaded selector exposes all feasible stable oriented vertical blocks and
lets the policy choose the oriented block and EMS anchor.

```bash
python train.py \
  --policy-mode cascaded_block_selector \
  --stack-only \
  --use-simple-blocks \
  --output-name cascaded-block-selector
```

## Evaluation

Use the matching policy mode when evaluating a checkpoint.

```bash
python test.py \
  --policy-mode cascaded_block_selector \
  --checkpoint outputs/train_outputs/cascaded-block-selector/policy_step.pth \
  --stack-only \
  --use-simple-blocks
```

Compare final utilization, blocked step, packed source boxes, selected stack
height distribution, and inference time per packing decision.
```

- [ ] **Step 2: Update design spec implementation status**

Append to `docs/superpowers/specs/2026-05-27-cascaded-block-selector-design.md`:

```markdown
## Implementation Notes

The implementation plan is saved at
`docs/superpowers/plans/2026-05-27-cascaded-block-selector.md`.
```

- [ ] **Step 3: Run docs check**

Run:

```bash
rg -n "cascaded_block_selector|largest_block_baseline" docs configs train.py test.py packing
```

Expected: output includes the new experiment doc, configs, CLI flags, and policy mode plumbing.

- [ ] **Step 4: Commit**

```bash
git add docs/cascaded_block_selector_experiment.md docs/superpowers/specs/2026-05-27-cascaded-block-selector-design.md
git commit -m "Document cascaded block selector experiment"
```

## Final Verification

- [ ] **Step 1: Run full tests**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

Expected: all tests pass.

- [ ] **Step 2: Run baseline smoke command**

```bash
python test.py --policy-mode largest_block_baseline --num-sequences 1 --use-mcts false
```

Expected: script loads the baseline checkpoint and runs one sequence without policy-mode mismatch.

- [ ] **Step 3: Run cascaded env smoke without checkpoint**

```bash
python - <<'PY'
from packing_env.gym_env import PackingEnv

env = PackingEnv(
    k_placement=4,
    buffer_capacity=3,
    container_size=(600, 600, 600),
    stack_only=True,
    use_simple_blocks=True,
    policy_mode="cascaded_block_selector",
)
obs, _ = env.reset(seed=1)
print(obs["oriented_blocks"].shape)
print(obs["action_mask"].shape)
print(bool(obs["action_mask"].any()))
PY
```

Expected:

```text
(6, 8)
(6, 4)
True
```

- [ ] **Step 4: Review git state**

```bash
git status --short --branch
```

Expected: clean except for intentionally unrelated files that pre-existed before this implementation.
