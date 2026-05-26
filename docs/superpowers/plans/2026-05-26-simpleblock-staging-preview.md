# SimpleBlock Staging Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the largest generated SimpleBlock in the interactive simulator staging area as a preview without changing placement behavior.

**Architecture:** The backend exposes a nullable `stagingBlock` payload derived from existing buffer SimpleBlock candidates. The frontend renders that payload as a distinct first object in the right-side Three.js staging/buffer view and in the sidebar, while existing current-item placement paths stay untouched.

**Tech Stack:** Python simulator backend, pytest, browser JavaScript, Three.js, existing static HTML/CSS.

---

### Task 1: Backend StagingBlock Payload

**Files:**
- Modify: `interactive_simulator_app/simulator.py`
- Test: `tests/test_simpleblock_buffer.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_simpleblock_buffer.py`:

```python
def test_interactive_simulator_state_previews_largest_simpleblock():
    box = Orthogonal3D(100, 100, 50)
    simulator = InteractivePackingSimulator(buffer_capacity=3)
    simulator.env.buffer = Buffer(
        capacity=3,
        data_sampler=FakeSampler([box, box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )

    state = simulator.state()

    assert state["currentItem"] == {"dx": 100, "dy": 100, "dz": 50}
    assert state["stagingBlock"]["dx"] == 100
    assert state["stagingBlock"]["dy"] == 100
    assert state["stagingBlock"]["dz"] == 150
    assert state["stagingBlock"]["baseDx"] == 100
    assert state["stagingBlock"]["baseDy"] == 100
    assert state["stagingBlock"]["baseDz"] == 50
    assert state["stagingBlock"]["consumedCount"] == 3
    assert state["stagingBlock"]["stackDims"] == [1, 1, 3]
    assert state["stagingBlock"]["label"] == "SimpleBlock x3"
```

Also add this import near the top of the test file:

```python
from interactive_simulator_app.simulator import InteractivePackingSimulator
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_simpleblock_buffer.py::test_interactive_simulator_state_previews_largest_simpleblock -q`

Expected: FAIL with a missing `stagingBlock` key.

- [ ] **Step 3: Implement the backend payload**

In `interactive_simulator_app/simulator.py`, add `"stagingBlock": self._staging_block_payload(),` to the dictionary returned by `state()`.

Add this method near `_item_payload`:

```python
    def _staging_block_payload(self) -> dict | None:
        if not self.env.buffer.has_items or not self.env.buffer.all_blocks:
            return None
        block = self.env.buffer.sample_blocks(deterministic=True)
        dx, dy, dz = item_dims(block)
        base_dx, base_dy, base_dz = item_dims(block.box)
        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "baseDx": base_dx,
            "baseDy": base_dy,
            "baseDz": base_dz,
            "consumedCount": int(block.consumed_count),
            "stackDims": [int(value) for value in block.no_boxes_wrt_axis],
            "label": f"SimpleBlock x{block.consumed_count}",
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_simpleblock_buffer.py::test_interactive_simulator_state_previews_largest_simpleblock -q`

Expected: PASS.

### Task 2: Preview Does Not Change Placement Consumption

**Files:**
- Modify: `tests/test_simpleblock_buffer.py`

- [ ] **Step 1: Write the failing regression test**

Add this test to `tests/test_simpleblock_buffer.py`:

```python
def test_interactive_simulator_staging_preview_does_not_consume_simpleblock():
    box = Orthogonal3D(100, 100, 50)
    simulator = InteractivePackingSimulator(buffer_capacity=3)
    simulator.env.buffer = Buffer(
        capacity=3,
        data_sampler=FakeSampler([box, box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )

    before = simulator.env.buffer.dims()
    state = simulator.state()
    action = state["actions"][0]
    simulator.place(action["x"], action["y"], rotation=action["rotation"])

    assert before == [(100, 100, 50), (100, 100, 50), (100, 100, 50)]
    assert simulator.env.container.placed_items[0].Dim.raw().tolist() == [100, 100, 50]
    assert len(simulator.env.buffer.buffer) == 3
```

- [ ] **Step 2: Run the regression test**

Run: `pytest tests/test_simpleblock_buffer.py::test_interactive_simulator_staging_preview_does_not_consume_simpleblock -q`

Expected: PASS after Task 1, because placement already uses `_current_item()` and `env.buffer.update(source_item)`.

### Task 3: Frontend Staging Preview Rendering

**Files:**
- Modify: `interactive_simulator_app/static/simulator.js`
- Modify: `interactive_simulator_app/static/index.html`
- Modify: `interactive_simulator_app/static/style.css`

- [ ] **Step 1: Add sidebar markup**

In `interactive_simulator_app/static/index.html`, add a compact staging row near the current item display:

```html
<div class="stat-row">
  <span>Staging</span>
  <strong id="stagingBlock">-</strong>
</div>
```

- [ ] **Step 2: Update JS display helpers**

In `interactive_simulator_app/static/simulator.js`, add:

```javascript
    function stagingText(block) {
      if (!block) return "-";
      return `${block.label}: ${dimsText(block)}`;
    }
```

In `renderSidebar()`, set:

```javascript
      document.getElementById("stagingBlock").textContent = stagingText(state.stagingBlock);
```

- [ ] **Step 3: Render staging block first in the right-side Three.js scene**

In `renderBufferScene()`, before pushing `state.currentItem`, add:

```javascript
      if (state.stagingBlock) {
        items.push({ item: state.stagingBlock, label: state.stagingBlock.label, staging: true, current: false });
      }
```

Change the `addBox` opacity expression to:

```javascript
          entry.staging ? 0.94 : (entry.current ? 0.92 : 0.78),
```

Change the marker condition to:

```javascript
        if (entry.current || entry.staging) {
```

Change the marker color to:

```javascript
              color: entry.staging ? 0x7c3aed : 0x0f766e,
```

- [ ] **Step 4: Add sidebar overflow styling**

In `interactive_simulator_app/static/style.css`, ensure stat row values fit:

```css
    .stat-row strong {
      min-width: 0;
      overflow-wrap: anywhere;
      text-align: right;
    }
```

### Task 4: Verification

**Files:**
- No code changes.

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_simpleblock_buffer.py -q`

Expected: all tests in the file pass.

- [ ] **Step 2: Start the simulator server**

Run: `python -m interactive_simulator_app.server --buffer-capacity 3`

Expected: server starts and prints a local URL.

- [ ] **Step 3: Inspect browser payload manually**

Open `/state` from the local simulator URL.

Expected: JSON includes `stagingBlock` with `label`, dimensions, base dimensions, consumed count, and stack dimensions. Clicking placement candidates should still place one box at a time.
