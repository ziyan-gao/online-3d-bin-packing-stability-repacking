# SimpleBlock Staging Preview Design

## Goal

Show the generated SimpleBlock in the interactive simulator's staging area as a preview only. Placement behavior must remain unchanged: clicking an anchor or grid candidate still places the current single buffered item and consumes the buffer through the existing FIFO path.

## Recommended Approach

Expose one preview block from the backend state and render it in the right-side staging/buffer Three.js panel before the current item. The preview should use the largest generated SimpleBlock available from the buffer because that matches the existing deterministic block selection behavior used elsewhere.

## Backend

`InteractivePackingSimulator.state()` will include a nullable `stagingBlock` payload. The payload will be derived from `env.buffer.sample_blocks(deterministic=True)` when generated blocks exist. If the buffer is empty or no generated blocks are available, `stagingBlock` is `None`.

The payload will include:

- `dx`, `dy`, `dz`: preview dimensions.
- `baseDx`, `baseDy`, `baseDz`: source box dimensions.
- `consumedCount`: how many buffered boxes the block represents.
- `stackDims`: the SimpleBlock stack dimensions.
- `label`: compact display text.

The preview computation must not mutate `env.buffer`, `env.buffer.simple_blocks`, `env.ems_list`, placement candidates, or selected rotation.

## Frontend

`renderBufferScene()` will render `state.stagingBlock` first when present. It will have a distinct visual treatment from the current item, such as a stronger opacity and a small marker above it. The current item and remaining buffer items stay visible after it.

The sidebar will show a compact staging row with the preview dimensions and label. Existing controls remain unchanged.

## Error Handling

If generated blocks are unavailable, the backend returns `stagingBlock: null` and the frontend simply omits the preview. This keeps the simulator usable when the buffer is empty or SimpleBlock generation produces no container-fitting block.

## Tests

Add backend tests that verify:

- State includes a staging block when SimpleBlock candidates exist.
- The staging block represents the largest deterministic generated block.
- Placing an item still consumes only the current single buffered item, not the preview block.
