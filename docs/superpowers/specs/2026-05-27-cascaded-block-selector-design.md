# Cascaded Block Selector DRL Design

## Purpose

The current block-wise packing baseline generates vertical block candidates,
filters them by geometric feasibility and stability, chooses the largest usable
block, and lets the learned policy choose the placement. This design introduces
an experimental DRL model that learns the block choice itself.

The goal is to compare:

- A heuristic block selector that always chooses the largest usable block.
- A learned cascaded selector that chooses among all currently feasible stable
  oriented vertical blocks, then chooses where to place the selected block.

The comparison should isolate the block-selection decision. Both modes should
share candidate generation, EMS construction, feasibility filtering, stability
checks, reward definition, and evaluation metrics wherever possible.

## Scope

This design covers a new experimental policy path for vertical blocks only.
Horizontal or multi-axis composite blocks are out of scope. The existing
largest-usable-block behavior remains available as the baseline.

The new policy path should support training, checkpointing, deterministic
inference, and side-by-side evaluation against the baseline.

## Architecture

The cascaded policy factors one packing decision into two learned decisions:

```text
P(oriented_block, ems | state)
  = P(oriented_block | state)
    * P(ems | state, oriented_block)
```

An oriented block is a vertical block candidate plus one horizontal orientation.
For each vertical stack candidate, the model sees both normal and x/y-transposed
forms when each form has at least one feasible stable placement.

The first policy head selects the oriented block. The second policy head
conditions on the selected oriented block and selects an EMS anchor. The
selected oriented block already contains the rotation choice, so the loading
head does not choose orientation again.

The external action remains a single discrete value for PPO compatibility:

```text
action = oriented_block_index * k_placement + ems_index
```

The environment decodes:

```text
oriented_block_index = action // k_placement
ems_index = action % k_placement
```

This keeps the RL interface close to the current discrete-action setup while
allowing the model to learn an interpretable two-stage decision.

## Candidate Generation

At each decision step, the buffer is summarized by box type. For each available
box type, vertical stack candidates are generated from stack height 1 up to the
number of available boxes of that type.

Each vertical block candidate has:

- True physical dimensions.
- Reserved virtual dimensions for clearance-aware collision and EMS checks.
- Source box type.
- Stack height / consumed box count.

Each block is expanded into up to two oriented candidates:

- Normal x/y orientation.
- Transposed x/y orientation.

Only oriented candidates with at least one feasible stable EMS placement are
exposed to the policy. This means every selectable first-stage action is
meaningful.

With buffer size 12 and vertical stacks only, the number of block candidates is
bounded by the buffer size. The number of oriented candidates is therefore at
most 24 before feasibility filtering.

## Observation Design

The cascaded mode needs a block-candidate axis in the observation. The intended
observation contains:

```text
oriented_blocks: [max_oriented_blocks, block_feature_dim]
ems:             [k_placement, ems_feature_dim]
block_mask:      [max_oriented_blocks]
loading_mask:    [max_oriented_blocks, k_placement]
```

`oriented_blocks` contains normalized features for each valid oriented block.
Initial block features should include:

- True dimensions.
- Reserved dimensions or clearance value.
- Stack height.
- Consumed box count.
- Volume ratio relative to the container.

`ems` contains the same normalized EMS geometry used by the current policy,
unless later experiments show that additional features are needed.

`block_mask` marks exposed oriented blocks. Because impossible blocks are
filtered before observation construction, each valid block should have at least
one valid loading action.

`loading_mask` marks which EMS anchors are feasible for each oriented block.
Feasibility requires reserved-dimension fit, height-map compatibility, and
stability acceptance at the same EMS anchor.

## Actor Design

The actor should embed oriented blocks and EMS candidates, then allow block and
EMS representations to interact. The exact encoder can reuse the current
transformer pattern, but it must support a variable block-candidate axis rather
than assuming one selected item or block.

The actor has two heads:

1. Block selector head:

   ```text
   logits_block: [max_oriented_blocks]
   ```

   Invalid entries are masked by `block_mask`.

2. Loading selector head:

   ```text
   logits_ems: [k_placement]
   ```

   This head is conditioned on the selected oriented block and masked by the
   selected row of `loading_mask`.

During stochastic training, the policy samples or receives a flat action,
decodes it into oriented-block and EMS indices, and computes:

```text
log_prob = log_prob(oriented_block)
         + log_prob(ems | oriented_block)
```

Entropy should also include both stages so that exploration occurs over both
block choice and loading position:

```text
entropy = entropy_block + entropy_ems_given_block
```

During deterministic inference, the simplest behavior is:

1. Choose the highest-scoring valid oriented block.
2. Choose the highest-scoring valid EMS for that selected oriented block.

Later experiments may compare this greedy cascade with maximizing the joint
score over all valid `(oriented_block, ems)` pairs.

## Critic Design

The critic should estimate one value for the full state, not one value per
block. A single state-value head keeps PPO conventional and keeps the actor
responsible for the block-selection policy.

The critic can pool over encoded block and EMS features, respecting masks, then
predict the scalar state value. A per-block auxiliary value or quality head is
out of scope for the first implementation but can be added later if training
signals are weak.

## Environment Integration

The cascaded mode should be added as a separate policy mode rather than
replacing the current behavior. The baseline mode continues to preselect the
largest usable block and exposes placement choices for that block.

The cascaded mode instead exposes all feasible stable oriented block candidates
and decodes the selected flat action into:

- Selected oriented block.
- Selected EMS anchor.

The selected oriented block determines the true placed dimensions, source box
type, stack height, consumed count, and whether the block is transposed. The
placement stage must not apply another orientation decision.

After placement, buffer consumption follows the selected block's consumed count.
The candidate set is regenerated from the updated buffer before the next
decision.

## Configuration And Checkpoints

Training and evaluation should include an explicit policy mode setting:

```text
policy_mode: largest_block_baseline | cascaded_block_selector
```

Checkpoint metadata should record the policy mode and relevant block settings.
Evaluation should reject incompatible checkpoints instead of silently loading a
baseline checkpoint into a cascaded model or the reverse.

The cascaded model should write outputs to a separate checkpoint namespace so
baseline and experimental runs are easy to compare.

## Failure Handling

If no feasible stable oriented block exists, the state is blocked.

If an oriented block has no feasible EMS, it must not be exposed as selectable
in the block selector.

Masks should prevent invalid actions during training and inference. The action
decoder should still validate selected indices and fail loudly during
development if an invalid flat action is encountered.

## Comparison Plan

Compare the baseline and cascaded modes under matched conditions:

- Same item distribution.
- Same container size.
- Same buffer size.
- Same clearance setting.
- Same training seeds where practical.
- Same PPO settings where practical.
- Same candidate generator and geometric filters.

Primary metrics:

- Final utilization.
- Number of packed source boxes.
- Blocked step.
- Number of placement operations.
- Success rate against the target utilization.

Block-specific metrics:

- Average selected stack height.
- Distribution of selected stack heights.
- Fraction of height-1 selections.
- Selected block volume over largest feasible block volume.
- Frequency of choosing a smaller-than-largest feasible block.

Training/runtime metrics:

- Reward curve.
- Utilization curve during evaluation.
- Policy entropy for block selection and EMS selection.
- Inference time per packing decision.

The most important qualitative question is whether the learned policy chooses
smaller stacks in some states to avoid future blockage, and whether those
choices improve final utilization or MCTS recovery.

## Test Plan

Focused tests should verify:

- Vertical blocks expand into normal and transposed oriented candidates.
- Only feasible stable oriented blocks are exposed.
- Every exposed oriented block has at least one feasible EMS.
- Loading masks have the expected shape and validity.
- Flat action decoding selects the intended oriented block and EMS.
- Selected orientation is applied exactly once.
- Buffer consumption matches the selected stack height.
- Baseline largest-block mode remains unchanged.
- Checkpoint/config mismatches are rejected.

## Non-Goals

- General multi-axis composite block generation.
- Changing the physical clearance semantics.
- Replacing PPO or the discrete-action training interface.
- Adding MCTS-aware training rewards in the first implementation.
- Adding per-block auxiliary value heads in the first implementation.
