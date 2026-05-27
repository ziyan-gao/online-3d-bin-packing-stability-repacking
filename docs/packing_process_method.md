# Packing Process Method

This document summarizes the packing method at the algorithm level. It avoids
implementation names and describes the process as a sequence of modeling and
decision steps.

## Overview

The packing process maintains a 3D container state, a set of empty maximal
spaces, a buffer of upcoming boxes, and geometric maps used for height and
stability reasoning. At each step, the method generates feasible placement
candidates, asks the learned policy to choose among them, places the selected
box or vertical block, and then updates all geometric state before continuing.

The process uses two dimensions for each packed object:

- The true physical dimension, used for height, stability, support, and
  utilization.
- The reserved virtual dimension, used for collision and empty-space
  feasibility when clearance is enabled.

For a vertical block, the true footprint remains the footprint of one box while
the height is multiplied by the stack height. Clearance is applied around the
whole stack footprint, not between boxes inside the stack.

## 1. Initial State Construction

The method begins with an empty container. The container occupancy state is
clear, the height map is flat, and the stability map is reset. The empty-space
set initially contains one space: the entire container volume.

The item buffer is then filled from the selected item distribution. This buffer
represents the currently visible upcoming boxes. The method repeatedly consumes
items from this buffer and refills it so that the policy always reasons over a
bounded local horizon rather than the full future sequence.

If clearance is enabled, each candidate keeps both its true dimensions and its
reserved footprint. The reserved footprint extends only in the horizontal
directions:

```text
reserved_dx = true_dx + clearance
reserved_dy = true_dy + clearance
reserved_dz = true_dz
```

## 2. Vertical Block Candidate Generation

When block-wise packing is enabled, the buffer is summarized by box type. For
each box type, the method counts how many identical boxes are currently
available. It then creates vertical stack candidates with stack heights from one
box up to the available count.

For example, if four boxes of size `(dx, dy, dz)` are visible, the method can
form candidates with true dimensions:

```text
(dx, dy, dz)
(dx, dy, 2 * dz)
(dx, dy, 3 * dz)
(dx, dy, 4 * dz)
```

Each vertical block candidate also has a reserved dimension:

```text
(dx + clearance, dy + clearance, stack_height * dz)
```

Only candidates whose reserved dimensions can fit inside the container are kept
as initial block candidates.

## 3. Empty Maximal Space Maintenance

The method maintains a dynamic set of empty maximal spaces. Each empty space is
a rectangular cuboid that represents a currently available region of the
container. At the beginning, the whole container is one empty space.

After every placement, the reserved occupied volume of the placed object is
subtracted from the selected empty space. The remaining free volume is split
into new candidate empty spaces. Invalid spaces, tiny spaces, and optionally
spaces fully contained inside other spaces are removed. This produces the empty
space set used for the next decision.

The key point is that empty-space maintenance uses the reserved virtual
dimension. This means a placed item or block removes its physical volume plus
its clearance buffer from future collision candidates.

## 4. Candidate Space Filtering

Before the learned policy is asked to choose an action, the method filters the
empty-space set for the current candidate items or blocks.

An empty space is considered relevant if at least one candidate can fit inside
it in either horizontal orientation. Fitting is checked with reserved
dimensions, so clearance is respected before stability is considered.

When there are many relevant spaces, the method ranks and caps them to a fixed
maximum number. Ranking favors lower placements first, then deterministic
horizontal ordering, while also considering space volume. This converts a
variable-size geometric search space into a fixed-size decision surface for the
policy.

## 5. Usable Block Filtering

For vertical block candidates, the method further filters the generated block
set to keep only candidates that are actually usable in the current container
state.

A vertical block is usable only if all of the following hold:

1. Its reserved dimensions fit inside at least one currently relevant empty
   space.
2. The true footprint of the block is compatible with the current height map at
   that placement.
3. The stability check accepts the placement using the true physical block
   dimensions.
4. The stable placement returned by the stability reasoning agrees with the
   empty-space anchor being tested.

The largest usable block is preferred. If no block survives this filtering, the
method treats the state as blocked for block-wise packing.

## 6. Feasibility Mask Construction

For each candidate item or block, the method evaluates possible placements over
the filtered empty-space list. Each empty space contributes up to two actions:
one for the normal horizontal orientation and one for the rotated horizontal
orientation.

Each action is marked feasible only if:

- The reserved dimensions fit inside the empty space.
- The true dimensions pass the stability check at the empty-space anchor.
- The stability result corresponds exactly to the same anchor.

The result is a binary feasibility mask. The policy can only choose actions
that are marked feasible. This prevents the learned model from selecting
placements that violate geometric fit, clearance, or stability constraints.

## 7. Policy Decision

The policy receives a fixed-size representation containing:

- The normalized candidate item or block dimensions.
- The normalized empty-space candidates.
- The feasibility mask for normal and rotated orientations.
- The clearance value associated with the candidate.

The decision has two parts:

1. Select which candidate item or block to place.
2. Select which feasible empty-space and orientation to use.

The chosen action is decoded into a physical placement: a front-left-bottom
anchor, a true placed dimension, and a rotation flag.

## 8. Placement State Update

Once a placement is selected, the method updates all geometric state in a fixed
order.

First, the stability representation is updated using the true physical
dimensions. This preserves the distinction between physical support and reserved
clearance.

Next, the container occupancy grid is updated using the reserved virtual
dimension. This prevents later placements from entering the clearance region.

Then, the height map is updated using the true physical dimensions. The height
map therefore reflects the actual top surface of packed material rather than
the virtual clearance area.

Finally, the empty-space set is updated by subtracting the reserved occupied
volume from the selected empty space and regenerating the available spaces.

## 9. Buffer Consumption and Candidate Regeneration

After a successful placement, the consumed item or block is removed from the
buffer.

For a single item, one box is consumed. For a vertical block, the number of
consumed boxes equals the stack height. These boxes are removed from the current
buffer, new boxes are sampled to refill the buffer, and the vertical block
candidate set is regenerated from the updated buffer contents.

This regeneration is important because the availability of vertical blocks
depends directly on how many identical boxes remain visible in the buffer.

## 10. Repetition Until Success or Blockage

The process repeats:

1. Generate or refresh candidate items and vertical blocks.
2. Filter empty spaces that can contain them.
3. Filter usable vertical blocks.
4. Construct the feasibility mask.
5. Ask the policy to select a feasible placement.
6. Place the selected object.
7. Update container, height, stability, empty spaces, and buffer.

The loop stops when the target utilization is reached, the maximum step count is
reached, or no feasible placement remains.

## 11. Blockage and Repacking Search

If no feasible placement remains before the target utilization is reached, the
state is considered blocked. In that case, an optional search phase can attempt
to recover.

The search phase reasons over unpack and repack operations. It selects removable
items, temporarily removes them, and explores whether the blocked incoming item
or block can be inserted after rearrangement. Candidate search states reuse the
same geometric rules: true dimensions for physical support and stability,
reserved dimensions for collision and empty-space feasibility.

If a valid rearrangement is found, the method converts the search result into an
execution sequence, replays the operations, and validates the final packing
state.

## 12. Method Summary

The method combines deterministic geometric filtering with learned action
selection. Geometry determines what is allowed; the policy chooses among the
allowed placements.

The central design choice is the separation between true physical dimensions and
reserved virtual dimensions. This allows the method to reserve clearance around
packed objects while still computing height, stability, support, and utilization
from the real packed material.

For the current vertical-block setting, this means a stack behaves like one
taller physical object with one reserved horizontal clearance footprint around
the whole stack.
