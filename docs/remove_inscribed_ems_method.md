# Remove Inscribed EMS Method

When `remove_inscribed_ems` is enabled, the EMS set is pruned after each
placement so that redundant empty spaces are removed before the next decision.

## What Changes

After an item or block is packed, the EMS manager updates the empty-space list:

1. It identifies the selected EMS that contained the placed virtual item.
2. It splits or subtracts the occupied virtual volume from that EMS.
3. It clips any other EMSs that overlap the placed virtual volume.
4. It removes invalid residual spaces, such as spaces with non-positive
   dimensions or spaces below the minimum size thresholds.
5. It removes exact duplicate EMSs.
6. If `remove_inscribed_ems` is `true`, it removes any EMS that is completely
   contained inside another EMS.

An EMS is considered inscribed when all of its lower bounds are greater than or
equal to another EMS's lower bounds, and all of its upper bounds are less than or
equal to that other EMS's upper bounds. In that case, the inner EMS does not add
new placement opportunity because every item that fits inside it also fits
inside the larger containing EMS.

## Why It Can Be Faster

Without this pruning, the EMS list can grow quickly. In CJ-sized containers,
many generated spaces can be nested inside larger spaces. Even if only the top
`k_placement` EMS candidates are exposed to the policy, the method still has to
filter, rank, and test a much larger internal EMS list before that cap is
applied.

With inscribed EMS removal, the internal EMS list stays much smaller. In your
observation, the unpruned list can grow beyond 2000 EMSs, while the pruned list
stays around 100 EMSs. That reduction speeds up later placement decisions
because fewer EMSs need to be checked for fit, stability, ranking, and policy
visibility.

## Tradeoff

The pruning step itself has a cost because it compares EMSs against one another
to find contained spaces. However, for CJ-style packing this cost is usually
worth paying because it prevents the much larger repeated cost of carrying
thousands of redundant EMSs through every later step.

## Effect On Packing Behavior

The policy may see a different ordered EMS candidate set when pruning is enabled.
That means training and testing should use the same value for
`remove_inscribed_ems`. If the policy is trained with pruning disabled but tested
with pruning enabled, the observation distribution can shift because EMS
candidates are filtered differently.

For CJ experiments, `remove_inscribed_ems: true` is recommended because it keeps
the EMS list compact and makes training and testing faster while preserving the
meaningful placement opportunities.
