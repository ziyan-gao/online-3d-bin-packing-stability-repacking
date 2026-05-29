# Stable EMS Pruning Design

## Purpose

The baseline should keep `remove_inscribed_ems: false` because some inscribed EMSs can still fit real items and satisfy the stability constraint. Removing all inscribed EMSs is too aggressive.

However, disabling inscribed-EMS removal can cause EMS growth. Some generated EMSs are meaningless because they are floating or cannot stably support any item currently available to pack. This feature automatically prunes those unusable EMSs after every packing update while preserving useful inscribed EMSs.

## Scope

The feature applies to the true EMS list maintained by `packing_env.heu_ems.EMS`. It should run after every packing update, regardless of the `remove_inscribed_ems` setting.

The pruning rule uses:

- the updated height map,
- the updated feasibility/stability map,
- the unique raw item types currently present in the buffer.

It should not inspect repeated buffer entries one by one, and it should not inspect generated simpleblocks. Simpleblock selection remains a later policy decision.

## Pruning Rule

For each EMS, keep it only if at least one unique current buffer item type can be placed stably at the EMS lower-front-left-bottom point.

The rule is:

1. Run a cheap height-map support prefilter. If no cell under the EMS footprint has height equal to `EMS.FLB.z`, remove the EMS.
2. Iterate over unique item types from the current buffer.
3. For each item type, test the normal XY orientation and the XY-rotated orientation. If `dx == dy`, test only once.
4. For an orientation to keep the EMS:
   - the oriented item dimensions must fit inside the EMS,
   - the existing feasibility/stability heuristic must report a stable placement at `EMS.FLB`,
   - the returned stable coordinate must equal `EMS.FLB`.
5. If no item type and orientation passes, remove the EMS.

This uses real buffer item dimensions instead of a synthetic probe item. For the CJ dataset this is cheap because the item type set is small.

## Integration

EMS should own the final pruning of its EMS list, but it should receive the needed context from `PackingEnv`.

Add an EMS method shaped like:

```python
prune_unstable(
    hm: HeightMap,
    feasibility_map,
    item_types: Iterable[Orthogonal3D],
) -> None
```

The method should not receive the whole `Buffer`. Passing only `item_types` keeps EMS decoupled from buffer internals and repeated buffer entries.

`PackingEnv` should call the method after every packing update.

For `_step(...)`, the update order should be:

```text
1. update stability map for placed item
2. add item to container
3. update height map
4. update buffer, consuming the placed source item
5. update EMS geometry
6. prune EMSs using the updated buffer item types
```

The pruning happens after `buffer.update(source_item)` so the item just consumed no longer influences EMS survival, and any refill is reflected.

For `pack(...)`, there is no buffer consumption. It should still call unstable pruning after EMS geometry update, using the current buffer item types as-is. This keeps replay and direct-pack paths from accumulating meaningless EMSs.

## Relationship To `remove_inscribed_ems`

`remove_inscribed_ems` remains independent.

The cleanup order inside the EMS update/prune flow should be:

```text
deduplicate EMSs
optionally remove inscribed EMSs if remove_inscribed_ems is true
always prune unstable EMSs after PackingEnv supplies context
```

With the preferred CJ baseline setting:

```yaml
remove_inscribed_ems: false
```

the effective behavior is:

```text
deduplicate EMSs
prune unstable EMSs
```

This preserves stable inscribed EMSs while removing floating or unusable EMSs.

## Stability Check

The pruning method should mirror the existing stability logic used by `PackingEnv.get_stable_lps_mask(...)` and `Buffer._is_block_usable(...)`.

For a single EMS and oriented item:

```text
candidate placement = EMS.FLB
candidate item dimension = normal or XY-rotated raw item type
```

Keep the EMS if:

```text
EMS.include(candidate item dimension)
and feasibility_map(candidate item dimension, hm, [EMS.FLB.topix()]) says stable
and stable_coord == EMS.FLB
```

If the item does not fit, skip the stability call for that orientation.

## Testing

Add focused tests for:

- height-map support prefilter removes EMSs with no footprint cell at `EMS.FLB.z`;
- a stable unique item type keeps an EMS;
- item types that do not fit, or that fail stability, cause the EMS to be removed;
- repeated buffer items are reduced to unique dimensions before pruning;
- `_step(...)` calls pruning after `buffer.update(...)`;
- `pack(...)` calls pruning with current buffer item types;
- with `remove_inscribed_ems=False`, an inscribed EMS survives when it can stably fit a current item type.

Tests should use small deterministic containers and synthetic EMS/item cases. They should isolate the pruning rule with monkeypatched stability checks where that makes the intent clearer, and use at least one real stability-path test to guard integration with `Heu_Stable`.
