from __future__ import annotations

from packing_env.data_type.item import Item
from packing_env.lbcp.contact_patch import contact_patch_between, item_bottom_z


def direct_supporters(upper: Item, placed: list[Item]) -> list[Item]:
    supporters: list[Item] = []
    for candidate in placed:
        if candidate is upper:
            continue
        if contact_patch_between(upper, candidate) is not None:
            supporters.append(candidate)
    return supporters


def bottommost_items(items: list[Item]) -> list[Item]:
    if not items:
        return []
    min_z = min(item_bottom_z(item) for item in items)
    return [item for item in items if abs(item_bottom_z(item) - min_z) < 0.5]


def collect_support_chain(top: Item, placed: list[Item]) -> list[Item]:
    """Return ordered list top + transitive supporters down to (excluding) bottommost."""
    chain = [top]
    seen = {top.to_key()}
    frontier = direct_supporters(top, placed)
    while frontier:
        nxt = frontier.pop(0)
        key = nxt.to_key()
        if key in seen:
            continue
        seen.add(key)
        chain.append(nxt)
        frontier.extend(direct_supporters(nxt, placed))
    return chain
