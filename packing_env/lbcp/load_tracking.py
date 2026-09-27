from __future__ import annotations

from packing_env.data_type.item import Item
from packing_env.lbcp.contact_patch import contact_patch_between, item_bottom_z


def is_on_floor(item: Item) -> bool:
    return item_bottom_z(item) <= 0.5


def items_directly_on(lower: Item, all_items: list[Item]) -> list[Item]:
    return [
        upper
        for upper in all_items
        if upper is not lower and contact_patch_between(upper, lower) is not None
    ]
