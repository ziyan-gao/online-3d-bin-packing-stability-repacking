from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class TrackedItem:
    handle: int
    size: list[float]
    pose: list[float]
    raw: dict[str, Any]


class BeltMonitor:
    def __init__(self, sim: object):
        self.sim = sim
        self.was_full = False

    def set_robot_picking(self, enabled: bool) -> None:
        self.sim.setInt32Signal("robotPicking", 1 if enabled else 0)
        if not enabled:
            self.was_full = False

    def read_full_items(self) -> list[TrackedItem]:
        data = self._read_belt_data()
        if not data or data.get("full") is not True:
            return []
        return self._items_from_data(data)

    def full_transition_items(self) -> list[TrackedItem]:
        data = self._read_belt_data()
        is_full = bool(data and data.get("full") is True)
        was_full = self.was_full
        self.was_full = is_full
        if is_full and not was_full:
            return self._items_from_data(data)
        return []

    def _read_belt_data(self) -> dict[str, Any] | None:
        packed = self.sim.getStringSignal("trackedItemsData")
        if not packed:
            return None
        if isinstance(packed, dict):
            return packed
        if isinstance(packed, bytes):
            packed = packed.decode("utf-8")
        if isinstance(packed, str):
            try:
                data = json.loads(packed)
            except json.JSONDecodeError:
                return self.sim.unpackTable(packed)
            if isinstance(data, dict):
                return data
            return None
        return self.sim.unpackTable(packed)

    def _items_from_data(self, data: dict[str, Any]) -> list[TrackedItem]:
        items = []
        for item in data.get("items", []):
            handle = item.get("handle")
            if handle is None:
                continue
            items.append(
                TrackedItem(
                    handle=int(handle),
                    size=list(item.get("size", [])),
                    pose=list(item.get("pose", [])),
                    raw=item,
                )
            )
        return items
