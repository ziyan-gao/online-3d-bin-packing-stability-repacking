from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from shapely.geometry.base import BaseGeometry

from packing_env.data_type.item import Item
from packing_env.lbcp.load_bounds import (
    GlobalLoadBoundsResult,
    GlobalLoadStatus,
    ItemKey,
    solve_global_load_bounds,
)


@dataclass(frozen=True)
class LBCPConfig:
    material_density: float  # kg/mm³
    payload_capacity: float  # N, shared by every item
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.material_density <= 0:
            raise ValueError("material_density must be positive")
        if self.payload_capacity < 0:
            raise ValueError("payload_capacity must be non-negative")
        if self.gravity <= 0:
            raise ValueError("gravity must be positive")


@dataclass(frozen=True)
class CommitLPTiming:
    commit_index: int
    packed_items: int
    interfaces: int
    variables: int
    highs_calls: int
    total_ms: float
    highs_ms: float
    overhead_ms: float
    status: GlobalLoadStatus

    @classmethod
    def from_result(
        cls,
        *,
        commit_index: int,
        packed_items: int,
        result: GlobalLoadBoundsResult,
    ) -> "CommitLPTiming":
        total_ms = result.solve_time_sec * 1000.0
        highs_ms = result.linprog_time_sec * 1000.0
        overhead_ms = total_ms - highs_ms
        if overhead_ms < -1e-6:
            raise ValueError("HiGHS time exceeds total commit solve time")
        return cls(
            commit_index=commit_index,
            packed_items=packed_items,
            interfaces=result.interface_count,
            variables=result.variable_count,
            highs_calls=result.linprog_calls,
            total_ms=total_ms,
            highs_ms=highs_ms,
            overhead_ms=max(0.0, overhead_ms),
            status=result.status,
        )


class PayloadSafety(str, Enum):
    GUARANTEED_SAFE = "guaranteed_safe"
    GUARANTEED_OVERLOAD = "guaranteed_overload"
    MECHANICALLY_UNCERTAIN = "mechanically_uncertain"


class LBCPValidator:
    """Whole-configuration LP validator for LBCP-constrained payload bounds."""

    def __init__(self, config: LBCPConfig) -> None:
        self.config = config
        self.last_result = GlobalLoadBoundsResult(status=GlobalLoadStatus.OPTIMAL)
        self.last_candidate_result: GlobalLoadBoundsResult | None = None
        self.commit_timing_history: list[CommitLPTiming] = []

    def reset_state(self) -> None:
        self.last_result = GlobalLoadBoundsResult(status=GlobalLoadStatus.OPTIMAL)
        self.last_candidate_result = None
        self.commit_timing_history.clear()

    def item_weight(self, item: Item) -> float:
        return (
            self.config.material_density
            * float(item.Dim.Volume)
            * self.config.gravity
        )

    def solve(
        self,
        items: Sequence[Item],
        *,
        lbcp_polygons: Mapping[
            ItemKey,
            BaseGeometry | Sequence[tuple[float, float]],
        ],
        payload_max_only: bool = False,
        feasibility_only: bool = False,
    ) -> GlobalLoadBoundsResult:
        return solve_global_load_bounds(
            items,
            lbcp_polygons=lbcp_polygons,
            material_density=self.config.material_density,
            gravity=self.config.gravity,
            payload_max_only=payload_max_only,
            feasibility_only=feasibility_only,
        )

    def classify_item(
        self,
        result: GlobalLoadBoundsResult,
        item: Item,
    ) -> PayloadSafety:
        bounds = result.item_payload_bounds[item.to_key()]
        capacity = self.config.payload_capacity
        if bounds.maximum <= capacity + 1e-9:
            return PayloadSafety.GUARANTEED_SAFE
        if bounds.minimum is not None and bounds.minimum > capacity + 1e-9:
            return PayloadSafety.GUARANTEED_OVERLOAD
        return PayloadSafety.MECHANICALLY_UNCERTAIN

    def is_guaranteed_safe(
        self,
        result: GlobalLoadBoundsResult,
        items: Sequence[Item],
    ) -> bool:
        if result.status is not GlobalLoadStatus.OPTIMAL:
            return False
        return all(
            self.classify_item(result, item) is PayloadSafety.GUARANTEED_SAFE
            for item in items
        )

    def rebuild_state_from_placed(
        self,
        placed: Sequence[Item],
        *,
        lbcp_polygons: Mapping[
            ItemKey,
            BaseGeometry | Sequence[tuple[float, float]],
        ],
    ) -> GlobalLoadBoundsResult:
        result = self.solve(placed, lbcp_polygons=lbcp_polygons)
        self.last_result = result
        return result

    def check(
        self,
        candidate: Item,
        placed: Sequence[Item],
        *,
        lbcp_polygons: Mapping[
            ItemKey,
            BaseGeometry | Sequence[tuple[float, float]],
        ],
    ) -> bool:
        items = [*placed, candidate]
        if sum(self.item_weight(item) for item in items) <= (
            self.config.payload_capacity + 1e-9
        ):
            result = self.solve(
                items,
                lbcp_polygons=lbcp_polygons,
                feasibility_only=True,
            )
            self.last_candidate_result = result
            return result.status is GlobalLoadStatus.OPTIMAL
        result = self.solve(
            items,
            lbcp_polygons=lbcp_polygons,
            payload_max_only=True,
        )
        self.last_candidate_result = result
        return self.is_guaranteed_safe(result, items)

    def commit(
        self,
        item: Item,
        placed: Sequence[Item],
        *,
        lbcp_polygons: Mapping[
            ItemKey,
            BaseGeometry | Sequence[tuple[float, float]],
        ],
    ) -> bool:
        del item
        result = self.solve(placed, lbcp_polygons=lbcp_polygons)
        self.commit_timing_history.append(
            CommitLPTiming.from_result(
                commit_index=len(self.commit_timing_history) + 1,
                packed_items=len(placed),
                result=result,
            )
        )
        if not self.is_guaranteed_safe(result, placed):
            return False
        self.last_result = result
        return True

    def sync_state_from_placed(
        self,
        placed: Sequence[Item],
        *,
        lbcp_polygons: Mapping[
            ItemKey,
            BaseGeometry | Sequence[tuple[float, float]],
        ],
    ) -> GlobalLoadBoundsResult:
        return self.rebuild_state_from_placed(
            placed,
            lbcp_polygons=lbcp_polygons,
        )
