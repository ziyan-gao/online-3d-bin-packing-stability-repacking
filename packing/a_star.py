from copy import deepcopy
from dataclasses import dataclass, field
import heapq
import math

from packing_env.gym_env import PackingEnv
from packing.plans import (
    ExecutionPlan,
    MCTSPlan,
    PackOperation,
    RepackOperation,
    UnpackOperation,
    execution_plan_from_mcts_plan,
    is_pack_operation_ready,
    pack_precedence_keys,
)


@dataclass(frozen=True)
class SearchContext:
    plan: MCTSPlan
    precedence_keys_by_pack_idx: dict[int, set[tuple[int, int, int, int, int, int]]]
    source_unpack_idx: dict[tuple[int, int, int, int, int, int], int]

    def heuristic(self, pending_unpacks: frozenset[int], pending_packs: frozenset[int]) -> int:
        operated_items = {
            ("item", self.plan.unpack_sequence[idx].item.to_key())
            for idx in pending_unpacks
        }
        for idx in pending_packs:
            pack_op = self.plan.pack_sequence[idx]
            if pack_op.source_item is None:
                operated_items.add(("incoming", idx))
            else:
                operated_items.add(("item", pack_op.source_item.to_key()))
        return len(operated_items)


@dataclass(order=True)
class AStarNode:
    priority: int
    tie: int
    env: PackingEnv = field(compare=False)
    pending_unpacks: frozenset[int] = field(compare=False)
    pending_packs: frozenset[int] = field(compare=False)
    steps: list[UnpackOperation | PackOperation | RepackOperation] = field(compare=False)
    cost: int = field(compare=False, default=0)

    def state_key(self) -> tuple:
        return (self.pending_unpacks, self.pending_packs, self.env.to_key())

    def is_goal(self) -> bool:
        return not self.pending_unpacks and not self.pending_packs

    def expand(self, ctx: SearchContext, next_tie: int) -> tuple[list["AStarNode"], int]:
        children, tie = self._expand_pack_ops(ctx, next_tie)
        unpack_children, tie = self._expand_unpack_ops(ctx, tie)
        children.extend(unpack_children)
        return children, tie

    # Pack expansion
    def _expand_pack_ops(
        self,
        ctx: SearchContext,
        next_tie: int,
    ) -> tuple[list["AStarNode"], int]:
        children: list[AStarNode] = []
        tie = next_tie
        for pack_idx in self.pending_packs:
            pack_op = ctx.plan.pack_sequence[pack_idx]
            pack_children = self._pack_successors(pack_idx, pack_op, ctx, tie)
            children.extend(pack_children)
            tie += len(pack_children)
        return children, tie

    def _pack_successors(
        self,
        pack_idx: int,
        pack_op: PackOperation,
        ctx: SearchContext,
        tie: int,
    ) -> list["AStarNode"]:
        if pack_op.source_item is None:
            child = self._make_incoming_pack_child(pack_idx, pack_op, ctx, tie)
            return [] if child is None else [child]
        return self._make_source_pack_children(pack_idx, pack_op, ctx, tie)

    def _make_source_pack_children(
        self,
        pack_idx: int,
        pack_op: PackOperation,
        ctx: SearchContext,
        tie: int,
    ) -> list["AStarNode"]:
        children: list[AStarNode] = []
        hold_child = self._make_holding_pack_child(pack_idx, pack_op, ctx, tie)
        if hold_child is not None:
            children.append(hold_child)
        repack_child = self._make_repack_child(pack_idx, pack_op, ctx, tie + len(children))
        if repack_child is not None:
            children.append(repack_child)
        return children

    def _make_incoming_pack_child(
        self,
        pack_idx: int,
        pack_op: PackOperation,
        ctx: SearchContext,
        tie: int,
    ) -> "AStarNode | None":
        if not self._pack_ready(pack_idx, pack_op, ctx):
            return None
        child_env = self._pack_into_child_env(pack_op)
        return self._make_child(
            child_env,
            self.pending_unpacks,
            self.pending_packs - {pack_idx},
            self.steps + [pack_op],
            ctx,
            tie,
        )

    def _make_holding_pack_child(
        self,
        pack_idx: int,
        pack_op: PackOperation,
        ctx: SearchContext,
        tie: int,
    ) -> "AStarNode | None":
        if not self._source_in_holding(pack_op) or not self._pack_ready(pack_idx, pack_op, ctx):
            return None
        child_env = self._pack_into_child_env(pack_op, source_item=pack_op.source_item)
        return self._make_child(
            child_env,
            self.pending_unpacks,
            self.pending_packs - {pack_idx},
            self.steps + [pack_op],
            ctx,
            tie,
        )

    def _make_repack_child(
        self,
        pack_idx: int,
        pack_op: PackOperation,
        ctx: SearchContext,
        tie: int,
    ) -> "AStarNode | None":
        source_key = pack_op.source_item.to_key()
        unpack_idx = ctx.source_unpack_idx.get(source_key)
        if unpack_idx not in self.pending_unpacks:
            return None
        if not self._is_unpackable_in_env(self.env, pack_op.source_item):
            return None

        child_env = deepcopy(self.env)
        placed = self._find_placed_item(child_env, pack_op.source_item)
        if placed is None:
            return None

        child_env.unpack(placed)
        if not self._pack_ready_in_env(child_env, pack_idx, pack_op, ctx):
            return None

        selected_ems = self._selected_ems_for_env(child_env, pack_op)
        child_env.pack(
            deepcopy(pack_op.packed_box),
            selected_ems=selected_ems,
            pruning_item_types=child_env._ems_pruning_item_types_after_holding_remove(
                pack_op.source_item
            ),
        )
        child_env.remove_holding_item(pack_op.source_item)
        return self._make_child(
            child_env,
            self.pending_unpacks - {unpack_idx},
            self.pending_packs - {pack_idx},
            self.steps + [
                RepackOperation(
                    source_item=pack_op.source_item,
                    packed_box=pack_op.packed_box,
                    selected_ems=selected_ems,
                )
            ],
            ctx,
            tie,
        )

    # Unpack expansion
    def _expand_unpack_ops(
        self,
        ctx: SearchContext,
        next_tie: int,
    ) -> tuple[list["AStarNode"], int]:
        children: list[AStarNode] = []
        tie = next_tie
        for unpack_idx in self.pending_unpacks:
            child = self._make_unpack_child(unpack_idx, ctx, tie)
            if child is None:
                continue
            children.append(child)
            tie += 1
        return children, tie

    def _make_unpack_child(
        self,
        unpack_idx: int,
        ctx: SearchContext,
        tie: int,
    ) -> "AStarNode | None":
        unpack_op = ctx.plan.unpack_sequence[unpack_idx]
        if not self._is_unpackable_in_env(self.env, unpack_op.item):
            return None
        child_env = deepcopy(self.env)
        placed = self._find_placed_item(child_env, unpack_op.item)
        if placed is None:
            return None
        child_env.unpack(placed)
        return self._make_child(
            child_env,
            self.pending_unpacks - {unpack_idx},
            self.pending_packs,
            self.steps + [unpack_op],
            ctx,
            tie,
        )

    # Shared checks and child construction
    def _make_child(
        self,
        env: PackingEnv,
        pending_unpacks: frozenset[int],
        pending_packs: frozenset[int],
        steps: list[UnpackOperation | PackOperation | RepackOperation],
        ctx: SearchContext,
        tie: int,
    ) -> "AStarNode":
        cost = self.cost + 1
        return AStarNode(
            priority=cost + ctx.heuristic(pending_unpacks, pending_packs),
            tie=tie,
            env=env,
            pending_unpacks=pending_unpacks,
            pending_packs=pending_packs,
            steps=steps,
            cost=cost,
        )

    def _source_in_holding(self, pack_op: PackOperation) -> bool:
        return (
            self.env.container.find_matching_item(
                pack_op.source_item,
                self.env.container.holding_list,
            )
            is not None
        )

    def _pack_ready(self, pack_idx: int, pack_op: PackOperation, ctx: SearchContext) -> bool:
        return self._pack_ready_in_env(self.env, pack_idx, pack_op, ctx)

    def _pack_ready_in_env(
        self,
        env: PackingEnv,
        pack_idx: int,
        pack_op: PackOperation,
        ctx: SearchContext,
    ) -> bool:
        return is_pack_operation_ready(env, pack_op, ctx.precedence_keys_by_pack_idx[pack_idx])

    def _find_placed_item(self, env: PackingEnv, item) -> object | None:
        return env.container.find_matching_item(item)

    def _is_unpackable_in_env(self, env: PackingEnv, item) -> bool:
        return env.container.find_matching_item(item, env.container.unpackable_boxes) is not None

    def _pack_into_child_env(self, pack_op: PackOperation, source_item=None) -> PackingEnv:
        child_env = deepcopy(self.env)
        pruning_item_types = None
        if source_item is not None:
            pruning_item_types = child_env._ems_pruning_item_types_after_holding_remove(
                source_item
            )
        child_env.pack(
            deepcopy(pack_op.packed_box),
            selected_ems=self._selected_ems_for_env(child_env, pack_op),
            pruning_item_types=pruning_item_types,
        )
        if source_item is not None:
            child_env.remove_holding_item(source_item)
        return child_env

    def _selected_ems_for_env(self, env: PackingEnv, pack_op: PackOperation):
        if pack_op.selected_ems is None:
            return None
        for ems in env.heu_ems.get_ems_list():
            if ems == pack_op.selected_ems:
                return ems
        return None


def optimize_execution_plan(initial_env: PackingEnv, plan: MCTSPlan) -> ExecutionPlan:
    """Find a short executable sequence for a MCTS plan."""
    ctx = SearchContext(
        plan=plan,
        precedence_keys_by_pack_idx=pack_precedence_keys(initial_env, plan),
        source_unpack_idx={
            unpack_op.item.to_key(): idx
            for idx, unpack_op in enumerate(plan.unpack_sequence)
        },
    )

    start_pending_unpacks = frozenset(range(len(plan.unpack_sequence)))
    start_pending_packs = frozenset(range(len(plan.pack_sequence)))
    start = AStarNode(
        priority=ctx.heuristic(start_pending_unpacks, start_pending_packs),
        tie=0,
        env=deepcopy(initial_env),
        pending_unpacks=start_pending_unpacks,
        pending_packs=start_pending_packs,
        steps=[],
        cost=0,
    )

    frontier = [start]
    best_cost = {start.state_key(): 0}
    tie = 1

    while frontier:
        node = heapq.heappop(frontier)
        if node.cost > best_cost.get(node.state_key(), math.inf):
            continue
        if node.is_goal():
            return ExecutionPlan(steps=node.steps)

        children, tie = node.expand(ctx, tie)
        for child in children:
            child_key = child.state_key()
            if child.cost >= best_cost.get(child_key, math.inf):
                continue
            best_cost[child_key] = child.cost
            heapq.heappush(frontier, child)

    return execution_plan_from_mcts_plan(plan)
