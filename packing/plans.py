from copy import deepcopy
from dataclasses import dataclass

from packing_env.gym_env import PackingEnv
from packing_env.data_type.ems import EmptyMaximalSpace
from packing_env.data_type.item import Item


@dataclass
class UnpackOperation:
    item: Item


@dataclass
class PackOperation:
    packed_box: Item
    source_item: Item | None = None  # None means the incoming conveyor item
    selected_ems: EmptyMaximalSpace | None = None

    @property
    def is_incoming(self) -> bool:
        return self.source_item is None


@dataclass
class RepackOperation:
    source_item: Item
    packed_box: Item
    selected_ems: EmptyMaximalSpace | None = None


@dataclass
class MCTSPlan:
    unpack_sequence: list[UnpackOperation]
    pack_sequence: list[PackOperation]

    @property
    def unpacked_items(self) -> list[Item]:
        return [operation.item for operation in self.unpack_sequence]


@dataclass
class ExecutionPlan:
    steps: list[UnpackOperation | PackOperation | RepackOperation]


def execution_plan_from_mcts_plan(plan: MCTSPlan) -> ExecutionPlan:
    return ExecutionPlan(steps=[*plan.unpack_sequence, *plan.pack_sequence])


def _commit_replay_state(replay_env: PackingEnv) -> None:
    replay_env.validate_packing_state()


def _replay_unpack(
    replay_env: PackingEnv,
    source_item: Item,
) -> None:
    item_to_unpack = replay_env.find_placed_item(source_item)
    replay_env.unpack(item_to_unpack)
    _commit_replay_state(replay_env)


def _replay_pack(
    replay_env: PackingEnv,
    packed_box: Item,
    source_item: Item | None,
    selected_ems: EmptyMaximalSpace | None = None,
) -> None:
    pruning_item_types = None
    if source_item is not None:
        pruning_item_types = replay_env._ems_pruning_item_types_after_holding_remove(source_item)
    replay_env.pack(
        deepcopy(packed_box),
        selected_ems=_selected_ems_for_env(replay_env, selected_ems),
        pruning_item_types=pruning_item_types,
    )
    if source_item is not None:
        replay_env.remove_holding_item(source_item)
    _commit_replay_state(replay_env)


def _selected_ems_for_env(
    env: PackingEnv,
    selected_ems: EmptyMaximalSpace | None,
) -> EmptyMaximalSpace | None:
    if selected_ems is None:
        return None
    for ems in env.heu_ems.get_ems_list():
        if ems == selected_ems:
            return ems
    return None


def replay_execution_plan(
    env: PackingEnv,
    execution_plan: ExecutionPlan,
) -> PackingEnv:
    replay_env = deepcopy(env)
    for step in execution_plan.steps:
        if isinstance(step, UnpackOperation):
            _replay_unpack(replay_env, step.item)
        elif isinstance(step, RepackOperation):
            item_to_unpack = replay_env.find_placed_item(step.source_item)
            replay_env.unpack(item_to_unpack)
            replay_env.remove_holding_item(step.source_item)
            replay_env.pack(
                deepcopy(step.packed_box),
                selected_ems=_selected_ems_for_env(replay_env, step.selected_ems),
            )
            _commit_replay_state(replay_env)
        elif isinstance(step, PackOperation):
            _replay_pack(
                replay_env,
                step.packed_box,
                step.source_item,
                selected_ems=step.selected_ems,
            )
        else:
            raise TypeError(f"Unsupported execution step: {step!r}")

    return replay_env


def is_pack_operation_ready(
    env: PackingEnv,
    operation: PackOperation,
    precedence_keys: set[tuple[int, int, int, int, int, int]],
) -> bool:
    placed_keys = {item.to_key() for item in env.container.placed_items}
    return precedence_keys.issubset(placed_keys) and env.container.is_placeable(operation.packed_box)


def pack_precedence_keys(
    initial_env: PackingEnv,
    plan: MCTSPlan,
) -> dict[int, set[tuple[int, int, int, int, int, int]]]:
    replay_env = deepcopy(initial_env)
    for operation in plan.unpack_sequence:
        item = replay_env.container.find_matching_item(operation.item)
        if item is None:
            raise ValueError(f"Cannot find unpack item in replay env: {operation.item}")
        replay_env.unpack(item)

    precedence_keys_by_pack_idx = {}
    for idx, operation in enumerate(plan.pack_sequence):
        pruning_item_types = None
        if operation.source_item is not None:
            pruning_item_types = replay_env._ems_pruning_item_types_after_holding_remove(
                operation.source_item
            )
        replay_env.pack(
            deepcopy(operation.packed_box),
            selected_ems=operation.selected_ems,
            pruning_item_types=pruning_item_types,
        )
        placed_item = replay_env.container.find_matching_item(operation.packed_box)
        if placed_item is None:
            raise ValueError(f"Cannot find packed item in replay env: {operation.packed_box}")
        precedence_keys_by_pack_idx[idx] = {
            parent.to_key() for parent in replay_env.container.get_parents(placed_item)
        }
        if operation.source_item is not None:
            replay_env.remove_holding_item(operation.source_item)

    return precedence_keys_by_pack_idx
