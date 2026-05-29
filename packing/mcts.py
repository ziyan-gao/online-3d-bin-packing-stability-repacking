from copy import deepcopy
import math
import random
import time

import numpy as np

from packing_env.data_type.geometry import GRID_HEIGHT, GRID_SIZE
from packing_env.gym_env import PackingEnv
from packing.agents import PackingAgent
from packing.plans import MCTSPlan, PackOperation, UnpackOperation


def _placement_key(ems, dims: tuple[int, int, int]) -> tuple[int, int, int, int, int, int]:
    return (
        int(ems.FLB.Gx),
        int(ems.FLB.Gy),
        int(ems.FLB.Gz),
        int(dims[0] // GRID_SIZE),
        int(dims[1] // GRID_SIZE),
        int(dims[2] // GRID_HEIGHT),
    )


def _mask_no_op_holding_repack(env: PackingEnv, pack_data: dict) -> bool:
    """Remove actions that put a holding item back in the same pose."""
    mask = np.asarray(pack_data["mask"], dtype=bool).copy()
    holding_items = list(env.container.holding_list)
    for candidate_idx, source_item in enumerate(holding_items):
        source_key = source_item.to_key()
        dims = source_item.to_dim_key()
        rotated_dims = (dims[1], dims[0], dims[2])
        for placement_idx, ems in enumerate(env.ems_list):
            if _placement_key(ems, dims) == source_key:
                mask[candidate_idx, 0, placement_idx] = False
            if _placement_key(ems, rotated_dims) == source_key:
                mask[candidate_idx, 1, placement_idx] = False

    pack_data["mask"] = mask
    pack_data["action_mask"] = mask
    pack_data["placable"] = bool(mask.any())
    pack_data["done"] = ~mask.reshape(mask.shape[0], -1).any(axis=1)
    return pack_data["placable"]


def rollout(
    env: PackingEnv,
    agent: PackingAgent,
    new_item,
    value_deterministic=True,
    logits_deterministic=True,
    w_operation=0.01,
    w_value=50,
    Uti_0=0.7,
    Uti_requirement=0.8,
):
    operations = []
    candidate_dims = [item.to_dim_key() for item in env.container.holding_list]
    candidate_dims.append(new_item.to_dim_key())

    incoming_item_is_available = True
    last_value = -1

    while candidate_dims:
        pack_data = env.get_pack_data(np.asarray(candidate_dims, dtype=np.int32))
        if not pack_data["placable"] or not _mask_no_op_holding_repack(env, pack_data):
            break

        packed_box, (candidate_idx, action_idx, last_value) = agent.predict(
            pack_data,
            value_deterministic=value_deterministic,
            logits_deterministic=logits_deterministic,
        )
        selected_ems = env.ems_list[action_idx % env.k_placement]
        post_pack_candidate_dims = list(candidate_dims)
        post_pack_candidate_dims.pop(candidate_idx)

        env.pack(
            packed_box,
            selected_ems=selected_ems,
            pruning_item_types=post_pack_candidate_dims,
        )
        candidate_dims.pop(candidate_idx)

        selected_incoming_item = (
            incoming_item_is_available
            and candidate_idx == len(candidate_dims)
        )
        if selected_incoming_item:
            source_item = None
            incoming_item_is_available = False
        else:
            source_item = env.container.holding_list[candidate_idx]
            env.remove_holding_item(source_item)

        operations.append(
            PackOperation(
                packed_box=packed_box,
                source_item=source_item,
                selected_ems=deepcopy(selected_ems),
            )
        )

    utilization_gain = env.container.utilization - Uti_0
    success = len(candidate_dims) == 0 or env.container.utilization > Uti_requirement
    reward = w_value * last_value + utilization_gain
    return success, operations, reward


class MCT_Node:
    def __init__(self, state, parent=None, unpack_actions=None, max_child=3):
        self.state = state
        self.parent = parent
        self.children = []
        self.untried_unpack_items = list(self.state.container.unpackable_boxes)
        self.unpacked_items = list(unpack_actions or [])
        self.visits = 0
        self.total_reward = 0.0
        self.max_child = max_child

    @property
    def unpack_actions(self):
        return self.unpacked_items

    @property
    def depth(self):
        return len(self.unpacked_items)

    def compute_hash(self):
        return hash(self.state.to_key())

    def is_fully_expanded(self):
        return (
            len(self.untried_unpack_items) == 0
            or len(self.children) >= self.max_child
        )

    def sample_untried_unpack_item(self):
        item = random.choice(self.untried_unpack_items)
        self.untried_unpack_items.remove(item)
        return item

    def best_child(self, exploration_weight=1.0):
        def ucb_score(child):
            exploitation = child.total_reward / (child.visits + 1e-6)
            exploration = math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            return exploitation + exploration_weight * exploration

        return max(self.children, key=ucb_score)

    def backpropagate(self, reward):
        node = self
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent


def _select_leaf(root: MCT_Node) -> MCT_Node:
    node = root
    while node.is_fully_expanded() and node.children:
        node = node.best_child()
    return node


def _expand_once(
    node: MCT_Node,
    visited_states: set[int],
    max_unpack: int,
    timing: dict[str, float] | None = None,
) -> tuple[MCT_Node, bool]:
    if node.depth >= max_unpack:
        node.untried_unpack_items = []
        return node, False

    while node.untried_unpack_items and len(node.children) < node.max_child:
        item_to_unpack = node.sample_untried_unpack_item()
        child_copy_start = time.perf_counter()
        child_state = deepcopy(node.state)
        if timing is not None:
            timing["child_deepcopy_sec"] += time.perf_counter() - child_copy_start

        child_unpack_start = time.perf_counter()
        child_state.unpack(item_to_unpack)
        if timing is not None:
            timing["child_unpack_sec"] += time.perf_counter() - child_unpack_start

        child = MCT_Node(
            child_state,
            parent=node,
            unpack_actions=node.unpacked_items + [item_to_unpack],
            max_child=node.max_child,
        )
        state_hash = child.compute_hash()
        if state_hash in visited_states:
            continue

        visited_states.add(state_hash)
        node.children.append(child)
        return child, True

    return node, False


def mcts(
    initial_state,
    agent,
    new_item,
    iterations=30,
    max_unpack=10,
    Uti_requirement=0.8,
    return_info=False,
    max_child=3,
):
    max_child = max(1, int(max_child))
    root = MCT_Node(initial_state, max_child=max_child)
    visited_states = {root.compute_hash()}
    successful_rollouts = []
    stats = {
        "iterations_requested": iterations,
        "iterations_run": 0,
        "rollouts": 0,
        "expanded_nodes": 0,
        "visited_states": 1,
        "max_child": max_child,
        "root_branches": 0,
        "success_iteration": None,
        "best_reward": None,
        "best_depth": None,
        "timing": {
            "select_sec": 0.0,
            "expand_sec": 0.0,
            "child_deepcopy_sec": 0.0,
            "child_unpack_sec": 0.0,
            "rollout_deepcopy_sec": 0.0,
            "rollout_sec": 0.0,
            "backprop_sec": 0.0,
        },
    }

    while stats["iterations_run"] < iterations and not successful_rollouts:
        stats["iterations_run"] += 1

        select_start = time.perf_counter()
        leaf = _select_leaf(root)
        stats["timing"]["select_sec"] += time.perf_counter() - select_start

        expand_start = time.perf_counter()
        simulation_node, expanded = _expand_once(
            leaf,
            visited_states,
            max_unpack,
            timing=stats["timing"],
        )
        stats["timing"]["expand_sec"] += time.perf_counter() - expand_start
        if expanded:
            stats["expanded_nodes"] += 1
        stats["root_branches"] = len(root.children)

        rollout_copy_start = time.perf_counter()
        rollout_env = deepcopy(simulation_node.state)
        stats["timing"]["rollout_deepcopy_sec"] += time.perf_counter() - rollout_copy_start

        rollout_start = time.perf_counter()
        success, operations, reward = rollout(
            env=rollout_env,
            agent=agent,
            new_item=new_item,
            Uti_0=initial_state.container.utilization,
            Uti_requirement=Uti_requirement,
        )
        stats["timing"]["rollout_sec"] += time.perf_counter() - rollout_start
        stats["rollouts"] += 1
        stats["visited_states"] = len(visited_states)

        backprop_start = time.perf_counter()
        simulation_node.backpropagate(reward)
        stats["timing"]["backprop_sec"] += time.perf_counter() - backprop_start

        if success:
            stats["success_iteration"] = stats["rollouts"]
            successful_rollouts.append((reward, operations, simulation_node.unpack_actions))

    if not successful_rollouts:
        if return_info:
            return None, stats
        return None

    best_reward, best_operations, best_unpacked_items = max(
        successful_rollouts,
        key=lambda result: result[0],
    )
    stats["best_reward"] = best_reward
    stats["best_depth"] = len(best_unpacked_items)
    result = MCTSPlan(
        unpack_sequence=[UnpackOperation(item=item) for item in best_unpacked_items],
        pack_sequence=best_operations,
    )
    if return_info:
        return result, stats
    return result
