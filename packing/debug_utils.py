from packing_env.gym_env import PackingEnv

def print_mcts_steps(plan) -> None:
    print("\nMCTS final action sequence:")
    if not plan.unpack_sequence:
        print("  no unpack actions")
    for idx, operation in enumerate(plan.unpack_sequence, start=1):
        print(f"  unpack {idx}: {operation.item}")

    if not plan.pack_sequence:
        print("  no pack actions")
    for idx, operation in enumerate(plan.pack_sequence, start=1):
        source = "incoming item" if operation.is_incoming else f"holding item {operation.source_item}"
        print(f"  pack {idx}: {source} -> {operation.packed_box}")


def print_execution_steps(execution_plan) -> None:
    print("\nOptimized execution sequence:")
    for idx, step in enumerate(execution_plan.steps, start=1):
        step_type = type(step).__name__
        if step_type == "RepackOperation":
            print(f"  {idx}: repack {step.source_item} -> {step.packed_box}")
        elif step_type == "UnpackOperation":
            print(f"  {idx}: unpack {step.item}")
        elif step_type == "PackOperation":
            source = "incoming item" if step.is_incoming else f"holding item {step.source_item}"
            print(f"  {idx}: pack {source} -> {step.packed_box}")


def print_mcts_stats(stats: dict) -> None:
    print("\nMCTS search stats:")
    print(f"  iterations requested: {stats['iterations_requested']}")
    print(f"  iterations run: {stats['iterations_run']}")
    print(f"  rollouts: {stats['rollouts']}")
    print(f"  expanded nodes: {stats['expanded_nodes']}")
    print(f"  visited states: {stats['visited_states']}")
    print(f"  max children per node: {stats['max_child']}")
    print(f"  root branches: {stats['root_branches']}")
    print(f"  success iteration: {stats['success_iteration']}")
    print(f"  best unpack depth: {stats['best_depth']}")
    print(f"  best reward: {stats['best_reward']}")
    if "elapsed_sec" in stats:
        print(f"  elapsed time: {stats['elapsed_sec']:.4f}s")
    if "avg_rollout_sec" in stats:
        print(f"  avg time per rollout: {stats['avg_rollout_sec']:.4f}s")
    timing = stats.get("timing")
    if timing:
        divisor = max(stats["rollouts"], 1)
        print("  timing breakdown:")
        for name in (
            "select_sec",
            "expand_sec",
            "child_deepcopy_sec",
            "child_unpack_sec",
            "rollout_deepcopy_sec",
            "rollout_sec",
            "backprop_sec",
        ):
            value = timing.get(name, 0.0)
            label = name.removesuffix("_sec").replace("_", " ")
            print(f"    {label}: {value:.4f}s total, {value / divisor:.4f}s/rollout")


def print_bin_state(env: PackingEnv) -> None:
    print(
        "bin state: utilization={:.4f}, placed={}, unpackable={}, holding={}, buffer={}".format(
            env.container.utilization,
            len(env.container.placed_items),
            len(env.container.unpackable_boxes),
            len(env.container.holding_list),
            len(env.buffer.items),
        )
    )
