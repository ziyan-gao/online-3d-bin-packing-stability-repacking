import os
import time
from copy import deepcopy
from dataclasses import dataclass, fields

from omegaconf import OmegaConf

from packing_env.gym_env import PackingEnv
from packing.agents import PackingAgent
from packing.debug_utils import (
    print_execution_steps,
    print_mcts_stats,
    print_mcts_steps,
)
from packing.a_star import optimize_execution_plan
from packing.mcts import mcts
from packing.plans import execution_plan_from_mcts_plan
from packing.policy_loader import set_eval_seed
from packing.visualizer import Visualizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEST_CONFIG = os.path.join(PROJECT_ROOT, "configs", "test_cj_cascade.yaml")


@dataclass(frozen=True)
class TestConfig:
    ds_name: str = "random"
    checkpoint: str | None = "outputs/train_outputs/random/policy_step.pth"
    device: str | None = None
    seed: int = 101
    num_sequences: int = 1
    iterations: int = 100
    max_unpack: int = 6
    use_mcts: bool = True
    target_util: float = 0.8
    max_steps: int = 300
    container_size: tuple[int, int, int] = (600, 600, 600)
    buffer_space: int = 10
    remove_inscribed_ems: bool = False
    stack_only: bool = False
    use_simple_blocks: bool = False
    policy_mode: str = "largest_block_baseline"
    layered_achievability: bool = False
    layered_num_chunks: int = 3
    visualize: bool = False
    visual_dir: str = "outputs/three_live/mcts"
    visual_z_max: float = 610.0
    visual_port: int = 8766
    visual_delay_sec: float = 0.5
    visual_bind_host: str = "127.0.0.1"
    visual_public_host: str = "127.0.0.1"
    visual_poll_ms: int = 300
    show_ems: bool = False
    ems_visual_mode: str = "raw"
    hold_visual: bool = False
    save_replay: bool = False
    replay_path: str | None = None
    replay_interval_ms: int = 700
    mcts_max_child: int = 3
    optimize_sequence: bool = False


def load_test_config(config_path: str = DEFAULT_TEST_CONFIG) -> TestConfig:
    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"Test config must be a mapping: {config_path}")

    valid_keys = {field.name for field in fields(TestConfig)}
    unknown_keys = set(data).difference(valid_keys)
    if unknown_keys:
        raise KeyError(f"Unknown test config keys in {config_path}: {sorted(unknown_keys)}")

    if "container_size" in data:
        data["container_size"] = tuple(data["container_size"])
    return TestConfig(**data)


def build_agent(config: TestConfig):
    if config.checkpoint is None:
        raise ValueError("checkpoint is required for policy validation")
    agent = PackingAgent(
        device=config.device,
        checkpoint_path=config.checkpoint,
        policy_mode=config.policy_mode,
    )
    print(f"loaded policy weights from {config.checkpoint} on {agent.device}")
    return agent


def build_env(config: TestConfig, seed: int) -> PackingEnv:
    env = PackingEnv(
        ds_name=config.ds_name,
        container_size=tuple(config.container_size),
        item_buffer_space=config.buffer_space,
        remove_inscribed_ems=config.remove_inscribed_ems,
        stack_only=config.stack_only,
        use_simple_blocks=config.use_simple_blocks,
        policy_mode=config.policy_mode,
        layered_achievability=config.layered_achievability,
        layered_num_chunks=config.layered_num_chunks,
    )
    env.reset(seed=seed)
    return env


def make_summary(
    seed: int,
    status: str,
    before_env: PackingEnv,
    after_env: PackingEnv | None = None,
    unpacked: int = 0,
    operations: int = 0,
) -> dict:
    if after_env is None:
        after_env = before_env
    return {
        "seed": seed,
        "status": status,
        "before_util": before_env.container.utilization,
        "after_util": after_env.container.utilization,
        "placed_before": len(before_env.container.placed_items),
        "placed_after": len(after_env.container.placed_items),
        "unpacked": unpacked,
        "operations": operations,
    }


def selected_buffer_indices(item, buffer_before) -> list[int]:
    if getattr(item, "preserve_order", False):
        return [0]
    if not hasattr(item, "box"):
        return [0]

    indices = []
    consume_count = int(getattr(item, "consumed_count", 1))
    for index, dims in enumerate(buffer_before):
        if tuple(map(int, dims)) == tuple(map(int, item.box.raw())):
            indices.append(index)
            if len(indices) == consume_count:
                break
    return indices


def record_pack_step(box, item, buffer_before, env: PackingEnv) -> tuple:
    return (
        (int(box.FLB.x), int(box.FLB.y), int(box.FLB.z)),
        (int(box.Dim.dx), int(box.Dim.dy), int(box.Dim.dz)),
        bool(getattr(box, "rot", False)),
        [tuple(map(int, dims)) for dims in buffer_before],
        item,
        selected_buffer_indices(item, buffer_before),
        env.buffer.dims(),
    )


def contained_item_count(env: PackingEnv) -> int:
    return sum(int(getattr(item, "source_item_count", 1)) for item in env.container.placed_items)


def annotate_source_item_count(box, source_item) -> None:
    box.source_item_count = int(getattr(source_item, "consumed_count", 1))


def accumulate_step_reward(
    reward_total: float,
    reward: float,
    step: int,
    utilization: float,
    placed_count: int,
) -> float:
    reward_total += float(reward)
    print(
        "policy step={}, reward={:.6f}, episode_reward={:.6f}, utilization={:.4f}, placed={}".format(
            step,
            float(reward),
            reward_total,
            utilization,
            placed_count,
        )
    )
    return reward_total


def pack_until_blocked(config: TestConfig, env: PackingEnv, agent, seed: int, visualizer: Visualizer):
    pack_history = []
    episode_reward = 0.0
    for step in range(config.max_steps):
        if getattr(config, "policy_mode", "largest_block_baseline") == "cascaded_block_selector":
            obs = env.get_next_observation()
            if not obs["placable"]:
                print(
                    "blocked at step={}, utilization={:.4f}, placed={}, unpackable={}".format(
                        step,
                        env.container.utilization,
                        len(env.container.placed_items),
                        len(env.container.unpackable_boxes),
                    )
                )
                blocked_title = f"Blocked Before MCTS at Pack Step {step} - seed {seed}"
                visualizer.push(env, blocked_title)
                return None, pack_history, False, episode_reward

            action = agent.predict(obs)
            _, _, selected_block, selected_ems = env.decode_cascaded_action(action)
            buffer_before = env.buffer.dims()
            pack_title = f"Pack Step {step + 1} - seed {seed}"
            visualizer.push(env, pack_title)
            box = selected_block.to_item(selected_ems.FLB)
            annotate_source_item_count(box, selected_block.block)
            _, reward, _, _, _ = env.step(action)
            pack_history.append(
                record_pack_step(box, selected_block.block, buffer_before, env)
            )
            env.validate_packing_state()
            episode_reward = accumulate_step_reward(
                episode_reward,
                reward,
                step,
                env.container.utilization,
                len(env.container.placed_items),
            )

            if env.container.utilization >= config.target_util:
                print("target utilization reached before MCTS was needed")
                return None, pack_history, True, episode_reward
            continue

        if config.use_simple_blocks:
            env.select_largest_policy_block()
            if env.buffer.all_blocks:
                candidate_items = [env.buffer.sample_blocks(deterministic=True)]
                obs = env.get_pack_data(candidate_items[0])
            else:
                print(
                    "blocked at step={}, utilization={:.4f}, placed={}, unpackable={}".format(
                        step,
                        env.container.utilization,
                        len(env.container.placed_items),
                        len(env.container.unpackable_boxes),
                    )
                )
                blocked_title = f"Blocked Before MCTS at Pack Step {step} - seed {seed}"
                visualizer.push(env, blocked_title)
                return None, pack_history, False, episode_reward
        else:
            candidate_items = [env.buffer.sample_item()]
            obs = env.get_pack_data(candidate_items[0])

        if not obs["placable"]:
            print(
                "blocked at step={}, utilization={:.4f}, placed={}, unpackable={}".format(
                    step,
                    env.container.utilization,
                    len(env.container.placed_items),
                    len(env.container.unpackable_boxes),
                )
            )
            blocked_title = f"Blocked Before MCTS at Pack Step {step} - seed {seed}"
            visualizer.push(env, blocked_title)
            return (
                candidate_items[0] if candidate_items else None,
                pack_history,
                False,
                episode_reward,
            )

        box, (item_idx, action_idx, _) = agent.predict(obs)
        item = candidate_items[item_idx]
        annotate_source_item_count(box, item)
        buffer_before = env.buffer.dims()
        pack_title = f"Pack Step {step + 1} - seed {seed}"
        if hasattr(visualizer, "push_buffer_selection"):
            visualizer.push_buffer_selection(env, pack_title, item)
        else:
            visualizer.push(env, pack_title)
        reward = box.Dim.Volume / env.container.Volume
        env.pack(box, selected_ems=env.ems_list[action_idx % env.k_placement])
        env.buffer.update(item)
        pack_history.append(record_pack_step(box, item, buffer_before, env))
        env.validate_packing_state()
        episode_reward = accumulate_step_reward(
            episode_reward,
            reward,
            step,
            env.container.utilization,
            len(env.container.placed_items),
        )

        if env.container.utilization >= config.target_util:
            print("target utilization reached before MCTS was needed")
            return None, pack_history, True, episode_reward

    raise AssertionError(f"no blocked item found within {config.max_steps} steps")


def run_mcts_search(config: TestConfig, env: PackingEnv, agent, blocked_item):
    mcts_start = time.perf_counter()
    result, mcts_stats = mcts(
        deepcopy(env),
        agent,
        new_item=blocked_item,
        iterations=config.iterations,
        max_unpack=config.max_unpack,
        Uti_requirement=config.target_util,
        return_info=True,
        max_child=config.mcts_max_child,
    )
    mcts_elapsed = time.perf_counter() - mcts_start
    mcts_stats["elapsed_sec"] = mcts_elapsed
    mcts_stats["avg_rollout_sec"] = (
        mcts_elapsed / mcts_stats["rollouts"] if mcts_stats["rollouts"] else 0.0
    )
    print_mcts_stats(mcts_stats)
    return result


def reject_unsupported_cascaded_mcts(config: TestConfig, target_reached: bool) -> None:
    if (
        getattr(config, "policy_mode", "largest_block_baseline") == "cascaded_block_selector"
        and getattr(config, "use_mcts", True)
        and not target_reached
    ):
        raise ValueError(
            "cascaded_block_selector validation does not support use_mcts=True yet"
        )


def replay_search_result(config: TestConfig, env: PackingEnv, seed: int, visualizer: Visualizer, result):
    plan = result
    print_mcts_steps(plan)
    if config.optimize_sequence:
        execution_plan = optimize_execution_plan(env, plan)
        print_execution_steps(execution_plan)
        step_prefix = "Optimized Step"
    else:
        execution_plan = execution_plan_from_mcts_plan(plan)
        step_prefix = "MCTS Step"
    replay_env = visualizer.replay_execution_plan(
        env,
        execution_plan,
        seed=seed,
        step_prefix=step_prefix,
    )
    replay_env.validate_packing_state()
    return replay_env, plan, execution_plan, step_prefix


def print_final_episode_reward(seed: int, episode_reward: float) -> None:
    print("final episode reward seed={}: {:.6f}".format(seed, float(episode_reward)))


def validate_sequence(config: TestConfig, seed: int) -> dict:
    set_eval_seed(seed)

    print(f"\n=== sequence seed={seed} ===")
    env = build_env(config, seed)
    agent = build_agent(config)
    visualizer = Visualizer.from_args(config)
    initial_buffer = env.buffer.dims()
    visualizer.push(env, f"Initial State - seed {seed}")

    blocked_item, pack_history, target_reached, episode_reward = pack_until_blocked(
        config,
        env,
        agent,
        seed,
        visualizer,
    )
    reject_unsupported_cascaded_mcts(config, target_reached)

    if target_reached:
        print_final_episode_reward(seed, episode_reward)
        visualizer.save_sequence_replay(
            seed,
            initial_buffer,
            pack_history,
            mcts_used=False,
        )
        visualizer.hold()
        return make_summary(seed, "target_before_mcts", env)

    if not config.use_mcts:
        print_final_episode_reward(seed, episode_reward)
        visualizer.save_sequence_replay(
            seed,
            initial_buffer,
            pack_history,
            mcts_used=False,
        )
        visualizer.hold()
        return make_summary(seed, "blocked_no_mcts", env)

    result = run_mcts_search(config, env, agent, blocked_item)
    if result is None:
        print_final_episode_reward(seed, episode_reward)
        visualizer.save_sequence_replay(
            seed,
            initial_buffer,
            pack_history,
            mcts_used=True,
        )
        return make_summary(seed, "mcts_not_found", env)

    replay_env, plan, execution_plan, step_prefix = replay_search_result(
        config,
        env,
        seed,
        visualizer,
        result,
    )
    visualizer.save_sequence_replay(
        seed,
        initial_buffer,
        pack_history,
        mcts_used=True,
        execution_plan=execution_plan,
        step_prefix=step_prefix,
    )
    visualizer.hold()
    print_final_episode_reward(seed, episode_reward)

    packed_incoming = any(operation.is_incoming for operation in plan.pack_sequence)
    assert packed_incoming or replay_env.container.utilization >= config.target_util

    result_summary = make_summary(
        seed,
        "mcts_passed",
        env,
        after_env=replay_env,
        unpacked=len(plan.unpack_sequence),
        operations=len(plan.pack_sequence),
    )
    print(
        "MCTS validation passed: unpacked={}, operations={}, utilization {:.4f} -> {:.4f}, placed {} -> {}".format(
            result_summary["unpacked"],
            result_summary["operations"],
            result_summary["before_util"],
            result_summary["after_util"],
            result_summary["placed_before"],
            result_summary["placed_after"],
        )
    )
    return result_summary
