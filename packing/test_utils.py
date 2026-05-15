import os
import time
from copy import deepcopy
from dataclasses import dataclass, fields

from omegaconf import OmegaConf

from packing_env.gym_env import PackingEnv
from packing.agents import GreedyValidationAgent, PackingAgent
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
DEFAULT_TEST_CONFIG = os.path.join(PROJECT_ROOT, "configs", "test_default.yaml")


@dataclass(frozen=True)
class TestConfig:
    ds_name: str = "random"
    agent: str = "policy"
    checkpoint: str | None = "train_outputs/random/policy_step.pth"
    device: str | None = None
    seed: int = 101
    num_sequences: int = 1
    iterations: int = 100
    max_unpack: int = 6
    target_util: float = 0.8
    max_steps: int = 300
    container_size: tuple[int, int, int] = (600, 600, 600)
    buffer_space: int = 10
    remove_inscribed_ems: bool = False
    visualize: bool = False
    visual_dir: str = "_plotly_live/mcts"
    visual_z_max: float = 610.0
    visual_port: int = 8766
    visual_delay_sec: float = 0.5
    visual_bind_host: str = "127.0.0.1"
    visual_public_host: str = "127.0.0.1"
    visual_poll_ms: int = 300
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
    agent_name = config.agent
    if agent_name == "policy":
        if config.checkpoint is None:
            raise ValueError("--checkpoint is required when --agent policy is used")
        agent = PackingAgent(device=config.device, checkpoint_path=config.checkpoint)
        print(f"loaded policy weights from {config.checkpoint} on {agent.device}")
        return agent
    return GreedyValidationAgent()


def build_env(config: TestConfig, seed: int) -> PackingEnv:
    env = PackingEnv(
        ds_name=config.ds_name,
        container_size=tuple(config.container_size),
        item_buffer_space=config.buffer_space,
        remove_inscribed_ems=config.remove_inscribed_ems,
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


def record_pack_step(box, env: PackingEnv) -> tuple:
    return (
        (int(box.FLB.x), int(box.FLB.y), int(box.FLB.z)),
        (int(box.Dim.dx), int(box.Dim.dy), int(box.Dim.dz)),
        bool(getattr(box, "rot", False)),
        env.buffer.dims(),
    )


def pack_until_blocked(config: TestConfig, env: PackingEnv, agent, seed: int, visualizer: Visualizer):
    pack_history = []
    for step in range(config.max_steps):
        item = env.buffer.sample_item()
        obs = env.get_pack_data(item)
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
            return item, pack_history, False

        box, (_, action_idx, _) = agent.predict(obs)
        env.pack(box, selected_ems=env.ems_list[action_idx % env.k_placement])
        env.buffer.update(item)
        pack_history.append(record_pack_step(box, env))
        env.validate_packing_state()
        print(
            "{} step={}, utilization={:.4f}, placed={}".format(
                config.agent,
                step,
                env.container.utilization,
                len(env.container.placed_items),
            )
        )
        pack_title = f"Pack Step {step + 1} - seed {seed}"
        visualizer.push(env, pack_title)

        if env.container.utilization >= config.target_util:
            print("target utilization reached before MCTS was needed")
            return None, pack_history, True

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


def validate_sequence(config: TestConfig, seed: int) -> dict:
    set_eval_seed(seed)

    print(f"\n=== sequence seed={seed} ===")
    env = build_env(config, seed)
    agent = build_agent(config)
    visualizer = Visualizer.from_args(config)
    initial_buffer = env.buffer.dims()
    visualizer.push(env, f"Initial State - seed {seed}")

    blocked_item, pack_history, target_reached = pack_until_blocked(
        config,
        env,
        agent,
        seed,
        visualizer,
    )

    if target_reached:
        visualizer.save_sequence_replay(
            seed,
            initial_buffer,
            pack_history,
            mcts_used=False,
        )
        visualizer.hold()
        return make_summary(seed, "target_before_mcts", env)

    result = run_mcts_search(config, env, agent, blocked_item)
    if result is None:
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
