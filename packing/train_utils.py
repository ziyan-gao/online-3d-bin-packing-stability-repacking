from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from packing_env.data_type.data_sampler import DataSampler
from packing.policy_loader import resolve_runtime_device

if TYPE_CHECKING:
    from tianshou.policy import PPOPolicy


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_CONFIG = os.path.join(PROJECT_ROOT, "configs", "train_default.yaml")
VALID_POLICY_MODES = {"largest_block_baseline", "cascaded_block_selector"}


@dataclass(frozen=True)
class TrainConfig:
    data_name: str = "random"
    buffer_size: int = 12
    container_dx: int = 600
    container_dy: int = 600
    container_dz: int = 600
    k_placement: int = 80
    remove_inscribed_ems: bool = False
    stack_only: bool = False
    use_simple_blocks: bool = False
    policy_mode: str = "largest_block_baseline"
    layered_achievability: bool = False
    layered_num_chunks: int = 3
    train_env_num: int = 64
    test_env_num: int = 32
    train_env_seed: int = 5
    max_epoch: int = 1000
    step_per_epoch: int = 800 * 40
    step_per_collect: int = 4000
    episode_per_test: int = 128
    batch_size: int = 256
    learning_rate: float = 7e-5
    output_root: str = os.path.join(PROJECT_ROOT, "outputs", "train_outputs")
    output_name: str | None = None
    tb_log_dir: str | None = None
    resume_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.policy_mode not in VALID_POLICY_MODES:
            raise ValueError(
                f"policy_mode must be one of {sorted(VALID_POLICY_MODES)}, "
                f"got {self.policy_mode!r}"
            )
        if self.policy_mode == "cascaded_block_selector" and not self.use_simple_blocks:
            object.__setattr__(self, "use_simple_blocks", True)
        if self.policy_mode == "cascaded_block_selector" and not self.stack_only:
            object.__setattr__(self, "stack_only", True)
        if int(self.layered_num_chunks) <= 0:
            raise ValueError("layered_num_chunks must be a positive integer.")


def load_train_config(config_path: str = DEFAULT_TRAIN_CONFIG) -> TrainConfig:
    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"Train config must be a mapping: {config_path}")

    valid_keys = {field.name for field in fields(TrainConfig)}
    unknown_keys = set(data).difference(valid_keys)
    if unknown_keys:
        raise KeyError(f"Unknown train config keys in {config_path}: {sorted(unknown_keys)}")

    return TrainConfig(**data)


def output_label(config: TrainConfig) -> str:
    if config.output_name:
        return config.output_name
    return os.path.splitext(os.path.basename(config.data_name))[0]


def get_output_dir(output_root: str, label: str) -> str:
    output_dir = os.path.join(output_root, label)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_data_distribution(output_dir: str, config: TrainConfig) -> str:
    sampler = DataSampler(config.data_name)

    boxes = [list(map(int, box)) for box in sampler.box_set]
    dist_cfg = getattr(sampler, "_dist_cfg", None)
    if isinstance(dist_cfg, str) and dist_cfg == "random":
        probabilities = None
        distribution_mode = "random_per_env_dirichlet"
    elif getattr(sampler, "dataset_episodes", None) is not None:
        probabilities = None
        distribution_mode = "dataset_replay"
    elif dist_cfg is None:
        probabilities = (np.ones(len(boxes)) / len(boxes)).tolist()
        distribution_mode = "uniform"
    else:
        probabilities = np.asarray(dist_cfg, dtype=float).tolist()
        distribution_mode = "fixed"

    payload = {
        "data_name": config.data_name,
        "distribution_mode": distribution_mode,
        "buffer_size": config.buffer_size,
        "container_size": [config.container_dx, config.container_dy, config.container_dz],
        "k_placement": config.k_placement,
        "remove_inscribed_ems": config.remove_inscribed_ems,
        "stack_only": config.stack_only,
        "use_simple_blocks": config.use_simple_blocks,
        "boxes": boxes,
        "probabilities": probabilities,
    }
    out_path = os.path.join(output_dir, "data_distribution.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def make_envs(config: TrainConfig):
    from tianshou.env import SubprocVectorEnv

    container_size = (config.container_dx, config.container_dy, config.container_dz)
    train_envs = SubprocVectorEnv(
        [
            lambda: gym.make(
                "OnlinePack-v2",
                k_placement=config.k_placement,
                ds_name=config.data_name,
                buffer_capacity=config.buffer_size,
                container_size=container_size,
                remove_inscribed_ems=config.remove_inscribed_ems,
                stack_only=config.stack_only,
                use_simple_blocks=config.use_simple_blocks,
                policy_mode=config.policy_mode,
                layered_achievability=config.layered_achievability,
                layered_num_chunks=config.layered_num_chunks,
            )
            for _ in range(config.train_env_num)
        ]
    )
    test_envs = SubprocVectorEnv(
        [
            lambda: gym.make(
                "OnlinePack-v2",
                k_placement=config.k_placement,
                ds_name=config.data_name,
                buffer_capacity=config.buffer_size,
                container_size=container_size,
                remove_inscribed_ems=config.remove_inscribed_ems,
                stack_only=config.stack_only,
                use_simple_blocks=config.use_simple_blocks,
                policy_mode=config.policy_mode,
                layered_achievability=config.layered_achievability,
                layered_num_chunks=config.layered_num_chunks,
            )
            for _ in range(config.test_env_num)
        ]
    )
    train_envs.seed(config.train_env_seed)
    test_envs.seed(config.train_env_seed)

    return train_envs, test_envs


def make_single_env(config: TrainConfig):
    return gym.make(
        "OnlinePack-v2",
        k_placement=config.k_placement,
        ds_name=config.data_name,
        buffer_capacity=config.buffer_size,
        container_size=container_size(config),
        remove_inscribed_ems=config.remove_inscribed_ems,
        stack_only=config.stack_only,
        use_simple_blocks=config.use_simple_blocks,
        policy_mode=config.policy_mode,
        layered_achievability=config.layered_achievability,
        layered_num_chunks=config.layered_num_chunks,
    )


def select_training_device() -> str:
    return str(resolve_runtime_device())


def container_size(config: TrainConfig) -> tuple[int, int, int]:
    return (config.container_dx, config.container_dy, config.container_dz)


def training_checkpoint_metadata(config: TrainConfig) -> dict:
    return {
        "data_name": config.data_name,
        "container_size": container_size(config),
        "buffer_size": config.buffer_size,
        "k_placement": config.k_placement,
        "remove_inscribed_ems": config.remove_inscribed_ems,
        "stack_only": config.stack_only,
        "use_simple_blocks": config.use_simple_blocks,
        "policy_mode": config.policy_mode,
    }


def build_training_policy(config: TrainConfig, env, device: str):
    from tianshou.policy import PPOPolicy
    from tianshou.utils.net.common import ActorCritic
    from tianshou.policy.modelfree.ppo import PPOTrainingStats

    from model.cascaded_policy import CascadedCategoricalMasked
    from packing.policy_loader import CategoricalMasked, build_net

    actor, critic = build_net(device=device, policy_mode=config.policy_mode)
    actor_critic = ActorCritic(actor, critic)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.learning_rate)
    dist_fn = (
        CascadedCategoricalMasked
        if config.policy_mode == "cascaded_block_selector"
        else CategoricalMasked
    )

    policy: PPOPolicy[PPOTrainingStats] = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        dist_fn=dist_fn,
        discount_factor=1.0,
        eps_clip=0.3,
        vf_coef=0.5,
        ent_coef=0.003,
        reward_normalization=False,
        advantage_normalization=False,
        recompute_advantage=False,
        dual_clip=None,
        value_clip=False,
        gae_lambda=0.96,
        action_space=env.action_space,
        action_scaling=False,
        deterministic_eval=True,
    )
    return policy, optimizer


def make_collectors(policy, train_envs, test_envs):
    from tianshou.data import Collector, CollectStats, VectorReplayBuffer

    replay_buffer = VectorReplayBuffer(1000000, len(train_envs))
    replay_buffer_test = VectorReplayBuffer(100000, len(test_envs))
    train_collector = Collector[CollectStats](
        policy,
        train_envs,
        replay_buffer,
    )
    test_collector = Collector[CollectStats](
        policy,
        test_envs,
        replay_buffer_test,
    )
    return train_collector, test_collector, replay_buffer, replay_buffer_test


def reset_training_runtime(
    train_envs,
    test_envs,
    train_collector,
    test_collector,
    replay_buffer,
    replay_buffer_test,
) -> None:
    train_collector.reset()
    train_envs.reset()
    test_collector.reset()
    test_envs.reset()
    replay_buffer.reset()
    replay_buffer_test.reset()


def load_training_checkpoint(
    checkpoint_path: str,
    policy: PPOPolicy,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: str,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required_keys = {"model", "optim", "epoch", "env_step", "gradient_step"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise KeyError(
            f"Checkpoint {checkpoint_path!r} is missing required keys: {sorted(missing_keys)}"
        )

    expected = training_checkpoint_metadata(config)
    checkpoint_defaults = {
        "stack_only": False,
        "use_simple_blocks": False,
        "policy_mode": "largest_block_baseline",
    }
    mismatches = {
        key: (checkpoint.get(key, checkpoint_defaults.get(key)), value)
        for key, value in expected.items()
        if checkpoint.get(key, checkpoint_defaults.get(key)) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={ckpt_value!r}, current={current_value!r}"
            for key, (ckpt_value, current_value) in mismatches.items()
        )
        raise ValueError(
            f"Checkpoint configuration does not match current training config ({details})."
        )

    policy.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optim"])
    return checkpoint


def maybe_load_training_checkpoint(config: TrainConfig, policy, optimizer, device: str) -> dict | None:
    if not config.resume_checkpoint:
        return None

    resume_state = load_training_checkpoint(
        checkpoint_path=config.resume_checkpoint,
        policy=policy,
        optimizer=optimizer,
        config=config,
        device=device,
    )
    print(
        "Loaded checkpoint "
        f"{config.resume_checkpoint} "
        f"(epoch={resume_state['epoch']}, "
        f"env_step={resume_state['env_step']}, "
        f"gradient_step={resume_state['gradient_step']})"
    )
    return resume_state


def prepare_training_output(config: TrainConfig) -> tuple[str, str]:
    label = output_label(config)
    save_dir = get_output_dir(config.output_root, label)
    save_data_distribution(save_dir, config)
    return label, save_dir


def make_training_callbacks(
    config: TrainConfig,
    save_dir: str,
    policy,
    optimizer: torch.optim.Optimizer,
):
    metadata = training_checkpoint_metadata(config)

    def train_fn(epoch, env_step):
        pass

    def save_best_fn(policy):
        torch.save(
            {
                "model": policy.state_dict(),
                **metadata,
            },
            os.path.join(save_dir, "policy_step.pth"),
        )

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(save_dir, "checkpoint.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optimizer.state_dict(),
                **metadata,
                "epoch": epoch,
                "env_step": env_step,
                "gradient_step": gradient_step,
            },
            ckpt_path,
        )
        return ckpt_path

    return train_fn, save_best_fn, save_checkpoint_fn


def make_training_logger(config: TrainConfig, label: str, resume_state: dict | None):
    from tianshou.utils import TensorboardLogger

    tb_label = config.tb_log_dir or label
    writer = SummaryWriter(log_dir=os.path.join("log", "tensorboard", tb_label))
    logger = TensorboardLogger(writer)
    if resume_state is not None:
        logger.restore_data = lambda: (
            int(resume_state["epoch"]),
            int(resume_state["env_step"]),
            int(resume_state["gradient_step"]),
        )
    return writer, logger


def run_onpolicy_training(
    config: TrainConfig,
    policy,
    train_collector,
    test_collector,
    logger,
    callbacks,
    resume_state: dict | None,
) -> None:
    from tianshou.trainer import OnpolicyTrainer

    train_fn, save_best_fn, save_checkpoint_fn = callbacks
    OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=config.max_epoch,
        step_per_epoch=config.step_per_epoch,
        repeat_per_collect=1,
        episode_per_test=config.episode_per_test,
        step_per_collect=config.step_per_collect,
        batch_size=config.batch_size,
        train_fn=train_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        resume_from_log=resume_state is not None,
        logger=logger,
    ).run()


def close_training_runtime(writer, train_envs, test_envs, env) -> None:
    writer.close()
    train_envs.close()
    test_envs.close()
    env.close()


def register_training_envs() -> None:
    register(
        id="OnlinePack-v2",
        entry_point="packing_env:PackingEnv",
    )
