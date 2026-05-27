"""
refer:
- https://github.com/albertcity/OCARL
- https://github.com/pioneer-innovation/Real-3D-Embodied-Dataset
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace

from packing.train_utils import (
    DEFAULT_TRAIN_CONFIG,
    TrainConfig,
    build_training_policy,
    close_training_runtime,
    load_train_config,
    make_collectors,
    make_envs,
    make_single_env,
    make_training_callbacks,
    make_training_logger,
    maybe_load_training_checkpoint,
    prepare_training_output,
    register_training_envs,
    reset_training_runtime,
    run_onpolicy_training,
    select_training_device,
)

warnings.filterwarnings("ignore")


def parse_train_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the item-wise packing policy.")
    parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--data-name")
    parser.add_argument("--max-epoch", type=int)
    parser.add_argument("--output-name")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument(
        "--remove-inscribed-ems",
        action=argparse.BooleanOptionalAction,
        help="Remove EMS candidates that are fully contained in another EMS.",
    )
    parser.add_argument(
        "--stack-only",
        action=argparse.BooleanOptionalAction,
        help="Generate SimpleBlock candidates as vertical stacks of identical boxes.",
    )
    parser.add_argument(
        "--use-simple-blocks",
        action=argparse.BooleanOptionalAction,
        help="Use generated SimpleBlock candidates instead of FIFO single-box sampling.",
    )
    args = parser.parse_args()

    config = load_train_config(args.config)
    overrides = {
        "data_name": args.data_name,
        "max_epoch": args.max_epoch,
        "output_name": args.output_name,
        "resume_checkpoint": args.resume_checkpoint,
        "remove_inscribed_ems": args.remove_inscribed_ems,
        "stack_only": args.stack_only,
        "use_simple_blocks": args.use_simple_blocks,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **overrides)


def train(config: TrainConfig) -> None:
    # Runtime objects: environments, policy, optimizer, and collectors.
    device = select_training_device()
    env = make_single_env(config)
    train_envs, test_envs = make_envs(config=config)

    policy, optimizer = build_training_policy(config, env, device)
    resume_state = maybe_load_training_checkpoint(config, policy, optimizer, device)

    (
        train_collector,
        test_collector,
        replay_buffer,
        replay_buffer_test,
    ) = make_collectors(
        policy,
        train_envs,
        test_envs,
    )

    reset_training_runtime(
        train_envs,
        test_envs,
        train_collector,
        test_collector,
        replay_buffer,
        replay_buffer_test,
    )

    # Output objects: checkpoints, dataset metadata, TensorBoard logging.
    label, save_dir = prepare_training_output(config)
    callbacks = make_training_callbacks(
        config,
        save_dir,
        optimizer,
    )
    writer, logger = make_training_logger(config, label, resume_state)

    # Tianshou owns the training loop; the script owns setup and cleanup.
    try:
        run_onpolicy_training(
            config,
            policy,
            train_collector,
            test_collector,
            logger,
            callbacks,
            resume_state,
        )
    finally:
        close_training_runtime(writer, train_envs, test_envs, env)


if __name__ == "__main__":
    register_training_envs()
    train(parse_train_args())
