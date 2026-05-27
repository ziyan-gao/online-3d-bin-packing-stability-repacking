import argparse
import os
from dataclasses import replace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from packing.test_utils import DEFAULT_TEST_CONFIG, TestConfig, load_test_config, validate_sequence


def parse_test_args() -> TestConfig:
    parser = argparse.ArgumentParser(description="Validate MCTS unpack/repack search.")
    parser.add_argument("--config", default=DEFAULT_TEST_CONFIG)
    parser.add_argument(
        "--checkpoint",
        help="Policy checkpoint to load.",
    )
    parser.add_argument(
        "--device",
        help="Torch device for policy inference. Defaults to cuda when available.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-sequences", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--max-unpack", type=int)
    parser.add_argument(
        "--use-mcts",
        action=argparse.BooleanOptionalAction,
        help="Run MCTS unpack/repack search after the initial packing loop blocks.",
    )
    parser.add_argument("--target-util", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--container-size", type=int, nargs=3)
    parser.add_argument(
        "--buffer-space",
        "--item-buffer-space",
        dest="buffer_space",
        type=int,
        help="Per-side x/y clearance in millimeters for each packed item.",
    )
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
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--show-ems",
        action=argparse.BooleanOptionalAction,
        help="Draw EMS cuboids in the left Three.js scene.",
    )
    parser.add_argument(
        "--ems-visual-mode",
        choices=("raw", "policy"),
        help="Choose whether EMS visualization shows raw capped EMSs or policy-exposed EMSs.",
    )
    parser.add_argument("--save-replay", action=argparse.BooleanOptionalAction)
    parser.add_argument("--replay-path", help="Interactive replay HTML path or output directory.")
    parser.add_argument("--optimize-sequence", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    config = load_test_config(args.config)
    overrides = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "seed": args.seed,
        "num_sequences": args.num_sequences,
        "iterations": args.iterations,
        "max_unpack": args.max_unpack,
        "use_mcts": args.use_mcts,
        "target_util": args.target_util,
        "max_steps": args.max_steps,
        "container_size": tuple(args.container_size) if args.container_size is not None else None,
        "buffer_space": args.buffer_space,
        "remove_inscribed_ems": args.remove_inscribed_ems,
        "stack_only": args.stack_only,
        "use_simple_blocks": args.use_simple_blocks,
        "visualize": args.visualize,
        "show_ems": args.show_ems,
        "ems_visual_mode": args.ems_visual_mode,
        "save_replay": args.save_replay,
        "replay_path": args.replay_path,
        "optimize_sequence": args.optimize_sequence,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **overrides)


def main() -> None:
    args = parse_test_args()
    if args.num_sequences < 1:
        raise ValueError("--num-sequences must be at least 1")

    summaries = []
    for offset in range(args.num_sequences):
        summaries.append(validate_sequence(args, seed=args.seed + offset))

    if args.num_sequences > 1:
        print("\n=== summary ===")
        for summary in summaries:
            print(
                "seed={seed}: {status}, util {before_util:.4f}->{after_util:.4f}, "
                "placed {placed_before}->{placed_after}, unpacked={unpacked}, operations={operations}".format(
                    **summary
                )
            )


if __name__ == "__main__":
    main()
