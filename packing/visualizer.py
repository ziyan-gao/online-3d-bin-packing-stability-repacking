import os
import time
from copy import deepcopy

from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item
from packing_env.gym_env import PackingEnv

from .interactive_replay import InteractiveReplayRecorder
from .live_plot import LivePlotServer, make_live_server
from .plans import PackOperation, RepackOperation, UnpackOperation
from .three_scene import build_three_scene


class Visualizer:
    def __init__(
        self,
        args,
        live_server: LivePlotServer | None = None,
    ) -> None:
        self.args = args
        self.live_server = live_server
        self.visual_z_max = args.visual_z_max
        self.delay_sec = args.visual_delay_sec
        self.last_selected_count = 0

    @classmethod
    def from_args(cls, args) -> "Visualizer":
        live_server = make_live_server(args) if args.visualize else None
        return cls(args, live_server=live_server)

    def build(
        self,
        env: PackingEnv,
        title: str,
        reset_history: bool = False,
        *,
        highlight_selected: bool = True,
        selected_block=None,
        selected_indices: list[int] | None = None,
    ):
        self._refresh_policy_ems_for_visualization(
            env,
            highlight_selected=highlight_selected,
            selected_block=selected_block,
            selected_indices=selected_indices,
        )
        if highlight_selected and selected_block is not None:
            self.last_selected_count = int(getattr(selected_block, "consumed_count", 1))
        env.visual_selected_count = self.last_selected_count
        return build_three_scene(
            env,
            title,
            show_ems=getattr(self.args, "show_ems", False),
            ems_mode=getattr(self.args, "ems_visual_mode", "raw"),
            visual_z_max=self.visual_z_max,
        )

    def _refresh_policy_ems_for_visualization(
        self,
        env: PackingEnv,
        *,
        highlight_selected: bool = True,
        selected_block=None,
        selected_indices: list[int] | None = None,
    ) -> None:
        show_policy_ems = getattr(self.args, "ems_visual_mode", "raw") == "policy"
        if getattr(env, "policy_mode", "largest_block_baseline") == "cascaded_block_selector":
            env.visual_selected_block = selected_block if highlight_selected else None
            env.visual_selected_block_indices = list(selected_indices or [])
            env.buffer.visual_highlight_indices = env.visual_selected_block_indices
            return

        if not env.buffer.has_items:
            if show_policy_ems:
                env.ems_list = []
            env.visual_selected_block = None
            env.visual_selected_block_indices = []
            env.buffer.visual_highlight_indices = []
            return

        if env.use_simple_blocks:
            previous_blocks = {
                box_type: list(blocks)
                for box_type, blocks in env.buffer.simple_blocks.items()
            }
            try:
                if selected_block is None:
                    env.select_largest_policy_block()
                    if len(env.buffer.all_blocks) == 0:
                        if show_policy_ems:
                            env.ems_list = []
                        env.visual_selected_block = None
                        env.visual_selected_block_indices = []
                        env.buffer.visual_highlight_indices = []
                        return
                    item = env.buffer.sample_blocks(deterministic=True)
                else:
                    item = selected_block
                if item is None:
                    if show_policy_ems:
                        env.ems_list = []
                    env.visual_selected_block = None
                    env.visual_selected_block_indices = []
                    env.buffer.visual_highlight_indices = []
                    return
                if show_policy_ems:
                    env.ems_list = env._get_item_fit_ems_list([item])
                indices = (
                    list(selected_indices)
                    if selected_indices is not None
                    else self._buffer_indices_for_block(env, item)
                )
                env.visual_selected_block = item if highlight_selected else None
                env.visual_selected_block_indices = indices if highlight_selected else []
                env.buffer.visual_highlight_indices = (
                    env.visual_selected_block_indices if highlight_selected else []
                )
            finally:
                env.buffer.simple_blocks = previous_blocks
            return

        item = selected_block if selected_block is not None else env.buffer.sample_item()
        if show_policy_ems:
            env.ems_list = env._get_item_fit_ems_list([item])
        env.visual_selected_block = item if highlight_selected else None
        env.visual_selected_block_indices = (
            list(selected_indices) if selected_indices is not None else [0]
        ) if highlight_selected else []
        env.buffer.visual_highlight_indices = env.visual_selected_block_indices

    @staticmethod
    def _buffer_indices_for_block(env: PackingEnv, block) -> list[int]:
        indices = []
        for index, buffered_dim in enumerate(env.buffer.buffer):
            if buffered_dim == block.box:
                indices.append(index)
                if len(indices) == block.consumed_count:
                    break
        return indices

    def push(self, env: PackingEnv, title: str) -> None:
        self.push_scene(env, title)

    def push_scene(
        self,
        env: PackingEnv,
        title: str,
        *,
        highlight_selected: bool = True,
        selected_block=None,
    ) -> None:
        if self.live_server is None:
            return
        self.live_server.push(
            self.build(
                env,
                title,
                highlight_selected=highlight_selected,
                selected_block=selected_block,
            )
        )
        print(f"updated visual: {title}")
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)

    def push_buffer_selection(self, env: PackingEnv, title: str, selected_block) -> None:
        self.push_scene(
            env,
            f"{title}: Buffer",
            highlight_selected=False,
            selected_block=selected_block,
        )
        self.push_scene(
            env,
            f"{title}: Selected Buffer",
            highlight_selected=True,
            selected_block=selected_block,
        )

    def hold(self) -> None:
        if self.args.hold_visual and self.live_server is not None:
            input("Press Enter to stop the live visualization server...")

    def replay_execution_plan(
        self,
        env: PackingEnv,
        execution_plan,
        seed: int | None = None,
        snapshots: list[tuple] | None = None,
        push_live: bool = True,
        step_prefix: str = "Step",
    ) -> PackingEnv:
        replay_env = deepcopy(env)
        for idx, step in enumerate(execution_plan.steps, start=1):
            if isinstance(step, UnpackOperation):
                item_to_unpack = replay_env.find_placed_item(step.item)
                replay_env.unpack(item_to_unpack)
                replay_env.validate_packing_state()
                title = f"{step_prefix} {idx}: Unpack - seed {seed}"
                self._emit_visual_state(replay_env, title, snapshots, push_live)
                continue

            if isinstance(step, RepackOperation):
                item_to_unpack = replay_env.find_placed_item(step.source_item)
                replay_env.unpack(item_to_unpack)
                replay_env.remove_holding_item(step.source_item)
                replay_env.pack(
                    deepcopy(step.packed_box),
                    selected_ems=self._selected_ems_for_env(replay_env, step.selected_ems),
                )
                replay_env.validate_packing_state()
                title = f"{step_prefix} {idx}: Repack - seed {seed}"
                self._emit_visual_state(replay_env, title, snapshots, push_live)
                continue

            if isinstance(step, PackOperation):
                pruning_item_types = None
                if step.source_item is not None:
                    pruning_item_types = replay_env._ems_pruning_item_types_after_holding_remove(
                        step.source_item
                    )
                elif replay_env.buffer.has_items:
                    pruning_item_types = replay_env._ems_pruning_item_types_after_buffer_update(
                        replay_env.buffer.sample_item()
                    )
                replay_env.pack(
                    deepcopy(step.packed_box),
                    selected_ems=self._selected_ems_for_env(replay_env, step.selected_ems),
                    pruning_item_types=pruning_item_types,
                )
                if step.source_item is not None:
                    replay_env.remove_holding_item(step.source_item)
                elif replay_env.buffer.has_items:
                    replay_env.buffer.update(replay_env.buffer.sample_item())
                replay_env.validate_packing_state()
                title = f"{step_prefix} {idx}: Pack - seed {seed}"
                self._emit_visual_state(replay_env, title, snapshots, push_live)
                continue

            raise TypeError(f"Unsupported execution step: {step!r}")

        return replay_env

    def save_replay(
        self,
        seed: int,
        snapshots: list[tuple],
        mcts_used: bool,
    ) -> None:
        if not self.args.save_replay:
            return
        replay_path = self.args.replay_path
        if replay_path is None:
            replay_path = self._default_replay_path(seed, mcts_used)
        elif os.path.splitext(replay_path)[1].lower() != ".html":
            replay_path = os.path.join(
                replay_path,
                self._default_replay_filename(seed, mcts_used),
            )
        recorder = InteractiveReplayRecorder(
            out_path=replay_path,
            interval_ms=self.args.replay_interval_ms,
        )
        for idx, (env, title, *metadata) in enumerate(snapshots):
            options = metadata[0] if metadata and isinstance(metadata[0], dict) else {}
            recorder.capture(
                title,
                self.build(
                    env,
                    title,
                    reset_history=idx == 0,
                    highlight_selected=bool(options.get("highlight_selected", False)),
                    selected_block=options.get("selected_block"),
                    selected_indices=options.get("selected_indices"),
                ),
            )
        recorder.save()

    def save_sequence_replay(
        self,
        seed: int,
        initial_buffer: list[tuple[int, int, int]],
        pack_history: list[tuple],
        mcts_used: bool,
        execution_plan=None,
        step_prefix: str = "Step",
        start_from_blocked: bool = False,
    ) -> None:
        if not self.args.save_replay:
            return
        snapshots = []
        replay_env = PackingEnv(
            ds_name=self.args.ds_name,
            container_size=tuple(self.args.container_size),
            item_buffer_space=getattr(self.args, "buffer_space", 0),
            remove_inscribed_ems=getattr(self.args, "remove_inscribed_ems", False),
            stack_only=getattr(self.args, "stack_only", False),
            use_simple_blocks=getattr(self.args, "use_simple_blocks", False),
        )
        replay_env.reset(seed=seed)
        replay_env.buffer.buffer = [Orthogonal3D(*dims) for dims in initial_buffer]
        if not start_from_blocked:
            snapshots.append((
                deepcopy(replay_env),
                f"Initial State - seed {seed}",
                {"highlight_selected": False},
            ))

        for step, pack_step in enumerate(pack_history, start=1):
            if len(pack_step) == 7:
                (
                    box_pos,
                    box_dims,
                    box_rot,
                    buffer_before,
                    selected_block,
                    selected_indices,
                    buffer_after,
                ) = pack_step
            elif len(pack_step) == 6:
                box_pos, box_dims, box_rot, buffer_before, selected_block, buffer_after = pack_step
                selected_indices = None
            else:
                box_pos, box_dims, box_rot, buffer_after = pack_step
                buffer_before = None
                selected_block = None
                selected_indices = None
            box = Item(
                FLB=Point3D(*box_pos),
                Dim=Orthogonal3D(*box_dims),
                buffer_space=getattr(self.args, "buffer_space", 0),
            )
            box.rot = box_rot
            if selected_block is not None:
                box.source_item_count = int(getattr(selected_block, "consumed_count", 1))
            if not start_from_blocked and buffer_before is not None:
                replay_env.buffer.buffer = [Orthogonal3D(*dims) for dims in buffer_before]
                snapshots.append((
                    deepcopy(replay_env),
                    f"Pack Step {step}: Buffer - seed {seed}",
                    {"highlight_selected": False},
                ))
                snapshots.append((
                    deepcopy(replay_env),
                    f"Pack Step {step}: Selected Buffer - seed {seed}",
                    {
                        "highlight_selected": selected_block is not None,
                        "selected_block": selected_block,
                        "selected_indices": selected_indices,
                    },
                ))
            replay_env.pack(
                box,
                pruning_item_types=[Orthogonal3D(*dims) for dims in buffer_after],
            )
            replay_env.buffer.buffer = [Orthogonal3D(*dims) for dims in buffer_after]
            if not start_from_blocked and buffer_before is None:
                snapshots.append((
                    deepcopy(replay_env),
                    f"Pack Step {step}: Packed - seed {seed}",
                    {"highlight_selected": False},
                ))

        snapshots.append(
            (
                deepcopy(replay_env),
                f"Blocked Before MCTS at Pack Step {len(pack_history)} - seed {seed}",
                {"highlight_selected": False},
            )
        )

        if execution_plan is not None:
            self.replay_execution_plan(
                replay_env,
                execution_plan,
                seed=seed,
                snapshots=snapshots,
                push_live=False,
                step_prefix=step_prefix,
            )

        self.save_replay(seed, snapshots, mcts_used=mcts_used)

    def _emit_visual_state(
        self,
        env: PackingEnv,
        title: str,
        snapshots: list[tuple] | None,
        push_live: bool,
    ) -> None:
        if snapshots is not None:
            snapshots.append((deepcopy(env), title, {"highlight_selected": False}))
        if push_live:
            self.push(env, title)

    @staticmethod
    def _selected_ems_for_env(env: PackingEnv, selected_ems):
        if selected_ems is None:
            return None
        for ems in env.heu_ems.get_ems_list():
            if ems == selected_ems:
                return ems
        return None

    def _default_replay_path(self, seed: int, mcts_used: bool) -> str:
        return os.path.join(
            self.args.visual_dir,
            self._default_replay_filename(seed, mcts_used),
        )

    def _default_replay_filename(self, seed: int, mcts_used: bool) -> str:
        return (
            f"run_seed_{seed}_"
            f"mcts_{str(bool(mcts_used)).lower()}_"
            f"optimize_{str(bool(self.args.optimize_sequence)).lower()}.html"
        )
