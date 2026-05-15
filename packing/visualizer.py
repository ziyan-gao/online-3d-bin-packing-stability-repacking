import os
import time
from copy import deepcopy

from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item
from packing_env.gym_env import PackingEnv
from packing_env.visualization import PackVisualizer

from .interactive_replay import InteractiveReplayRecorder
from .live_plot import LivePlotServer, make_live_server
from .plans import PackOperation, RepackOperation, UnpackOperation


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
        self._pack_visualizer: PackVisualizer | None = None

    @classmethod
    def from_args(cls, args) -> "Visualizer":
        live_server = make_live_server(args) if args.visualize else None
        return cls(args, live_server=live_server)

    def build(self, env: PackingEnv, title: str, reset_history: bool = False):
        if self._pack_visualizer is None:
            self._pack_visualizer = PackVisualizer(
                env,
                title=title,
                show_anchor=True,
                show_ems=False,
            )
        self._pack_visualizer.env = env
        self._pack_visualizer.title = title
        if reset_history:
            self._pack_visualizer.reset_history()
        fig, _ = self._pack_visualizer.refresh()
        self._set_visual_z_range(fig)
        return fig

    def push(self, env: PackingEnv, title: str) -> None:
        if self.live_server is None:
            return
        self.live_server.push(self.build(env, title))
        print(f"updated visual: {title}")
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)

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
                replay_env.pack(
                    deepcopy(step.packed_box),
                    selected_ems=self._selected_ems_for_env(replay_env, step.selected_ems),
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
        for idx, (env, title, *_) in enumerate(snapshots):
            recorder.capture(title, self.build(env, title, reset_history=idx == 0))
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
        )
        replay_env.reset(seed=seed)
        replay_env.buffer.buffer = [Orthogonal3D(*dims) for dims in initial_buffer]
        if not start_from_blocked:
            snapshots.append((deepcopy(replay_env), f"Initial State - seed {seed}"))

        for step, (box_pos, box_dims, box_rot, buffer_after) in enumerate(pack_history, start=1):
            box = Item(
                FLB=Point3D(*box_pos),
                Dim=Orthogonal3D(*box_dims),
                buffer_space=getattr(self.args, "buffer_space", 0),
            )
            box.rot = box_rot
            replay_env.pack(box)
            replay_env.buffer.buffer = [Orthogonal3D(*dims) for dims in buffer_after]
            if not start_from_blocked:
                snapshots.append((deepcopy(replay_env), f"Pack Step {step} - seed {seed}"))

        snapshots.append(
            (
                deepcopy(replay_env),
                f"Blocked Before MCTS at Pack Step {len(pack_history)} - seed {seed}",
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
            snapshots.append((deepcopy(env), title))
        if push_live:
            self.push(env, title)

    def _set_visual_z_range(self, fig) -> None:
        fig.update_layout(
            scene2=dict(zaxis=dict(range=[0, self.visual_z_max])),
            scene3=dict(zaxis=dict(range=[0, self.visual_z_max])),
        )

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
