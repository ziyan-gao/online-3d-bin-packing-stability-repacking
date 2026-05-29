import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.cascaded_actor import CascadedActor
from model.cascaded_critic import CascadedCritic
from model.cascaded_policy import CascadedCategoricalMasked
from packing.agents import PackingAgent
from packing.policy_loader import build_net, load_checkpoint_state_dict
from packing.train_utils import (
    PROJECT_ROOT,
    TrainConfig,
    load_train_config,
    load_training_checkpoint,
    make_training_callbacks,
    select_training_device,
    training_checkpoint_metadata,
)
from packing.visualizer import Visualizer
import packing.test_utils as test_utils
from packing.test_utils import (
    DEFAULT_TEST_CONFIG,
    PROJECT_ROOT as TEST_PROJECT_ROOT,
    accumulate_step_reward,
    build_agent,
    build_env,
    load_test_config,
    reject_unsupported_cascaded_mcts,
)
from packing_env.data_type.buffer import Buffer
from packing_env.data_type.geometry import Orthogonal3D
from packing_env.data_type.oriented_block import OrientedBlock
from packing_env.gym_env import PackingEnv


class LocalFakeSampler:
    is_random_distribution = False

    def __init__(self, items):
        self.items = list(items)
        self.cursor = 0

    def sample(self, n):
        sampled = []
        for _ in range(n):
            sampled.append(self.items[self.cursor % len(self.items)])
            self.cursor += 1
        return sampled


def make_cascaded_obs():
    return SimpleNamespace(
        oriented_blocks=torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3, 0.11, 0.21, 0.3, 0.25, 0.0],
                    [0.2, 0.1, 0.3, 0.21, 0.11, 0.3, 0.25, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
        ems=torch.zeros((1, 4, 6), dtype=torch.float32),
        block_mask=torch.tensor([[True, True, False, False]]),
        loading_mask=torch.tensor(
            [
                [
                    [True, False, False, False],
                    [False, True, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            ]
        ),
        action_mask=torch.tensor(
            [
                [
                    [True, False, False, False],
                    [False, True, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            ]
        ),
    )


def test_train_config_accepts_cascaded_policy_mode():
    config = TrainConfig(policy_mode="cascaded_block_selector")

    assert config.policy_mode == "cascaded_block_selector"


def test_train_config_accepts_layered_achievability_fields():
    config = TrainConfig(
        use_simple_blocks=True,
        stack_only=True,
        layered_achievability=True,
        layered_num_chunks=4,
    )

    assert config.layered_achievability is True
    assert config.layered_num_chunks == 4


def test_train_config_rejects_non_positive_layered_num_chunks():
    with pytest.raises(ValueError, match="layered_num_chunks"):
        TrainConfig(layered_num_chunks=0)


def test_train_config_rejects_non_integer_layered_num_chunks():
    with pytest.raises(ValueError, match="layered_num_chunks"):
        TrainConfig(layered_num_chunks=2.5)


def test_training_checkpoint_metadata_includes_layered_achievability_fields():
    config = TrainConfig(
        use_simple_blocks=True,
        stack_only=True,
        layered_achievability=True,
        layered_num_chunks=4,
    )

    metadata = training_checkpoint_metadata(config)

    assert metadata["layered_achievability"] is True
    assert metadata["layered_num_chunks"] == 4


def test_train_config_rejects_unknown_policy_mode():
    with pytest.raises(ValueError, match="policy_mode"):
        TrainConfig(policy_mode="not_a_policy")


def test_train_cj_default_enables_block_baseline_defaults():
    config = load_train_config(str(Path(PROJECT_ROOT) / "configs/train_cj_default.yaml"))

    assert config.data_name == "CJ"
    assert (config.container_dx, config.container_dy, config.container_dz) == (
        1200,
        1000,
        1350,
    )
    assert config.stack_only
    assert config.use_simple_blocks
    assert config.policy_mode == "largest_block_baseline"
    assert config.output_name == "baseline-blocks"
    assert config.train_env_num == 64
    assert config.test_env_num == 32
    assert config.step_per_collect == 1000
    assert config.episode_per_test == 128


def test_test_config_accepts_default_policy_mode():
    config = load_test_config(DEFAULT_TEST_CONFIG)

    assert config.policy_mode == "cascaded_block_selector"


def test_test_config_stores_cascaded_policy_mode():
    config = test_utils.TestConfig(policy_mode="cascaded_block_selector")

    assert config.policy_mode == "cascaded_block_selector"


def test_test_config_accepts_layered_achievability_fields():
    config = test_utils.TestConfig(
        use_simple_blocks=True,
        stack_only=True,
        layered_achievability=True,
        layered_num_chunks=5,
    )

    assert config.layered_achievability is True
    assert config.layered_num_chunks == 5


def test_test_config_rejects_non_positive_layered_num_chunks():
    with pytest.raises(ValueError, match="layered_num_chunks"):
        test_utils.TestConfig(layered_num_chunks=0)


def test_test_config_rejects_non_integer_layered_num_chunks():
    with pytest.raises(ValueError, match="layered_num_chunks"):
        test_utils.TestConfig(layered_num_chunks=2.5)


def test_layered_achievability_constructor_initializes_state():
    env = PackingEnv(
        layered_achievability=True,
        layered_num_chunks=4,
        use_simple_blocks=True,
        stack_only=True,
        policy_mode="largest_block_baseline",
    )

    assert env.layered_achievability is True
    assert env.layered_num_chunks == 4
    assert env.layered_stage == 1
    assert env._policy_ems_source_by_id == {}


def test_layered_achievability_requires_simple_block_baseline():
    with pytest.raises(ValueError, match="largest_block_baseline"):
        PackingEnv(
            layered_achievability=True,
            use_simple_blocks=True,
            policy_mode="cascaded_block_selector",
        )

    with pytest.raises(ValueError, match="use_simple_blocks"):
        PackingEnv(
            layered_achievability=True,
            use_simple_blocks=False,
            policy_mode="largest_block_baseline",
        )


def test_layered_num_chunks_must_be_positive():
    with pytest.raises(ValueError, match="layered_num_chunks"):
        PackingEnv(
            layered_achievability=True,
            layered_num_chunks=0,
            use_simple_blocks=True,
            policy_mode="largest_block_baseline",
        )


def test_layered_num_chunks_must_be_integer():
    with pytest.raises(ValueError, match="layered_num_chunks"):
        PackingEnv(
            layered_achievability=True,
            layered_num_chunks=2.5,
            use_simple_blocks=True,
            policy_mode="largest_block_baseline",
        )


def test_test_cj_default_uses_baseline_block_checkpoint():
    config = load_test_config(str(Path(TEST_PROJECT_ROOT) / "configs/test_cj_default.yaml"))

    assert config.checkpoint == "outputs/train_outputs/baseline-blocks/policy_step.pth"
    assert config.ds_name == "CJ"
    assert config.stack_only
    assert config.use_simple_blocks
    assert config.policy_mode == "largest_block_baseline"
    assert config.use_mcts is False


def test_test_cj_cascade_uses_cascaded_checkpoint():
    config = load_test_config(str(Path(TEST_PROJECT_ROOT) / "configs/test_cj_cascade.yaml"))

    assert config.checkpoint == (
        "outputs/train_outputs/cascaded-block-selector/policy_step.pth"
    )
    assert config.ds_name == "CJ"
    assert config.stack_only
    assert config.use_simple_blocks
    assert config.policy_mode == "cascaded_block_selector"
    assert config.use_mcts is False


def test_accumulate_step_reward_prints_step_and_total(capsys):
    reward_total = accumulate_step_reward(
        reward_total=0.25,
        reward=0.125,
        step=2,
        utilization=0.5,
        placed_count=4,
    )

    captured = capsys.readouterr()
    assert reward_total == pytest.approx(0.375)
    assert "policy step=2" in captured.out
    assert "reward=0.125000" in captured.out
    assert "episode_reward=0.375000" in captured.out


def test_checkpoint_policy_mode_mismatch_is_rejected(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch = pytest.importorskip("torch")
    torch.save(
        {
            "model": {},
            "optim": {},
            "epoch": 1,
            "env_step": 2,
            "gradient_step": 3,
            "data_name": "random",
            "container_size": (600, 600, 600),
            "buffer_size": 12,
            "k_placement": 80,
            "remove_inscribed_ems": False,
            "stack_only": True,
            "use_simple_blocks": True,
            "policy_mode": "largest_block_baseline",
        },
        checkpoint_path,
    )

    class DummyPolicy:
        def load_state_dict(self, state):
            pass

    class DummyOptim:
        def load_state_dict(self, state):
            pass

    config = TrainConfig(
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    with pytest.raises(ValueError, match="policy_mode"):
        load_training_checkpoint(
            str(checkpoint_path),
            DummyPolicy(),
            DummyOptim(),
            config,
            "cpu",
        )


def test_training_checkpoint_accepts_legacy_metadata_without_layered_fields(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model": {},
            "optim": {},
            "epoch": 1,
            "env_step": 2,
            "gradient_step": 3,
            "data_name": "random",
            "container_size": (600, 600, 600),
            "buffer_size": 12,
            "k_placement": 80,
            "remove_inscribed_ems": False,
            "stack_only": True,
            "use_simple_blocks": True,
            "policy_mode": "largest_block_baseline",
        },
        checkpoint_path,
    )

    class DummyPolicy:
        def load_state_dict(self, state):
            self.state = state

    class DummyOptim:
        def load_state_dict(self, state):
            self.state = state

    config = TrainConfig(
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="largest_block_baseline",
    )
    checkpoint = load_training_checkpoint(
        str(checkpoint_path),
        DummyPolicy(),
        DummyOptim(),
        config,
        "cpu",
    )

    assert "layered_achievability" not in checkpoint
    assert "layered_num_chunks" not in checkpoint


def test_policy_checkpoint_policy_mode_mismatch_is_rejected(tmp_path):
    checkpoint_path = tmp_path / "policy_step.pth"
    torch.save(
        {
            "model": {},
            "policy_mode": "largest_block_baseline",
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="policy_mode"):
        load_checkpoint_state_dict(
            str(checkpoint_path),
            "cpu",
            expected_metadata={"policy_mode": "cascaded_block_selector"},
        )


def test_best_policy_checkpoint_carries_policy_metadata(tmp_path):
    class DummyPolicy:
        def state_dict(self):
            return {
                "actor.weight": torch.tensor([1.0]),
                "critic.weight": torch.tensor([2.0]),
            }

    class DummyOptimizer:
        def state_dict(self):
            return {"lr": 0.001}

    config = TrainConfig(
        data_name="random",
        container_dx=600,
        container_dy=600,
        container_dz=600,
        buffer_size=12,
        k_placement=80,
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    _, save_best_fn, _ = make_training_callbacks(
        config,
        str(tmp_path),
        policy=DummyPolicy(),
        optimizer=DummyOptimizer(),
    )

    save_best_fn(DummyPolicy())

    checkpoint = torch.load(
        tmp_path / "policy_step.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["model"]["actor.weight"].item() == 1.0
    assert checkpoint["policy_mode"] == "cascaded_block_selector"
    assert checkpoint["container_size"] == (600, 600, 600)
    assert checkpoint["buffer_size"] == 12


def test_periodic_training_checkpoint_captures_policy_state(tmp_path):
    class DummyPolicy:
        def state_dict(self):
            return {
                "actor.weight": torch.tensor([3.0]),
                "critic.weight": torch.tensor([4.0]),
            }

    class DummyOptimizer:
        def state_dict(self):
            return {"lr": 0.002}

    config = TrainConfig(
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    _, _, save_checkpoint_fn = make_training_callbacks(
        config,
        str(tmp_path),
        policy=DummyPolicy(),
        optimizer=DummyOptimizer(),
    )

    ckpt_path = save_checkpoint_fn(epoch=1, env_step=2, gradient_step=3)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["model"]["actor.weight"].item() == 3.0
    assert checkpoint["optim"] == {"lr": 0.002}
    assert checkpoint["epoch"] == 1
    assert checkpoint["env_step"] == 2
    assert checkpoint["gradient_step"] == 3
    assert checkpoint["policy_mode"] == "cascaded_block_selector"


def test_select_training_device_uses_runtime_compatibility_guard(monkeypatch):
    monkeypatch.setattr(
        "packing.train_utils.resolve_runtime_device",
        lambda requested_device=None: torch.device("cpu"),
    )

    assert select_training_device() == "cpu"


def test_cascaded_actor_outputs_flat_logits_and_masks():
    actor = CascadedActor(
        block_feature_dim=8,
        ems_feature_dim=6,
        embed_size=32,
        device=torch.device("cpu"),
    )
    obs = make_cascaded_obs()

    out, _ = actor(obs)

    assert out.logits.shape == (1, 16)
    assert out.action_mask.shape == (1, 4, 4)
    assert out.block_logits.shape == (1, 4)


def test_cascaded_distribution_masks_invalid_flat_actions():
    actor = CascadedActor(
        block_feature_dim=8,
        ems_feature_dim=6,
        embed_size=32,
        device=torch.device("cpu"),
    )
    out, _ = actor(make_cascaded_obs())
    dist = CascadedCategoricalMasked(out)

    assert dist.probs.shape == (1, 16)
    assert torch.isclose(
        dist.probs[0, 0] + dist.probs[0, 5],
        torch.tensor(1.0),
        atol=1e-5,
    )
    assert torch.all(dist.probs[0, [1, 2, 3, 4, 6, 7]] == 0)


def test_cascaded_critic_returns_batch_value():
    critic = CascadedCritic(
        block_feature_dim=8,
        ems_feature_dim=6,
        embed_size=32,
        device=torch.device("cpu"),
    )

    value = critic(make_cascaded_obs())

    assert value.shape == (1,)


def test_build_net_returns_cascaded_models_for_cascaded_policy_mode():
    actor, critic = build_net(device="cpu", policy_mode="cascaded_block_selector")

    assert isinstance(actor, CascadedActor)
    assert isinstance(critic, CascadedCritic)


def test_test_utils_build_env_propagates_cascaded_policy_mode():
    config = test_utils.TestConfig(
        ds_name="random",
        container_size=(600, 600, 600),
        buffer_space=0,
        remove_inscribed_ems=False,
        stack_only=False,
        use_simple_blocks=False,
        policy_mode="cascaded_block_selector",
    )

    env = build_env(config, seed=1)

    assert env.policy_mode == "cascaded_block_selector"
    assert env.layered_achievability is False
    assert env.layered_num_chunks == 3


def test_test_utils_build_agent_propagates_cascaded_policy_mode(monkeypatch):
    created = {}

    class RecordingAgent:
        device = "cpu"

        def __init__(self, *, device, checkpoint_path, policy_mode):
            created["device"] = device
            created["checkpoint_path"] = checkpoint_path
            created["policy_mode"] = policy_mode

    monkeypatch.setattr(test_utils, "PackingAgent", RecordingAgent)
    config = SimpleNamespace(
        device="cpu",
        checkpoint="checkpoint.pth",
        policy_mode="cascaded_block_selector",
    )

    agent = build_agent(config)

    assert isinstance(agent, RecordingAgent)
    assert created == {
        "device": "cpu",
        "checkpoint_path": "checkpoint.pth",
        "policy_mode": "cascaded_block_selector",
    }


def test_cascaded_mcts_validation_is_rejected_when_target_not_reached():
    config = SimpleNamespace(
        policy_mode="cascaded_block_selector",
        use_mcts=True,
    )

    with pytest.raises(ValueError, match="cascaded_block_selector.*use_mcts"):
        reject_unsupported_cascaded_mcts(config, target_reached=False)


def test_cascaded_no_mcts_validation_is_allowed_when_target_not_reached():
    config = SimpleNamespace(
        policy_mode="cascaded_block_selector",
        use_mcts=False,
    )

    reject_unsupported_cascaded_mcts(config, target_reached=False)


def test_cascaded_env_step_can_use_flat_policy_action():
    box = Orthogonal3D(100, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=3,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.reset(seed=1)
    env.buffer = Buffer(
        capacity=3,
        data_sampler=LocalFakeSampler([box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    obs = env.get_next_observation()
    action = int(np.flatnonzero(obs["action_mask"].reshape(-1))[0])

    _, reward, _, _, info = env.step(action)

    assert reward > 0
    assert info["selected_stack_height"] >= 1


def test_cascaded_agent_predict_accepts_unbatched_env_observation():
    box = Orthogonal3D(100, 100, 50)
    env = PackingEnv(
        k_placement=4,
        buffer_capacity=3,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.reset(seed=1)
    env.buffer = Buffer(
        capacity=3,
        data_sampler=LocalFakeSampler([box, box, box]),
        stack_only=True,
        container_size=(600, 600, 600),
    )
    obs = env.get_next_observation()
    agent = PackingAgent(
        device="cpu",
        k_placement=4,
        policy_mode="cascaded_block_selector",
    )

    action = agent.predict(obs)

    assert isinstance(action, int)
    assert obs["action_mask"].reshape(-1)[action]


def test_cascaded_observation_masks_padded_ems_columns(monkeypatch):
    box = Orthogonal3D(100, 100, 50)
    env = PackingEnv(
        k_placement=8,
        buffer_capacity=3,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.reset(seed=1)
    real_ems = env.heu_ems.get_all_ems()[:1]
    block = env.buffer.all_blocks[0]
    oriented = OrientedBlock.from_block(
        block,
        source_index=0,
        rotate_xy=False,
    )
    loading_rows = np.zeros((1, env.k_placement), dtype=bool)
    loading_rows[0, 5] = True

    monkeypatch.setattr(
        env,
        "get_cascaded_block_candidates",
        lambda: ([oriented], real_ems, loading_rows),
    )

    obs = env.get_next_observation()

    assert len(env.ems_list) == 1
    assert not obs["action_mask"][:, 1:].any()
    assert env.done


def test_policy_ems_visualization_preserves_cascaded_ems_list(monkeypatch):
    env = PackingEnv(
        k_placement=8,
        buffer_capacity=3,
        container_size=(600, 600, 600),
        stack_only=True,
        use_simple_blocks=True,
        policy_mode="cascaded_block_selector",
    )
    env.reset(seed=1)
    obs = env.get_next_observation()
    action = int(np.flatnonzero(obs["action_mask"].reshape(-1))[0])
    before_ems = list(env.ems_list)

    def fail_baseline_refresh():
        raise AssertionError("cascaded visualization must not refresh baseline EMS")

    monkeypatch.setattr(env, "select_largest_policy_block", fail_baseline_refresh)
    visualizer = Visualizer(
        SimpleNamespace(
            visual_z_max=610.0,
            visual_delay_sec=0.0,
            show_ems=True,
            ems_visual_mode="policy",
        )
    )

    visualizer.build(env, "cascaded policy EMS")

    assert env.ems_list == before_ems
    env.decode_cascaded_action(action)
