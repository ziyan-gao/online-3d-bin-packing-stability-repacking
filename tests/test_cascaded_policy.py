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
from packing.policy_loader import build_net
from packing.train_utils import TrainConfig, load_training_checkpoint
from packing.test_utils import DEFAULT_TEST_CONFIG, load_test_config
from packing_env.data_type.buffer import Buffer
from packing_env.data_type.geometry import Orthogonal3D
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


def test_train_config_rejects_unknown_policy_mode():
    with pytest.raises(ValueError, match="policy_mode"):
        TrainConfig(policy_mode="not_a_policy")


def test_test_config_accepts_default_policy_mode():
    config = load_test_config(DEFAULT_TEST_CONFIG)

    assert config.policy_mode == "largest_block_baseline"


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
