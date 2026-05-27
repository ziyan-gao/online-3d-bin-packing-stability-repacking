import os
import random
from types import SimpleNamespace

import numpy as np
import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


CHECKPOINT_METADATA_DEFAULTS = {
    "stack_only": False,
    "use_simple_blocks": False,
    "policy_mode": "largest_block_baseline",
}


class CategoricalMasked(torch.distributions.Categorical):
    def __init__(self, actor_output):
        self.device = actor_output.logits.device
        self.masks = actor_output.action_mask.type(torch.BoolTensor).to(self.device)
        self.batch_size = actor_output.logits.shape[0]

        logits = actor_output.logits.clone().float()
        reshaped_masks = self.masks.reshape(self.batch_size, -1)
        all_masked = ~reshaped_masks.any(dim=1)
        if all_masked.any():
            reshaped_masks[all_masked, 0] = True

        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs * reshaped_masks.float()
        probs_sum = probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
        probs = probs / probs_sum
        probs = torch.nan_to_num(probs, nan=1e-10, posinf=1e-10, neginf=1e-10)

        super().__init__(probs=probs, validate_args=False)
        self.reshaped_masks = reshaped_masks

    def entropy(self):
        if len(self.masks) == 0:
            return super().entropy()
        p_log_p = self.probs * torch.log(self.probs.clamp(min=1e-10))
        p_log_p = torch.where(
            self.reshaped_masks,
            p_log_p,
            torch.tensor(0.0, device=self.device),
        )
        return -p_log_p.sum(-1)


def resolve_runtime_device(requested_device: str | None = None):
    if requested_device is None:
        requested_device = os.environ.get("PACKING_DEVICE", "cuda")

    device = torch.device(requested_device)
    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    try:
        dev_index = 0 if device.index is None else device.index
        major, minor = torch.cuda.get_device_capability(dev_index)
        sm_tag = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        if supported_arches and sm_tag not in supported_arches:
            gpu_name = torch.cuda.get_device_name(dev_index)
            print(
                f"CUDA arch mismatch for {gpu_name} ({sm_tag}); "
                f"this PyTorch build supports {sorted(supported_arches)}. "
                "Falling back to CPU."
            )
            return torch.device("cpu")
    except Exception as exc:
        print(
            f"Could not validate CUDA compatibility ({type(exc).__name__}: {exc}). "
            "Falling back to CPU."
        )
        return torch.device("cpu")

    return device


def set_eval_seed(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def extract_prefixed_state_dict(state_dict: dict, prefix: str) -> dict:
    extracted = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not extracted:
        raise ValueError(f"Checkpoint does not contain parameters with prefix {prefix!r}")
    return extracted


def validate_checkpoint_metadata(checkpoint: dict, expected_metadata: dict | None) -> None:
    if not expected_metadata:
        return

    mismatches = {
        key: (checkpoint.get(key, CHECKPOINT_METADATA_DEFAULTS.get(key)), value)
        for key, value in expected_metadata.items()
        if (
            key in checkpoint
            or key in CHECKPOINT_METADATA_DEFAULTS
        )
        and checkpoint.get(key, CHECKPOINT_METADATA_DEFAULTS.get(key)) != value
    }
    if not mismatches:
        return

    details = ", ".join(
        f"{key}: checkpoint={ckpt_value!r}, expected={expected_value!r}"
        for key, (ckpt_value, expected_value) in mismatches.items()
    )
    raise ValueError(
        f"Policy checkpoint configuration does not match requested policy ({details})."
    )


def load_checkpoint_state_dict(
    checkpoint_path: str,
    device,
    expected_metadata: dict | None = None,
):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        validate_checkpoint_metadata(checkpoint, expected_metadata)
        return checkpoint["model"]
    return checkpoint


def build_net(device: str = "cuda", policy_mode: str = "largest_block_baseline"):
    runtime_device = torch.device(device)

    if policy_mode == "cascaded_block_selector":
        from model.cascaded_actor import CascadedActor
        from model.cascaded_critic import CascadedCritic

        actor = CascadedActor(device=runtime_device)
        critic = CascadedCritic(device=runtime_device)
        return actor, critic

    if policy_mode != "largest_block_baseline":
        raise ValueError(f"Unknown policy_mode: {policy_mode!r}")

    from model.actor import Actor
    from model.critic import Critic
    from model.packing_transformer import PackingTransformer
    from model.space_embed import SpaceEmbed

    space_embed = SpaceEmbed(embed_dim=128)
    pack_transform = PackingTransformer(
        embed_dim=128,
        ffn_expansion_factor=2,
        num_heads=4,
        num_layers=3,
    )
    actor = Actor(space_embed, pack_transform, embed_size=128, device=runtime_device)
    critic = Critic(space_embed, pack_transform, embed_size=128, device=runtime_device)
    return actor, critic


def build_policy(
    checkpoint_path: str,
    device: str | None = None,
    k_placement: int = 80,
    policy_mode: str = "largest_block_baseline",
):
    runtime_device = resolve_runtime_device(device)
    actor, critic = build_net(device=str(runtime_device), policy_mode=policy_mode)
    load_policy_weights(
        actor,
        critic,
        checkpoint_path,
        runtime_device,
        expected_metadata={"policy_mode": policy_mode},
    )
    return actor, critic


def load_policy_weights(
    actor,
    critic,
    checkpoint_path: str,
    device,
    expected_metadata: dict | None = None,
) -> None:
    state_dict = load_checkpoint_state_dict(
        checkpoint_path,
        device,
        expected_metadata=expected_metadata,
    )
    actor.load_state_dict(extract_prefixed_state_dict(state_dict, "actor."))
    critic.load_state_dict(extract_prefixed_state_dict(state_dict, "critic."))
    actor.eval()
    critic.eval()


def masked_mode(logits, action_mask):
    batch_size = logits.shape[0]
    flattened_mask = torch.as_tensor(
        action_mask,
        dtype=torch.bool,
        device=logits.device,
    ).reshape(batch_size, -1)
    masked_logits = logits.clone().float()
    masked_logits = masked_logits.masked_fill(~flattened_mask, -torch.inf)

    all_masked = ~flattened_mask.any(dim=1)
    if all_masked.any():
        masked_logits[all_masked, 0] = 0

    return masked_logits.argmax(dim=-1)


def select_action(actor, critic, obs) -> tuple[int, int]:
    data = SimpleNamespace(
        new_item=obs["new_item"],
        ems=obs["ems"],
        action_mask=obs["action_mask"],
    )
    act_out, _ = actor(data)
    _ = critic(data).detach().cpu().numpy().reshape(-1,)
    item_idx = 0
    action_idx = int(masked_mode(act_out.logits, act_out.action_mask)[item_idx])
    return item_idx, action_idx
