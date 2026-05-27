from types import SimpleNamespace

import numpy as np
import torch

from packing_env.data_type.geometry import Orthogonal3D, Point3D
from packing_env.data_type.item import Item

from .policy_loader import (
    CategoricalMasked,
    build_net,
    load_policy_weights,
    resolve_runtime_device,
)


def _idx2pos(idx, candidates, k_placement: int):
    if idx > k_placement - 1:
        idx = idx - k_placement
        rot = True
    else:
        rot = False
    pos = candidates[idx][:3]
    return pos, rot


def _predict_from_outputs(
    act_out,
    predicted_value: np.ndarray,
    obs,
    k_placement: int,
    value_deterministic=True,
    logits_deterministic=True,
    row_offset: int = 0,
):
    placable_mask = np.asarray(obs["mask"]).sum(axis=(1, 2)) != 0
    if not placable_mask.any():
        raise ValueError("No placable item is available in the observation mask.")

    if value_deterministic:
        masked_value = np.where(placable_mask, predicted_value, -np.inf)
        item_idx = int(masked_value.argmax())
    else:
        placable_indices = np.arange(len(placable_mask))[placable_mask]
        probs = _softmax(predicted_value[placable_mask])
        item_idx = int(np.random.choice(placable_indices, p=probs))

    action_dist = CategoricalMasked(act_out)
    global_item_idx = row_offset + item_idx
    if logits_deterministic:
        act_idx = action_dist.mode[global_item_idx]
    else:
        act_idx = action_dist.sample()[global_item_idx]
    act_idx = int(act_idx.detach().cpu().item() if torch.is_tensor(act_idx) else act_idx)

    pack_item = np.asarray(obs["item_raw"][item_idx], dtype=np.int32)
    buffer_space = int(
        np.asarray(
            obs.get("buffer_space", np.zeros(len(obs["item_raw"]), dtype=np.int32))
        ).reshape(-1)[item_idx]
    )
    pos, rot = _idx2pos(act_idx, obs["ems_unnorm"][item_idx].reshape(-1, 6), k_placement)
    pack_item = pack_item[[1, 0, 2]] if rot else pack_item
    box = Item(
        FLB=Point3D(*pos),
        Dim=Orthogonal3D(*map(int, pack_item)),
        buffer_space=buffer_space,
    )
    return box, (item_idx, act_idx, predicted_value[item_idx])


class PackingAgent:
    def __init__(
        self,
        device: str | None = None,
        k_placement: int = 80,
        checkpoint_path: str | None = None,
    ):
        self.device = resolve_runtime_device(device)
        self.actor, self.critic = build_net(device=str(self.device))
        self.k_placement = k_placement
        if checkpoint_path is not None:
            load_policy_weights(self.actor, self.critic, checkpoint_path, self.device)
        else:
            self.actor.eval()
            self.critic.eval()

    def predict(
        self,
        obs,
        value_deterministic=True,
        logits_deterministic=True,
    ):
        data = SimpleNamespace(
            new_item=obs["item"],
            ems=obs["ems"],
            action_mask=obs["mask"],
        )

        with torch.no_grad():
            act_out, _ = self.actor(data)
            predicted_value = self.critic(data).detach().cpu().numpy().reshape(-1)

        return _predict_from_outputs(
            act_out=act_out,
            predicted_value=predicted_value,
            obs=obs,
            k_placement=self.k_placement,
            value_deterministic=value_deterministic,
            logits_deterministic=logits_deterministic,
        )

    def idx2pos(self, idx, candidates):
        return _idx2pos(idx, candidates, self.k_placement)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()
