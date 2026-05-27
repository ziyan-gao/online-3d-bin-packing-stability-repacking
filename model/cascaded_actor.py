from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class CascadedActorOutput:
    logits: torch.Tensor
    action_mask: torch.Tensor
    block_logits: torch.Tensor
    ems_logits: torch.Tensor
    block_mask: torch.Tensor
    loading_mask: torch.Tensor


class CascadedActor(nn.Module):
    def __init__(
        self,
        block_feature_dim: int = 8,
        ems_feature_dim: int = 6,
        embed_size: int = 128,
        device: torch.device | str = torch.device("cpu"),
        dtype=torch.float32,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.block_encoder = nn.Sequential(
            nn.Linear(block_feature_dim, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
        )
        self.ems_encoder = nn.Sequential(
            nn.Linear(ems_feature_dim, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size),
            nn.LeakyReLU(),
        )
        self.block_head = nn.Linear(embed_size, 1)
        self.loading_head = nn.Sequential(
            nn.Linear(embed_size * 3, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, 1),
        )
        self.to(device=self.device, dtype=self.dtype)

    @property
    def num_param(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, obs: Any, state: Any = None, info: Any | None = None):
        blocks = torch.as_tensor(
            obs.oriented_blocks,
            dtype=self.dtype,
            device=self.device,
        )
        ems = torch.as_tensor(obs.ems, dtype=self.dtype, device=self.device)
        block_mask = torch.as_tensor(
            obs.block_mask,
            dtype=torch.bool,
            device=self.device,
        )
        loading_mask = torch.as_tensor(
            obs.loading_mask,
            dtype=torch.bool,
            device=self.device,
        )

        block_embed = self.block_encoder(blocks)
        ems_embed = self.ems_encoder(ems)
        block_logits = self.block_head(block_embed).squeeze(-1)

        block_mask_float = block_mask.unsqueeze(-1).to(dtype=block_embed.dtype)
        block_context = (block_embed * block_mask_float).sum(
            dim=1,
            keepdim=True,
        )
        block_context = block_context / block_mask.sum(
            dim=1,
            keepdim=True,
        ).clamp(min=1).unsqueeze(-1)
        block_context = block_context.expand(-1, block_embed.shape[1], -1)

        block_count = block_embed.shape[1]
        ems_count = ems_embed.shape[1]
        block_pair = block_embed[:, :, None, :].expand(
            -1,
            block_count,
            ems_count,
            -1,
        )
        ems_pair = ems_embed[:, None, :, :].expand(
            -1,
            block_count,
            ems_count,
            -1,
        )
        context_pair = block_context[:, :, None, :].expand(
            -1,
            block_count,
            ems_count,
            -1,
        )
        pair_features = torch.cat([block_pair, ems_pair, context_pair], dim=-1)
        ems_logits = self.loading_head(pair_features).squeeze(-1)

        joint_logits = block_logits[:, :, None] + ems_logits
        action_mask = loading_mask & block_mask[:, :, None]
        flat_logits = joint_logits.reshape(joint_logits.shape[0], -1)
        return (
            CascadedActorOutput(
                logits=flat_logits,
                action_mask=action_mask,
                block_logits=block_logits,
                ems_logits=ems_logits,
                block_mask=block_mask,
                loading_mask=loading_mask,
            ),
            state,
        )
