from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CascadedCritic(nn.Module):
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
        self.value_head = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, 1),
        )
        self.to(device=self.device, dtype=self.dtype)

    @property
    def num_param(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, obs: Any, state: Any = None, info: Any | None = None) -> torch.Tensor:
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
        ems_mask = loading_mask.any(dim=1)

        block_embed = self.block_encoder(blocks)
        ems_embed = self.ems_encoder(ems)
        block_pool = (
            block_embed * block_mask.unsqueeze(-1).to(dtype=block_embed.dtype)
        ).sum(dim=1)
        block_pool = block_pool / block_mask.sum(dim=1, keepdim=True).clamp(min=1)
        ems_pool = (
            ems_embed * ems_mask.unsqueeze(-1).to(dtype=ems_embed.dtype)
        ).sum(dim=1)
        ems_pool = ems_pool / ems_mask.sum(dim=1, keepdim=True).clamp(min=1)
        value = self.value_head(torch.cat([block_pool, ems_pool], dim=-1))
        return value.squeeze(-1)
