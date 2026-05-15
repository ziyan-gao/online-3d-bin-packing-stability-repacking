from __future__ import annotations
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .blocks.ems_embed import EMSEmbed
from .blocks.item_embed import ItemEmbed
from dataclasses import dataclass

@dataclass
class SpaceEmbedOutput:
    'output of ems_embed of space embed'
    ems:Tensor
    'output of item_embed of space embed'
    item:Tensor

class SpaceEmbed(nn.Module):

    ems_encoder: EMSEmbed
    item_encoder: ItemEmbed

    def __init__(
        self,
        embed_dim=128,
    ):
        super().__init__()
        self.ems_embed = EMSEmbed(
            input_dim=6,
            embed_dim=embed_dim,
        )

        self.item_embed = ItemEmbed(
            input_dim=3,
            embed_dim=embed_dim,
        )

    @property
    def num_param(self):
        c = 0
        c += sum(p.numel() for p in self.ems_embed.mlp.parameters())
        c += sum(p.numel() for p in self.item_embed.mlp.parameters())
        return c

    def forward(
        self,
        ems:Optional[Tensor | np.ndarray],
        new_item:Optional[Tensor | np.ndarray]
    ) -> SpaceEmbedOutput:
        
        out_ems = self.ems_embed(ems)
        out_item = self.item_embed(new_item)
        
        return SpaceEmbedOutput(ems=out_ems, item=out_item)
