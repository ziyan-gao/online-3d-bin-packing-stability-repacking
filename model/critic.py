from __future__ import annotations
import numpy as np
import torch
from torch import nn, device
from typing import Any
from torch.nn.functional import log_softmax
from .blocks.projection import Linear
from .blocks.ems_embed import _init_orthogonal_weight_and_bias
from dataclasses import dataclass
from .space_embed import SpaceEmbed
from .packing_transformer import PackingTransformer
from torch import as_tensor, Tensor

class Critic(nn.Module):
    """Actor head for Actor-Critic model

    Args:
        encoder (nn.Module): encoder module
        embed_size (int): embedding size
        device (Union[str, int, torch.device]): device
    """

    space_embed: SpaceEmbed
    pack_transform:PackingTransformer
    ems_layer: nn.Module
    
    def __init__(
        self,
        space_embed: SpaceEmbed,
        pack_transform:PackingTransformer,
        embed_size: int = 128,
        device=device,
        dtype=torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.space_embed = space_embed.to(self.device)
        self.pack_transform = pack_transform.to(self.device)

        self.layer_1 = nn.Sequential(
            Linear(
                embed_size,
                embed_size,
                bias=True,
                init_fn=_init_orthogonal_weight_and_bias,
                device=device,
                dtype=dtype,
            ),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            Linear(
                embed_size,
                embed_size,
                bias=True,
                init_fn=_init_orthogonal_weight_and_bias,
                device=device,
                dtype=dtype,
            ),
            nn.LeakyReLU(),
        )
        self.ems_layer = nn.Sequential(
            Linear(
                embed_size*2,
                embed_size,
                bias=True,
                init_fn=_init_orthogonal_weight_and_bias,
                device=device,
                dtype=dtype,
            ),
            nn.LeakyReLU(),
            Linear(
                embed_size,
                embed_size,
                bias=True,
                init_fn=_init_orthogonal_weight_and_bias,
                device=device,
                dtype=dtype,
            ),
            nn.LeakyReLU(),
            Linear(
                embed_size,
                1,
                bias=True,
                init_fn=_init_orthogonal_weight_and_bias,
            ),
        ).to(self.device)
        self.data_tf = lambda x:as_tensor(x, dtype=self.dtype, device=self.device)
        self.mask_tf = lambda x:as_tensor(x, dtype=torch.bool, device=self.device)


    @property
    def num_param(self):
        c = 0
        c +=  sum(p.numel() for p in self.ems_layer.parameters())
        return c + self.space_embed.num_param + self.pack_transform.num_param

    def forward(
        self,
        obs: Any,
        state: Any = None,
        info={}
    ) -> torch.Tensor:
        ems, new_item, action_mask = list(map(self.data_tf, 
                                 [obs.ems, 
                                  obs.new_item,
                                  obs.action_mask
                                  ]))

        mask = self.mask_tf(obs.action_mask)
        ems_mask = mask.sum(1).bool()
        
        new_item = torch.cat([new_item[:, None,:], new_item[:,None, [1,0,2]]], 1)
        ems = ems.view(-1, action_mask.shape[2], 6)

        space_embed_out = self.space_embed(ems, new_item)
        
        attn = self.pack_transform(space_embed_out, ems_mask=None)
        item_feature = self.layer_1(attn.crs_attn.item.sum(1))
        ems_feature = self.layer_2(torch.sum(attn.crs_attn.ems * ems_mask[:,:,None], dim=1))
        ems_item = torch.cat([ems_feature, item_feature], dim=-1)
        critic = self.ems_layer(ems_item).squeeze()

        return critic
