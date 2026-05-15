from __future__ import annotations
import torch
from torch import nn
from torch import as_tensor, Tensor, device
from typing import Any
from .blocks.projection import Linear
from .blocks.ems_embed import _init_orthogonal_weight_and_bias
from .space_embed import SpaceEmbed
from .packing_transformer import PackingTransformer
from dataclasses import dataclass

class Actor(nn.Module):

    space_embed: SpaceEmbed
    pack_transform:PackingTransformer
    ems_layer: nn.Module
    item_layer: nn.Module

    def __init__(
        self,
        space_embed:SpaceEmbed, 
        pack_transform:PackingTransformer,
        embed_size: int = 128,
        device=device,
        dtype=torch.float32
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.space_embed = space_embed.to(self.device)
        self.pack_transform = pack_transform.to(self.device)
        self.ems_layer = nn.Sequential(
            Linear(
                embed_size,
                embed_size,
                bias=True,
                init_fn=_init_orthogonal_weight_and_bias,
                device=device,
                dtype=dtype,
            ),
            nn.LeakyReLU(),
        ).to(self.device)
        self.item_layer = nn.Sequential(
            Linear(
                embed_size,
                embed_size,
                bias=True,
                init_fn=_init_orthogonal_weight_and_bias,
                device=device,
                dtype=dtype,
            ),
            nn.LeakyReLU(),
        ).to(self.device)
        self.data_tf = lambda x:as_tensor(x, dtype=self.dtype, device=self.device)
        self.mask_tf = lambda x:as_tensor(x, dtype=torch.bool, device=self.device)

    @property
    def num_param(self):
        c = 0
        c +=  sum(p.numel() for p in self.ems_layer.parameters())
        c +=  sum(p.numel() for p in self.item_layer.parameters())
        return c + self.space_embed.num_param + self.pack_transform.num_param


    def forward(
        self,
        obs: Any,
        state: Any = None,
        info={}
    ):
        ems, new_item, action_mask = list(map(self.data_tf, 
                                 [obs.ems, 
                                  obs.new_item,
                                  obs.action_mask
                                  ]))
        # batch_size = ems.shape[0]
        batch_size = ems.shape[0]
        mask = self.mask_tf(obs.action_mask)
        ems_mask = mask.sum(1).bool()
        new_item = torch.cat([new_item[:, None,:], new_item[:,None, [1,0,2]]], 1)
        ems = ems.view(-1, action_mask.shape[2], 6)
        # if not isinstance(obs.action_mask, Tensor):
            # action_mask = as_tensor(obs.action_mask, dtype=self.dtype, device=self.device)
        #ems:(batch, max_ems_len, embed_dim), item:(batch, embed_dim)
        space_embed_out = self.space_embed(ems, new_item)
        attn = self.pack_transform(space_embed_out, ems_mask)
        item_embed = self.item_layer(attn.crs_attn.item)  # [B, T1, C]
        ems_embed = self.ems_layer(attn.crs_attn.ems).permute(0, 2, 1)  # [B, C, T2]
        # logits = torch.bmm(item_embed, ems_embed).view(batch_size, -1)  # [B, T1, T2]
        logits = torch.bmm(item_embed, ems_embed).reshape(batch_size, -1)
        # Check for NaN values
        # if torch.isnan(logits).any():
        #     print("Tensor contains NaN values.")
        return ActorOutput(logits=logits, action_mask=action_mask), state

@dataclass
class ActorOutput:

    logits: torch.Tensor
    """ logits """

    action_mask:torch.Tensor
    """ log_prob """
