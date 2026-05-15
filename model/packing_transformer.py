from __future__ import annotations
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .blocks.pack_transform_layer import PackingTransformerEncoderLayer
from .blocks.pack_transform_layer import SelfAttenOutput, CrossAttenOutput
from .blocks.norm_order import TransformerNormOrder
from .blocks.projection import init_bert_projection
from .space_embed import SpaceEmbedOutput
from dataclasses import dataclass
import torch

@dataclass
class PackingTransformerOutput:
    slf_attn:SelfAttenOutput
    crs_attn:CrossAttenOutput

class PackingTransformer(nn.Module):
    layers: nn.ModuleList[PackingTransformerEncoderLayer]
    ems_layer_norm: nn.LayerNorm
    item_layer_norm: nn.LayerNorm

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        ffn_expansion_factor: int = 4,
        dropout_p: float = 0.,
        num_layers: int = 6,
        norm_order: TransformerNormOrder = TransformerNormOrder.PRE,
    ):

        super().__init__()
        self.layers = nn.ModuleList(
            [
                PackingTransformerEncoderLayer(
                    model_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    dropout_p=dropout_p,
                    norm_order=norm_order,
                    proj_init_fn=init_bert_projection,
                )
                for _ in range(num_layers)
            ]
        )

        self.ems_layer_norm = nn.LayerNorm(embed_dim)
        self.item_layer_norm = nn.LayerNorm(embed_dim)

    
    @property
    def num_param(self):
        total_params = sum(p.numel() for p in self.layers.parameters())
        return total_params

    def forward(
        self,
        space_embed:SpaceEmbedOutput,
        ems_mask: Optional[Tensor | np.ndarray] = None,
    ) -> PackingTransformerOutput:
        # ems_mask_modified = ems_mask.clone().detach()
        # invalid_indices =  ems_mask.all(-1)
        # if invalid_indices.sum()!=0:
        #     ems_mask_modified[invalid_indices] = torch.zeros((invalid_indices.sum(), ems_mask.shape[-1]), device=ems_mask.device).bool()
        ems_embed = self.ems_layer_norm(space_embed.ems)
        item_embed = self.item_layer_norm(space_embed.item)
        self_attn_out = self.layers[0].compute_slf_attn(ems_embed, 
                                                        item_embed, 
                                                        ems_mask)
        crs_attn_out = self.layers[0].compute_crs_attn(self_attn_out.ems, 
                                                        self_attn_out.item,
                                                        ems_mask)
        for layer in self.layers[1:]:
            self_attn_out = layer.compute_slf_attn(crs_attn_out.ems, 
                                                    crs_attn_out.item, 
                                                            ems_mask)
            crs_attn_out = layer.compute_crs_attn(self_attn_out.ems, 
                                                self_attn_out.item,
                                                ems_mask)

        
        return PackingTransformerOutput(slf_attn=self_attn_out,
                                        crs_attn=crs_attn_out)


    # def forward(
    #     self,
    #     space_embed:SpaceEmbedOutput,
    #     ems_mask: Optional[Tensor | np.ndarray] = None,
    # ) -> PackingTransformerOutput:
    #     # ems_mask_modified = ems_mask.clone().detach()
    #     # invalid_indices =  ems_mask.all(-1)
    #     # if invalid_indices.sum()!=0:
    #     #     ems_mask_modified[invalid_indices] = torch.zeros((invalid_indices.sum(), ems_mask.shape[-1]), device=ems_mask.device).bool()
    #     ems_embed = self.ems_layer_norm(space_embed.ems)
    #     item_embed = self.item_layer_norm(space_embed.item)
    #     self_attn_out = self.layers[0].compute_slf_attn(ems_embed, 
    #                                                     item_embed, 
    #                                                     ems_mask)
    #     crs_attn_out = self.layers[0].compute_crs_attn(self_attn_out.ems, 
    #                                                     self_attn_out.item,
    #                                                     ems_mask)
    #     for layer in self.layers[1:]:
    #         self_attn_out = layer.compute_slf_attn(crs_attn_out.ems, 
    #                                                 crs_attn_out.item, 
    #                                                         ems_mask)
    #         crs_attn_out = layer.compute_crs_attn(self_attn_out.ems, 
    #                                             self_attn_out.item,
    #                                             ems_mask)

        
    #     return PackingTransformerOutput(slf_attn=self_attn_out,
    #                                     crs_attn=crs_attn_out)


