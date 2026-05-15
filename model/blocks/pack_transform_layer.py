from __future__ import annotations

from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
# from .multiheaded_attention import (
#     MultiHeadedAttention,
#     init_qkv_projection,
#     init_output_projection,
# )
from .ffn import FeedForwardNetwork
from .projection import Linear
from .norm_order import TransformerNormOrder
from dataclasses import dataclass

@dataclass
class SelfAttenOutput:
    
    ems: torch.Tensor
    """ ems after self attention"""
    item: torch.Tensor
    """ item logits after cross attention"""

@dataclass
class CrossAttenOutput:
    
    ems: torch.Tensor
    """ems after interact with item"""
    item: torch.Tensor
    """item after interact with ems"""
    
# # Safe MultiheadAttention function
# def safe_multihead_attention(mha_layer, query, key, value, key_padding_mask):
#     batch_size, seq_len, _ = query.shape
    
#     # Detect fully masked sequences
#     fully_masked = key_padding_mask.all(dim=1)  # Shape: (batch_size,)
    
#     # Allocate output tensors
#     attn_output = torch.zeros_like(query)  # Initialize with zeros
#     # attn_weights = torch.zeros(batch_size, seq_len, seq_len, device=query.device)  # Shape: (batch_size, seq_len, seq_len)

#     # Process valid sequences
#     if not fully_masked.all():  # If some sequences are valid
#         valid_indices = torch.where(~fully_masked)[0]  # Indices of valid sequences
#         valid_query = query[valid_indices]
#         valid_key = key[valid_indices]
#         valid_value = value[valid_indices]
#         valid_mask = key_padding_mask[valid_indices]

#         # Compute attention for valid sequences
#         valid_attn_output, _ = mha_layer(valid_query, valid_key, valid_value, key_padding_mask=valid_mask)

#         # Insert valid outputs back into the full batch output
#         attn_output[valid_indices] = valid_attn_output
#         # attn_weights[valid_indices] = valid_attn_weights

#     # Return outputs (zeroed-out for fully masked sequences)
#     return attn_output, 0


class ResidualConnection(nn.Module):
    """Residual connection module.

    Args:
        sublayer: Sublayer module.
        dropout: Dropout probability.
    """

    def __init__(
        self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0, single_input=True
    ):
        super().__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor
        self.single_input = single_input

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.single_input:
            return (self.module(x, *args, **kwargs) * self.module_factor) + (
                x * self.input_factor
            )
        else:
            return (self.module(x, *args, **kwargs)[0] * self.module_factor) + (
                x * self.input_factor
            )


class PackingTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer.

    Args:
        model_dim: Model dimensionality.
        num_heads: Number of attention heads.
        ffn_expansion_factor: Feed-forward network expansion factor.
        dropout_p: Dropout probability.
        proj_init_fn: Projection initialization function.
    """

    _ems_self_attn_ffn: ResidualConnection
    _ems_self_attn: ResidualConnection
    _item_self_attn_ffn: ResidualConnection
    _item_self_attn: ResidualConnection
    _ems_cross_attn_ffn: ResidualConnection
    _ems_cross_attn: ResidualConnection
    _item_cross_attn_ffn: ResidualConnection
    _item_cross_attn: ResidualConnection
    ems_self_attn_layer_norm: nn.LayerNorm
    ems_cross_attn_layer_norm: nn.LayerNorm
    item_self_attn_layer_norm: nn.LayerNorm
    item_cross_attn_layer_norm: nn.LayerNorm
    ems_self_attn_ffn_layer_norm: nn.LayerNorm
    item_self_attn_ffn_layer_norm: nn.LayerNorm
    ems_cross_attn_ffn_layer_norm: nn.LayerNorm
    item_cross_attn_ffn_layer_norm: nn.LayerNorm
    norm_order: TransformerNormOrder

    def __init__(
        self,
        model_dim: int = 128,
        num_heads: int = 8,
        ffn_expansion_factor: int = 4,
        dropout_p: float = 0.1,
        norm_order: TransformerNormOrder = TransformerNormOrder.PRE,
        proj_init_fn: Optional[Callable[[Linear], None]] = None,
    ):
        super().__init__()

        self._ems_self_attn_ffn = ResidualConnection(
            FeedForwardNetwork(
                model_dim=model_dim,
                inner_dim=model_dim * ffn_expansion_factor,
                bias=True,
                inner_dropout_p=dropout_p,
                proj_init_fn=proj_init_fn if proj_init_fn is not None else None,
            )
        )

        self._ems_self_attn = ResidualConnection(
            nn.MultiheadAttention(embed_dim=model_dim,
                                  num_heads=num_heads,
                                  batch_first=True,
                                  dropout=dropout_p,
                                  ), single_input=False
            )
        
        self._item_self_attn_ffn = ResidualConnection(
            FeedForwardNetwork(
                model_dim=model_dim,
                inner_dim=model_dim * ffn_expansion_factor,
                bias=True,
                inner_dropout_p=dropout_p,
                proj_init_fn=proj_init_fn if proj_init_fn is not None else None,
            )
        )

        self._item_self_attn = ResidualConnection(
            nn.MultiheadAttention(embed_dim=model_dim,
                                  num_heads=num_heads,
                                  batch_first=True,
                                  dropout=dropout_p,
                                  ), single_input=False
        )

        self._ems_cross_attn_ffn = ResidualConnection(
            FeedForwardNetwork(
                model_dim=model_dim,
                inner_dim=model_dim * ffn_expansion_factor,
                bias=True,
                inner_dropout_p=dropout_p,
                proj_init_fn=proj_init_fn if proj_init_fn is not None else None,
            )
        )

        self._ems_cross_attn = ResidualConnection(
            nn.MultiheadAttention(embed_dim=model_dim,
                                  num_heads=num_heads,
                                  batch_first=True,
                                  dropout=dropout_p,
                                  ), single_input=False
        )

        self._item_cross_attn_ffn = ResidualConnection(
            FeedForwardNetwork(
                model_dim=model_dim,
                inner_dim=model_dim * ffn_expansion_factor,
                bias=True,
                inner_dropout_p=dropout_p,
                proj_init_fn=proj_init_fn if proj_init_fn is not None else None,
            )
        )

        self._item_cross_attn = ResidualConnection(
            nn.MultiheadAttention(embed_dim=model_dim,
                                  num_heads=num_heads,
                                  batch_first=True,
                                  dropout=dropout_p,
                                  ), single_input=False
        )

        self.ems_self_attn_layer_norm = nn.LayerNorm(model_dim)
        self.ems_cross_attn_layer_norm = nn.LayerNorm(model_dim)
        self.item_self_attn_layer_norm = nn.LayerNorm(model_dim)
        self.item_cross_attn_layer_norm = nn.LayerNorm(model_dim)

        self.ems_self_attn_ffn_layer_norm = nn.LayerNorm(model_dim)
        self.item_self_attn_ffn_layer_norm = nn.LayerNorm(model_dim)
        self.ems_cross_attn_ffn_layer_norm = nn.LayerNorm(model_dim)
        self.item_cross_attn_ffn_layer_norm = nn.LayerNorm(model_dim)

        self.norm_order = norm_order

    def _forward_ems_self_attn(
        self, ems_query: Tensor, ems_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        if self.norm_order != TransformerNormOrder.POST:
            ems_query = self.ems_self_attn_layer_norm(ems_query)
        # seqs = self._ems_self_attn(ems_query, ems_query, ems_query, key_padding_mask = ems_padding_mask)
        seqs = self._ems_self_attn(ems_query, ems_query, ems_query, key_padding_mask = ems_padding_mask)
        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.ems_self_attn_layer_norm(seqs)

        return seqs

    def _forward_item_self_attn(self, item_query: Tensor) -> Tensor:
        if self.norm_order != TransformerNormOrder.POST:
            item_query = self.item_self_attn_layer_norm(item_query)
        
        # key_padding_mask= torch.zeros(item_query.shape[:2], device=item_query.device).bool()
        # seqs = self._item_self_attn(item_query, item_query, item_query, key_padding_mask=key_padding_mask)
        seqs = self._item_self_attn(item_query, item_query, item_query, key_padding_mask=None)

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.item_self_attn_layer_norm(seqs)

        return seqs

    def _forward_ems_cross_attn(
        self,
        ems_query: Tensor,
        item_kv: Tensor,
    ) -> Tensor:
        if self.norm_order != TransformerNormOrder.POST:
            ems_query = self.ems_cross_attn_layer_norm(ems_query)

        # key_padding_mask= torch.zeros(item_kv.shape[:2], device=item_kv.device).bool()
        # seqs = self._ems_cross_attn(ems_query, item_kv, item_kv,key_padding_mask=key_padding_mask)
        seqs = self._ems_cross_attn(ems_query, item_kv, item_kv,key_padding_mask=None)

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.ems_cross_attn_layer_norm(seqs)

        return seqs

    def _forward_item_cross_attn(
        self,
        item_query: Tensor,
        ems_kv: Tensor,
        ems_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.norm_order != TransformerNormOrder.POST:
            item_query = self.item_cross_attn_layer_norm(item_query)
        # seqs = self._item_cross_attn(item_query, ems_kv, ems_kv, key_padding_mask = ems_padding_mask)
        seqs = self._item_cross_attn(item_query, ems_kv, ems_kv, key_padding_mask = ems_padding_mask)
        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.item_cross_attn_layer_norm(seqs)

        return seqs

    def _forward_ems_self_ffn(self, ems: Tensor) -> Tensor:
        if self.norm_order != TransformerNormOrder.POST:
            ems = self.ems_self_attn_ffn_layer_norm(ems)

        seqs = self._ems_self_attn_ffn(ems)

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.ems_self_attn_ffn_layer_norm(seqs)

        return seqs

    def _forward_item_self_ffn(self, item: Tensor) -> Tensor:
        if self.norm_order != TransformerNormOrder.POST:
            item = self.item_self_attn_ffn_layer_norm(item)

        seqs = self._item_self_attn_ffn(item)

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.item_self_attn_ffn_layer_norm(seqs)

        return seqs

    def _forward_ems_cross_ffn(self, ems: Tensor) -> Tensor:
        if self.norm_order != TransformerNormOrder.POST:
            ems = self.ems_cross_attn_ffn_layer_norm(ems)

        seqs = self._ems_cross_attn_ffn(ems)

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.ems_cross_attn_ffn_layer_norm(seqs)

        return seqs

    def _forward_item_cross_ffn(self, item: Tensor) -> Tensor:
        if self.norm_order != TransformerNormOrder.POST:
            item = self.item_cross_attn_ffn_layer_norm(item)

        seqs = self._item_cross_attn_ffn(item)

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.item_cross_attn_ffn_layer_norm(seqs)

        return seqs

    def compute_slf_attn(
        self,
        ems: Tensor,
        item: Tensor,
        ems_padding_mask: Optional[Tensor] = None,
    ) -> SelfAttenOutput:
        item = item.view(ems.shape[0],
                         -1, 
                         item.shape[-1])
        # self-attention
        slf_attn_ems = self._forward_ems_self_attn(ems, ems_padding_mask)
        slf_attn_item = self._forward_item_self_attn(item)

        slf_attn_ems = self._forward_ems_self_ffn(slf_attn_ems)
        slf_attn_item = self._forward_item_self_ffn(slf_attn_item)

        return SelfAttenOutput(
            ems=slf_attn_ems,
            item=slf_attn_item,
        )
        
    def compute_crs_attn(
        self,
        ems: Tensor,
        item: Tensor,
        ems_padding_mask: Optional[Tensor] = None,
    ) -> CrossAttenOutput:
        item = item.view(ems.shape[0],
                         -1, 
                         item.shape[-1])
        # mutually cross-attention
        crs_attn_ems = self._forward_ems_cross_attn(ems, item)
        crs_attn_item = self._forward_item_cross_attn(item, ems, ems_padding_mask)

        crs_attn_ems = self._forward_ems_cross_ffn(crs_attn_ems)
        crs_attn_item = self._forward_item_cross_ffn(crs_attn_item)

        return CrossAttenOutput(
            ems=crs_attn_ems,
            item=crs_attn_item,
        )
    
    

