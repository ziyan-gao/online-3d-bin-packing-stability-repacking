from __future__ import annotations
from typing import Optional
from typing import Callable

from functools import partial
import torch
import torch.nn as nn
from .projection import Linear
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend
from torch.nn.functional import scaled_dot_product_attention

# Hardcoded SDP backends
SDP_BACKEND_MAP = {
    "enable_flash": SDPBackend.FLASH_ATTENTION,
    "enable_mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "enable_math": SDPBackend.MATH,
    "enable_cudnn": SDPBackend.CUDNN_ATTENTION,
}


class MultiHeadedAttention(nn.Module):
    """Standard multi-headed attention.

    Args:
        embed_dim: int, dimension of embedding
        xpos_scale_base: int, base scale of xpos, default 512
        num_heads: int, number of heads, default 8
        dropout_prob: float, dropout probability, default 0.1
        self_attention: bool, whether self-attention, default True
        subln: bool, whether to apply pre layer normalization, default False
    """

    embed_dim: int
    num_heads: int
    head_dim: int
    scaling: float
    dropout_p: float
    w_k: Linear
    w_v: Linear
    w_q: Linear
    out_proj: Linear
    inner_attn_ln: nn.LayerNorm
    dropout: nn.Dropout | None
    _sdpa_backend: partial

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout_prob: int = 0.1,
        qkv_init_fn: Callable[[Linear], None] = None,
        output_proj_init_fn: Callable[[Linear], None] = None,
        subln: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout_p = dropout_prob

        self.w_k = Linear(
            embed_dim,
            embed_dim,
            bias=True,
            init_fn=init_qkv_projection if qkv_init_fn is None else qkv_init_fn,
        )
        self.w_v = Linear(
            embed_dim,
            embed_dim,
            bias=True,
            init_fn=init_qkv_projection if qkv_init_fn is None else qkv_init_fn,
        )
        self.w_q = Linear(
            embed_dim,
            embed_dim,
            bias=True,
            init_fn=init_qkv_projection if qkv_init_fn is None else qkv_init_fn,
        )
        self.out_proj = Linear(
            embed_dim,
            embed_dim,
            bias=True,
            init_fn=(
                init_output_projection
                if output_proj_init_fn is None
                else output_proj_init_fn
            ),
        )

        self.dropout = (
            nn.Dropout(dropout_prob)
            if dropout_prob > 0
            else self.register_module("dropout", None)
        )

        # flash attention related context manager
        flash_kwargs = {
            "enable_flash": True,
            "enable_math": True,
            "enable_mem_efficient": True,
        }
        sdpa_backends = [
            SDP_BACKEND_MAP[enable_str]
            for enable_str, enable in flash_kwargs.items()
            if enable
        ]
        self._sdpa_backend = partial(torch.nn.attention.sdpa_kernel, sdpa_backends)

    def attention_ops(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask=None,
    ):

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)  # [B, H, 1, T]
        else:
            attn_mask = None

        with self._sdpa_backend():
            attn = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p,
                is_causal=False,
            )  # [B, H, T, D]

        return attn

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ):
        """Forward pass.

        Args:
            query (Tensor): query, [B, T, C]
            key (Tensor): key, [B, T, C]
            value (Tensor): value, [B, T, C]
            key_padding_mask (Optional[Tensor]): key padding [B, T]
        Returns:
            attn: attention output, [B, T, C]
        """
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q, k, v = (
            rearrange(x, "b t (h d) -> b h t d", h=self.num_heads) for x in (q, k, v)
        )

        attn = self.attention_ops(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
        )

        attn = rearrange(attn, "b h t d -> b t (h d)")

        attn = self.out_proj(attn)

        return attn


def init_qkv_projection(proj: Linear) -> None:
    """Initialize ``proj`` as a multi-head attention input projection."""
    # Empirically observed the convergence to be much better with the scaled
    # initialization.
    nn.init.xavier_uniform_(proj.weight, gain=2**-0.5)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)


def init_output_projection(proj: Linear) -> None:
    """Initialize ``proj`` as a multi-head attention output projection."""
    nn.init.xavier_uniform_(proj.weight)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)
