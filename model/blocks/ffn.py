from collections.abc import Callable
from typing import Optional

from torch import Tensor
from torch.nn import Dropout, Module, SiLU, LayerNorm

from .projection import Linear
from .norm_order import TransformerNormOrder


class FeedForwardNetwork(Module):
    """Represents a Transformer feed-forward network as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    inner_proj: Linear
    inner_activation: Module
    inner_dropout: Optional[Dropout]
    inner_norm: Optional[LayerNorm]
    output_proj: Linear

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        bias: bool,
        inner_dropout_p: float = 0.1,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        proj_init_fn: Optional[Callable[[Linear], None]] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param inner_dim:
            The dimensionality of the inner projection layer.
        :param bias:
            If ``True``, both the inner and output projection learn an additive
            bias.
        :param inner_activation:
            The activation to apply to outputs of the inner projection layer. If
            ``None``, :func:`~torch.nn.ReLU` will be used.
        :param inner_dropout_p:
            The dropout probability on outputs of the inner projection layer.
        :param norm_order:
            The Layer Normalization order.
        :param layer_norm_factory:
            The factory to construct the Layer Normalization module.
        """
        super().__init__()

        self.inner_proj = Linear(
            model_dim,
            inner_dim,
            bias,
            init_fn=proj_init_fn,
        )

        self.inner_activation = SiLU()

        if inner_dropout_p > 0.0:
            self.inner_dropout = Dropout(inner_dropout_p)
        else:
            self.register_module("inner_dropout", None)

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.inner_layer_norm = LayerNorm(inner_dim)
        else:
            self.register_module("inner_layer_norm", None)

        self.output_proj = Linear(
            inner_dim,
            model_dim,
            bias,
            init_fn=proj_init_fn,
        )

    @property
    def len_params(self):
        return self.inner_proj.num_param + self.output_proj.num_param

    
    def forward(self, seqs: Tensor) -> Tensor:

        seqs = self.inner_proj(seqs)

        seqs = self.inner_activation(seqs)

        if self.inner_layer_norm is not None:
            seqs = self.inner_layer_norm(seqs)

        if self.inner_dropout is not None:
            seqs = self.inner_dropout(seqs)

        seqs = self.output_proj(seqs)

        return seqs
