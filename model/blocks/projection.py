from __future__ import annotations
import torch
import math
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.functional import linear
from typing import Callable, Optional
from torch import device as Device
from torch import dtype as DataType


class Linear(Module):
    """Applies a linear transformation to incoming data using weights and bias.

    Unless overridden by a subclass, the weights and bias are initialized from
    :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
    :math:`k = \\frac{1}{\\text{input_dim}}`.

    .. note::
        This class is identical to :class:`torch.nn.Linear`.
    """

    weight: Parameter
    bias: Optional[Parameter]
    init_fn: Optional[Callable[[Linear], None]]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool,
        *,
        init_fn: Optional[Callable[[Linear], None]] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of projected outputs.
        :param bias:
            If ``True``, learns an additive bias.
        :param init_fn:
            The callable to use for parameter initialization.
        """
        super().__init__()

        self.weight = Parameter(
            torch.empty((output_dim, input_dim), device=device, dtype=dtype)
        )

        if bias:
            self.bias = Parameter(
                torch.empty((output_dim,), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.init_fn = init_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.init_fn is not None:
            self.init_fn(self.weight, self.bias)
        else:
            _init_uniform_weight_and_bias(self.weight, self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)

    @property
    def num_param(self):
        w_shape = self.weight.shape
        return w_shape[0] * w_shape[1] + len(self.bias)


def _init_uniform_weight_and_bias(weight: Tensor, bias: Tensor | None) -> None:
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    if bias is not None:
        fan_in = weight.size(1)

        m = 1
        if weight.ndim > 2:
            for s in weight.shape[2:]:
                m *= s

        fan_in *= m

        # We do not calculate the true standard deviation of the uniform
        # distribution (i.e. multiply with sqrt(3)). See
        # https://github.com/pytorch/pytorch/issues/57109#issuecomment-828847575.
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

        nn.init.uniform_(bias, -bound, bound)


def init_bert_projection(weight: Tensor, bias: Optional[Tensor]) -> None:
    """Initialize ``proj`` as a projection to be used in BERT-like models."""
    nn.init.normal_(weight, mean=0.0, std=0.02)

    if bias is not None:
        nn.init.zeros_(bias)
