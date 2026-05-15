# Description: Space encoder component for EMS
from torch import nn
from .projection import Linear
from typing import Callable, Optional
from torch import Tensor, as_tensor, float32
from torch import dtype as DataType
import torch


class EMSEmbed(nn.Module):
    dtype: DataType = float32
    """EMS encoder

    Args:
        input_dim (int): input dimension
        embed_dim (int): embedding dimension
    """

    def __init__(
        self,
        input_dim: int = 6,
        embed_dim: int = 128,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            Linear(
                input_dim,
                embed_dim,
                bias=True,
                init_fn=_init_orthogonal_weight_and_bias,
            ),
            nn.LeakyReLU(),
            # nn.LayerNorm(embed_dim),
            Linear(
                embed_dim,
                embed_dim,
                bias=True,
                init_fn=_init_orthogonal_weight_and_bias,
            ),
        )

    def forward(self, x):
        # params = list(self.mlp.parameters())[0]
        # if torch.isnan(params).any():
        #     print("Tensor contains NaN values.")
        return self.mlp(x)


def _init_orthogonal_weight_and_bias(weight: Tensor, bias: Optional[Tensor]) -> None:
    nn.init.orthogonal_(weight)

    if bias is not None:
        nn.init.constant_(bias, 0)
