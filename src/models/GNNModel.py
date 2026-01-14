from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel


class GNNBlock(nn.Module):
    """Simple message-passing style block over a 2D grid.

    Neighbour aggregation is implemented with a 3x3 convolution, which
    is equivalent to message passing on a regular grid graph.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.use_residual = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        if self.use_residual:
            out = out + x
        return out


class GNNModel(BaseModel):
    """GNN-style model operating on the spatial grid.

    The temporal dimension is flattened into the channel dimension (via the
    ``flatten_temporal_dimension`` flag in the BaseModel), then several
    message-passing style blocks aggregate information over the grid.
    """

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            *args,
            **kwargs,
        )

        self.save_hyperparameters()

        # Project (possibly very high dimensional) input features to a
        # manageable hidden size before message passing.
        self.input_proj = nn.Conv2d(n_channels, hidden_dim, kernel_size=1)

        layers = []
        for _ in range(num_layers):
            layers.append(GNNBlock(hidden_dim, hidden_dim, dropout=dropout))
        self.gnn_layers = nn.ModuleList(layers)

        # Final classifier head to produce a single logit per pixel.
        self.output_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, doys: torch.Tensor | None = None) -> torch.Tensor:
        # Mirror BaseModel behaviour: if the temporal dimension is present and
        # we want to flatten it, merge it into the channel dimension.
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            # B, T, C, H, W -> B, T*C, H, W
            x = x.flatten(start_dim=1, end_dim=2)

        # x is now [B, C_flat, H, W]
        h = self.input_proj(x)
        for layer in self.gnn_layers:
            h = layer(h)

        logits = self.output_head(h)
        return logits

