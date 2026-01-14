from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel


class ChannelAttention(nn.Module):
    """Lightweight channel attention mechanism (SE-Net style)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Average pooling branch
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1)
        # Max pooling branch
        y_max = self.max_pool(x).view(b, c)
        y_max = self.fc(y_max).view(b, c, 1, 1)
        # Combine and apply
        y = y_avg + y_max
        return x * y


class SpatialAttention(nn.Module):
    """Lightweight spatial attention mechanism."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and apply convolution
        x_att = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv(x_att)
        x_att = self.sigmoid(x_att)
        return x * x_att


class LightGNNBlock(nn.Module):
    """Lightweight GNN block with attention mechanisms.
    
    Combines spatial message passing with channel and spatial attention
    for improved feature learning while remaining lightweight.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, use_attention: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.use_residual = in_channels == out_channels
        
        # Attention mechanisms
        self.use_attention = use_attention
        if use_attention:
            self.channel_att = ChannelAttention(out_channels, reduction=4)
            self.spatial_att = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = F.relu(out, inplace=True)
        
        # Apply attention mechanisms
        if self.use_attention:
            out = self.channel_att(out)  # Channel attention first
            out = self.spatial_att(out)  # Then spatial attention
        
        out = self.dropout(out)

        if self.use_residual:
            out = out + x
        return out


class SmallGNNModel(BaseModel):
    """Small, lightweight GNN-style model with attention mechanisms.
    
    This is a compact version of the GNN model with:
    - Smaller hidden dimensions (32 vs 64)
    - Fewer layers (2 vs 4)
    - Channel and spatial attention for improved feature learning
    - Minimal architecture for fast training and inference
    """

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.05,
        use_attention: bool = True,
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

        # Lightweight input projection
        self.input_proj = nn.Conv2d(n_channels, hidden_dim, kernel_size=1)

        # GNN layers with attention
        layers = []
        for _ in range(num_layers):
            layers.append(LightGNNBlock(hidden_dim, hidden_dim, dropout=dropout, use_attention=use_attention))
        self.gnn_layers = nn.ModuleList(layers)

        # Simple output head
        self.output_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, doys: torch.Tensor | None = None) -> torch.Tensor:
        # Handle temporal dimension flattening
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            # B, T, C, H, W -> B, T*C, H, W
            x = x.flatten(start_dim=1, end_dim=2)

        # x is now [B, C_flat, H, W]
        h = self.input_proj(x)
        for layer in self.gnn_layers:
            h = layer(h)

        logits = self.output_head(h)
        return logits
