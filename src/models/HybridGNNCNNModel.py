from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel


class CNNBlock(nn.Module):
    """Standard CNN block with two convolutions and residual connection."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Projection for residual connection if channel dimensions differ
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.use_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        out = out + residual
        return out


class GNNBlock(nn.Module):
    """GNN-style message passing block with multi-scale aggregation."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        # Multi-scale convolutions for richer spatial aggregation
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.use_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        
        # Multi-scale feature extraction
        out1 = self.conv1x1(x)
        out3 = self.conv3x3(x)
        out5 = self.conv5x5(x)
        
        # Concatenate multi-scale features
        out = torch.cat([out1, out3, out5], dim=1)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        
        out = out + residual
        return out


class AttentionGate(nn.Module):
    """Attention gate for combining encoder and decoder features."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class HybridGNNCNNModel(BaseModel):
    """Deep hybrid model combining CNN encoder-decoder with GNN message passing.
    
    Architecture:
    1. CNN encoder extracts hierarchical features at multiple scales
    2. GNN layers perform spatial message passing at each scale
    3. CNN decoder with skip connections and attention gates
    4. Final classification head
    """

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        base_channels: int = 64,
        num_encoder_blocks: int = 4,
        num_gnn_layers_per_scale: int = 2,
        dropout: float = 0.1,
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

        # Input projection
        self.input_proj = nn.Conv2d(n_channels, base_channels, kernel_size=3, padding=1)

        # CNN Encoder blocks (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.gnn_blocks_encoder = nn.ModuleList()
        
        in_ch = base_channels
        for i in range(num_encoder_blocks):
            out_ch = base_channels * (2 ** i)
            # CNN feature extraction
            self.encoder_blocks.append(CNNBlock(in_ch, out_ch, dropout=dropout))
            # GNN message passing at this scale
            gnn_layers = nn.ModuleList([
                GNNBlock(out_ch, out_ch, dropout=dropout) 
                for _ in range(num_gnn_layers_per_scale)
            ])
            self.gnn_blocks_encoder.append(gnn_layers)
            # Downsampling (except for last block)
            if i < num_encoder_blocks - 1:
                self.downsample_layers.append(
                    nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2)
                )
            else:
                self.downsample_layers.append(nn.Identity())
            in_ch = out_ch

        # Bottleneck: additional GNN layers at the deepest scale
        bottleneck_ch = base_channels * (2 ** (num_encoder_blocks - 1))
        self.bottleneck_gnn = nn.ModuleList([
            GNNBlock(bottleneck_ch, bottleneck_ch, dropout=dropout)
            for _ in range(num_gnn_layers_per_scale * 2)  # More layers in bottleneck
        ])

        # CNN Decoder blocks (upsampling) with skip connections
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.gnn_blocks_decoder = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention else None
        
        for i in range(num_encoder_blocks - 1, 0, -1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i - 1))
            skip_ch = out_ch  # Skip connection from encoder
            
            # Attention gate (if enabled)
            if use_attention:
                self.attention_gates.append(AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=out_ch))
            
            # Upsampling
            self.upsample_layers.append(
                nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
            )
            # Decoder CNN block
            self.decoder_blocks.append(CNNBlock(in_ch + skip_ch, out_ch, dropout=dropout))
            # GNN message passing in decoder
            gnn_layers = nn.ModuleList([
                GNNBlock(out_ch, out_ch, dropout=dropout)
                for _ in range(num_gnn_layers_per_scale)
            ])
            self.gnn_blocks_decoder.append(gnn_layers)

        # Final output head
        self.output_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, doys: torch.Tensor | None = None) -> torch.Tensor:
        # Handle temporal dimension flattening
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            # B, T, C, H, W -> B, T*C, H, W
            x = x.flatten(start_dim=1, end_dim=2)

        # x is now [B, C_flat, H, W]
        h = self.input_proj(x)
        
        # Store skip connections from encoder
        skip_connections = []
        
        # CNN Encoder + GNN at each scale
        for i, (encoder_block, gnn_layers, downsample) in enumerate(
            zip(self.encoder_blocks, self.gnn_blocks_encoder, self.downsample_layers)
        ):
            # CNN feature extraction
            h = encoder_block(h)
            # GNN message passing
            for gnn_layer in gnn_layers:
                h = gnn_layer(h)
            # Store for skip connection (before downsampling)
            skip_connections.append(h)
            # Downsample
            h = downsample(h)

        # Bottleneck: additional GNN processing at deepest scale
        for gnn_layer in self.bottleneck_gnn:
            h = gnn_layer(h)

        # CNN Decoder + GNN with skip connections
        for i, (upsample, decoder_block, gnn_layers) in enumerate(
            zip(self.upsample_layers, self.decoder_blocks, self.gnn_blocks_decoder)
        ):
            # Upsample
            h = upsample(h)
            # Get skip connection (reverse order)
            skip = skip_connections[-(i + 2)]
            
            # Apply attention gate if enabled
            if self.attention_gates is not None:
                skip = self.attention_gates[i](h, skip)
            
            # Concatenate with skip connection
            h = torch.cat([h, skip], dim=1)
            # Decoder CNN block
            h = decoder_block(h)
            # GNN message passing
            for gnn_layer in gnn_layers:
                h = gnn_layer(h)

        # Final classification head
        logits = self.output_head(h)
        return logits
