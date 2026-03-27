"""
CNN encoder with Dueling DQN head.
"""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


def _init_weights(module: nn.Module) -> None:
    """Xavier-uniform init for Linear layers, Kaiming for Conv2d."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class CNNEncoder(nn.Module):
    """3-layer CNN encoder for the (C, H, W) spatial grid.

    The scalar observations are concatenated *after* flattening and
    processed through a small FC merge layer so the CNN sees only the
    spatial grid.
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        channels: Tuple[int, ...],
        merge_hidden: int,
        spatial_channels: int,
        scalar_dim: int,
    ):
        super().__init__()
        self.spatial_channels = spatial_channels
        self.scalar_dim = scalar_dim
        c1, c2, c3 = channels

        self.conv = nn.Sequential(
            nn.Conv2d(spatial_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LayerNorm([c1, grid_h, grid_w]),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LayerNorm([c2, grid_h, grid_w]),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        # flattened conv output size at runtime, more robust than hardcoding
        with torch.no_grad():
            dummy = torch.zeros(1, spatial_channels, grid_h, grid_w)
            conv_out = self.conv(dummy).view(1, -1).size(1)

        self.merge = nn.Sequential(
            nn.Linear(conv_out + scalar_dim, merge_hidden),
            nn.ReLU(),
            nn.LayerNorm(merge_hidden),
        )
        self.output_size = merge_hidden
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accept flat observation vector, split into spatial + scalars."""
        spatial_dim = self.spatial_channels * self.grid_h * self.grid_w
        spatial_flat = x[:, :spatial_dim]
        scalars = x[:, spatial_dim:]

        spatial = spatial_flat.view(-1, self.spatial_channels, self.grid_h, self.grid_w)
        conv_features = self.conv(spatial).view(x.size(0), -1)
        merged = self.merge(torch.cat([conv_features, scalars], dim=1))
        return merged


class DuelingHead(nn.Module):
    """Dueling Q-value head: V(s) + A(s,a) − mean A."""

    def __init__(self, feature_size: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, hidden), nn.ReLU(), nn.Linear(hidden, action_size),
        )
        self.apply(_init_weights)
        # Scale output layers so initial Q-values start near zero
        nn.init.orthogonal_(self.value_stream[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.advantage_stream[-1].weight, gain=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.value_stream(features)          # (B, 1)
        advantage = self.advantage_stream(features)  # (B, A)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DQNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


def create_network(
    action_size: int,
    merge_hidden: int,
    head_hidden: int,
    cnn_channels: Tuple[int, ...],
    grid_h: int,
    grid_w: int,
    spatial_channels: int,
    scalar_dim: int,
) -> DQNetwork:
    """Factory function to build a CNN-encoder + Dueling-head DQN."""
    encoder = CNNEncoder(
        grid_h=grid_h,
        grid_w=grid_w,
        channels=cnn_channels,
        merge_hidden=merge_hidden,
        spatial_channels=spatial_channels,
        scalar_dim=scalar_dim,
    )

    feature_size = encoder.output_size

    head = DuelingHead(feature_size, action_size, hidden=head_hidden)
    return DQNetwork(encoder, head)
