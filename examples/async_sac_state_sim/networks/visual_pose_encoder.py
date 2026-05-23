from __future__ import annotations

from typing import Tuple

import torch as th
from torch import nn


class VisualPoseEncoder(nn.Module):
    """Lightweight depth encoder for auxiliary socket relative pose prediction."""

    def __init__(
        self,
        depth_shape: Tuple[int, int, int],
        z_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        channels, height, width = depth_shape
        self.z_dim = z_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, channels, height, width)
            cnn_dim = self.cnn(sample).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(cnn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
            nn.ReLU(),
        )
        self.pose_head = nn.Linear(z_dim, 3)

    def forward(self, depth: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        z_vis = self.mlp(self.cnn(depth))
        pose_hat = self.pose_head(z_vis)
        return pose_hat, z_vis
