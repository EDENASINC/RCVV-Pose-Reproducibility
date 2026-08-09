from __future__ import annotations

import torch
from torch import nn


class ResidualLinearBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class BaselinePoseLifter(nn.Module):
    """
    Frame-based residual MLP for 15-joint monocular 2D-to-3D lifting.

    Input:
        [B, 15, 2] root-centered normalized camera coordinates.

    Output:
        [B, 15, 3] root-relative 3D coordinates in metres.
    """

    def __init__(
        self,
        *,
        num_joints: int = 15,
        hidden_dim: int = 1024,
        num_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_joints <= 0:
            raise ValueError("num_joints must be greater than zero.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero.")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.num_joints = num_joints
        input_dim = num_joints * 2
        output_dim = num_joints * 3

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualLinearBlock(
                    hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pose2d: torch.Tensor) -> torch.Tensor:
        if pose2d.ndim != 3:
            raise ValueError(
                f"Expected pose2d shape [B, J, 2], found {tuple(pose2d.shape)}."
            )
        if pose2d.shape[1:] != (self.num_joints, 2):
            raise ValueError(
                f"Expected pose2d shape [B, {self.num_joints}, 2], "
                f"found {tuple(pose2d.shape)}."
            )

        batch_size = pose2d.shape[0]
        features = pose2d.reshape(batch_size, -1)
        features = self.input_layer(features)
        features = self.residual_blocks(features)
        pose3d = self.output_layer(features)
        return pose3d.reshape(batch_size, self.num_joints, 3)
