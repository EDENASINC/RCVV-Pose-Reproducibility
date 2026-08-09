from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class ObservedOnlyPoseLifter(nn.Module):
    """
    Root-relative single-view 2D-to-3D pose lifter.

    Input:
        [B, 15, 2] camera-normalized, pelvis-rooted 2D pose.

    Output:
        [B, 15, 3] pelvis-rooted 3D pose in millimetres.

    The pelvis joint is fixed to zero. Only the remaining 14 joints are
    predicted.
    """

    def __init__(
        self,
        hidden_width: int = 1024,
        residual_blocks: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.hidden_width = int(hidden_width)
        self.residual_blocks = int(residual_blocks)
        self.dropout = float(dropout)

        self.input_layer = nn.Sequential(
            nn.Linear(14 * 2, self.hidden_width),
            nn.LayerNorm(self.hidden_width),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(self.hidden_width, self.dropout)
                for _ in range(self.residual_blocks)
            ]
        )
        self.output_layer = nn.Linear(self.hidden_width, 14 * 3)

    def forward(self, pose2d_nonroot: torch.Tensor) -> torch.Tensor:
        if pose2d_nonroot.ndim != 3 or pose2d_nonroot.shape[1:] != (14, 2):
            raise ValueError(
                "Expected pose2d_nonroot with shape [B,14,2], "
                f"got {tuple(pose2d_nonroot.shape)}."
            )
        x = pose2d_nonroot.flatten(start_dim=1)
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x).reshape(-1, 14, 3)
