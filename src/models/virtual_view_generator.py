from __future__ import annotations

import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, *, dropout: float = 0.0) -> None:
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.block(features)


class GeometryConditionedVirtualViewGenerator(nn.Module):
    """
    Predict target-camera root-centred 2D pose from one observed 2D pose
    and relative camera geometry.
    """

    def __init__(
        self,
        *,
        num_joints: int = 15,
        pose_hidden_dim: int = 512,
        pose_blocks: int = 2,
        geometry_hidden_dim: int = 128,
        fusion_hidden_dim: int = 1024,
        fusion_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        pose_dim = num_joints * 2

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden_dim),
            nn.LayerNorm(pose_hidden_dim),
            nn.GELU(),
            *[
                ResidualMLPBlock(pose_hidden_dim, dropout=dropout)
                for _ in range(pose_blocks)
            ],
        )

        self.geometry_encoder = nn.Sequential(
            nn.Linear(12, geometry_hidden_dim),
            nn.LayerNorm(geometry_hidden_dim),
            nn.GELU(),
            nn.Linear(geometry_hidden_dim, geometry_hidden_dim),
            nn.LayerNorm(geometry_hidden_dim),
            nn.GELU(),
        )

        self.fusion_input = nn.Sequential(
            nn.Linear(
                pose_hidden_dim + geometry_hidden_dim,
                fusion_hidden_dim,
            ),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
        )
        self.fusion_blocks = nn.Sequential(
            *[
                ResidualMLPBlock(fusion_hidden_dim, dropout=dropout)
                for _ in range(fusion_blocks)
            ]
        )
        self.output_layer = nn.Linear(fusion_hidden_dim, pose_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        source_pose2d: torch.Tensor,
        relative_rotation: torch.Tensor,
        relative_translation: torch.Tensor,
    ) -> torch.Tensor:
        if source_pose2d.ndim != 3:
            raise ValueError(
                f"Expected source pose [B,J,2], found {tuple(source_pose2d.shape)}."
            )

        batch_size = source_pose2d.shape[0]
        if source_pose2d.shape[1:] != (self.num_joints, 2):
            raise ValueError(
                f"Expected source pose [B,{self.num_joints},2], "
                f"found {tuple(source_pose2d.shape)}."
            )
        if relative_rotation.shape != (batch_size, 3, 3):
            raise ValueError(
                f"Expected relative rotation [{batch_size},3,3], "
                f"found {tuple(relative_rotation.shape)}."
            )
        if relative_translation.shape != (batch_size, 3):
            raise ValueError(
                f"Expected relative translation [{batch_size},3], "
                f"found {tuple(relative_translation.shape)}."
            )

        source_flat = source_pose2d.reshape(batch_size, -1)
        pose_features = self.pose_encoder(source_flat)

        geometry_input = torch.cat(
            [
                relative_rotation.reshape(batch_size, 9),
                relative_translation,
            ],
            dim=1,
        )
        geometry_features = self.geometry_encoder(geometry_input)

        fused = torch.cat([pose_features, geometry_features], dim=1)
        fused = self.fusion_input(fused)
        fused = self.fusion_blocks(fused)

        delta = self.output_layer(fused)
        predicted = source_flat + delta
        return predicted.reshape(batch_size, self.num_joints, 2)
