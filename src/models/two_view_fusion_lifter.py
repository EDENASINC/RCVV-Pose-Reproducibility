from __future__ import annotations

import torch
from torch import nn


class ResidualFeatureBlock(nn.Module):
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.block(features)


class SharedViewEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_joints: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints

        self.input_layer = nn.Sequential(
            nn.Linear(num_joints * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualFeatureBlock(
                    hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, pose2d: torch.Tensor) -> torch.Tensor:
        if pose2d.ndim != 3:
            raise ValueError(
                f"Expected [B, J, 2], found {tuple(pose2d.shape)}."
            )
        if pose2d.shape[1:] != (self.num_joints, 2):
            raise ValueError(
                f"Expected [B, {self.num_joints}, 2], "
                f"found {tuple(pose2d.shape)}."
            )

        features = pose2d.reshape(pose2d.shape[0], -1)
        return self.blocks(self.input_layer(features))


class GeometryConditionedTwoViewLifter(nn.Module):
    """
    Geometry-conditioned two-view 2D-to-3D lifter.

    Both 2D views are processed by a shared encoder. Relative camera geometry
    is encoded separately and fused with source, second-view, difference, and
    element-wise product features.

    Inputs
    ------
    source_pose2d:
        [B, J, 2], standardized root-centred camera-normalized coordinates.

    second_pose2d:
        [B, J, 2], standardized using the same statistics.

    relative_rotation:
        [B, 3, 3], source-camera coordinates to second-camera coordinates.

    relative_translation_m:
        [B, 3], source-camera origin expressed in the second-camera frame,
        in metres.

    Output
    ------
    [B, J, 3], source-camera root-relative 3D pose in metres.
    """

    def __init__(
        self,
        *,
        num_joints: int = 15,
        view_hidden_dim: int = 512,
        view_blocks: int = 2,
        geometry_hidden_dim: int = 128,
        fusion_hidden_dim: int = 1024,
        fusion_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_joints = num_joints

        self.view_encoder = SharedViewEncoder(
            num_joints=num_joints,
            hidden_dim=view_hidden_dim,
            num_blocks=view_blocks,
            dropout=dropout,
        )

        self.geometry_encoder = nn.Sequential(
            nn.Linear(12, geometry_hidden_dim),
            nn.LayerNorm(geometry_hidden_dim),
            nn.GELU(),
            nn.Linear(geometry_hidden_dim, geometry_hidden_dim),
            nn.LayerNorm(geometry_hidden_dim),
            nn.GELU(),
        )

        fusion_input_dim = (
            view_hidden_dim * 4
            + geometry_hidden_dim
        )

        self.fusion_input = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
        )
        self.fusion_blocks = nn.Sequential(
            *[
                ResidualFeatureBlock(
                    fusion_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(fusion_blocks)
            ]
        )
        self.output_layer = nn.Linear(
            fusion_hidden_dim,
            num_joints * 3,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        source_pose2d: torch.Tensor,
        second_pose2d: torch.Tensor,
        relative_rotation: torch.Tensor,
        relative_translation_m: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = source_pose2d.shape[0]

        if second_pose2d.shape != source_pose2d.shape:
            raise ValueError(
                "source_pose2d and second_pose2d must have the same shape."
            )
        if relative_rotation.shape != (batch_size, 3, 3):
            raise ValueError(
                "Expected relative_rotation shape "
                f"({batch_size}, 3, 3), found "
                f"{tuple(relative_rotation.shape)}."
            )
        if relative_translation_m.shape != (batch_size, 3):
            raise ValueError(
                "Expected relative_translation_m shape "
                f"({batch_size}, 3), found "
                f"{tuple(relative_translation_m.shape)}."
            )

        source_features = self.view_encoder(source_pose2d)
        second_features = self.view_encoder(second_pose2d)

        geometry_input = torch.cat(
            [
                relative_rotation.reshape(batch_size, 9),
                relative_translation_m,
            ],
            dim=1,
        )
        geometry_features = self.geometry_encoder(geometry_input)

        fused = torch.cat(
            [
                source_features,
                second_features,
                second_features - source_features,
                second_features * source_features,
                geometry_features,
            ],
            dim=1,
        )
        fused = self.fusion_input(fused)
        fused = self.fusion_blocks(fused)

        output = self.output_layer(fused)
        return output.reshape(batch_size, self.num_joints, 3)
