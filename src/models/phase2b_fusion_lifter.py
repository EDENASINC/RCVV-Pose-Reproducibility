from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
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
        return x + self.net(x)


class Phase2BTwoViewFusionLifter(nn.Module):
    """
    Shared-encoder two-view 2D-to-3D fusion model.

    Inputs
    ------
    source_pose2d:
        [B, J, 2], standardized source-view root-relative 2D pose.

    second_pose2d:
        [B, J, 2], standardized duplicate, synthesized, or Oracle 2D pose.

    relative_rotation:
        [B, 3, 3], source-to-second camera rotation.

    relative_translation:
        [B, 3], standardized source-to-second camera translation.

    Output
    ------
    [B, J, 3], standardized root-relative 3D pose in the source camera.
    """

    def __init__(
        self,
        *,
        num_joints: int = 15,
        view_hidden_dim: int = 256,
        view_blocks: int = 2,
        geometry_hidden_dim: int = 64,
        fusion_hidden_dim: int = 512,
        fusion_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints

        pose_input_dim = num_joints * 2
        output_dim = num_joints * 3

        self.view_input = nn.Sequential(
            nn.Linear(pose_input_dim, view_hidden_dim),
            nn.LayerNorm(view_hidden_dim),
            nn.GELU(),
        )
        self.view_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    view_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(view_blocks)
            ]
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
                ResidualBlock(
                    fusion_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(fusion_blocks)
            ]
        )
        self.output_layer = nn.Linear(
            fusion_hidden_dim,
            output_dim,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_view(
        self,
        pose2d: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = pose2d.shape[0]
        flat = pose2d.reshape(batch_size, -1)
        features = self.view_input(flat)
        return self.view_blocks(features)

    def forward(
        self,
        source_pose2d: torch.Tensor,
        second_pose2d: torch.Tensor,
        relative_rotation: torch.Tensor,
        relative_translation: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = source_pose2d.shape[0]
        expected_pose_shape = (
            batch_size,
            self.num_joints,
            2,
        )

        if source_pose2d.shape != expected_pose_shape:
            raise ValueError(
                f"Expected source pose {expected_pose_shape}, "
                f"found {tuple(source_pose2d.shape)}."
            )
        if second_pose2d.shape != expected_pose_shape:
            raise ValueError(
                f"Expected second pose {expected_pose_shape}, "
                f"found {tuple(second_pose2d.shape)}."
            )
        if relative_rotation.shape != (batch_size, 3, 3):
            raise ValueError(
                "Invalid relative rotation shape: "
                f"{tuple(relative_rotation.shape)}."
            )
        if relative_translation.shape != (batch_size, 3):
            raise ValueError(
                "Invalid relative translation shape: "
                f"{tuple(relative_translation.shape)}."
            )

        source_features = self.encode_view(source_pose2d)
        second_features = self.encode_view(second_pose2d)

        geometry_input = torch.cat(
            [
                relative_rotation.reshape(batch_size, 9),
                relative_translation,
            ],
            dim=1,
        )
        geometry_features = self.geometry_encoder(
            geometry_input
        )

        fusion_features = torch.cat(
            [
                source_features,
                second_features,
                second_features - source_features,
                second_features * source_features,
                geometry_features,
            ],
            dim=1,
        )
        fusion_features = self.fusion_input(
            fusion_features
        )
        fusion_features = self.fusion_blocks(
            fusion_features
        )

        output = self.output_layer(fusion_features)
        return output.reshape(
            batch_size,
            self.num_joints,
            3,
        )
