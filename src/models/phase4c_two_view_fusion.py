from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
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
        return x + self.net(x)


class ViewEncoder(nn.Module):
    def __init__(
        self,
        feature_width: int,
        residual_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(14 * 2, feature_width),
            nn.LayerNorm(feature_width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(feature_width, dropout)
                for _ in range(residual_blocks)
            ]
        )

    def forward(self, pose2d_nonroot: torch.Tensor) -> torch.Tensor:
        if pose2d_nonroot.ndim != 3 or pose2d_nonroot.shape[1:] != (14, 2):
            raise ValueError(
                "Expected [B,14,2], got "
                f"{tuple(pose2d_nonroot.shape)}."
            )
        x = pose2d_nonroot.flatten(start_dim=1)
        return self.blocks(self.input_layer(x))


class GeometryAwareTwoViewLifter(nn.Module):
    """
    Shared architecture for Duplicate, Oracle, and Synthesized experiments.

    Reference view defines the 3D output coordinate frame.
    Geometry maps auxiliary camera coordinates into the reference camera:
        X_reference = R_aux_to_ref @ X_auxiliary + t_aux_to_ref
    """

    def __init__(
        self,
        view_width: int = 256,
        view_residual_blocks: int = 2,
        geometry_width: int = 128,
        fusion_width: int = 1024,
        fusion_residual_blocks: int = 2,
        dropout: float = 0.20,
        translation_scale_m: float = 5.0,
    ) -> None:
        super().__init__()
        self.translation_scale_m = float(translation_scale_m)

        self.view_encoder = ViewEncoder(
            feature_width=view_width,
            residual_blocks=view_residual_blocks,
            dropout=dropout,
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(12, geometry_width),
            nn.LayerNorm(geometry_width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(geometry_width, geometry_width),
            nn.LayerNorm(geometry_width),
            nn.ReLU(inplace=True),
        )

        fusion_input_width = view_width * 3 + geometry_width
        self.fusion_input = nn.Sequential(
            nn.Linear(fusion_input_width, fusion_width),
            nn.LayerNorm(fusion_width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fusion_blocks = nn.Sequential(
            *[
                ResidualBlock(fusion_width, dropout)
                for _ in range(fusion_residual_blocks)
            ]
        )
        self.output_layer = nn.Linear(fusion_width, 14 * 3)

    def forward(
        self,
        reference_pose2d_nonroot: torch.Tensor,
        auxiliary_pose2d_nonroot: torch.Tensor,
        auxiliary_to_reference_rotation: torch.Tensor,
        auxiliary_to_reference_translation_m: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = reference_pose2d_nonroot.shape[0]

        if auxiliary_pose2d_nonroot.shape != reference_pose2d_nonroot.shape:
            raise ValueError("Reference and auxiliary 2D shapes differ.")
        if auxiliary_to_reference_rotation.shape != (batch_size, 3, 3):
            raise ValueError(
                "Expected rotation [B,3,3], got "
                f"{tuple(auxiliary_to_reference_rotation.shape)}."
            )
        if auxiliary_to_reference_translation_m.shape != (batch_size, 3):
            raise ValueError(
                "Expected translation [B,3], got "
                f"{tuple(auxiliary_to_reference_translation_m.shape)}."
            )

        reference_feature = self.view_encoder(reference_pose2d_nonroot)
        auxiliary_feature = self.view_encoder(auxiliary_pose2d_nonroot)

        identity = torch.eye(
            3,
            dtype=auxiliary_to_reference_rotation.dtype,
            device=auxiliary_to_reference_rotation.device,
        ).expand(batch_size, -1, -1)

        geometry_input = torch.cat(
            [
                (
                    auxiliary_to_reference_rotation - identity
                ).flatten(start_dim=1),
                auxiliary_to_reference_translation_m
                / self.translation_scale_m,
            ],
            dim=1,
        )
        geometry_feature = self.geometry_encoder(geometry_input)

        x = torch.cat(
            [
                reference_feature,
                auxiliary_feature,
                auxiliary_feature - reference_feature,
                geometry_feature,
            ],
            dim=1,
        )
        x = self.fusion_input(x)
        x = self.fusion_blocks(x)
        return self.output_layer(x).reshape(-1, 14, 3)
