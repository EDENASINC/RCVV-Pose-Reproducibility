from __future__ import annotations

import torch
from torch import nn


EDGES = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8),
    (7, 9), (9, 10), (10, 11),
    (7, 12), (12, 13), (13, 14),
)


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


def compute_quality_features(
    source_pose2d: torch.Tensor,
    second_pose2d: torch.Tensor,
    relative_rotation: torch.Tensor,
    relative_translation: torch.Tensor,
) -> torch.Tensor:
    """
    Compute inference-available reliability cues.

    All inputs are raw root-relative camera-normalized coordinates.
    No Oracle pose or 3D target is used.
    """
    delta = second_pose2d - source_pose2d
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)

    source_radius = torch.linalg.vector_norm(
        source_pose2d,
        dim=-1,
    )
    second_radius = torch.linalg.vector_norm(
        second_pose2d,
        dim=-1,
    )

    source_bones = torch.stack(
        [
            torch.linalg.vector_norm(
                source_pose2d[:, end] - source_pose2d[:, start],
                dim=-1,
            )
            for start, end in EDGES
        ],
        dim=1,
    )
    second_bones = torch.stack(
        [
            torch.linalg.vector_norm(
                second_pose2d[:, end] - second_pose2d[:, start],
                dim=-1,
            )
            for start, end in EDGES
        ],
        dim=1,
    )

    relative_bone_error = (
        (second_bones - source_bones).abs()
        / source_bones.clamp_min(1e-4)
    )

    trace = (
        relative_rotation[:, 0, 0]
        + relative_rotation[:, 1, 1]
        + relative_rotation[:, 2, 2]
    )
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    rotation_angle = torch.acos(cosine)

    translation_norm = torch.linalg.vector_norm(
        relative_translation,
        dim=-1,
    )

    scale_ratio = (
        second_radius.mean(dim=1)
        / source_radius.mean(dim=1).clamp_min(1e-4)
    )

    return torch.stack(
        [
            delta_norm.mean(dim=1),
            delta_norm.std(dim=1),
            delta_norm.max(dim=1).values,
            source_radius.mean(dim=1),
            second_radius.mean(dim=1),
            scale_ratio,
            relative_bone_error.mean(dim=1),
            relative_bone_error.max(dim=1).values,
            rotation_angle,
            translation_norm,
        ],
        dim=1,
    )


class ConfidenceGatedFusionLifter(nn.Module):
    """
    Source-only base prediction plus confidence-gated virtual-view correction.

    final_3d = base_3d + gate * correction_3d

    The learned gate receives only inference-available inputs.
    """

    def __init__(
        self,
        *,
        num_joints: int = 15,
        quality_dim: int = 10,
        view_hidden_dim: int = 256,
        view_blocks: int = 2,
        geometry_hidden_dim: int = 64,
        fusion_hidden_dim: int = 512,
        fusion_blocks: int = 4,
        dropout: float = 0.1,
        initial_gate_bias: float = -1.5,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints

        pose_input_dim = num_joints * 2
        pose_output_dim = num_joints * 3

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

        self.base_head = nn.Sequential(
            nn.Linear(view_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            ResidualBlock(
                fusion_hidden_dim,
                dropout=dropout,
            ),
            nn.Linear(fusion_hidden_dim, pose_output_dim),
        )

        fusion_input_dim = (
            view_hidden_dim * 4
            + geometry_hidden_dim
            + quality_dim
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

        self.correction_head = nn.Linear(
            fusion_hidden_dim,
            pose_output_dim,
        )
        self.gate_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.reset_parameters(initial_gate_bias)

    def reset_parameters(
        self,
        initial_gate_bias: float,
    ) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(
            self.correction_head.weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.correction_head.bias)

        final_gate_layer = self.gate_head[-1]
        nn.init.normal_(
            final_gate_layer.weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.constant_(
            final_gate_layer.bias,
            initial_gate_bias,
        )

    def encode_view(
        self,
        pose2d: torch.Tensor,
    ) -> torch.Tensor:
        flat = pose2d.reshape(pose2d.shape[0], -1)
        return self.view_blocks(
            self.view_input(flat)
        )

    def forward(
        self,
        source_pose2d_standardized: torch.Tensor,
        second_pose2d_standardized: torch.Tensor,
        relative_rotation: torch.Tensor,
        relative_translation_standardized: torch.Tensor,
        quality_features_standardized: torch.Tensor,
        *,
        gate_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = source_pose2d_standardized.shape[0]

        source_features = self.encode_view(
            source_pose2d_standardized
        )
        second_features = self.encode_view(
            second_pose2d_standardized
        )

        geometry_input = torch.cat(
            [
                relative_rotation.reshape(batch_size, 9),
                relative_translation_standardized,
            ],
            dim=1,
        )
        geometry_features = self.geometry_encoder(
            geometry_input
        )

        fused = torch.cat(
            [
                source_features,
                second_features,
                second_features - source_features,
                second_features * source_features,
                geometry_features,
                quality_features_standardized,
            ],
            dim=1,
        )
        fused = self.fusion_blocks(
            self.fusion_input(fused)
        )

        base = self.base_head(source_features).reshape(
            batch_size,
            self.num_joints,
            3,
        )
        correction = self.correction_head(fused).reshape(
            batch_size,
            self.num_joints,
            3,
        )

        gate_logits = self.gate_head(fused).reshape(
            batch_size,
            1,
            1,
        )
        learned_gate = torch.sigmoid(gate_logits)

        gate = (
            learned_gate
            if gate_override is None
            else gate_override.reshape(batch_size, 1, 1)
        )
        final = base + gate * correction

        return {
            "base": base,
            "correction": correction,
            "gate_logits": gate_logits,
            "learned_gate": learned_gate,
            "used_gate": gate,
            "final": final,
        }
