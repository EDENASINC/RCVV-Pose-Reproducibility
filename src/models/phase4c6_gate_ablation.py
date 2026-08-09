from __future__ import annotations

import math

import torch
from torch import nn


class TrainableConstantGate(nn.Module):
    """
    A single learned scalar shared by every sample and every joint.

    This isolates the benefit of choosing a global interpolation level from
    the benefit of input-dependent confidence estimation.
    """

    uses_cycle_feature = False
    gate_mode = "constant"

    def __init__(self, initial_gate: float = 0.25, **_: object) -> None:
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be in (0, 1).")
        initial_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.gate_logit = nn.Parameter(
            torch.tensor(initial_logit, dtype=torch.float32)
        )

    def forward(
        self,
        observed_pose2d_standardized: torch.Tensor,
        synthesized_pose2d_standardized: torch.Tensor,
        source_to_target_rotation: torch.Tensor,
        source_to_target_translation_m: torch.Tensor,
        observed_prediction_standardized: torch.Tensor,
        synthesized_prediction_standardized: torch.Tensor,
        cycle_error_standardized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del (
            synthesized_pose2d_standardized,
            source_to_target_rotation,
            source_to_target_translation_m,
            cycle_error_standardized,
        )
        batch_size, joint_count, _ = observed_pose2d_standardized.shape
        gate = torch.sigmoid(self.gate_logit).expand(
            batch_size,
            joint_count,
            1,
        )
        prediction_delta = (
            synthesized_prediction_standardized
            - observed_prediction_standardized
        )
        prediction = (
            observed_prediction_standardized
            + gate * prediction_delta
        )
        return prediction, gate


class AblationConfidenceGate(nn.Module):
    """
    Same learned gate architecture and parameter count for both ablations.

    with_cycle:
        Uses cycle error locally and in global summary statistics.

    without_cycle:
        Replaces all cycle inputs by exact zeros while preserving network
        dimensions, parameter count, initialization order, and all other
        features.
    """

    def __init__(
        self,
        use_cycle_feature: bool,
        global_width: int = 128,
        joint_width: int = 128,
        dropout: float = 0.10,
        initial_gate: float = 0.25,
        translation_scale_m: float = 5.0,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be in (0, 1).")

        self.use_cycle_feature = bool(use_cycle_feature)
        self.uses_cycle_feature = bool(use_cycle_feature)
        self.gate_mode = (
            "with_cycle" if self.use_cycle_feature else "without_cycle"
        )
        self.translation_scale_m = float(translation_scale_m)

        # Geometry 12 + cycle/delta/prediction summaries 3*3 = 21.
        self.global_encoder = nn.Sequential(
            nn.Linear(21, global_width),
            nn.LayerNorm(global_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(global_width, global_width),
            nn.LayerNorm(global_width),
            nn.GELU(),
        )

        # Local width remains 18 in both modes.
        self.joint_network = nn.Sequential(
            nn.Linear(18 + global_width, joint_width),
            nn.LayerNorm(joint_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(joint_width, joint_width // 2),
            nn.GELU(),
            nn.Linear(joint_width // 2, 1),
        )

        output = self.joint_network[-1]
        nn.init.normal_(output.weight, mean=0.0, std=1e-3)
        nn.init.constant_(
            output.bias,
            math.log(initial_gate / (1.0 - initial_gate)),
        )

    @staticmethod
    def _summary(values: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                values.mean(dim=1),
                values.amax(dim=1),
                values.std(dim=1, unbiased=False),
            ],
            dim=1,
        )

    def forward(
        self,
        observed_pose2d_standardized: torch.Tensor,
        synthesized_pose2d_standardized: torch.Tensor,
        source_to_target_rotation: torch.Tensor,
        source_to_target_translation_m: torch.Tensor,
        observed_prediction_standardized: torch.Tensor,
        synthesized_prediction_standardized: torch.Tensor,
        cycle_error_standardized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = observed_pose2d_standardized.shape[0]
        expected_2d = (batch_size, 14, 2)
        expected_3d = (batch_size, 14, 3)

        if observed_pose2d_standardized.shape != expected_2d:
            raise ValueError("Observed 2D must have shape [B,14,2].")
        if synthesized_pose2d_standardized.shape != expected_2d:
            raise ValueError("Synthesized 2D must have shape [B,14,2].")
        if observed_prediction_standardized.shape != expected_3d:
            raise ValueError("Observed 3D must have shape [B,14,3].")
        if synthesized_prediction_standardized.shape != expected_3d:
            raise ValueError("Synthesized 3D must have shape [B,14,3].")
        if cycle_error_standardized.shape != (batch_size, 14):
            raise ValueError("Cycle error must have shape [B,14].")

        cycle_for_model = (
            cycle_error_standardized
            if self.use_cycle_feature
            else torch.zeros_like(cycle_error_standardized)
        )

        pose_delta = (
            synthesized_pose2d_standardized
            - observed_pose2d_standardized
        )
        pose_delta_norm = torch.linalg.vector_norm(
            pose_delta,
            dim=-1,
        )
        prediction_delta = (
            synthesized_prediction_standardized
            - observed_prediction_standardized
        )
        prediction_delta_norm = torch.linalg.vector_norm(
            prediction_delta,
            dim=-1,
        )

        identity = torch.eye(
            3,
            dtype=source_to_target_rotation.dtype,
            device=source_to_target_rotation.device,
        ).expand(batch_size, -1, -1)

        global_feature = torch.cat(
            [
                (
                    source_to_target_rotation - identity
                ).flatten(start_dim=1),
                source_to_target_translation_m
                / self.translation_scale_m,
                self._summary(cycle_for_model),
                self._summary(pose_delta_norm),
                self._summary(prediction_delta_norm),
            ],
            dim=1,
        )
        global_context = self.global_encoder(global_feature)
        global_context = global_context[:, None, :].expand(-1, 14, -1)

        local_feature = torch.cat(
            [
                observed_pose2d_standardized,
                synthesized_pose2d_standardized,
                pose_delta,
                pose_delta_norm[..., None],
                torch.log1p(
                    torch.clamp(cycle_for_model, min=0.0)
                )[..., None],
                prediction_delta_norm[..., None],
                observed_prediction_standardized,
                synthesized_prediction_standardized,
                prediction_delta,
            ],
            dim=-1,
        )

        gate_logits = self.joint_network(
            torch.cat([local_feature, global_context], dim=-1)
        )
        gate = torch.sigmoid(gate_logits)
        prediction = (
            observed_prediction_standardized
            + gate * prediction_delta
        )
        return prediction, gate


def build_gate_model(
    gate_mode: str,
    global_width: int = 128,
    joint_width: int = 128,
    dropout: float = 0.10,
    initial_gate: float = 0.25,
) -> nn.Module:
    if gate_mode == "constant":
        return TrainableConstantGate(initial_gate=initial_gate)
    if gate_mode == "without_cycle":
        return AblationConfidenceGate(
            use_cycle_feature=False,
            global_width=global_width,
            joint_width=joint_width,
            dropout=dropout,
            initial_gate=initial_gate,
        )
    if gate_mode == "with_cycle":
        return AblationConfidenceGate(
            use_cycle_feature=True,
            global_width=global_width,
            joint_width=joint_width,
            dropout=dropout,
            initial_gate=initial_gate,
        )
    raise ValueError(f"Unsupported gate_mode: {gate_mode}")
