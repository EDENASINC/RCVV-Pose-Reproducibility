from __future__ import annotations

import math

import torch
from torch import nn


def _inverse_tanh_fraction(value: float, bound: float) -> float:
    fraction = max(-0.95, min(0.95, float(value) / float(bound)))
    return math.atanh(fraction)


class GlobalContinuousCameraPolicy(nn.Module):
    """One bounded yaw/pitch pair shared by every sample."""

    policy_name = "global_continuous_policy"

    def __init__(
        self,
        initial_yaw_degrees: float,
        initial_pitch_degrees: float,
        yaw_bound_degrees: float = 45.0,
        pitch_bound_degrees: float = 10.0,
    ) -> None:
        super().__init__()
        self.yaw_bound_degrees = float(yaw_bound_degrees)
        self.pitch_bound_degrees = float(pitch_bound_degrees)
        self.raw_yaw = nn.Parameter(
            torch.tensor(
                _inverse_tanh_fraction(
                    initial_yaw_degrees,
                    self.yaw_bound_degrees,
                ),
                dtype=torch.float32,
            )
        )
        self.raw_pitch = nn.Parameter(
            torch.tensor(
                _inverse_tanh_fraction(
                    initial_pitch_degrees,
                    self.pitch_bound_degrees,
                ),
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        observed_pose2d_standardized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = observed_pose2d_standardized.shape[0]
        yaw = self.yaw_bound_degrees * torch.tanh(self.raw_yaw)
        pitch = self.pitch_bound_degrees * torch.tanh(self.raw_pitch)
        return yaw.expand(batch_size), pitch.expand(batch_size)


class PoseConditionedContinuousCameraPolicy(nn.Module):
    """A small MLP predicts bounded yaw and pitch from the observed pose."""

    policy_name = "pose_conditioned_continuous_policy"

    def __init__(
        self,
        initial_yaw_degrees: float,
        initial_pitch_degrees: float,
        hidden_width: int = 128,
        dropout: float = 0.10,
        yaw_bound_degrees: float = 45.0,
        pitch_bound_degrees: float = 10.0,
    ) -> None:
        super().__init__()
        self.yaw_bound_degrees = float(yaw_bound_degrees)
        self.pitch_bound_degrees = float(pitch_bound_degrees)
        self.network = nn.Sequential(
            nn.Linear(14 * 2, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, 2),
        )
        output = self.network[-1]
        nn.init.zeros_(output.weight)
        with torch.no_grad():
            output.bias.copy_(
                torch.tensor(
                    [
                        _inverse_tanh_fraction(
                            initial_yaw_degrees,
                            self.yaw_bound_degrees,
                        ),
                        _inverse_tanh_fraction(
                            initial_pitch_degrees,
                            self.pitch_bound_degrees,
                        ),
                    ],
                    dtype=output.bias.dtype,
                )
            )

    def forward(
        self,
        observed_pose2d_standardized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(
            observed_pose2d_standardized.flatten(start_dim=1)
        )
        yaw = self.yaw_bound_degrees * torch.tanh(raw[:, 0])
        pitch = self.pitch_bound_degrees * torch.tanh(raw[:, 1])
        return yaw, pitch


def build_policy(
    policy_name: str,
    *,
    initial_yaw_degrees: float,
    initial_pitch_degrees: float,
    hidden_width: int = 128,
    dropout: float = 0.10,
) -> nn.Module:
    if policy_name == "global_continuous_policy":
        return GlobalContinuousCameraPolicy(
            initial_yaw_degrees=initial_yaw_degrees,
            initial_pitch_degrees=initial_pitch_degrees,
        )
    if policy_name == "pose_conditioned_continuous_policy":
        return PoseConditionedContinuousCameraPolicy(
            initial_yaw_degrees=initial_yaw_degrees,
            initial_pitch_degrees=initial_pitch_degrees,
            hidden_width=hidden_width,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported policy_name={policy_name!r}.")


def yaw_pitch_to_rotation(
    yaw_degrees: torch.Tensor,
    pitch_degrees: torch.Tensor,
) -> torch.Tensor:
    """
    Exact SO(3) rotation using the project convention:
        R = Ry(yaw) @ Rx(pitch)
    """
    if yaw_degrees.shape != pitch_degrees.shape:
        raise ValueError("Yaw and pitch shapes differ.")
    yaw = torch.deg2rad(yaw_degrees)
    pitch = torch.deg2rad(pitch_degrees)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    zeros = torch.zeros_like(cy)
    ones = torch.ones_like(cy)

    ry = torch.stack(
        [
            cy, zeros, sy,
            zeros, ones, zeros,
            -sy, zeros, cy,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    rx = torch.stack(
        [
            ones, zeros, zeros,
            zeros, cp, -sp,
            zeros, sp, cp,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    return torch.matmul(ry, rx)
