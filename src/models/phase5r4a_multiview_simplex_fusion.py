from __future__ import annotations

import math

import torch
from torch import nn


class StaticMultiViewSimplexFusion(nn.Module):
    """
    Convex fusion over one observed and several frozen virtual-view branches.

    global:
        One softmax weight vector is shared by every non-root joint.

    jointwise:
        Each non-root joint has its own softmax weight vector.  This is the
        main Phase 5R.4A hypothesis because the Official-Test diagnostic
        showed upper-limb gains but ankle regressions.
    """

    def __init__(
        self,
        *,
        mode: str,
        view_count: int = 5,
        joint_count: int = 14,
        observed_initial_weight: float = 0.50,
    ) -> None:
        super().__init__()
        if mode not in {"global", "jointwise"}:
            raise ValueError(f"Unsupported fusion mode: {mode!r}")
        if view_count < 2:
            raise ValueError("view_count must be at least 2.")
        if joint_count < 1:
            raise ValueError("joint_count must be positive.")
        if not 0.0 < observed_initial_weight < 1.0:
            raise ValueError("observed_initial_weight must be in (0,1).")

        self.mode = mode
        self.view_count = int(view_count)
        self.joint_count = int(joint_count)
        remaining = (1.0 - observed_initial_weight) / (view_count - 1)
        initial = torch.full(
            (view_count,),
            math.log(remaining),
            dtype=torch.float32,
        )
        initial[0] = math.log(observed_initial_weight)
        if mode == "global":
            self.logits = nn.Parameter(initial)
        else:
            self.logits = nn.Parameter(
                initial.unsqueeze(0).repeat(joint_count, 1)
            )

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def forward(
        self,
        branch_predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_tail = (
            self.view_count,
            self.joint_count,
            3,
        )
        if (
            branch_predictions.ndim != 4
            or tuple(branch_predictions.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "Expected branch predictions [B,"
                f"{self.view_count},{self.joint_count},3], got "
                f"{tuple(branch_predictions.shape)}."
            )
        weights = self.weights()
        if self.mode == "global":
            prediction = (
                branch_predictions
                * weights[None, :, None, None]
            ).sum(dim=1)
        else:
            prediction = (
                branch_predictions
                * weights.transpose(0, 1)[None, :, :, None]
            ).sum(dim=1)
        return prediction, weights
