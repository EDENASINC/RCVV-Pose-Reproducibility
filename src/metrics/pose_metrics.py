from __future__ import annotations

import torch


def mpjpe(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Mean per-joint position error.

    predicted and target must have shape [B, J, 3] and use the same unit.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: predicted={tuple(predicted.shape)}, "
            f"target={tuple(target.shape)}."
        )
    if predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError(
            f"Expected [B, J, 3], found {tuple(predicted.shape)}."
        )
    return torch.linalg.vector_norm(predicted - target, dim=-1).mean()


def per_joint_position_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return per-joint Euclidean errors with shape [B, J]."""
    if predicted.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: predicted={tuple(predicted.shape)}, "
            f"target={tuple(target.shape)}."
        )
    return torch.linalg.vector_norm(predicted - target, dim=-1)


def pa_mpjpe(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Procrustes-aligned MPJPE with per-sample similarity alignment.

    The function aligns predicted poses to target poses using translation,
    rotation, and uniform scale, then returns the mean Euclidean error.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: predicted={tuple(predicted.shape)}, "
            f"target={tuple(target.shape)}."
        )
    if predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError(
            f"Expected [B, J, 3], found {tuple(predicted.shape)}."
        )

    mu_pred = predicted.mean(dim=1, keepdim=True)
    mu_tgt = target.mean(dim=1, keepdim=True)

    pred_centered = predicted - mu_pred
    tgt_centered = target - mu_tgt

    pred_norm = torch.linalg.vector_norm(
        pred_centered.reshape(predicted.shape[0], -1),
        dim=1,
        keepdim=True,
    ).clamp_min(eps)

    tgt_norm = torch.linalg.vector_norm(
        tgt_centered.reshape(target.shape[0], -1),
        dim=1,
        keepdim=True,
    ).clamp_min(eps)

    pred_normalized = pred_centered / pred_norm[:, None, :]
    tgt_normalized = tgt_centered / tgt_norm[:, None, :]

    covariance = pred_normalized.transpose(1, 2) @ tgt_normalized
    u, singular_values, vh = torch.linalg.svd(covariance)

    v = vh.transpose(1, 2)
    ut = u.transpose(1, 2)

    det = torch.det(v @ ut)
    correction = torch.ones(
        predicted.shape[0],
        3,
        device=predicted.device,
        dtype=predicted.dtype,
    )
    correction[:, -1] = torch.where(det < 0, -1.0, 1.0)
    correction_matrix = torch.diag_embed(correction)

    rotation = v @ correction_matrix @ ut

    trace = (
        singular_values * correction
    ).sum(dim=1, keepdim=True)

    scale = (
        trace * tgt_norm / pred_norm
    ).reshape(-1, 1, 1)

    aligned = (
        scale * (pred_centered @ rotation.transpose(1, 2))
        + mu_tgt
    )

    return torch.linalg.vector_norm(aligned - target, dim=-1).mean()
