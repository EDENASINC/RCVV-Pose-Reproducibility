from __future__ import annotations

import argparse
import bisect
import csv
import gc
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PHASE = "9C-R1D"
SPLITS = ("split_a", "split_b", "split_c")
SEEDS = (42, 123, 2026)
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
ARMS = (
    "detector_aware_observed",
    "detector_conditioned_virtual_view_no_confidence",
    "bounded_reliability_dual_modulation",
)
EXPECTED_B1_SHA256 = "fc1ae8f11ec6fb1b75c8d8c730e61a67f1b1aa45422227364e4bce71eb72b673"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_b1(root: Path, expected_hash: str):
    path = root / "scripts/train/run_phase9c_b1_four_arm_smoke.py"
    actual = sha256_file(path)
    if actual != expected_hash or actual != EXPECTED_B1_SHA256:
        raise ValueError(f"Phase 9C-B1 V3 source hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("phase9c_b1_v3_r1d", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_strict_gpu(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value in {"cuda", "cuda:0"}:
        value = "cuda:0"
    device = torch.device(value)
    if device.type != "cuda" or device.index not in {None, 0}:
        raise RuntimeError("Phase 9C-R1D is locked to --device cuda:0.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden.")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_worker_r1d(_: int) -> None:
    # This initializer belongs to the real script module and is Windows-spawn safe.
    torch.set_num_threads(1)


def loader_kwargs(workers: int) -> dict[str, Any]:
    return {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 1,
        "worker_init_fn": init_worker_r1d,
    }


def iterate_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    iterator = iter(loader)
    try:
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                break
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()


class DetectorConditionedSource(Dataset):
    def __init__(
        self,
        *,
        base: Dataset,
        virtual_cache: dict[str, np.ndarray],
        partition: str,
    ) -> None:
        code = {"train": 0, "val": 1, "test": 2}[partition]
        mask = virtual_cache["partition_code"] == code
        cache_dataset = virtual_cache["dataset_index"][mask].astype(np.int64, copy=False)
        cache_detector = virtual_cache["detector_sample_index"][mask].astype(np.int64, copy=False)
        cache_rows = np.flatnonzero(mask).astype(np.int64, copy=False)
        base_dataset = np.asarray(base.dataset_index, dtype=np.int64)
        base_detector = np.asarray(base.detector_index, dtype=np.int64)
        if not np.array_equal(cache_dataset, base_dataset):
            raise ValueError(f"{partition}: R1A dataset_index alignment mismatch.")
        if not np.array_equal(cache_detector, base_detector):
            raise ValueError(f"{partition}: R1A detector_sample_index alignment mismatch.")
        self.base = base
        self.cache_rows = cache_rows
        self.virtual = virtual_cache["virtual_pose_camera_root"]
        self.full_aligned_samples = int(cache_dataset.size)

    def __len__(self) -> int:
        return int(self.cache_rows.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        source = self.base[item]
        source.pop("teacher_virtual", None)
        virtual = self.virtual[:, int(self.cache_rows[item])].astype(np.float32, copy=False)
        source["detector_virtual"] = torch.from_numpy(virtual)
        return source


class MaterializedDataset(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        sizes = {int(value.shape[0]) for value in tensors.values()}
        if len(sizes) != 1:
            raise ValueError(f"Materialized tensor length mismatch: {sorted(sizes)}")
        self.tensors = tensors
        self.size = sizes.pop()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        return {name: value[item] for name, value in self.tensors.items()}


def materialize_full(source: DetectorConditionedSource, partition: str) -> MaterializedDataset:
    size = len(source)
    if size <= 0:
        raise ValueError(f"Empty {partition} dataset.")
    print(f"[MATERIALIZE] partition={partition} samples={size} workers=0 preallocated=yes")
    tensors = {
        "clean_pose2d": torch.empty((size, 15, 2), dtype=torch.float32),
        "target3d_mm": torch.empty((size, 15, 3), dtype=torch.float32),
        "intrinsic": torch.empty((size, 3, 3), dtype=torch.float32),
        "detector_index": torch.empty((size,), dtype=torch.long),
        "detector_virtual": torch.empty((size, 2, 15, 2), dtype=torch.float32),
    }
    for index in range(size):
        item = source[index]
        for name, destination in tensors.items():
            destination[index].copy_(item[name])
        if (index + 1) % 8192 == 0 or index + 1 == size:
            print(f"[MATERIALIZE] partition={partition} progress={index + 1}/{size}")
    expected = {
        "clean_pose2d": (15, 2),
        "target3d_mm": (15, 3),
        "intrinsic": (3, 3),
        "detector_index": (),
        "detector_virtual": (2, 15, 2),
    }
    if set(tensors) != set(expected):
        raise ValueError(f"Materialized fields mismatch: {sorted(tensors)}")
    for name, suffix in expected.items():
        tensor = tensors[name]
        if tuple(tensor.shape[1:]) != suffix:
            raise ValueError(f"Unexpected {partition}/{name}: {tuple(tensor.shape)}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite values in {partition}/{name}")
    root = tensors["detector_virtual"][:, :, 0, :]
    if not bool(torch.equal(root, torch.zeros_like(root))):
        raise ValueError(f"{partition}: detector-conditioned virtual root is not zero.")
    return MaterializedDataset(tensors)


def risk_joint_scale(residual_risk: np.ndarray, floor: float) -> np.ndarray:
    if residual_risk.shape != (1, 2, 15, 10):
        raise ValueError(f"Unexpected split-sliced residual risk: {residual_risk.shape}")
    scale = np.empty((2, 15), dtype=np.float32)
    for detector_index in range(2):
        for joint in range(15):
            values = residual_risk[0, detector_index, joint]
            positive = values[np.isfinite(values) & (values > 0)]
            scale[detector_index, joint] = max(
                float(np.median(positive)) if positive.size else floor,
                floor,
            )
    if not np.isfinite(scale).all() or bool((scale < floor).any()):
        raise ValueError("Invalid bounded-risk joint scale.")
    return scale


def feature_dim(arm: str) -> int:
    return {
        "detector_aware_observed": 28,
        "detector_conditioned_virtual_view_no_confidence": 56,
        "bounded_reliability_dual_modulation": 126,
    }[arm]


def make_features(
    b1: Any,
    batch: dict[str, torch.Tensor],
    *,
    split: str,
    arm: str,
    detector: str,
    detector_index: int,
    detector_cache: dict[str, np.ndarray],
    calibration_models: dict[str, Any],
    residual_risk: np.ndarray,
    joint_scale: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    intrinsic = batch["intrinsic"].to(device, non_blocking=True)
    indices = batch["detector_index"].cpu().numpy().astype(np.int64)
    prediction_px = torch.from_numpy(
        detector_cache["prediction_c15_px"][detector_index, indices]
    ).to(device=device, dtype=torch.float32)
    observed = b1.normalized_root_from_pixels_torch(prediction_px, intrinsic)[:, 1:, :]
    virtual = batch["detector_virtual"][:, detector_index, 1:, :].to(device, non_blocking=True)
    observed_flat = observed.reshape(observed.shape[0], -1)
    virtual_flat = virtual.reshape(virtual.shape[0], -1)
    if arm == "detector_aware_observed":
        features = observed_flat
    elif arm == "detector_conditioned_virtual_view_no_confidence":
        features = torch.cat([observed_flat, virtual_flat], dim=1)
    else:
        raw_conf = detector_cache["confidence_c15"][detector_index, indices].astype(np.float32)
        calibrated = b1.calibration_probability(calibration_models, split, detector, raw_conf)[:, 1:]
        risk = b1.residual_risk_features(residual_risk, 0, detector_index, raw_conf)[:, 1:]
        scale = joint_scale[detector_index, 1:][None, :]
        quality = 1.0 / (1.0 + np.maximum(risk, 0.0) / np.maximum(scale, 1e-6))
        reliability = np.clip(calibrated * quality, 0.0, 1.0)
        for name, values in {
            "calibrated": calibrated,
            "quality": quality,
            "reliability": reliability,
        }.items():
            if not np.isfinite(values).all() or bool((values < 0).any()) or bool((values > 1).any()):
                raise RuntimeError(f"Invalid bounded confidence tensor: {name}")
        reliability_t = torch.from_numpy(reliability.astype(np.float32)).to(device)
        gate = reliability_t.unsqueeze(-1)
        if arm == "bounded_reliability_dual_modulation":
            features = torch.cat([
                observed_flat,
                virtual_flat,
                (observed * gate).reshape(observed.shape[0], -1),
                (virtual * gate).reshape(virtual.shape[0], -1),
                reliability_t,
            ], dim=1)
        else:
            raise ValueError(arm)
    if features.shape[1] != feature_dim(arm):
        raise RuntimeError(f"Feature dimension mismatch: {arm} -> {features.shape}")
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError(f"Non-finite features: {split}/{arm}/{detector}")
    return features


def pa_mpjpe_per_sample_mm(prediction_mm: torch.Tensor, target_mm: torch.Tensor) -> torch.Tensor:
    x = prediction_mm.double()
    y = target_mm.double()
    mean_x = x.mean(dim=1, keepdim=True)
    mean_y = y.mean(dim=1, keepdim=True)
    x0 = x - mean_x
    y0 = y - mean_y
    norm_x = torch.sqrt(torch.clamp((x0 * x0).sum((1, 2), keepdim=True), min=1e-12))
    norm_y = torch.sqrt(torch.clamp((y0 * y0).sum((1, 2), keepdim=True), min=1e-12))
    u, singular, vh = torch.linalg.svd((x0 / norm_x).transpose(1, 2) @ (y0 / norm_y))
    rotation = u @ vh
    sign = torch.where(
        torch.det(rotation) < 0,
        -torch.ones(x.shape[0], device=x.device, dtype=x.dtype),
        torch.ones(x.shape[0], device=x.device, dtype=x.dtype),
    )
    u[:, :, -1] *= sign[:, None]
    singular[:, -1] *= sign
    rotation = u @ vh
    scale = (
        singular.sum(dim=1, keepdim=True)
        * norm_y.flatten(start_dim=1)
        / norm_x.flatten(start_dim=1)
    ).reshape(-1, 1, 1)
    aligned = scale * (x0 @ rotation) + mean_y
    return torch.linalg.vector_norm(aligned - y, dim=-1).mean(dim=1).float()


@torch.inference_mode()
def evaluate(
    *,
    b1: Any,
    model: nn.Module,
    dataset: Dataset,
    split: str,
    arm: str,
    detector: str,
    detector_index: int,
    detector_cache: dict[str, np.ndarray],
    calibration_models: dict[str, Any],
    residual_risk: np.ndarray,
    joint_scale: np.ndarray,
    device: torch.device,
    batch_size: int,
    workers: int,
    collect_per_sample: bool,
) -> tuple[dict[str, float | int], dict[str, np.ndarray] | None]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(workers))
    model.eval()
    mpjpe_sum = pa_sum = 0.0
    sample_sum = 0
    index_parts: list[np.ndarray] = []
    mpjpe_parts: list[np.ndarray] = []
    pa_parts: list[np.ndarray] = []
    for batch in iterate_loader(loader):
        features = make_features(
            b1, batch, split=split, arm=arm, detector=detector,
            detector_index=detector_index, detector_cache=detector_cache,
            calibration_models=calibration_models, residual_risk=residual_risk,
            joint_scale=joint_scale, device=device,
        )
        prediction = model(features) * 1000.0
        target = batch["target3d_mm"].to(device, non_blocking=True)[:, 1:, :]
        mpjpe = torch.linalg.vector_norm(prediction - target, dim=-1).mean(dim=1)
        pa = pa_mpjpe_per_sample_mm(prediction, target)
        if not bool(torch.isfinite(mpjpe).all()) or not bool(torch.isfinite(pa).all()):
            raise RuntimeError(f"Non-finite validation metric: {split}/{arm}/{detector}")
        mpjpe_sum += float(mpjpe.sum().item())
        pa_sum += float(pa.sum().item())
        sample_sum += int(prediction.shape[0])
        if collect_per_sample:
            index_parts.append(batch["detector_index"].cpu().numpy().astype(np.int64))
            mpjpe_parts.append(mpjpe.cpu().numpy().astype(np.float32))
            pa_parts.append(pa.cpu().numpy().astype(np.float32))
    if sample_sum != len(dataset):
        raise RuntimeError(f"Validation coverage mismatch: {sample_sum} != {len(dataset)}")
    metrics = {
        "mpjpe_mm": mpjpe_sum / sample_sum,
        "pa_mpjpe_mm": pa_sum / sample_sum,
        "samples": sample_sum,
    }
    per_sample = None
    if collect_per_sample:
        per_sample = {
            "detector_sample_index": np.concatenate(index_parts),
            "mpjpe_mm": np.concatenate(mpjpe_parts),
            "pa_mpjpe_mm": np.concatenate(pa_parts),
        }
        if not np.isfinite(per_sample["mpjpe_mm"]).all() or not np.isfinite(per_sample["pa_mpjpe_mm"]).all():
            raise RuntimeError("Non-finite per-sample metric.")
    return metrics, per_sample


def selection_score(metrics: dict[str, float | int]) -> float:
    mpjpe = float(metrics["mpjpe_mm"])
    pa = float(metrics["pa_mpjpe_mm"])
    if not math.isfinite(mpjpe) or not math.isfinite(pa) or min(mpjpe, pa) <= 0:
        raise ValueError(f"Invalid validation metrics: {metrics}")
    return math.sqrt(mpjpe * pa)


def save_npz(path: Path, payload: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return sha256_file(path)


def valid_existing_run(path: Path, protocol_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        report = load_json(path)
        if report.get("status") != "PASS" or report.get("format") != "phase9c_r1d_run_report_v1":
            return False
        if report.get("protocol_sha256") != protocol_sha:
            return False
        for artifact in report.get("artifact_hashes", []):
            target = Path(artifact["path"])
            if not target.is_file() or sha256_file(target) != artifact["sha256"]:
                return False
        return True
    except Exception:
        return False


def train_one(
    *,
    b1: Any,
    split: str,
    seed: int,
    arm: str,
    train_detector: str,
    train_detector_index: int,
    datasets: dict[str, Dataset],
    detector_cache: dict[str, np.ndarray],
    calibration_models: dict[str, Any],
    residual_risk: np.ndarray,
    joint_scale: np.ndarray,
    device: torch.device,
    output_dir: Path,
    protocol: dict[str, Any],
    protocol_sha: str,
) -> dict[str, Any]:
    report_path = output_dir / "phase9c_r1d_run_report.json"
    if valid_existing_run(report_path, protocol_sha):
        print(f"[REUSE] split={split} seed={seed} arm={arm} detector={train_detector}")
        return load_json(report_path)
    budget = protocol["training_budget"]
    set_seed(seed)
    model = b1.TinyPoseLifter(feature_dim(arm), hidden=int(budget["hidden_width"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(budget["learning_rate"]),
        weight_decay=float(budget["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "training_history.csv"
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = 0
    stale = 0
    started = time.time()
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, int(budget["max_epochs"]) + 1):
        train_loader = DataLoader(
            datasets["train"],
            batch_size=int(budget["batch_size"]),
            shuffle=True,
            drop_last=True,
            generator=generator,
            **loader_kwargs(int(budget["num_workers"])),
        )
        model.train()
        loss_sum = 0.0
        sample_sum = 0
        batch_count = 0
        for batch in iterate_loader(train_loader):
            features = make_features(
                b1, batch, split=split, arm=arm, detector=train_detector,
                detector_index=train_detector_index, detector_cache=detector_cache,
                calibration_models=calibration_models, residual_risk=residual_risk,
                joint_scale=joint_scale, device=device,
            )
            target = batch["target3d_mm"].to(device, non_blocking=True)[:, 1:, :] / 1000.0
            loss = loss_fn(model(features), target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss: {split}/{arm}/{train_detector}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(budget["gradient_clip_norm"]))
            optimizer.step()
            count = int(target.shape[0])
            loss_sum += float(loss.item()) * count
            sample_sum += count
            batch_count += 1
        matched, _ = evaluate(
            b1=b1, model=model, dataset=datasets["val"], split=split, arm=arm,
            detector=train_detector, detector_index=train_detector_index,
            detector_cache=detector_cache, calibration_models=calibration_models,
            residual_risk=residual_risk, joint_scale=joint_scale, device=device,
            batch_size=int(budget["eval_batch_size"]), workers=int(budget["num_workers"]),
            collect_per_sample=False,
        )
        score = selection_score(matched)
        improved = score < best_score - 1e-9
        checkpoint = {
            "format": "phase9c_r1d_locked_full_checkpoint_v1",
            "phase": PHASE,
            "split": split,
            "seed": seed,
            "arm": arm,
            "train_detector": train_detector,
            "epoch": epoch,
            "validation_selection_score": score,
            "validation_metrics": matched,
            "feature_dim": feature_dim(arm),
            "hidden_width": int(budget["hidden_width"]),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "protocol_sha256": protocol_sha,
        }
        torch.save(checkpoint, last_path)
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(checkpoint, best_path)
        else:
            stale += 1
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(1, sample_sum),
            "train_samples": sample_sum,
            "train_batches": batch_count,
            "val_mpjpe_mm": matched["mpjpe_mm"],
            "val_pa_mpjpe_mm": matched["pa_mpjpe_mm"],
            "val_selection_score": score,
            "improved": improved,
        })
        print(
            f"[EPOCH] {split} seed={seed} arm={arm} detector={train_detector} epoch={epoch} "
            f"loss={history[-1]['train_loss']:.7f} val={matched['mpjpe_mm']:.3f}/"
            f"{matched['pa_mpjpe_mm']:.3f} best_epoch={best_epoch}"
        )
        if stale >= int(budget["early_stopping_patience"]):
            print(f"[EARLY-STOP] epoch={epoch} best_epoch={best_epoch}")
            break
    if not best_path.is_file():
        raise RuntimeError("Best checkpoint was not created.")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    evaluations: dict[str, Any] = {}
    artifact_hashes: list[dict[str, str]] = []
    for eval_index, eval_detector in enumerate(DETECTORS):
        evaluations[eval_detector] = {}
        for partition in ("val", "test"):
            metrics, per_sample = evaluate(
                b1=b1, model=model, dataset=datasets[partition], split=split, arm=arm,
                detector=eval_detector, detector_index=eval_index,
                detector_cache=detector_cache, calibration_models=calibration_models,
                residual_risk=residual_risk, joint_scale=joint_scale, device=device,
                batch_size=int(budget["eval_batch_size"]), workers=int(budget["num_workers"]),
                collect_per_sample=True,
            )
            if per_sample is None:
                raise RuntimeError("Per-sample metrics were not collected.")
            per_sample_path = output_dir / f"per_sample_{eval_detector}_{partition}.npz"
            digest = save_npz(per_sample_path, per_sample)
            evaluations[eval_detector][partition] = {
                **metrics,
                "per_sample_path": str(per_sample_path),
                "per_sample_sha256": digest,
            }
            artifact_hashes.append({"path": str(per_sample_path), "sha256": digest})
    for path in (best_path, last_path, history_path):
        artifact_hashes.append({"path": str(path), "sha256": sha256_file(path)})
    report = {
        "status": "PASS",
        "format": "phase9c_r1d_run_report_v1",
        "phase": PHASE,
        "scientific_role": "locked_multisplit_multiseed_confirmation_and_post_selection_development_test",
        "split": split,
        "seed": seed,
        "arm": arm,
        "train_detector": train_detector,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_selection_score": best_score,
        "checkpoint_selection": "matched_validation_geometric_mean_mpjpe_pa_mpjpe",
        "feature_dim": feature_dim(arm),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "evaluations": evaluations,
        "test_metrics_computed": True,
        "test_used_for_selection": False,
        "protocol_sha256": protocol_sha,
        "artifact_hashes": artifact_hashes,
        "elapsed_sec": time.time() - started,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 9C-R1D locked full confirmation for one split.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-name", choices=SPLITS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.project_root.resolve()
    split = args.split_name
    output_root = root / "outputs/phase9c_r1d_locked_full_confirmation"
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "configs/phase9c_r1d_locked_full_confirmation_protocol.json"
    protocol = load_json(protocol_path)
    preflight = load_json(output_root / "phase9c_r1d_preflight_report.json")
    if protocol.get("status") != "LOCKED" or protocol.get("format") != "phase9c_r1d_locked_full_confirmation_protocol_v1":
        raise ValueError("Invalid Phase 9C-R1D protocol lock.")
    if preflight.get("status") != "PASS" or preflight.get("scientific_decision") != "READY_FOR_PHASE9C_R1D_FULL_RUNS":
        raise ValueError("Phase 9C-R1D preflight is not PASS.")
    protocol_sha = sha256_file(protocol_path)
    if preflight.get("protocol_sha256") != protocol_sha:
        raise ValueError("Protocol changed after preflight; rerun preflight.")
    if tuple(protocol["arms"]) != ARMS or tuple(protocol["detectors"]) != DETECTORS:
        raise ValueError("Locked arm/detector matrix mismatch.")
    budget = protocol["training_budget"]
    if int(budget["num_workers"]) != 8 or int(budget["max_train_samples"]) != 0 or int(budget["max_eval_samples"]) != 0:
        raise ValueError("R1D requires workers=8 and full train/validation/test cohorts.")

    device = resolve_strict_gpu(args.device)
    b1 = load_b1(root, str(protocol["source_b1_script_sha256"]))
    b1.SPLIT = split
    artifacts = b1.find_required_artifacts(root)
    phase9ca_qc = load_json(artifacts["phase9c_a_qc"])
    phase9ca_build = load_json(artifacts["phase9c_a_build"])
    phase9cb0_qc = load_json(artifacts["phase9c_b0_qc"])
    if phase9ca_qc.get("status") != "PASS" or phase9cb0_qc.get("status") != "PASS":
        raise ValueError("Phase 9C-A/B0 prerequisite gate failed.")
    with np.load(artifacts["phase9c_b0_join_cache"], allow_pickle=False) as handle:
        join_cache = {name: handle[name] for name in handle.files}
    detector_cache, detector_location, detector_hash = b1.load_phase9b_cache(
        root, str(phase9ca_build["source_hashes"]["phase9b_detector_cache.npz"])
    )
    detection_audit, common_valid = b1.build_common_detection_audit(detector_cache, join_cache)
    calibration_models = load_json(artifacts["phase9c_a_models"])
    with np.load(artifacts["phase9c_a_residual_bank"], allow_pickle=False) as handle:
        residual_bank = {name: handle[name] for name in handle.files}
    all_risk = b1.residual_risk_table(residual_bank)
    split_index = SPLITS.index(split)
    residual_risk = all_risk[split_index : split_index + 1].copy()
    joint_scale = risk_joint_scale(
        residual_risk,
        float(protocol["calibration_and_risk_policy"]["risk_scale_floor"]),
    )

    r1a_root = root / "outputs/phase9c_r1a_detector_conditioned_virtual_cache"
    r1a_cache_path = r1a_root / split / "phase9c_r1a_detector_virtual_cache.npz"
    r1a_report = load_json(r1a_cache_path.with_suffix(".report.json"))
    if r1a_report.get("protocol_sha256") != protocol["source_r1a_protocol_sha256"]:
        raise ValueError("R1A protocol hash mismatch.")
    if r1a_report.get("cache_sha256") != sha256_file(r1a_cache_path):
        raise ValueError("R1A cache hash mismatch.")
    with np.load(r1a_cache_path, allow_pickle=False) as handle:
        virtual_cache = {name: handle[name] for name in handle.files}
    if str(virtual_cache["source_input"].item()) != "phase9b_prediction_c15_px_real_detector":
        raise ValueError("R1A virtual view is not detector-conditioned.")
    if tuple(str(value) for value in virtual_cache["detector_id"].tolist()) != DETECTORS:
        raise ValueError("R1A detector order mismatch.")

    manifest = root / "data/processed/caches/phase5r1a_multigeometry_oof_25fps_v1" / split / "manifest.json"
    base = {
        partition: b1.JoinedDetectorDataset(
            manifest_path=manifest,
            join_cache=join_cache,
            common_valid_detector_sample=common_valid,
            partition=partition,
            max_samples=0,
            seed=42,
        )
        for partition in ("train", "val", "test")
    }
    sources = {
        partition: DetectorConditionedSource(base=base[partition], virtual_cache=virtual_cache, partition=partition)
        for partition in ("train", "val", "test")
    }
    alignment = {
        partition: {
            "full_aligned_samples": source.full_aligned_samples,
            "selected_samples": len(source),
        }
        for partition, source in sources.items()
    }
    datasets = {partition: materialize_full(source, partition) for partition, source in sources.items()}
    del base, sources, virtual_cache
    gc.collect()

    reports: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            for detector_index, detector in enumerate(DETECTORS):
                print(f"[RUN] split={split} seed={seed} arm={arm} detector={detector} workers=8")
                run_dir = output_root / split / f"seed{seed}" / arm / f"train_{detector}"
                reports.append(train_one(
                    b1=b1, split=split, seed=seed, arm=arm, train_detector=detector,
                    train_detector_index=detector_index, datasets=datasets,
                    detector_cache=detector_cache, calibration_models=calibration_models,
                    residual_risk=residual_risk, joint_scale=joint_scale, device=device,
                    output_dir=run_dir, protocol=protocol, protocol_sha=protocol_sha,
                ))
    split_report = {
        "status": "PASS",
        "format": "phase9c_r1d_split_report_v1",
        "phase": PHASE,
        "scientific_role": "locked_multisplit_multiseed_confirmation_and_post_selection_development_test",
        "split": split,
        "seeds": list(SEEDS),
        "run_count": len(reports),
        "workers": 8,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "dataset_alignment": alignment,
        "detector_coverage_audit": detection_audit,
        "detector_cache_location": detector_location,
        "detector_cache_sha256": detector_hash,
        "r1a_cache_sha256": sha256_file(r1a_cache_path),
        "residual_risk_split": split,
        "residual_risk_source_index": split_index,
        "residual_risk_scale": {
            detector: joint_scale[index].tolist() for index, detector in enumerate(DETECTORS)
        },
        "protocol_sha256": protocol_sha,
        "test_metrics_computed": True,
        "test_used_for_selection": False,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    write_json(output_root / split / "phase9c_r1d_split_report.json", split_report)
    print("=" * 112)
    print(f"PHASE 9C-R1D FULL CONFIRMATION {split.upper()}")
    print("=" * 112)
    print("Status                : PASS")
    print(f"Runs                  : {len(reports)} / 18")
    print("Checkpoint selection  : matched validation only")
    print("Development test role : post-selection evidence only")
    print("Official TS1-TS6 read : NO")
    print("=" * 112)


if __name__ == "__main__":
    main()
