from __future__ import annotations

import argparse
import bisect
import csv
import gc
import hashlib
import io
import json
import math
import random
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPLIT = "split_a"
SEED = 42
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
ARMS = (
    "clean_observed_legacy",
    "detector_aware_observed",
    "virtual_view_no_confidence",
    "calibrated_confidence_virtual_view",
)
PARTITIONS = ("train", "val", "test")
PARTITION_CODE = {name: index for index, name in enumerate(PARTITIONS)}
CANDIDATE_FIXED_M15 = 2


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_worker(_: int) -> None:
    # Workers only collate an already-materialized compact tensor cache.
    torch.set_num_threads(1)


def resolve_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        index = 0 if device.index is None else int(device.index)
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable.")
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
    return device


def canonical_partition(value: str) -> str:
    return {"validation": "val", "valid": "val"}.get(value.lower(), value.lower())


def find_required_artifacts(root: Path) -> dict[str, Path]:
    paths = {
        "phase9c_a_qc": root / "outputs/phase9c_a_calibrated_error_bank/phase9c_a_qc_report.json",
        "phase9c_a_build": root / "outputs/phase9c_a_calibrated_error_bank/phase9c_a_build_report.json",
        "phase9c_a_models": root / "outputs/phase9c_a_calibrated_error_bank/phase9c_a_calibration_models.json",
        "phase9c_a_residual_bank": root / "outputs/phase9c_a_calibrated_error_bank/phase9c_a_residual_bank.npz",
        "phase9c_b0_qc": root / "outputs/phase9c_b_detector_join_preflight/phase9c_b0_qc_report.json",
        "phase9c_b0_build": root / "outputs/phase9c_b_detector_join_preflight/phase9c_b0_join_build_report.json",
        "phase9c_b0_join_cache": root / "outputs/phase9c_b_detector_join_preflight/phase9c_b0_detector_join_cache.npz",
        "protocol": root / "configs/phase9c_b1_smoke_protocol.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 9C-B1 prerequisites: {missing}")
    return paths


def load_phase9b_cache(root: Path, expected_hash: str) -> tuple[dict[str, np.ndarray], str, str]:
    base = root / "outputs/phase9b_rgb_detector_benchmark"
    candidates = (
        base / "full/phase9b_detector_cache.npz",
        base / "phase9b_detector_cache.npz",
    )
    mismatches: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        digest = sha256_file(path)
        if digest != expected_hash:
            mismatches.append(f"{path}: {digest}")
            continue
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}, str(path), digest
    zip_path = base / "phase9b_rgb_detector_benchmark_results.zip"
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as archive:
            members = [name for name in archive.namelist() if name.endswith("phase9b_detector_cache.npz")]
            for member in sorted(members, key=lambda item: (0 if "/full/" in f"/{item}" else 1, len(item))):
                blob = archive.read(member)
                digest = sha256_bytes(blob)
                if digest != expected_hash:
                    mismatches.append(f"{zip_path}!{member}: {digest}")
                    continue
                with np.load(io.BytesIO(blob), allow_pickle=False) as data:
                    return {key: data[key] for key in data.files}, f"{zip_path}!{member}", digest
    raise FileNotFoundError("No Phase 9B detector cache matches Phase 9C-A hash.\n" + "\n".join(mismatches[:8]))


def build_common_detection_audit(
    detector_cache: dict[str, np.ndarray],
    join_cache: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    detector_ids = tuple(str(value) for value in detector_cache["detector_id"].tolist())
    if detector_ids != DETECTORS:
        raise ValueError(f"Unexpected detector order: {detector_ids}")
    predictions = detector_cache["prediction_c15_px"]
    confidence = detector_cache["confidence_c15"]
    detected = detector_cache["detected"].astype(bool, copy=False)
    if predictions.ndim != 4 or predictions.shape[0] != len(DETECTORS):
        raise ValueError(f"Unexpected detector prediction shape: {predictions.shape}")
    if confidence.shape != predictions.shape[:-1] or detected.shape != predictions.shape[:2]:
        raise ValueError(
            f"Detector cache shape mismatch: prediction={predictions.shape}, "
            f"confidence={confidence.shape}, detected={detected.shape}"
        )
    finite_prediction = np.isfinite(predictions).all(axis=(2, 3))
    finite_confidence = np.isfinite(confidence).all(axis=2)
    valid_by_detector = detected & finite_prediction & finite_confidence
    invalid_detected = detected & ~(finite_prediction & finite_confidence)
    if bool(invalid_detected.any()):
        counts = {
            DETECTORS[index]: int(invalid_detected[index].sum())
            for index in range(len(DETECTORS))
        }
        raise ValueError(f"Detected samples contain non-finite detector data: {counts}")
    common_valid = valid_by_detector.all(axis=0)
    partition_codes = join_cache[f"{SPLIT}_partition_code"]
    detector_indices = join_cache[f"{SPLIT}_detector_sample_index"].astype(np.int64, copy=False)
    if detector_indices.size != partition_codes.size:
        raise ValueError("Join cache partition/detector index length mismatch.")
    if detector_indices.size == 0 or int(detector_indices.min()) < 0 or int(detector_indices.max()) >= common_valid.size:
        raise ValueError("Join cache detector indices are outside the detector cache.")

    detector_stats: dict[str, Any] = {}
    mapped = detector_indices
    for detector_index, detector in enumerate(DETECTORS):
        detector_stats[detector] = {
            "joined_samples": int(mapped.size),
            "detected_samples": int(detected[detector_index, mapped].sum()),
            "missed_detections": int((~detected[detector_index, mapped]).sum()),
            "finite_prediction_samples": int(finite_prediction[detector_index, mapped].sum()),
            "nonfinite_prediction_samples": int((~finite_prediction[detector_index, mapped]).sum()),
            "finite_confidence_samples": int(finite_confidence[detector_index, mapped].sum()),
            "invalid_detected_samples": int(invalid_detected[detector_index, mapped].sum()),
            "detection_rate": float(detected[detector_index, mapped].mean()),
        }
    partition_stats: dict[str, Any] = {}
    for partition, code in PARTITION_CODE.items():
        mask = partition_codes == code
        indices = detector_indices[mask]
        eligible = common_valid[indices]
        partition_stats[partition] = {
            "joined_samples": int(indices.size),
            "paired_common_detected_samples": int(eligible.sum()),
            "excluded_any_detector_failure": int((~eligible).sum()),
            "paired_common_detection_rate": float(eligible.mean()) if indices.size else 0.0,
            "detectors": {
                detector: {
                    "detected_samples": int(detected[index, indices].sum()),
                    "missed_detections": int((~detected[index, indices]).sum()),
                }
                for index, detector in enumerate(DETECTORS)
            },
        }
    audit = {
        "policy": "paired_common_detection",
        "same_sample_cohort_across_arms_and_detectors": True,
        "ground_truth_imputation_for_detector_failures": False,
        "detectors": detector_stats,
        "common_cohort": {
            "joined_samples": int(mapped.size),
            "eligible_samples": int(common_valid[mapped].sum()),
            "excluded_any_detector_failure": int((~common_valid[mapped]).sum()),
            "eligibility_rate": float(common_valid[mapped].mean()),
        },
        "partitions": partition_stats,
    }
    return audit, common_valid


def normalized_root_from_pixels_torch(pixels: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((*pixels.shape[:-1], 1), dtype=pixels.dtype, device=pixels.device)
    homogeneous = torch.cat([pixels, ones], dim=-1)
    inverse = torch.linalg.inv(intrinsic.to(dtype=pixels.dtype))
    normalized_h = torch.einsum("bij,bkj->bki", inverse, homogeneous)
    normalized = normalized_h[..., :2] / torch.clamp(normalized_h[..., 2:3], min=1e-12)
    return normalized - normalized[:, 0:1, :]


class JoinedDetectorDataset(Dataset):
    def __init__(
        self,
        *,
        manifest_path: Path,
        join_cache: dict[str, np.ndarray],
        common_valid_detector_sample: np.ndarray,
        partition: str,
        max_samples: int,
        seed: int,
    ) -> None:
        self.manifest_path = manifest_path
        manifest = load_json(manifest_path)
        if manifest.get("format") != "phase5r1a_multigeometry_synthesized_manifest_v1":
            raise ValueError(f"Unexpected manifest format: {manifest_path}")
        if manifest.get("status") != "PASS" or bool(manifest.get("partial", False)):
            raise ValueError("Phase 9C-B1 requires a full PASS Phase 5R.1A cache.")
        self.shards = [
            record for record in manifest["shards"]
            if canonical_partition(str(record["split"])) == partition
        ]
        if not self.shards:
            raise ValueError(f"No shards for partition={partition}")
        starts: list[int] = []
        total = 0
        for record in self.shards:
            starts.append(total)
            total += int(record["sample_count"])
        self.starts = starts
        self.ends = [start + int(record["sample_count"]) for start, record in zip(starts, self.shards)]
        code = PARTITION_CODE[partition]
        partition_codes = join_cache[f"{SPLIT}_partition_code"]
        mask = np.flatnonzero(partition_codes == code)
        dataset_index = join_cache[f"{SPLIT}_dataset_index"][mask].astype(np.int64)
        detector_index = join_cache[f"{SPLIT}_detector_sample_index"][mask].astype(np.int64)
        order = np.argsort(dataset_index, kind="stable")
        dataset_index = dataset_index[order]
        detector_index = detector_index[order]
        eligible = common_valid_detector_sample[detector_index]
        self.joined_before_eligibility = int(dataset_index.size)
        dataset_index = dataset_index[eligible]
        detector_index = detector_index[eligible]
        self.excluded_by_common_detection = self.joined_before_eligibility - int(dataset_index.size)
        self.eligible_before_limit = int(dataset_index.size)
        if max_samples > 0 and dataset_index.size > max_samples:
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(dataset_index.size, size=max_samples, replace=False))
            dataset_index = dataset_index[selected]
            detector_index = detector_index[selected]
        self.dataset_index = dataset_index
        self.detector_index = detector_index
        self.cache: OrderedDict[int, tuple[dict[str, Any], dict[str, Any]]] = OrderedDict()

    def __len__(self) -> int:
        return int(self.dataset_index.size)

    def _load_shard(self, shard_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if shard_id in self.cache:
            item = self.cache.pop(shard_id)
            self.cache[shard_id] = item
            return item
        record = self.shards[shard_id]
        original = torch.load(record["original_path"], map_location="cpu", weights_only=False)
        cached = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
        if original.get("format") != "phase4b_multisubject_paired_shard_v1":
            raise ValueError(f"Unexpected original shard: {record['original_path']}")
        if cached.get("format") != "phase5r1a_multigeometry_synthesized_shard_v1":
            raise ValueError(f"Unexpected synthesized shard: {record['cache_path']}")
        self.cache[shard_id] = (original, cached)
        while len(self.cache) > 1:
            self.cache.popitem(last=False)
        return original, cached

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        global_index = int(self.dataset_index[item])
        shard_id = bisect.bisect_right(self.ends, global_index)
        if shard_id >= len(self.shards):
            raise IndexError(global_index)
        previous = self.starts[shard_id]
        local_index = global_index - previous
        original, cached = self._load_shard(shard_id)
        tensors = original["tensors"]
        detector_index = int(self.detector_index[item])
        result = {
            "clean_pose2d": tensors["target_pose2d_camera_root"][local_index].float(),
            "target3d_mm": tensors["target_pose3d_camera_root_mm"][local_index].float(),
            "intrinsic": tensors["target_intrinsic"][local_index].float(),
            "teacher_virtual": cached["synthesized_views_camera_root"][local_index, CANDIDATE_FIXED_M15].float(),
            "detector_index": torch.tensor(detector_index, dtype=torch.long),
        }
        return result


class MaterializedSmokeDataset(Dataset):
    """Small, spawn-safe tensor cache for DataLoader workers."""

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


def materialize_smoke_dataset(source: JoinedDetectorDataset, partition: str) -> MaterializedSmokeDataset:
    print(f"[MATERIALIZE] partition={partition} samples={len(source)} workers=0")
    columns: dict[str, list[torch.Tensor]] = {}
    for index in range(len(source)):
        item = source[index]
        for name, value in item.items():
            columns.setdefault(name, []).append(value)
        if (index + 1) % 4096 == 0 or index + 1 == len(source):
            print(f"[MATERIALIZE] partition={partition} progress={index + 1}/{len(source)}")
    tensors = {name: torch.stack(values, dim=0).contiguous() for name, values in columns.items()}
    expected_shapes = {
        "clean_pose2d": (15, 2),
        "target3d_mm": (15, 3),
        "intrinsic": (3, 3),
        "teacher_virtual": (15, 2),
        "detector_index": (),
    }
    if set(tensors) != set(expected_shapes):
        raise ValueError(f"Materialized fields mismatch: {sorted(tensors)}")
    for name, suffix in expected_shapes.items():
        if tuple(tensors[name].shape[1:]) != suffix:
            raise ValueError(f"Unexpected {partition}/{name} shape: {tuple(tensors[name].shape)}")
        if tensors[name].is_floating_point() and not bool(torch.isfinite(tensors[name]).all()):
            raise ValueError(f"Non-finite values in materialized {partition}/{name}")
    print(f"[PASS] materialized partition={partition} in main process")
    return MaterializedSmokeDataset(tensors)


def limited_batches(loader: DataLoader, max_batches: int):
    """Yield bounded batches and deterministically stop Windows workers."""
    iterator = iter(loader)
    try:
        for _ in range(max_batches):
            try:
                yield next(iterator)
            except StopIteration:
                break
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()


class TinyPoseLifter(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 14 * 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).reshape(-1, 14, 3)


def calibration_probability(models: dict[str, Any], split: str, detector: str, confidence: np.ndarray) -> np.ndarray:
    entry = models["splits"][split]["detectors"][detector]
    knots = np.asarray(entry["knots_raw_confidence"], dtype=np.float32)
    values = np.asarray(entry["values_probability_correct_at_0.10_torso"], dtype=np.float32)
    flat = confidence.reshape(-1)
    calibrated = np.interp(flat, knots, values, left=values[0], right=values[-1])
    return calibrated.reshape(confidence.shape).astype(np.float32)


def residual_risk_table(bank: dict[str, np.ndarray]) -> np.ndarray:
    residual = bank["residual_xy_torso"]
    counts = bank["reservoir_counts"]
    risk = np.zeros(residual.shape[:4], dtype=np.float32)
    for index in np.ndindex(risk.shape):
        count = int(counts[index])
        if count <= 0:
            continue
        vectors = residual[index][:count]
        risk[index] = float(np.mean(np.linalg.norm(vectors, axis=-1)))
    return risk


def residual_risk_features(
    risk: np.ndarray,
    split_index: int,
    detector_index: int,
    confidence: np.ndarray,
) -> np.ndarray:
    bins = np.clip((confidence * 10.0).astype(np.int64), 0, 9)
    out = np.zeros_like(confidence, dtype=np.float32)
    for joint in range(confidence.shape[1]):
        out[:, joint] = risk[split_index, detector_index, joint, bins[:, joint]]
    return out


def make_features(
    batch: dict[str, torch.Tensor],
    *,
    arm: str,
    detector: str,
    detector_index: int,
    detector_cache: dict[str, np.ndarray],
    calibration_models: dict[str, Any],
    residual_risk: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    clean = batch["clean_pose2d"].to(device, non_blocking=True)
    virtual = batch["teacher_virtual"].to(device, non_blocking=True)
    intrinsic = batch["intrinsic"].to(device, non_blocking=True)
    indices = batch["detector_index"].cpu().numpy().astype(np.int64)
    if arm == "clean_observed_legacy":
        features = clean[:, 1:, :].reshape(clean.shape[0], -1)
    else:
        prediction_px = torch.from_numpy(
            detector_cache["prediction_c15_px"][detector_index, indices]
        ).to(device=device, dtype=torch.float32)
        detector_pose = normalized_root_from_pixels_torch(prediction_px, intrinsic)
        raw_conf = detector_cache["confidence_c15"][detector_index, indices].astype(np.float32)
        calibrated = calibration_probability(calibration_models, SPLIT, detector, raw_conf)
        calibrated_t = torch.from_numpy(calibrated[:, 1:]).to(device=device, dtype=torch.float32)
        if arm == "detector_aware_observed":
            features = torch.cat([detector_pose[:, 1:, :].reshape(clean.shape[0], -1), calibrated_t], dim=1)
        elif arm == "virtual_view_no_confidence":
            features = torch.cat([
            detector_pose[:, 1:, :].reshape(clean.shape[0], -1),
            virtual[:, 1:, :].reshape(clean.shape[0], -1),
            ], dim=1)
        elif arm == "calibrated_confidence_virtual_view":
            risk_np = residual_risk_features(residual_risk, 0, detector_index, raw_conf)[:, 1:]
            risk_t = torch.from_numpy(risk_np).to(device=device, dtype=torch.float32)
            features = torch.cat([
                detector_pose[:, 1:, :].reshape(clean.shape[0], -1),
                virtual[:, 1:, :].reshape(clean.shape[0], -1),
                calibrated_t,
                risk_t,
            ], dim=1)
        else:
            raise ValueError(arm)
    if not bool(torch.isfinite(features).all()):
        bad = int((~torch.isfinite(features)).sum().item())
        raise RuntimeError(
            f"Non-finite model features after paired detection filtering: "
            f"arm={arm}, detector={detector}, values={bad}"
        )
    return features


def feature_dim(arm: str) -> int:
    return {
        "clean_observed_legacy": 28,
        "detector_aware_observed": 42,
        "virtual_view_no_confidence": 56,
        "calibrated_confidence_virtual_view": 84,
    }[arm]


def target_m(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    target = batch["target3d_mm"].to(device, non_blocking=True)[:, 1:, :] / 1000.0
    if not bool(torch.isfinite(target).all()):
        raise RuntimeError("Non-finite 3D training target.")
    return target


def pa_mpjpe_mm(prediction_mm: torch.Tensor, target_mm: torch.Tensor) -> torch.Tensor:
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
    sign = torch.where(torch.det(rotation) < 0, -torch.ones(x.shape[0], device=x.device), torch.ones(x.shape[0], device=x.device))
    u[:, :, -1] *= sign[:, None]
    singular[:, -1] *= sign
    rotation = u @ vh
    scale = (singular.sum(dim=1, keepdim=True) * norm_y.flatten(start_dim=1) / norm_x.flatten(start_dim=1)).reshape(-1, 1, 1)
    aligned = scale * (x0 @ rotation) + mean_y
    return torch.linalg.vector_norm(aligned - y, dim=-1).mean().float()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    arm: str,
    eval_detector: str,
    eval_detector_index: int,
    detector_cache: dict[str, np.ndarray],
    calibration_models: dict[str, Any],
    residual_risk: np.ndarray,
    device: torch.device,
    max_batches: int,
) -> dict[str, float | int]:
    model.eval()
    mpjpe_sum = 0.0
    pa_sum = 0.0
    count_sum = 0
    for batch in limited_batches(loader, max_batches):
        features = make_features(
            batch,
            arm=arm,
            detector=eval_detector,
            detector_index=eval_detector_index,
            detector_cache=detector_cache,
            calibration_models=calibration_models,
            residual_risk=residual_risk,
            device=device,
        )
        pred_mm = model(features) * 1000.0
        tgt_mm = batch["target3d_mm"].to(device, non_blocking=True)[:, 1:, :]
        if not bool(torch.isfinite(pred_mm).all()) or not bool(torch.isfinite(tgt_mm).all()):
            raise RuntimeError(f"Non-finite evaluation tensor: arm={arm}, detector={eval_detector}")
        count = int(pred_mm.shape[0])
        mpjpe = torch.linalg.vector_norm(pred_mm - tgt_mm, dim=-1).mean()
        pa = pa_mpjpe_mm(pred_mm, tgt_mm)
        mpjpe_sum += float(mpjpe.item()) * count
        pa_sum += float(pa.item()) * count
        count_sum += count
    if count_sum == 0:
        raise RuntimeError("Evaluation processed zero samples.")
    return {
        "mpjpe_mm": mpjpe_sum / count_sum,
        "pa_mpjpe_mm": pa_sum / count_sum,
        "samples": count_sum,
    }


def train_one(
    *,
    arm: str,
    train_detector: str,
    train_detector_index: int,
    loaders: dict[str, DataLoader],
    detector_cache: dict[str, np.ndarray],
    calibration_models: dict[str, Any],
    residual_risk: np.ndarray,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = TinyPoseLifter(feature_dim(arm), hidden=args.hidden_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    train_losses: list[float] = []
    start = time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for batch in limited_batches(loaders["train"], args.max_train_batches):
            features = make_features(
                batch,
                arm=arm,
                detector=train_detector,
                detector_index=train_detector_index,
                detector_cache=detector_cache,
                calibration_models=calibration_models,
                residual_risk=residual_risk,
                device=device,
            )
            loss = loss_fn(model(features), target_m(batch, device))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite training loss: arm={arm}, detector={train_detector}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
    evaluations: dict[str, Any] = {}
    eval_detectors = ("clean",) if arm == "clean_observed_legacy" else DETECTORS
    for eval_detector in eval_detectors:
        eval_index = 0 if eval_detector == "clean" else DETECTORS.index(eval_detector)
        evaluations[eval_detector] = {
            split: evaluate(
                model,
                loaders[split],
                arm=arm,
                eval_detector=eval_detector if eval_detector != "clean" else DETECTORS[0],
                eval_detector_index=eval_index,
                detector_cache=detector_cache,
                calibration_models=calibration_models,
                residual_risk=residual_risk,
                device=device,
                max_batches=args.max_eval_batches,
            )
            for split in ("val", "test")
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_smoke.pt"
    torch.save(
        {
            "format": "phase9c_b1_smoke_checkpoint_v3",
            "phase": "9C-B1",
            "arm": arm,
            "train_detector": train_detector,
            "split": SPLIT,
            "seed": SEED,
            "model_state_dict": model.state_dict(),
            "feature_dim": feature_dim(arm),
            "hidden_width": args.hidden_width,
        },
        checkpoint_path,
    )
    return {
        "status": "PASS",
        "arm": arm,
        "train_detector": train_detector,
        "checkpoint": str(checkpoint_path),
        "train_loss_first": train_losses[0] if train_losses else math.nan,
        "train_loss_last": train_losses[-1] if train_losses else math.nan,
        "train_batches": len(train_losses),
        "evaluations": evaluations,
        "elapsed_sec": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 9C-B1 four-arm smoke training.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase9c_b1_four_arm_smoke"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=8)
    parser.add_argument("--max-eval-batches", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=16384)
    parser.add_argument("--max-eval-samples", type=int, default=4096)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    if args.num_workers != 8:
        raise ValueError("Phase 9C-B1 is locked to num_workers=8.")
    project_root = args.project_root.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = find_required_artifacts(project_root)
    protocol = load_json(artifacts["protocol"])
    if protocol.get("status") != "LOCKED" or int(protocol["training_budget"]["num_workers"]) != 8:
        raise ValueError("Invalid Phase 9C-B1 protocol lock.")
    if protocol.get("format") != "phase9c_b1_smoke_protocol_lock_v3":
        raise ValueError(f"Unexpected protocol format: {protocol.get('format')}")
    resource_policy = protocol.get("windows_spawn_resource_policy", {})
    expected_resource_policy = {
        "num_workers": 8,
        "materialize_smoke_subset_in_main_process": True,
        "materialization_workers": 0,
        "detector_cache_in_workers": False,
        "training_shard_io_in_workers": False,
        "persistent_workers": False,
        "prefetch_factor": 1,
        "worker_cpu_threads": 1,
    }
    if resource_policy != expected_resource_policy:
        raise ValueError(f"Invalid Windows spawn resource policy: {resource_policy}")
    expected_missingness_policy = {
        "primary_metric_cohort": "paired_common_detection",
        "same_sample_cohort_across_arms_and_detectors": True,
        "exclude_only_if_any_detector_failed_or_nonfinite": True,
        "report_detection_coverage_separately": True,
        "ground_truth_imputation_for_detector_failures": False,
    }
    if protocol.get("detector_missingness_policy") != expected_missingness_policy:
        raise ValueError(f"Invalid detector missingness policy: {protocol.get('detector_missingness_policy')}")
    phase9ca_qc = load_json(artifacts["phase9c_a_qc"])
    phase9ca_build = load_json(artifacts["phase9c_a_build"])
    phase9cb0_qc = load_json(artifacts["phase9c_b0_qc"])
    phase9cb0_build = load_json(artifacts["phase9c_b0_build"])
    if phase9ca_qc.get("status") != "PASS":
        raise ValueError("Phase 9C-A QC is not PASS.")
    if phase9cb0_qc.get("status") != "PASS" or phase9cb0_qc.get("scientific_decision") != "READY_FOR_PHASE9C_B1_FOUR_ARM_SMOKE":
        raise ValueError("Phase 9C-B0 gate is not READY.")
    if phase9ca_build.get("official_TS1_to_TS6_read") is not False or phase9cb0_build.get("official_TS1_to_TS6_read") is not False:
        raise ValueError("Official TS1-TS6 isolation was violated upstream.")
    join_hash_expected = str(phase9cb0_qc["artifact_hashes"]["phase9c_b0_detector_join_cache.npz"])
    join_hash_actual = sha256_file(artifacts["phase9c_b0_join_cache"])
    if join_hash_actual != join_hash_expected:
        raise ValueError(f"Join cache hash mismatch: {join_hash_actual} != {join_hash_expected}")
    with np.load(artifacts["phase9c_b0_join_cache"], allow_pickle=False) as data:
        join_cache = {key: data[key] for key in data.files}
    detector_cache, detector_location, detector_hash = load_phase9b_cache(
        project_root,
        str(phase9ca_build["source_hashes"]["phase9b_detector_cache.npz"]),
    )
    detection_audit, common_valid_detector_sample = build_common_detection_audit(detector_cache, join_cache)
    common_summary = detection_audit["common_cohort"]
    print(
        "[DETECTION] paired_common "
        f"eligible={common_summary['eligible_samples']}/{common_summary['joined_samples']} "
        f"excluded={common_summary['excluded_any_detector_failure']}"
    )
    for detector in DETECTORS:
        stats = detection_audit["detectors"][detector]
        print(
            f"[DETECTION] detector={detector} detected={stats['detected_samples']}/"
            f"{stats['joined_samples']} missed={stats['missed_detections']} "
            f"rate={stats['detection_rate']:.6f}"
        )
    calibration_models = load_json(artifacts["phase9c_a_models"])
    with np.load(artifacts["phase9c_a_residual_bank"], allow_pickle=False) as data:
        residual_bank = {key: data[key] for key in data.files}
    risk = residual_risk_table(residual_bank)
    device = resolve_device(args.device)
    set_seed(SEED)
    manifest_path = project_root / "data/processed/caches/phase5r1a_multigeometry_oof_25fps_v1/split_a/manifest.json"
    source_datasets = {
        "train": JoinedDetectorDataset(
            manifest_path=manifest_path,
            join_cache=join_cache,
            common_valid_detector_sample=common_valid_detector_sample,
            partition="train",
            max_samples=args.max_train_samples,
            seed=SEED,
        ),
        "val": JoinedDetectorDataset(
            manifest_path=manifest_path,
            join_cache=join_cache,
            common_valid_detector_sample=common_valid_detector_sample,
            partition="val",
            max_samples=args.max_eval_samples,
            seed=SEED + 1,
        ),
        "test": JoinedDetectorDataset(
            manifest_path=manifest_path,
            join_cache=join_cache,
            common_valid_detector_sample=common_valid_detector_sample,
            partition="test",
            max_samples=args.max_eval_samples,
            seed=SEED + 2,
        ),
    }
    dataset_selection = {
        name: {
            "joined_before_eligibility": source.joined_before_eligibility,
            "excluded_by_common_detection": source.excluded_by_common_detection,
            "eligible_before_smoke_limit": source.eligible_before_limit,
        }
        for name, source in source_datasets.items()
    }
    # Read full shards only in the main process. Windows spawn workers receive
    # only the compact bounded smoke subset, so eight workers no longer clone
    # detector caches or retain large Phase 5R.1A shards.
    datasets = {
        name: materialize_smoke_dataset(dataset, name)
        for name, dataset in source_datasets.items()
    }
    for name, dataset in datasets.items():
        dataset_selection[name]["selected_after_smoke_limit"] = len(dataset)
    del source_datasets
    gc.collect()
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
        "prefetch_factor": 1,
        "worker_init_fn": init_worker,
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs),
        "val": DataLoader(datasets["val"], batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs),
        "test": DataLoader(datasets["test"], batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs),
    }
    run_specs: list[tuple[str, str, int]] = [("clean_observed_legacy", "clean", 0)]
    for arm in ARMS[1:]:
        for detector_index, detector in enumerate(DETECTORS):
            run_specs.append((arm, detector, detector_index))
    run_reports: list[dict[str, Any]] = []
    for arm, detector, detector_index in run_specs:
        print(f"[RUN] arm={arm} train_detector={detector} workers={args.num_workers}")
        run_dir = output_root / SPLIT / f"seed{SEED}" / arm / f"train_{detector}"
        run_reports.append(
            train_one(
                arm=arm,
                train_detector=detector,
                train_detector_index=detector_index,
                loaders=loaders,
                detector_cache=detector_cache,
                calibration_models=calibration_models,
                residual_risk=risk,
                output_dir=run_dir,
                device=device,
                args=args,
            )
        )
    metrics_path = output_root / "phase9c_b1_smoke_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "arm", "train_detector", "eval_detector", "partition",
            "mpjpe_mm", "pa_mpjpe_mm", "samples",
            "train_loss_first", "train_loss_last", "train_batches",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in run_reports:
            for eval_detector, partitions in report["evaluations"].items():
                for partition, metrics in partitions.items():
                    writer.writerow(
                        {
                            "arm": report["arm"],
                            "train_detector": report["train_detector"],
                            "eval_detector": eval_detector,
                            "partition": partition,
                            "mpjpe_mm": metrics["mpjpe_mm"],
                            "pa_mpjpe_mm": metrics["pa_mpjpe_mm"],
                            "samples": metrics["samples"],
                            "train_loss_first": report["train_loss_first"],
                            "train_loss_last": report["train_loss_last"],
                            "train_batches": report["train_batches"],
                        }
                    )
    report = {
        "status": "PASS",
        "format": "phase9c_b1_smoke_build_report_v3",
        "phase": "9C-B1",
        "scientific_role": "training_smoke_not_final_scientific_comparison",
        "source_gates": {
            "phase9c_a": phase9ca_qc.get("scientific_decision"),
            "phase9c_b0": phase9cb0_qc.get("scientific_decision"),
        },
        "split": SPLIT,
        "seed": SEED,
        "workers": {
            "requested": args.num_workers,
            "effective": args.num_workers,
            "peak_concurrent": args.num_workers,
            "persistent_workers": False,
            "prefetch_factor": 1,
        },
        "data_loading": {
            "materialized_before_worker_spawn": True,
            "materialization_workers": 0,
            "detector_cache_in_dataset_workers": False,
            "training_shard_io_in_dataset_workers": False,
            "worker_cpu_threads": 1,
            "sample_cohort": "paired_common_detection",
            "same_sample_cohort_across_arms_and_detectors": True,
        },
        "detector_coverage_audit": detection_audit,
        "device": str(device),
        "detector_cache_location": detector_location,
        "detector_cache_sha256": detector_hash,
        "join_cache_sha256": join_hash_actual,
        "arms": list(ARMS),
        "detectors": list(DETECTORS),
        "run_count": len(run_reports),
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "dataset_selection": dataset_selection,
        "training_budget": {
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_epochs": args.max_epochs,
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
        },
        "run_reports": run_reports,
        "artifacts": {
            "metrics_csv": str(metrics_path),
        },
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
        "next_phase_if_qc_pass": "Phase 9C-B2 full 4-arm, 3-split, 3-seed training and matched/cross-detector evaluation.",
    }
    report_path = output_root / "phase9c_b1_smoke_build_report.json"
    write_json(report_path, report)
    print("=" * 112)
    print("PHASE 9C-B1 - FOUR ARM TRAINING SMOKE")
    print("=" * 112)
    print(f"Status                  : PASS")
    print(f"Runs                    : {len(run_reports)}")
    print(f"Workers                 : {args.num_workers}")
    print(f"Output                  : {output_root}")
    print("Scientific decision     : READY_FOR_PHASE9C_B1_QC")
    print("=" * 112)


if __name__ == "__main__":
    main()
