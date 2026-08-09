from __future__ import annotations

import argparse
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
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PHASE = "9C-R1B"
SPLIT = "split_a"
SEED = 42
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
ARMS = (
    "clean_observed_legacy",
    "detector_aware_observed",
    "detector_conditioned_virtual_view_no_confidence",
    "detector_conditioned_calibrated_confidence_virtual_view",
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
    spec = importlib.util.spec_from_file_location("phase9c_b1_v3_r1b", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SPLIT = SPLIT
    return module


def resolve_strict_gpu(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value in {"cuda", "cuda:0"}:
        value = "cuda:0"
    device = torch.device(value)
    if device.type != "cuda" or (device.index not in {None, 0}):
        raise RuntimeError("Phase 9C-R1B is locked to --device cuda:0.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden.")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_worker_r1b(_: int) -> None:
    """Spawn-safe DataLoader initializer defined in this real script module.

    Do not pass ``b1.init_worker`` to a Windows DataLoader.  The B1 helper is
    loaded under a synthetic importlib name in the parent process, and a
    spawned worker cannot import that synthetic module while unpickling the
    callback.
    """
    torch.set_num_threads(1)


class DetectorConditionedSource(Dataset):
    def __init__(
        self,
        *,
        base: Dataset,
        virtual_cache: dict[str, np.ndarray],
        partition: str,
        max_samples: int,
        seed: int,
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
        positions = np.arange(base_dataset.size, dtype=np.int64)
        if max_samples > 0 and positions.size > max_samples:
            rng = np.random.default_rng(seed)
            positions = np.sort(rng.choice(positions.size, size=max_samples, replace=False))
        self.base = base
        self.base_positions = positions
        self.cache_rows = cache_rows[positions]
        self.virtual = virtual_cache["virtual_pose_camera_root"]
        self.full_aligned_samples = int(cache_dataset.size)

    def __len__(self) -> int:
        return int(self.base_positions.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        source = self.base[int(self.base_positions[item])]
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


def materialize(source: DetectorConditionedSource, partition: str) -> MaterializedDataset:
    print(f"[MATERIALIZE] partition={partition} samples={len(source)} workers=0")
    columns: dict[str, list[torch.Tensor]] = {}
    for index in range(len(source)):
        item = source[index]
        for name, value in item.items():
            columns.setdefault(name, []).append(value)
        if (index + 1) % 4096 == 0 or index + 1 == len(source):
            print(f"[MATERIALIZE] partition={partition} progress={index + 1}/{len(source)}")
    tensors = {name: torch.stack(values).contiguous() for name, values in columns.items()}
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
            raise ValueError(f"Unexpected {partition}/{name} shape: {tuple(tensor.shape)}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite values in {partition}/{name}")
    if not bool(torch.equal(tensors["detector_virtual"][:, :, 0, :], torch.zeros_like(tensors["detector_virtual"][:, :, 0, :]))):
        raise ValueError(f"{partition}: virtual root joint is not exactly zero.")
    return MaterializedDataset(tensors)


def probe_spawn_loaders(loaders: dict[str, DataLoader], expected_workers: int) -> dict[str, Any]:
    """Start real workers and fetch one batch from each loader before training."""
    evidence: dict[str, Any] = {
        "status": "PASS",
        "worker_model": "windows_spawn_safe_local_initializer",
        "requested_workers": int(expected_workers),
        "partitions": {},
    }
    for partition in ("train", "val"):
        iterator = iter(loaders[partition])
        try:
            batch = next(iterator)
            expected_fields = {
                "clean_pose2d", "target3d_mm", "intrinsic", "detector_index", "detector_virtual"
            }
            if set(batch) != expected_fields:
                raise RuntimeError(f"{partition}: spawn preflight fields mismatch: {sorted(batch)}")
            batch_size = int(batch["clean_pose2d"].shape[0])
            if batch_size <= 0:
                raise RuntimeError(f"{partition}: spawn preflight returned an empty batch.")
            for name, value in batch.items():
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"{partition}: non-finite spawn preflight tensor: {name}")
            evidence["partitions"][partition] = {
                "batch_size": batch_size,
                "fields": sorted(batch),
            }
        finally:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
    print(
        f"[PASS] DataLoader spawn preflight workers={expected_workers} "
        f"train_batch={evidence['partitions']['train']['batch_size']} "
        f"val_batch={evidence['partitions']['val']['batch_size']}"
    )
    return evidence


def make_features(
    b1: Any,
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
    intrinsic = batch["intrinsic"].to(device, non_blocking=True)
    indices = batch["detector_index"].cpu().numpy().astype(np.int64)
    if arm == "clean_observed_legacy":
        features = clean[:, 1:, :].reshape(clean.shape[0], -1)
    else:
        prediction_px = torch.from_numpy(detector_cache["prediction_c15_px"][detector_index, indices]).to(
            device=device, dtype=torch.float32
        )
        observed = b1.normalized_root_from_pixels_torch(prediction_px, intrinsic)
        raw_conf = detector_cache["confidence_c15"][detector_index, indices].astype(np.float32)
        calibrated = b1.calibration_probability(calibration_models, SPLIT, detector, raw_conf)
        calibrated_t = torch.from_numpy(calibrated[:, 1:]).to(device=device, dtype=torch.float32)
        if arm == "detector_aware_observed":
            features = torch.cat([observed[:, 1:, :].reshape(clean.shape[0], -1), calibrated_t], dim=1)
        else:
            virtual = batch["detector_virtual"][:, detector_index].to(device, non_blocking=True)
            base = [
                observed[:, 1:, :].reshape(clean.shape[0], -1),
                virtual[:, 1:, :].reshape(clean.shape[0], -1),
            ]
            if arm == "detector_conditioned_calibrated_confidence_virtual_view":
                risk_np = b1.residual_risk_features(residual_risk, 0, detector_index, raw_conf)[:, 1:]
                risk_t = torch.from_numpy(risk_np).to(device=device, dtype=torch.float32)
                base.extend([calibrated_t, risk_t])
            elif arm != "detector_conditioned_virtual_view_no_confidence":
                raise ValueError(arm)
            features = torch.cat(base, dim=1)
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError(f"Non-finite features: arm={arm}, detector={detector}")
    return features


def feature_dim(arm: str) -> int:
    return {
        "clean_observed_legacy": 28,
        "detector_aware_observed": 42,
        "detector_conditioned_virtual_view_no_confidence": 56,
        "detector_conditioned_calibrated_confidence_virtual_view": 84,
    }[arm]


@torch.inference_mode()
def evaluate(
    b1: Any,
    model: nn.Module,
    loader: DataLoader,
    *,
    arm: str,
    detector: str,
    detector_index: int,
    detector_cache: dict[str, np.ndarray],
    calibration_models: dict[str, Any],
    residual_risk: np.ndarray,
    device: torch.device,
    max_batches: int,
) -> dict[str, float | int]:
    model.eval()
    mpjpe_sum = pa_sum = 0.0
    count_sum = 0
    for batch in b1.limited_batches(loader, max_batches):
        features = make_features(
            b1, batch, arm=arm, detector=detector, detector_index=detector_index,
            detector_cache=detector_cache, calibration_models=calibration_models,
            residual_risk=residual_risk, device=device,
        )
        prediction = model(features) * 1000.0
        target = batch["target3d_mm"].to(device, non_blocking=True)[:, 1:, :]
        if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
            raise RuntimeError(f"Non-finite validation tensor: {arm}/{detector}")
        count = int(prediction.shape[0])
        mpjpe_sum += float(torch.linalg.vector_norm(prediction - target, dim=-1).mean().item()) * count
        pa_sum += float(b1.pa_mpjpe_mm(prediction, target).item()) * count
        count_sum += count
    if count_sum <= 0:
        raise RuntimeError("Validation processed zero samples.")
    return {"mpjpe_mm": mpjpe_sum / count_sum, "pa_mpjpe_mm": pa_sum / count_sum, "samples": count_sum}


def train_one(
    b1: Any,
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
    model = b1.TinyPoseLifter(feature_dim(arm), hidden=args.hidden_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    started = time.time()
    for _ in range(args.max_epochs):
        model.train()
        for batch in b1.limited_batches(loaders["train"], args.max_train_batches):
            features = make_features(
                b1, batch, arm=arm, detector=train_detector, detector_index=train_detector_index,
                detector_cache=detector_cache, calibration_models=calibration_models,
                residual_risk=residual_risk, device=device,
            )
            target = batch["target3d_mm"].to(device, non_blocking=True)[:, 1:, :] / 1000.0
            loss = loss_fn(model(features), target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite training loss: {arm}/{train_detector}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    eval_detectors = ("clean",) if arm == "clean_observed_legacy" else DETECTORS
    evaluations: dict[str, Any] = {}
    for eval_detector in eval_detectors:
        index = 0 if eval_detector == "clean" else DETECTORS.index(eval_detector)
        detector_name = DETECTORS[0] if eval_detector == "clean" else eval_detector
        evaluations[eval_detector] = evaluate(
            b1, model, loaders["val"], arm=arm, detector=detector_name, detector_index=index,
            detector_cache=detector_cache, calibration_models=calibration_models,
            residual_risk=residual_risk, device=device, max_batches=args.max_eval_batches,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint_smoke.pt"
    torch.save({
        "format": "phase9c_r1b_detector_conditioned_smoke_checkpoint_v1",
        "phase": PHASE,
        "arm": arm,
        "train_detector": train_detector,
        "split": SPLIT,
        "seed": SEED,
        "feature_dim": feature_dim(arm),
        "hidden_width": args.hidden_width,
        "model_state_dict": model.state_dict(),
    }, checkpoint)
    return {
        "status": "PASS",
        "arm": arm,
        "train_detector": train_detector,
        "checkpoint": str(checkpoint),
        "train_loss_first": losses[0] if losses else math.nan,
        "train_loss_last": losses[-1] if losses else math.nan,
        "train_batches": len(losses),
        "validation_evaluations": evaluations,
        "elapsed_sec": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 9C-R1B detector-conditioned smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase9c_r1b_detector_conditioned_smoke"))
    parser.add_argument("--device", default="cuda:0")
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

    root = args.project_root.resolve()
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "configs/phase9c_r1b_detector_conditioned_smoke_protocol.json"
    protocol = load_json(protocol_path)
    if protocol.get("status") != "LOCKED" or protocol.get("format") != "phase9c_r1b_detector_conditioned_smoke_protocol_lock_v1":
        raise ValueError("Invalid Phase 9C-R1B protocol lock.")
    budget = protocol["training_budget"]
    locked_args = {
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_epochs": args.max_epochs,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
    }
    for key, value in locked_args.items():
        if int(value) != int(budget[key]):
            raise ValueError(f"Locked training budget mismatch: {key}={value} != {budget[key]}")

    r1a_root = root / "outputs/phase9c_r1a_detector_conditioned_virtual_cache"
    r1a_qc_path = r1a_root / "phase9c_r1a_qc_report.json"
    r1a_cache_path = r1a_root / SPLIT / "phase9c_r1a_detector_virtual_cache.npz"
    r1a_report_path = r1a_cache_path.with_suffix(".report.json")
    r1a_qc = load_json(r1a_qc_path)
    r1a_report = load_json(r1a_report_path)
    if r1a_qc.get("status") != "PASS" or r1a_qc.get("scientific_decision") != protocol["source_gate"]:
        raise ValueError("Phase 9C-R1A pass gate is absent.")
    if r1a_qc.get("official_TS1_to_TS6_read") is not False or r1a_report.get("official_TS1_to_TS6_read") is not False:
        raise ValueError("Official-test isolation was violated upstream.")
    if r1a_report.get("protocol_sha256") != protocol["source_r1a_protocol_sha256"]:
        raise ValueError("Phase 9C-R1A protocol hash mismatch.")
    if r1a_report.get("cache_sha256") != sha256_file(r1a_cache_path):
        raise ValueError("Phase 9C-R1A cache hash mismatch.")
    with np.load(r1a_cache_path, allow_pickle=False) as handle:
        virtual_cache = {name: handle[name] for name in handle.files}
    if str(virtual_cache["format"].item()) != "phase9c_r1a_detector_conditioned_virtual_cache_v1":
        raise ValueError("Unexpected Phase 9C-R1A cache format.")
    if tuple(str(value) for value in virtual_cache["detector_id"].tolist()) != DETECTORS:
        raise ValueError("Detector order mismatch in R1A cache.")
    if str(virtual_cache["source_input"].item()) != "phase9b_prediction_c15_px_real_detector":
        raise ValueError("R1A cache is not detector-conditioned.")

    device = resolve_strict_gpu(args.device)
    b1 = load_b1(root, str(protocol["source_b1_script_sha256"]))
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
    residual_risk = b1.residual_risk_table(residual_bank)

    manifest = root / "data/processed/caches/phase5r1a_multigeometry_oof_25fps_v1/split_a/manifest.json"
    base_train = b1.JoinedDetectorDataset(
        manifest_path=manifest, join_cache=join_cache, common_valid_detector_sample=common_valid,
        partition="train", max_samples=0, seed=SEED,
    )
    base_val = b1.JoinedDetectorDataset(
        manifest_path=manifest, join_cache=join_cache, common_valid_detector_sample=common_valid,
        partition="val", max_samples=0, seed=SEED + 1,
    )
    sources = {
        "train": DetectorConditionedSource(
            base=base_train, virtual_cache=virtual_cache, partition="train",
            max_samples=args.max_train_samples, seed=SEED,
        ),
        "val": DetectorConditionedSource(
            base=base_val, virtual_cache=virtual_cache, partition="val",
            max_samples=args.max_eval_samples, seed=SEED + 1,
        ),
    }
    alignment = {
        name: {"full_aligned_samples": source.full_aligned_samples, "selected_samples": len(source)}
        for name, source in sources.items()
    }
    datasets = {name: materialize(source, name) for name, source in sources.items()}
    del sources, base_train, base_val
    gc.collect()

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 1,
        # Must be a callable from this importable script, not from the B1
        # helper loaded through importlib under a synthetic module name.
        "worker_init_fn": init_worker_r1b,
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs),
        "val": DataLoader(datasets["val"], batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs),
    }
    spawn_preflight = probe_spawn_loaders(loaders, args.num_workers)
    set_seed(SEED)
    specs: list[tuple[str, str, int]] = [(ARMS[0], "clean", 0)]
    for arm in ARMS[1:]:
        for index, detector in enumerate(DETECTORS):
            specs.append((arm, detector, index))
    reports = []
    for arm, detector, index in specs:
        print(f"[RUN] phase={PHASE} arm={arm} train_detector={detector}")
        run_dir = output / SPLIT / f"seed{SEED}" / arm / f"train_{detector}"
        reports.append(train_one(
            b1, arm=arm, train_detector=detector, train_detector_index=index,
            loaders=loaders, detector_cache=detector_cache, calibration_models=calibration_models,
            residual_risk=residual_risk, output_dir=run_dir, device=device, args=args,
        ))

    metrics_path = output / "phase9c_r1b_smoke_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "arm", "train_detector", "eval_detector", "partition", "mpjpe_mm", "pa_mpjpe_mm",
            "samples", "train_loss_first", "train_loss_last", "train_batches",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for eval_detector, metric in report["validation_evaluations"].items():
                writer.writerow({
                    "arm": report["arm"], "train_detector": report["train_detector"],
                    "eval_detector": eval_detector, "partition": "val", **metric,
                    "train_loss_first": report["train_loss_first"],
                    "train_loss_last": report["train_loss_last"],
                    "train_batches": report["train_batches"],
                })
    build = {
        "status": "PASS",
        "format": "phase9c_r1b_detector_conditioned_smoke_build_report_v1",
        "phase": PHASE,
        "scientific_role": "pipeline_smoke_without_model_or_claim_selection",
        "split": SPLIT,
        "seed": SEED,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "workers": args.num_workers,
        "dataloader_spawn_preflight": spawn_preflight,
        "dynamic_b1_callable_passed_to_workers": False,
        "source_gate": r1a_qc.get("scientific_decision"),
        "source_r1a_cache": str(r1a_cache_path),
        "source_r1a_cache_sha256": sha256_file(r1a_cache_path),
        "source_r1a_protocol_sha256": r1a_report.get("protocol_sha256"),
        "virtual_view_input": "phase9b_prediction_c15_px_real_detector",
        "legacy_clean_conditioned_virtual_view_used_as_input": False,
        "virtual_view_selected_by_eval_detector": True,
        "exact_cache_alignment_verified": True,
        "detector_cache_location": detector_location,
        "detector_cache_sha256": detector_hash,
        "detector_coverage_audit": detection_audit,
        "dataset_alignment": alignment,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "arms": list(ARMS),
        "detectors": list(DETECTORS),
        "run_count": len(reports),
        "run_reports": reports,
        "training_budget": {key: int(value) for key, value in locked_args.items()},
        "development_test_metrics_computed": False,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    build_path = output / "phase9c_r1b_smoke_build_report.json"
    write_json(build_path, build)
    print("=" * 108)
    print("PHASE 9C-R1B - DETECTOR-CONDITIONED TRAINING SMOKE")
    print("=" * 108)
    print("Status                : PASS")
    print(f"Runs                  : {len(reports)}")
    print(f"Device                : {device} / {build['gpu_name']}")
    print("Scientific decision   : READY_FOR_PHASE9C_R1B_QC")
    print("=" * 108)


if __name__ == "__main__":
    main()
