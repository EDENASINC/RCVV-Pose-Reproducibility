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
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPLITS = ("split_a", "split_b", "split_c")
SEEDS = (42, 123, 2026)
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
ARMS = (
    "clean_observed_legacy",
    "detector_aware_observed",
    "virtual_view_no_confidence",
    "calibrated_confidence_virtual_view",
)
PARTITIONS = ("train", "val", "test")
EXPECTED_B1_SHA256 = (
    "fc1ae8f11ec6fb1b75c8d8c730e61a67f1b1aa45422227364e4bce71eb72b673"
)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_b1_module(project_root: Path, expected_hash: str):
    path = project_root / "scripts/train/run_phase9c_b1_four_arm_smoke.py"
    actual = sha256_file(path)
    if actual != expected_hash or actual != EXPECTED_B1_SHA256:
        raise ValueError(f"Phase 9C-B1 V3 source hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("phase9c_b1_v3_locked", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CompactFullDataset(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        sizes = {int(value.shape[0]) for value in tensors.values()}
        if len(sizes) != 1:
            raise ValueError(f"Compact tensor length mismatch: {sorted(sizes)}")
        self.tensors = tensors
        self.size = sizes.pop()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        return {name: value[item] for name, value in self.tensors.items()}


def materialize_full_dataset(source: Dataset, partition: str) -> CompactFullDataset:
    size = len(source)
    if size <= 0:
        raise ValueError(f"Empty full dataset: {partition}")
    print(f"[MATERIALIZE] partition={partition} samples={size} workers=0 preallocated=yes")
    tensors = {
        "clean_pose2d": torch.empty((size, 15, 2), dtype=torch.float32),
        "target3d_mm": torch.empty((size, 15, 3), dtype=torch.float32),
        "intrinsic": torch.empty((size, 3, 3), dtype=torch.float32),
        "teacher_virtual": torch.empty((size, 15, 2), dtype=torch.float32),
        "detector_index": torch.empty((size,), dtype=torch.long),
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
        "teacher_virtual": (15, 2),
        "detector_index": (),
    }
    for name, suffix in expected.items():
        if tuple(tensors[name].shape[1:]) != suffix:
            raise ValueError(f"Unexpected {partition}/{name}: {tuple(tensors[name].shape)}")
        if tensors[name].is_floating_point() and not bool(torch.isfinite(tensors[name]).all()):
            raise ValueError(f"Non-finite materialized tensor: {partition}/{name}")
    print(f"[PASS] full materialization partition={partition}")
    return CompactFullDataset(tensors)


def init_worker(_: int) -> None:
    torch.set_num_threads(1)


def loader_kwargs(device: torch.device, workers: int) -> dict[str, Any]:
    return {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
        "prefetch_factor": 1,
        "worker_init_fn": init_worker,
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
    arm: str,
    detector: str,
    detector_index: int,
    detector_cache: dict[str, np.ndarray],
    calibration_models: dict[str, Any],
    residual_risk: np.ndarray,
    device: torch.device,
    batch_size: int,
    workers: int,
    collect_per_sample: bool,
) -> tuple[dict[str, float | int], dict[str, np.ndarray] | None]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs(device, workers),
    )
    model.eval()
    mpjpe_parts: list[np.ndarray] = []
    pa_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    mpjpe_sum = 0.0
    pa_sum = 0.0
    samples = 0
    for batch in iterate_loader(loader):
        features = b1.make_features(
            batch,
            arm=arm,
            detector=detector,
            detector_index=detector_index,
            detector_cache=detector_cache,
            calibration_models=calibration_models,
            residual_risk=residual_risk,
            device=device,
        )
        prediction = model(features) * 1000.0
        target = batch["target3d_mm"].to(device, non_blocking=True)[:, 1:, :]
        if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
            raise RuntimeError(f"Non-finite evaluation tensor: arm={arm}, detector={detector}")
        mpjpe = torch.linalg.vector_norm(prediction - target, dim=-1).mean(dim=1)
        pa = pa_mpjpe_per_sample_mm(prediction, target)
        count = int(prediction.shape[0])
        mpjpe_sum += float(mpjpe.sum().item())
        pa_sum += float(pa.sum().item())
        samples += count
        if collect_per_sample:
            mpjpe_parts.append(mpjpe.cpu().numpy().astype(np.float32))
            pa_parts.append(pa.cpu().numpy().astype(np.float32))
            index_parts.append(batch["detector_index"].cpu().numpy().astype(np.int64))
    if samples != len(dataset):
        raise RuntimeError(f"Evaluation coverage mismatch: {samples} != {len(dataset)}")
    metrics = {
        "mpjpe_mm": mpjpe_sum / samples,
        "pa_mpjpe_mm": pa_sum / samples,
        "samples": samples,
    }
    per_sample = None
    if collect_per_sample:
        per_sample = {
            "detector_sample_index": np.concatenate(index_parts),
            "mpjpe_mm": np.concatenate(mpjpe_parts),
            "pa_mpjpe_mm": np.concatenate(pa_parts),
        }
        if not np.isfinite(per_sample["mpjpe_mm"]).all() or not np.isfinite(per_sample["pa_mpjpe_mm"]).all():
            raise RuntimeError("Non-finite per-sample evaluation metric.")
    return metrics, per_sample


def selection_score(metrics: dict[str, float | int]) -> float:
    mpjpe = float(metrics["mpjpe_mm"])
    pa = float(metrics["pa_mpjpe_mm"])
    if not math.isfinite(mpjpe) or not math.isfinite(pa) or mpjpe <= 0 or pa <= 0:
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
        if report.get("status") != "PASS" or report.get("format") != "phase9c_b2_run_report_v1":
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
    device: torch.device,
    output_dir: Path,
    protocol: dict[str, Any],
    protocol_sha: str,
) -> dict[str, Any]:
    run_report_path = output_dir / "phase9c_b2_run_report.json"
    if valid_existing_run(run_report_path, protocol_sha):
        print(f"[REUSE] split={split} seed={seed} arm={arm} train_detector={train_detector}")
        return load_json(run_report_path)
    budget = protocol["training_budget"]
    set_seed(seed)
    model = b1.TinyPoseLifter(
        b1.feature_dim(arm), hidden=int(budget["hidden_width"])
    ).to(device)
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
    epochs_without_improvement = 0
    start = time.time()
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, int(budget["max_epochs"]) + 1):
        train_loader = DataLoader(
            datasets["train"],
            batch_size=int(budget["batch_size"]),
            shuffle=True,
            drop_last=True,
            generator=generator,
            **loader_kwargs(device, int(budget["num_workers"])),
        )
        model.train()
        loss_sum = 0.0
        sample_sum = 0
        batch_count = 0
        for batch in iterate_loader(train_loader):
            features = b1.make_features(
                batch,
                arm=arm,
                detector=train_detector if train_detector != "clean" else DETECTORS[0],
                detector_index=train_detector_index,
                detector_cache=detector_cache,
                calibration_models=calibration_models,
                residual_risk=residual_risk,
                device=device,
            )
            target = b1.target_m(batch, device)
            loss = loss_fn(model(features), target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"Non-finite loss: split={split} seed={seed} arm={arm} detector={train_detector}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(budget["gradient_clip_norm"]))
            optimizer.step()
            count = int(target.shape[0])
            loss_sum += float(loss.item()) * count
            sample_sum += count
            batch_count += 1
        eval_detector = train_detector if train_detector != "clean" else DETECTORS[0]
        val_metrics, _ = evaluate(
            b1=b1,
            model=model,
            dataset=datasets["val"],
            arm=arm,
            detector=eval_detector,
            detector_index=train_detector_index,
            detector_cache=detector_cache,
            calibration_models=calibration_models,
            residual_risk=residual_risk,
            device=device,
            batch_size=int(budget["eval_batch_size"]),
            workers=int(budget["num_workers"]),
            collect_per_sample=False,
        )
        score = selection_score(val_metrics)
        improved = score < best_score - 1e-9
        checkpoint = {
            "format": "phase9c_b2_full_checkpoint_v1",
            "phase": "9C-B2",
            "split": split,
            "seed": seed,
            "arm": arm,
            "train_detector": train_detector,
            "epoch": epoch,
            "validation_selection_score": score,
            "validation_metrics": val_metrics,
            "feature_dim": b1.feature_dim(arm),
            "hidden_width": int(budget["hidden_width"]),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "protocol_sha256": protocol_sha,
        }
        torch.save(checkpoint, last_path)
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, sample_sum),
            "train_samples": sample_sum,
            "train_batches": batch_count,
            "val_mpjpe_mm": val_metrics["mpjpe_mm"],
            "val_pa_mpjpe_mm": val_metrics["pa_mpjpe_mm"],
            "val_selection_score": score,
            "improved": improved,
        }
        history.append(row)
        print(
            f"[EPOCH] {split} seed={seed} arm={arm} detector={train_detector} "
            f"epoch={epoch} loss={row['train_loss']:.7f} "
            f"val={val_metrics['mpjpe_mm']:.3f}/{val_metrics['pa_mpjpe_mm']:.3f} "
            f"score={score:.6f} best_epoch={best_epoch}"
        )
        if epochs_without_improvement >= int(budget["early_stopping_patience"]):
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
    eval_detectors = ("clean",) if arm == "clean_observed_legacy" else DETECTORS
    for eval_detector in eval_detectors:
        actual_detector = DETECTORS[0] if eval_detector == "clean" else eval_detector
        actual_index = 0 if eval_detector == "clean" else DETECTORS.index(eval_detector)
        evaluations[eval_detector] = {}
        for partition in ("val", "test"):
            metrics, per_sample = evaluate(
                b1=b1,
                model=model,
                dataset=datasets[partition],
                arm=arm,
                detector=actual_detector,
                detector_index=actual_index,
                detector_cache=detector_cache,
                calibration_models=calibration_models,
                residual_risk=residual_risk,
                device=device,
                batch_size=int(budget["eval_batch_size"]),
                workers=int(budget["num_workers"]),
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
    artifact_hashes.extend(
        [
            {"path": str(best_path), "sha256": sha256_file(best_path)},
            {"path": str(last_path), "sha256": sha256_file(last_path)},
            {"path": str(history_path), "sha256": sha256_file(history_path)},
        ]
    )
    report = {
        "status": "PASS",
        "format": "phase9c_b2_run_report_v1",
        "phase": "9C-B2",
        "split": split,
        "seed": seed,
        "arm": arm,
        "train_detector": train_detector,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_selection_score": best_score,
        "checkpoint_selection": "matched_validation_geometric_mean_mpjpe_pa_mpjpe",
        "test_used_for_selection": False,
        "training_history": history,
        "evaluations": evaluations,
        "protocol_sha256": protocol_sha,
        "artifact_hashes": artifact_hashes,
        "elapsed_sec": time.time() - start,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    write_json(run_report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 9C-B2 full training for one split.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-name", choices=SPLITS, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = args.project_root.resolve()
    split = args.split_name
    output_root = root / "outputs/phase9c_b2_full_multisplit_multiseed"
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "configs/phase9c_b2_full_protocol.json"
    preflight_path = output_root / "phase9c_b2_preflight_report.json"
    protocol = load_json(protocol_path)
    preflight = load_json(preflight_path)
    if protocol.get("status") != "LOCKED" or protocol.get("format") != "phase9c_b2_full_protocol_lock_v1":
        raise ValueError("Invalid Phase 9C-B2 protocol lock.")
    if preflight.get("status") != "PASS" or preflight.get("scientific_decision") != "READY_FOR_PHASE9C_B2_FULL_RUNS":
        raise ValueError("Phase 9C-B2 preflight is not PASS.")
    if int(protocol["training_budget"]["num_workers"]) != 8:
        raise ValueError("Phase 9C-B2 is locked to workers=8.")
    protocol_sha = sha256_file(protocol_path)
    if preflight.get("protocol_sha256") != protocol_sha:
        raise ValueError("Protocol changed after preflight. Re-run preflight.")
    b1 = load_b1_module(root, str(protocol["source_b1_script_sha256"]))
    b1.SPLIT = split
    artifacts = b1.find_required_artifacts(root)
    phase9ca_qc = load_json(artifacts["phase9c_a_qc"])
    phase9ca_build = load_json(artifacts["phase9c_a_build"])
    phase9cb0_qc = load_json(artifacts["phase9c_b0_qc"])
    phase9cb0_build = load_json(artifacts["phase9c_b0_build"])
    if phase9ca_qc.get("status") != "PASS" or phase9cb0_qc.get("status") != "PASS":
        raise ValueError("Upstream Phase 9C-A/B0 gate is not PASS.")
    if phase9ca_build.get("official_TS1_to_TS6_read") is not False or phase9cb0_build.get("official_TS1_to_TS6_read") is not False:
        raise ValueError("Official TS1-TS6 isolation violated upstream.")
    join_expected = str(phase9cb0_qc["artifact_hashes"]["phase9c_b0_detector_join_cache.npz"])
    join_actual = sha256_file(artifacts["phase9c_b0_join_cache"])
    if join_actual != join_expected:
        raise ValueError("Join cache hash mismatch.")
    with np.load(artifacts["phase9c_b0_join_cache"], allow_pickle=False) as data:
        join_cache = {key: data[key] for key in data.files}
    detector_cache, detector_location, detector_hash = b1.load_phase9b_cache(
        root, str(phase9ca_build["source_hashes"]["phase9b_detector_cache.npz"])
    )
    audit, common_valid = b1.build_common_detection_audit(detector_cache, join_cache)
    common = audit["common_cohort"]
    print(
        f"[DETECTION] split={split} paired_common={common['eligible_samples']}/"
        f"{common['joined_samples']} excluded={common['excluded_any_detector_failure']}"
    )
    calibration_models = load_json(artifacts["phase9c_a_models"])
    with np.load(artifacts["phase9c_a_residual_bank"], allow_pickle=False) as data:
        residual_bank = {key: data[key] for key in data.files}
    risk_all_splits = b1.residual_risk_table(residual_bank)
    risk_split_index = SPLITS.index(split)
    if risk_all_splits.shape[0] != len(SPLITS):
        raise ValueError(f"Unexpected residual-risk split axis: {risk_all_splits.shape}")
    # The locked B1 V3 feature helper indexes residual risk at position zero
    # because smoke covered split_a only. Slice the requested split into
    # position zero so B2 cannot silently reuse split_a risk for split_b/c.
    risk = risk_all_splits[risk_split_index : risk_split_index + 1].copy()
    device = b1.resolve_device(args.device)
    manifest = (
        root / "data/processed/caches/phase5r1a_multigeometry_oof_25fps_v1"
        / split / "manifest.json"
    )
    source = {
        partition: b1.JoinedDetectorDataset(
            manifest_path=manifest,
            join_cache=join_cache,
            common_valid_detector_sample=common_valid,
            partition=partition,
            max_samples=0,
            seed=42,
        )
        for partition in PARTITIONS
    }
    selection = {
        name: {
            "joined_before_eligibility": dataset.joined_before_eligibility,
            "excluded_by_common_detection": dataset.excluded_by_common_detection,
            "eligible_full_samples": dataset.eligible_before_limit,
        }
        for name, dataset in source.items()
    }
    datasets = {name: materialize_full_dataset(dataset, name) for name, dataset in source.items()}
    del source
    gc.collect()
    run_specs: list[tuple[str, str, int]] = [("clean_observed_legacy", "clean", 0)]
    for arm in ARMS[1:]:
        for detector_index, detector in enumerate(DETECTORS):
            run_specs.append((arm, detector, detector_index))
    reports: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm, detector, detector_index in run_specs:
            print(
                f"[RUN] split={split} seed={seed} arm={arm} "
                f"train_detector={detector} workers=8"
            )
            run_dir = output_root / split / f"seed{seed}" / arm / f"train_{detector}"
            reports.append(
                train_one(
                    b1=b1,
                    split=split,
                    seed=seed,
                    arm=arm,
                    train_detector=detector,
                    train_detector_index=detector_index,
                    datasets=datasets,
                    detector_cache=detector_cache,
                    calibration_models=calibration_models,
                    residual_risk=risk,
                    device=device,
                    output_dir=run_dir,
                    protocol=protocol,
                    protocol_sha=protocol_sha,
                )
            )
    split_report = {
        "status": "PASS",
        "format": "phase9c_b2_split_report_v1",
        "phase": "9C-B2",
        "split": split,
        "run_count": len(reports),
        "seeds": list(SEEDS),
        "workers": 8,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "dataset_selection": selection,
        "residual_risk_split": split,
        "residual_risk_source_index": risk_split_index,
        "detector_coverage_audit": audit,
        "detector_cache_location": detector_location,
        "detector_cache_sha256": detector_hash,
        "join_cache_sha256": join_actual,
        "protocol_sha256": protocol_sha,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    write_json(output_root / split / "phase9c_b2_split_report.json", split_report)
    print("=" * 112)
    print(f"PHASE 9C-B2 FULL {split.upper()}")
    print("=" * 112)
    print("Status                : PASS")
    print(f"Runs                  : {len(reports)} / 21")
    print("Workers               : 8")
    print("Official TS1-TS6 read: NO")
    print("Next                  : run remaining split(s), then Phase 9C-B2 QC")
    print("=" * 112)


if __name__ == "__main__":
    main()
