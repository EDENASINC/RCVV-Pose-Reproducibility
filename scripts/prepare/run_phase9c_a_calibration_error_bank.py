from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


JOINT_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle", "l_hip",
    "l_knee", "l_ankle", "neck", "head", "l_shoulder",
    "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist",
]
COMMON12 = np.asarray([1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14])
REQUIRED_CACHE_KEYS = {
    "detector_id", "sample_index", "sample_key", "subject", "sequence",
    "camera_id", "frame_idx", "gt2d_px", "prediction_c15_px",
    "confidence_c15", "detected",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ece(probability: np.ndarray, target: np.ndarray, bins: int) -> float:
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        if index + 1 == bins:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        if np.any(mask):
            result += float(mask.mean()) * abs(
                float(probability[mask].mean()) - float(target[mask].mean())
            )
    return result


def brier(probability: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((probability - target) ** 2))


def fit_binned_isotonic(
    confidence: np.ndarray,
    target: np.ndarray,
    requested_bins: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    if confidence.size < requested_bins * 10:
        raise ValueError("Too few calibration observations.")
    order = np.argsort(confidence, kind="mergesort")
    x = confidence[order].astype(np.float64, copy=False)
    y = target[order].astype(np.float64, copy=False)
    edges = np.linspace(0, len(x), requested_bins + 1, dtype=np.int64)
    blocks: list[list[float]] = []
    for bin_index in range(requested_bins):
        start, stop = int(edges[bin_index]), int(edges[bin_index + 1])
        if stop <= start:
            continue
        blocks.append([
            float(x[start:stop].sum()),
            float(y[start:stop].sum()),
            float(stop - start),
        ])
        while (
            len(blocks) >= 2
            and blocks[-2][1] / blocks[-2][2]
            > blocks[-1][1] / blocks[-1][2]
        ):
            current = blocks.pop()
            blocks[-1] = [
                blocks[-1][0] + current[0],
                blocks[-1][1] + current[1],
                blocks[-1][2] + current[2],
            ]
    knots = np.asarray([item[0] / item[2] for item in blocks], dtype=np.float64)
    values = np.asarray([item[1] / item[2] for item in blocks], dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    summary = [
        {
            "mean_raw_confidence": float(item[0] / item[2]),
            "calibrated_probability": float(item[1] / item[2]),
            "count": int(item[2]),
        }
        for item in blocks
    ]
    return knots, values, summary


def apply_isotonic(
    confidence: np.ndarray,
    knots: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(
        confidence.astype(np.float64, copy=False),
        knots,
        values,
        left=float(values[0]),
        right=float(values[-1]),
    )


def flatten_calibration_data(
    raw_confidence: np.ndarray,
    radial_nme: np.ndarray,
    detected: np.ndarray,
    subjects: np.ndarray,
    selected_subjects: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    confidence = np.take(raw_confidence, COMMON12, axis=1)
    error = np.take(radial_nme, COMMON12, axis=1)
    mask = (
        np.isin(subjects, selected_subjects)[:, None]
        & detected[:, None]
        & np.isfinite(confidence)
        & np.isfinite(error)
    )
    return confidence[mask].astype(np.float64), (error[mask] <= 0.10).astype(np.float64)


def resolve(project_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else project_root / value


def valid_sha256_text(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def validate_phase9b_runtime(path: Path) -> tuple[bool, str]:
    try:
        runtime = load_json(path)
    except Exception as error:
        return False, f"invalid JSON: {error}"
    if runtime.get("format") != "phase9b_detector_runtime_v3":
        return False, f"format={runtime.get('format')!r}, expected phase9b_detector_runtime_v3"
    cuda = runtime.get("cuda", {})
    if cuda.get("required") is not True or cuda.get("cpu_fallback_allowed") is not False:
        return False, "strict CUDA/no-CPU-fallback provenance is missing"
    detectors = runtime.get("detectors", {})
    required_status = {"completed_on_gpu", "complete_cache_reused"}
    rtmpose = detectors.get("rtmpose_performance", {})
    yolo = detectors.get("yolo11l_pose", {})
    if rtmpose.get("status") not in required_status:
        return False, f"RTMPose status={rtmpose.get('status')!r}"
    if yolo.get("status") not in required_status:
        return False, f"YOLO status={yolo.get('status')!r}"
    rtmpose_files = rtmpose.get("model_files", [])
    roles = {
        item.get("role") for item in rtmpose_files
        if isinstance(item, dict) and valid_sha256_text(item.get("sha256"))
    }
    if roles != {"person_detector", "pose_estimator"}:
        return False, f"RTMPose hashed roles={sorted(str(role) for role in roles)}"
    yolo_files = yolo.get("model_files", [])
    if not any(
        isinstance(item, dict) and valid_sha256_text(item.get("sha256"))
        for item in yolo_files
    ):
        return False, "YOLO model SHA-256 is missing"
    return True, "Phase 9B-R3 detector runtime V3 with locked model hashes"


def extract_zip_member_atomic(archive: zipfile.ZipFile, member: zipfile.ZipInfo, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output, archive.open(member) as source:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def resolve_phase9b_artifact(
    phase9b_root: Path,
    materialized_root: Path,
    filename: str,
    expected_sha256: str | None = None,
    runtime_v3_required: bool = False,
) -> tuple[Path, str]:
    # Phase 9B-R3 keeps authoritative detector artifacts under full/. Older
    # packs may also expose flattened copies at the output root.
    candidates = [phase9b_root / "full" / filename, phase9b_root / filename]
    rejected: list[str] = []

    def candidate_is_valid(path: Path) -> bool:
        if not path.is_file():
            return False
        if expected_sha256 is not None:
            observed = sha256(path)
            if observed != expected_sha256:
                rejected.append(f"{path}: SHA-256 {observed}")
                return False
        if runtime_v3_required:
            passed, detail = validate_phase9b_runtime(path)
            if not passed:
                rejected.append(f"{path}: {detail}")
                return False
        return True

    for candidate in candidates:
        if candidate_is_valid(candidate):
            return candidate.resolve(), "filesystem"

    archive_path = phase9b_root / "phase9b_rgb_detector_benchmark_results.zip"
    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                (
                    info for info in archive.infolist()
                    if not info.is_dir() and Path(info.filename).name == filename
                ),
                key=lambda info: (
                    0 if "/full/" in f"/{info.filename.replace(chr(92), '/')}" else 1,
                    info.filename,
                ),
            )
            for index, member in enumerate(members):
                suffix = "" if index == 0 else f".{index}"
                extracted = materialized_root / f"{filename}{suffix}"
                extract_zip_member_atomic(archive, member, extracted)
                if candidate_is_valid(extracted):
                    return extracted.resolve(), f"zip:{archive_path.name}!{member.filename}"

    checked = ", ".join(str(path) for path in candidates + [archive_path])
    detail = "; ".join(rejected) if rejected else "no candidate file was found"
    raise FileNotFoundError(
        f"Cannot resolve locked Phase 9B-R3 artifact {filename}. "
        f"Checked: {checked}. Rejected: {detail}"
    )


def portable_source_location(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--phase9b-root", type=Path,
        default=Path("outputs/phase9b_rgb_detector_benchmark"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/phase9c_a_calibrated_error_bank"),
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("configs/phase9c_a_protocol_lock.json"),
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    phase9b_root = resolve(project_root, args.phase9b_root).resolve()
    output_root = resolve(project_root, args.output_root).resolve()
    protocol_path = resolve(project_root, args.protocol).resolve()
    protocol = load_json(protocol_path)
    if protocol.get("status") != "LOCKED" or protocol.get("format") != "phase9c_a_protocol_lock_v1":
        raise ValueError("Phase 9C-A protocol is not locked.")

    source_lock = protocol["source_lock"]
    materialized_root = output_root / "_phase9b_r3_source_materialized"
    qc_path, qc_origin = resolve_phase9b_artifact(
        phase9b_root,
        materialized_root,
        "phase9b_qc_report.json",
        expected_sha256=source_lock["phase9b_qc_sha256"],
    )
    cache_path, cache_origin = resolve_phase9b_artifact(
        phase9b_root,
        materialized_root,
        "phase9b_detector_cache.npz",
        expected_sha256=source_lock["phase9b_detector_cache_sha256"],
    )
    runtime_path, runtime_origin = resolve_phase9b_artifact(
        phase9b_root,
        materialized_root,
        "phase9b_detector_runtime.json",
        runtime_v3_required=True,
    )
    qc = load_json(qc_path)
    runtime = load_json(runtime_path)
    source_hashes = {
        "phase9b_qc_report.json": sha256(qc_path),
        "phase9b_detector_cache.npz": sha256(cache_path),
        "phase9b_detector_runtime.json": sha256(runtime_path),
    }
    if source_hashes["phase9b_qc_report.json"] != source_lock["phase9b_qc_sha256"]:
        raise ValueError("Phase 9B QC hash differs from the R3 evidence lock.")
    if source_hashes["phase9b_detector_cache.npz"] != source_lock["phase9b_detector_cache_sha256"]:
        raise ValueError("Phase 9B detector cache hash differs from the R3 evidence lock.")
    if qc.get("status") != source_lock["required_qc_status"]:
        raise ValueError("Phase 9B QC is not PASS.")
    if qc.get("scientific_decision") != source_lock["required_scientific_decision"]:
        raise ValueError("Phase 9B scientific decision does not open Phase 9C.")
    if qc.get("official_TS1_to_TS6_read") is not False:
        raise ValueError("Official TS1-TS6 isolation was violated.")
    if qc.get("model_training_performed") is not False:
        raise ValueError("Unexpected training recorded in Phase 9B.")
    runtime_valid, runtime_detail = validate_phase9b_runtime(runtime_path)
    if not runtime_valid:
        raise ValueError(f"Phase 9B-R3 runtime provenance is invalid: {runtime_detail}")

    output_root.mkdir(parents=True, exist_ok=True)
    npz = np.load(cache_path, allow_pickle=False)
    missing = REQUIRED_CACHE_KEYS - set(npz.files)
    if missing:
        raise ValueError(f"Phase 9B cache missing keys: {sorted(missing)}")
    detector_ids = npz["detector_id"].astype(str)
    if detector_ids.tolist() != source_lock["required_detectors"]:
        raise ValueError(f"Detector order mismatch: {detector_ids.tolist()}")
    subjects = npz["subject"].astype(str)
    gt = npz["gt2d_px"].astype(np.float32)
    prediction = npz["prediction_c15_px"].astype(np.float32)
    raw_confidence = npz["confidence_c15"].astype(np.float32)
    detected = npz["detected"].astype(bool)
    sample_count = int(gt.shape[0])
    if sample_count != int(source_lock["required_sample_count"]):
        raise ValueError(f"Sample count mismatch: {sample_count}")
    if prediction.shape != (2, sample_count, 15, 2):
        raise ValueError(f"Prediction shape mismatch: {prediction.shape}")
    if raw_confidence.shape != (2, sample_count, 15):
        raise ValueError(f"Confidence shape mismatch: {raw_confidence.shape}")
    if detected.shape != (2, sample_count):
        raise ValueError(f"Detection shape mismatch: {detected.shape}")

    torso = np.linalg.norm(gt[:, 7] - gt[:, 0], axis=1).astype(np.float32)
    if not np.all(np.isfinite(torso)) or np.any(torso <= 1e-6):
        raise ValueError("Invalid GT pelvis-to-neck torso normalizer.")
    residual_xy = (prediction - gt[None, ...]) / torso[None, :, None, None]
    radial_nme = np.linalg.norm(residual_xy, axis=-1)

    split_names = list(protocol["development_splits"].keys())
    reservoir_capacity = int(protocol["residual_bank"]["reservoir_per_split_detector_joint_bin"])
    confidence_bins = int(protocol["residual_bank"]["confidence_bins"])
    reservoir = np.full(
        (len(split_names), 2, 15, confidence_bins, reservoir_capacity, 2),
        np.nan,
        dtype=np.float32,
    )
    reservoir_raw_confidence = np.full(reservoir.shape[:-1], np.nan, dtype=np.float32)
    reservoir_calibrated_probability = np.full(reservoir.shape[:-1], np.nan, dtype=np.float32)
    reservoir_counts = np.zeros((len(split_names), 2, 15, confidence_bins), dtype=np.int32)
    population_counts = np.zeros_like(reservoir_counts, dtype=np.int64)
    missing_pose_rate_train = np.zeros((len(split_names), 2), dtype=np.float64)
    missing_pose_rate_validation = np.zeros((len(split_names), 2), dtype=np.float64)

    models: dict[str, Any] = {
        "format": "phase9c_a_calibration_models_v1",
        "phase": "9C-A",
        "target": protocol["calibration"]["target"],
        "fit_subset": "direct_common12",
        "splits": {},
    }
    metric_rows: list[dict[str, Any]] = []
    rng_root = 9102026

    for split_index, split_name in enumerate(split_names):
        split = protocol["development_splits"][split_name]
        train_subjects = list(split["train_subjects"])
        validation_subjects = list(split["validation_subjects"])
        models["splits"][split_name] = {
            "fit_subjects": train_subjects,
            "validation_subjects": validation_subjects,
            "test_subjects_used": [],
            "detectors": {},
        }
        train_sample_mask = np.isin(subjects, train_subjects)
        validation_sample_mask = np.isin(subjects, validation_subjects)
        for detector_index, detector_id in enumerate(detector_ids):
            train_x, train_y = flatten_calibration_data(
                raw_confidence[detector_index], radial_nme[detector_index],
                detected[detector_index], subjects, train_subjects,
            )
            validation_x, validation_y = flatten_calibration_data(
                raw_confidence[detector_index], radial_nme[detector_index],
                detected[detector_index], subjects, validation_subjects,
            )
            knots, values, pava_blocks = fit_binned_isotonic(
                train_x, train_y, int(protocol["calibration"]["method"].split("_")[1])
            )
            calibrated_train = apply_isotonic(train_x, knots, values)
            calibrated_validation = apply_isotonic(validation_x, knots, values)
            bins = int(protocol["calibration"]["ece_bins"])
            metrics = {
                "train": {
                    "observations": int(train_x.size),
                    "raw_ece": ece(train_x, train_y, bins),
                    "calibrated_ece": ece(calibrated_train, train_y, bins),
                    "raw_brier": brier(train_x, train_y),
                    "calibrated_brier": brier(calibrated_train, train_y),
                },
                "validation": {
                    "observations": int(validation_x.size),
                    "raw_ece": ece(validation_x, validation_y, bins),
                    "calibrated_ece": ece(calibrated_validation, validation_y, bins),
                    "raw_brier": brier(validation_x, validation_y),
                    "calibrated_brier": brier(calibrated_validation, validation_y),
                },
            }
            models["splits"][split_name]["detectors"][detector_id] = {
                "method": "fixed_100_equal_frequency_bins_then_weighted_PAVA",
                "knots_raw_confidence": knots.tolist(),
                "values_probability_correct_at_0.10_torso": values.tolist(),
                "pava_blocks": pava_blocks,
                "metrics": metrics,
            }
            metric_rows.append({
                "split": split_name,
                "detector": detector_id,
                "fit_subjects": "+".join(train_subjects),
                "validation_subjects": "+".join(validation_subjects),
                "train_observations": metrics["train"]["observations"],
                "validation_observations": metrics["validation"]["observations"],
                "validation_raw_ece": metrics["validation"]["raw_ece"],
                "validation_calibrated_ece": metrics["validation"]["calibrated_ece"],
                "validation_ece_reduction": metrics["validation"]["raw_ece"] - metrics["validation"]["calibrated_ece"],
                "validation_raw_brier": metrics["validation"]["raw_brier"],
                "validation_calibrated_brier": metrics["validation"]["calibrated_brier"],
                "validation_brier_reduction": metrics["validation"]["raw_brier"] - metrics["validation"]["calibrated_brier"],
            })
            missing_pose_rate_train[split_index, detector_index] = 1.0 - float(
                detected[detector_index, train_sample_mask].mean()
            )
            missing_pose_rate_validation[split_index, detector_index] = 1.0 - float(
                detected[detector_index, validation_sample_mask].mean()
            )

            for joint_index in range(15):
                joint_conf = raw_confidence[detector_index, :, joint_index]
                joint_residual = residual_xy[detector_index, :, joint_index]
                finite = (
                    train_sample_mask
                    & detected[detector_index]
                    & np.isfinite(joint_conf)
                    & np.isfinite(joint_residual).all(axis=1)
                )
                for confidence_bin in range(confidence_bins):
                    lower = confidence_bin / confidence_bins
                    upper = (confidence_bin + 1) / confidence_bins
                    if confidence_bin + 1 == confidence_bins:
                        selected = finite & (joint_conf >= lower) & (joint_conf <= upper)
                    else:
                        selected = finite & (joint_conf >= lower) & (joint_conf < upper)
                    indices = np.flatnonzero(selected)
                    population_counts[split_index, detector_index, joint_index, confidence_bin] = len(indices)
                    if len(indices) == 0:
                        continue
                    local_rng = np.random.default_rng(
                        rng_root + split_index * 100000 + detector_index * 10000
                        + joint_index * 100 + confidence_bin
                    )
                    if len(indices) > reservoir_capacity:
                        indices = local_rng.choice(indices, size=reservoir_capacity, replace=False)
                    count = len(indices)
                    reservoir_counts[split_index, detector_index, joint_index, confidence_bin] = count
                    reservoir[split_index, detector_index, joint_index, confidence_bin, :count] = joint_residual[indices]
                    selected_confidence = joint_conf[indices]
                    reservoir_raw_confidence[split_index, detector_index, joint_index, confidence_bin, :count] = selected_confidence
                    reservoir_calibrated_probability[split_index, detector_index, joint_index, confidence_bin, :count] = apply_isotonic(
                        selected_confidence, knots, values
                    ).astype(np.float32)

    models_path = output_root / "phase9c_a_calibration_models.json"
    write_json(models_path, models)
    metrics_path = output_root / "phase9c_a_calibration_summary.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    bank_path = output_root / "phase9c_a_residual_bank.npz"
    np.savez_compressed(
        bank_path,
        split_name=np.asarray(split_names),
        detector_id=detector_ids,
        joint_name=np.asarray(JOINT_NAMES),
        confidence_bin_edges=np.linspace(0.0, 1.0, confidence_bins + 1, dtype=np.float32),
        residual_xy_torso=reservoir,
        raw_confidence=reservoir_raw_confidence,
        calibrated_probability=reservoir_calibrated_probability,
        reservoir_counts=reservoir_counts,
        population_counts=population_counts,
        missing_pose_rate_train=missing_pose_rate_train,
        missing_pose_rate_validation=missing_pose_rate_validation,
    )
    protocol_copy = output_root / "phase9c_a_protocol_lock.json"
    protocol_copy.write_bytes(protocol_path.read_bytes())
    report = {
        "status": "PASS",
        "format": "phase9c_a_build_report_v1",
        "phase": "9C-A",
        "source_phase": "9B-R3",
        "source_hashes": source_hashes,
        "source_locations": {
            "phase9b_qc_report.json": {
                "path": portable_source_location(qc_path, project_root),
                "origin": qc_origin,
            },
            "phase9b_detector_cache.npz": {
                "path": portable_source_location(cache_path, project_root),
                "origin": cache_origin,
            },
            "phase9b_detector_runtime.json": {
                "path": portable_source_location(runtime_path, project_root),
                "origin": runtime_origin,
                "validation": runtime_detail,
            },
        },
        "source_runtime_format": runtime.get("format"),
        "source_qc_status": qc["status"],
        "source_scientific_decision": qc["scientific_decision"],
        "sample_count": sample_count,
        "detectors": detector_ids.tolist(),
        "splits": split_names,
        "calibration_pairs": len(metric_rows),
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
        "test_subjects_used_for_calibration": False,
        "artifacts": {
            "protocol": str(protocol_copy),
            "models": str(models_path),
            "metrics": str(metrics_path),
            "residual_bank": str(bank_path),
        },
        "artifact_hashes": {
            "phase9c_a_protocol_lock.json": sha256(protocol_copy),
            "phase9c_a_calibration_models.json": sha256(models_path),
            "phase9c_a_calibration_summary.csv": sha256(metrics_path),
            "phase9c_a_residual_bank.npz": sha256(bank_path),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    report_path = output_root / "phase9c_a_build_report.json"
    write_json(report_path, report)
    print("=" * 96)
    print("PHASE 9C-A CALIBRATION + RESIDUAL BANK")
    print("=" * 96)
    for row in metric_rows:
        print(
            f"{row['split']} {row['detector']:<20} "
            f"ECE {row['validation_raw_ece']:.4f}->{row['validation_calibrated_ece']:.4f} "
            f"Brier {row['validation_raw_brier']:.4f}->{row['validation_calibrated_brier']:.4f}"
        )
    print(f"Residual bank : {bank_path}")
    print(f"Build report  : {report_path}")
    print("Status        : PASS")


if __name__ == "__main__":
    main()
