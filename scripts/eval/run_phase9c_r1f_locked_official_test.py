from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PHASE = "9C-R1F"
FORMAT = "phase9c_r1f_locked_official_test_protocol_v1"
OUTPUT_RELATIVE = Path("outputs/phase9c_r1f_locked_official_test")
PROTOCOL_RELATIVE = Path("configs/phase9c_r1f_locked_official_test_protocol.json")
SPLITS = ("split_a", "split_b", "split_c")
SEEDS = (42, 123, 2026)
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
ARMS = (
    "detector_aware_observed",
    "detector_conditioned_virtual_view_no_confidence",
    "bounded_reliability_observed_modulation",
    "bounded_reliability_dual_modulation",
)
FEATURE_DIMS = {
    "detector_aware_observed": 28,
    "detector_conditioned_virtual_view_no_confidence": 56,
    "bounded_reliability_observed_modulation": 70,
    "bounded_reliability_dual_modulation": 126,
}
THRESHOLDS_MM = np.arange(0.0, 151.0, 5.0, dtype=np.float64)
EXPECTED_B1_SHA256 = "fc1ae8f11ec6fb1b75c8d8c730e61a67f1b1aa45422227364e4bce71eb72b673"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def protocol_and_root(project_root: Path) -> tuple[Path, dict[str, Any], Path]:
    root = project_root.resolve()
    path = root / PROTOCOL_RELATIVE
    protocol = load_json(path)
    if protocol.get("status") != "LOCKED" or protocol.get("format") != FORMAT:
        raise ValueError("Invalid Phase 9C-R1F protocol lock.")
    return root, protocol, path


def load_b1(root: Path):
    path = root / "scripts/train/run_phase9c_b1_four_arm_smoke.py"
    actual = sha256_file(path)
    if actual != EXPECTED_B1_SHA256:
        raise ValueError(f"Phase 9C-B1 V3 source hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("phase9c_b1_v3_r1f", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reshape_pose(array: np.ndarray, frames: int, joints: int, dimensions: int, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.shape == (frames, 1, joints, dimensions):
        value = value[:, 0]
    elif value.shape == (dimensions, joints, 1, frames):
        value = value.transpose(3, 2, 1, 0)[:, 0]
    elif value.shape != (frames, joints, dimensions):
        raise ValueError(f"{name} shape={value.shape}; expected {frames}x{joints}x{dimensions}")
    return np.asarray(value, dtype=np.float32)


def valid_frame_indices(annotation: Path, expected_total: int, expected_valid: int) -> np.ndarray:
    import h5py

    with h5py.File(annotation, "r") as handle:
        if "valid_frame" not in handle:
            raise KeyError(f"Missing valid_frame: {annotation}")
        raw = np.asarray(handle["valid_frame"]).reshape(-1)
    if raw.size != expected_total:
        raise ValueError(f"Total-frame mismatch: {annotation}: {raw.size} != {expected_total}")
    indices = np.flatnonzero(raw.astype(np.int64) == 1).astype(np.int32)
    if indices.size != expected_valid:
        raise ValueError(f"Valid-frame mismatch: {annotation}: {indices.size} != {expected_valid}")
    return indices


def discover_subject_images(subject_dir: Path, valid_indices: np.ndarray) -> tuple[dict[int, Path], dict[str, Any]]:
    groups: dict[Path, dict[int, Path]] = defaultdict(dict)
    for path in subject_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = re.search(r"(\d+)$", path.stem)
        if match:
            groups[path.parent][int(match.group(1))] = path.resolve()
    target = set(int(value) for value in valid_indices)
    candidates: list[dict[str, Any]] = []
    for directory, numbered in groups.items():
        for offset in (-2, -1, 0, 1, 2):
            aligned = {number + offset: path for number, path in numbered.items()}
            coverage = sum(index in aligned for index in target)
            candidates.append({
                "coverage": coverage,
                "count": len(numbered),
                "offset": offset,
                "directory": directory,
                "aligned": aligned,
            })
    if not candidates:
        raise FileNotFoundError(f"No extracted RGB images found under {subject_dir}")
    best = max(
        candidates,
        key=lambda item: (
            int(item["coverage"]),
            int(item["count"]),
            -abs(int(item["offset"])),
            -int(item["offset"]),
            str(item["directory"]),
        ),
    )
    coverage = int(best["coverage"])
    count = int(best["count"])
    directory = Path(best["directory"])
    aligned = best["aligned"]
    best_offset = int(best["offset"])
    selected = {index: aligned[index] for index in target if index in aligned}
    return selected, {
        "directory": str(directory),
        "numbered_images": count,
        "selected_offset_image_number_to_zero_based_annotation": best_offset,
        "valid_images_found": coverage,
        "valid_images_required": len(target),
        "coverage": coverage / max(len(target), 1),
    }


def canonical_checkpoint(root: Path, split: str, seed: int, detector: str, arm: str) -> tuple[Path, Path]:
    if arm == "bounded_reliability_observed_modulation":
        base = root / "outputs/phase9c_r1e_synergy_attribution"
        report_name = "phase9c_r1e_run_report.json"
    else:
        base = root / "outputs/phase9c_r1d_locked_full_confirmation"
        report_name = "phase9c_r1d_run_report.json"
    run_dir = base / split / f"seed{seed}" / arm / f"train_{detector}"
    return run_dir / "best.pt", run_dir / report_name


def verify_checkpoint(root: Path, split: str, seed: int, detector: str, arm: str) -> dict[str, Any]:
    checkpoint, report_path = canonical_checkpoint(root, split, seed, detector, arm)
    report = load_json(report_path)
    if report.get("status") != "PASS" or report.get("split") != split:
        raise ValueError(f"Invalid run report: {report_path}")
    if int(report.get("seed", -1)) != seed or report.get("arm") != arm or report.get("train_detector") != detector:
        raise ValueError(f"Run matrix mismatch: {report_path}")
    records = [item for item in report.get("artifact_hashes", []) if Path(item.get("path", "")).name == "best.pt"]
    if len(records) != 1:
        raise ValueError(f"Expected one best.pt hash: {report_path}")
    actual = sha256_file(checkpoint)
    if actual != records[0]["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
    return {
        "split": split,
        "seed": seed,
        "detector": detector,
        "arm": arm,
        "checkpoint": str(checkpoint),
        "sha256": actual,
        "report": str(report_path),
    }


def diagnostic_zip(output_root: Path, names: Iterable[str]) -> Path:
    path = output_root / "phase9c_r1f_preflight_diagnostic.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            source = output_root / name
            if source.is_file():
                archive.write(source, source.name)
    return path


def run_preflight(root: Path, protocol: dict[str, Any], protocol_path: Path) -> int:
    output_root = root / OUTPUT_RELATIVE
    output_root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    image_subjects: list[str] = []
    image_frames: list[int] = []
    image_relpaths: list[str] = []
    focal: list[float] = []
    width: list[int] = []
    height: list[int] = []
    subject_audit: list[dict[str, Any]] = []
    try:
        r1e_root = root / "outputs/phase9c_r1e_synergy_attribution"
        r1e_qc = load_json(r1e_root / "phase9c_r1e_qc_report.json")
        r1e_stats = load_json(r1e_root / "phase9c_r1e_statistics.json")
        gate = protocol["source_gate"]
        if r1e_qc.get("status") != gate["required_qc_status"] or r1e_qc.get("scientific_decision") != gate["required_scientific_decision"]:
            raise ValueError("R1E gate does not authorize Official Test.")
        if r1e_stats.get("scientific_decision") != gate["required_scientific_decision"]:
            raise ValueError("R1E statistics decision mismatch.")
        if r1e_qc.get("official_TS1_to_TS6_read") is not False or r1e_stats.get("official_TS1_to_TS6_read") is not False:
            raise ValueError("R1E Official-Test isolation flag is not false.")
        source_protocol = root / "configs/phase9c_r1e_synergy_attribution_protocol.json"
        if sha256_file(source_protocol) != gate["phase9c_r1e_protocol_sha256"]:
            raise ValueError("R1E protocol hash mismatch.")
        for split in SPLITS:
            for seed in SEEDS:
                for detector in DETECTORS:
                    for arm in ARMS:
                        checkpoints.append(verify_checkpoint(root, split, seed, detector, arm))
        if len(checkpoints) != 72:
            raise ValueError(f"Checkpoint matrix incomplete: {len(checkpoints)}/72")
        load_b1(root)
        missing_packages = [
            name
            for name in ("torch", "h5py", "cv2", "rtmlib", "ultralytics", "onnxruntime")
            if importlib.util.find_spec(name) is None
        ]
        if missing_packages:
            raise ImportError(f"Missing Phase 9B/R1F runtime packages: {missing_packages}")
        for item in protocol["subjects"]:
            annotation = root / item["annotation_relpath"]
            if sha256_file(annotation) != item["annotation_sha256"]:
                raise ValueError(f"Official annotation hash mismatch: {item['subject']}")
            valid = valid_frame_indices(annotation, int(item["total_frames"]), int(item["valid_frames"]))
            mapping, audit = discover_subject_images(annotation.parent, valid)
            audit["subject"] = item["subject"]
            audit["annotation_sha256"] = item["annotation_sha256"]
            subject_audit.append(audit)
            if len(mapping) != len(valid):
                raise ValueError(f"{item['subject']}: valid RGB coverage {len(mapping)}/{len(valid)}")
            for frame_index in valid:
                path = mapping[int(frame_index)]
                image_subjects.append(item["subject"])
                image_frames.append(int(frame_index))
                try:
                    relpath = path.relative_to(root).as_posix()
                except ValueError:
                    relpath = str(path)
                image_relpaths.append(relpath)
                focal.append(float(item["focal_length_px"]))
                width.append(int(item["image_size"][0]))
                height.append(int(item["image_size"][1]))
                rows.append({"subject": item["subject"], "frame_index_zero_based": int(frame_index), "image_relpath": relpath})
        np.savez_compressed(
            output_root / "phase9c_r1f_official_rgb_index.npz",
            subject=np.asarray(image_subjects),
            frame_index_zero_based=np.asarray(image_frames, dtype=np.int32),
            image_relpath=np.asarray(image_relpaths),
            focal_length_px=np.asarray(focal, dtype=np.float32),
            image_width=np.asarray(width, dtype=np.int32),
            image_height=np.asarray(height, dtype=np.int32),
        )
        write_csv(output_root / "phase9c_r1f_official_rgb_index.csv", rows)
        write_csv(output_root / "phase9c_r1f_checkpoint_inventory.csv", checkpoints)
    except Exception as error:
        errors.append(repr(error))
    report = {
        "status": "PASS" if not errors else "BLOCKED",
        "format": "phase9c_r1f_preflight_report_v1",
        "phase": PHASE,
        "scientific_decision": "READY_TO_LOCK_OFFICIAL_RGB_DETECTOR_CACHE" if not errors else "REPAIR_BEFORE_OFFICIAL_INFERENCE",
        "protocol_sha256": sha256_file(protocol_path),
        "checkpoints_verified": len(checkpoints),
        "official_subjects": subject_audit,
        "official_valid_rgb_frames": len(image_frames),
        "scientific_3d_metrics_opened": False,
        "official_ground_truth_3d_read": False,
        "official_file_inventory_read": bool(subject_audit),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": package_version("torch"),
            "h5py": package_version("h5py"),
            "opencv": package_version("opencv-python") or package_version("opencv-python-headless"),
            "rtmlib": package_version("rtmlib"),
            "ultralytics": package_version("ultralytics"),
            "onnxruntime_gpu": package_version("onnxruntime-gpu"),
        },
        "errors": errors,
    }
    write_json(output_root / "phase9c_r1f_preflight_report.json", report)
    if errors:
        path = diagnostic_zip(output_root, ("phase9c_r1f_preflight_report.json", "phase9c_r1f_official_rgb_index.csv"))
        print(f"Status: BLOCKED\nDiagnostic: {path}\nErrors: {errors}")
        return 2
    print("=" * 104)
    print("PHASE 9C-R1F PRE-METRIC EVIDENCE LOCK")
    print("=" * 104)
    print("Status                    : PASS")
    print(f"Checkpoints verified      : {len(checkpoints)} / 72")
    print(f"Official valid RGB frames : {len(image_frames)}")
    print("Official 3D metrics       : SEALED")
    print("Next                       : run detector cache")
    print("=" * 104)
    return 0


def coco17_to_c15(keypoints: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(confidence, dtype=np.float32)
    if points.shape != (17, 2) or scores.shape != (17,):
        raise ValueError(f"Expected COCO17, got {points.shape}/{scores.shape}")
    result = np.zeros((15, 2), dtype=np.float32)
    result_conf = np.zeros(15, dtype=np.float32)
    direct = {1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15, 9: 5, 10: 7, 11: 9, 12: 6, 13: 8, 14: 10}
    for target, source in direct.items():
        result[target] = points[source]
        result_conf[target] = scores[source]
    result[0] = 0.5 * (points[11] + points[12])
    result_conf[0] = min(scores[11], scores[12])
    result[7] = 0.5 * (points[5] + points[6])
    result_conf[7] = min(scores[5], scores[6])
    result[8] = points[0]
    result_conf[8] = scores[0]
    return result, np.clip(result_conf, 0.0, 1.0)


def normalize_pose_output(keypoints: Any, scores: Any) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(keypoints, dtype=np.float32)
    confidence = np.asarray(scores, dtype=np.float32)
    if points.ndim == 2:
        points = points[None]
    if confidence.ndim == 1:
        confidence = confidence[None]
    confidence = np.squeeze(confidence)
    if confidence.ndim == 1:
        confidence = confidence[None]
    if points.ndim != 3 or points.shape[-2:] != (17, 2) or confidence.shape != points.shape[:2] or points.shape[0] == 0:
        return None
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    index = int(np.argmax(confidence.mean(axis=1)))
    return coco17_to_c15(points[index], confidence[index])


class RTMPoseDetector:
    detector_id = "rtmpose_performance"

    def __init__(self, use_cuda: bool) -> None:
        from rtmlib import Body

        self.device = "cuda" if use_cuda else "cpu"
        self.model = Body(to_openpose=False, mode="performance", backend="onnxruntime", device=self.device)

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        return normalize_pose_output(*self.model(image))


class YOLOPoseDetector:
    detector_id = "yolo11l_pose"

    def __init__(self, use_cuda: bool) -> None:
        from ultralytics import YOLO

        self.device = 0 if use_cuda else "cpu"
        self.model = YOLO("yolo11l-pose.pt")

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        result = self.model.predict(source=image, imgsz=640, conf=0.05, iou=0.7, device=self.device, verbose=False)[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return None
        points = result.keypoints.xy.detach().cpu().numpy()
        confidence = (
            np.ones(points.shape[:2], dtype=np.float32)
            if result.keypoints.conf is None
            else result.keypoints.conf.detach().cpu().numpy()
        )
        if points.shape[0] == 0:
            return None
        if result.boxes is not None and result.boxes.conf is not None:
            index = int(np.argmax(result.boxes.conf.detach().cpu().numpy()))
            return coco17_to_c15(points[index], confidence[index])
        return normalize_pose_output(points, confidence)


def model_file_hashes(model: Any) -> list[dict[str, Any]]:
    candidates: set[Path] = set()

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 4:
            return
        if isinstance(value, (str, os.PathLike)):
            path = Path(value)
            if path.suffix.lower() in {".onnx", ".pt", ".pth"} and path.is_file():
                candidates.add(path.resolve())
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, depth + 1)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, depth + 1)
        elif hasattr(value, "__dict__"):
            visit(vars(value), depth + 1)

    visit(model)
    return [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(candidates)]


def source_phase9b_model_hashes(root: Path, detector_id: str) -> tuple[set[str], list[str]]:
    hashes: set[str] = set()
    reports: list[str] = []
    phase9b_root = root / "outputs/phase9b_rgb_detector_benchmark"
    if not phase9b_root.is_dir():
        return hashes, reports
    for path in phase9b_root.rglob("phase9b_detector_runtime.json"):
        try:
            report = load_json(path)
            record = report.get("detectors", {}).get(detector_id, {})
            values = {str(item["sha256"]) for item in record.get("model_files", []) if item.get("sha256")}
            if values:
                hashes.update(values)
                reports.append(str(path))
        except Exception:
            continue
    return hashes, reports


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    item = json.loads(line)
                    result[int(item["sample_index"])] = item
                except Exception as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    return result


def run_detectors(root: Path, protocol: dict[str, Any], protocol_path: Path) -> int:
    import cv2

    output_root = root / OUTPUT_RELATIVE
    preflight = load_json(output_root / "phase9c_r1f_preflight_report.json")
    if preflight.get("status") != "PASS" or preflight.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("R1F preflight is not a matching PASS.")
    with np.load(output_root / "phase9c_r1f_official_rgb_index.npz", allow_pickle=False) as handle:
        index = {name: handle[name] for name in handle.files}
    total = len(index["subject"])
    detector_dir = output_root / "detector_cache_work"
    detector_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        yolo_cuda = bool(torch.cuda.is_available())
    except Exception:
        yolo_cuda = False
    try:
        import onnxruntime as ort
        rtmpose_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        rtmpose_cuda = False
    builders = (
        (DETECTORS[0], lambda: RTMPoseDetector(rtmpose_cuda)),
        (DETECTORS[1], lambda: YOLOPoseDetector(yolo_cuda)),
    )
    runtime: dict[str, Any] = {"format": "phase9c_r1f_detector_runtime_v1", "detectors": {}, "total_samples": total}
    for detector_id, builder in builders:
        path = detector_dir / f"{detector_id}.jsonl"
        completed = load_jsonl(path)
        detector = builder()
        started = time.time()
        new = failures = 0
        with path.open("a", encoding="utf-8") as handle:
            for sample_index in range(total):
                if sample_index in completed:
                    continue
                image_path = Path(str(index["image_relpath"][sample_index]))
                if not image_path.is_absolute():
                    image_path = root / image_path
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                item: dict[str, Any] = {"sample_index": sample_index}
                if image is None:
                    item.update({"detected": False, "error": "cv2_imread_failed"})
                    failures += 1
                else:
                    try:
                        prediction = detector.predict(image)
                        if prediction is None:
                            item.update({"detected": False, "error": "no_person_pose"})
                            failures += 1
                        else:
                            points, confidence = prediction
                            item.update({"detected": True, "keypoints_c15_px": points.tolist(), "confidence_c15": confidence.tolist()})
                    except Exception as error:
                        item.update({"detected": False, "error": repr(error)})
                        failures += 1
                handle.write(json.dumps(item, separators=(",", ":")) + "\n")
                handle.flush()
                new += 1
                if (sample_index + 1) % 100 == 0 or sample_index + 1 == total:
                    print(f"[{detector_id}] {sample_index + 1}/{total} new={new} failures={failures}")
        elapsed = time.time() - started
        current_files = model_file_hashes(detector)
        current_hashes = {item["sha256"] for item in current_files}
        source_hashes, source_reports = source_phase9b_model_hashes(root, detector_id)
        if source_hashes and current_hashes and current_hashes.isdisjoint(source_hashes):
            raise ValueError(f"{detector_id}: Official detector weights do not match any recorded Phase 9B model hash")
        runtime["detectors"][detector_id] = {
            "device": detector.device,
            "new_samples": new,
            "elapsed_sec": elapsed,
            "model_files": current_files,
            "source_phase9b_runtime_reports": source_reports,
            "source_phase9b_model_hashes": sorted(source_hashes),
            "phase9b_weight_parity_verified": bool(source_hashes and current_hashes and not current_hashes.isdisjoint(source_hashes)),
            "phase9b_weight_parity_note": "Verified when both current and Phase 9B runtime model-file hashes are discoverable; otherwise detector ID/configuration and current hashes are retained for audit.",
        }
        write_json(detector_dir / "phase9c_r1f_detector_runtime.json", runtime)
    predictions = np.zeros((2, total, 15, 2), dtype=np.float32)
    confidence = np.zeros((2, total, 15), dtype=np.float32)
    detected = np.zeros((2, total), dtype=bool)
    for sample_index in range(total):
        predictions[:, sample_index, :, 0] = float(index["image_width"][sample_index]) / 2.0
        predictions[:, sample_index, :, 1] = float(index["image_height"][sample_index]) / 2.0
    coverage: dict[str, Any] = {}
    for detector_index, detector_id in enumerate(DETECTORS):
        records = load_jsonl(detector_dir / f"{detector_id}.jsonl")
        if set(records) != set(range(total)):
            raise ValueError(f"Incomplete detector cache: {detector_id} {len(records)}/{total}")
        for sample_index, item in records.items():
            if not item.get("detected"):
                continue
            points = np.asarray(item["keypoints_c15_px"], dtype=np.float32)
            scores = np.asarray(item["confidence_c15"], dtype=np.float32)
            if points.shape != (15, 2) or scores.shape != (15,) or not np.isfinite(points).all() or not np.isfinite(scores).all():
                continue
            predictions[detector_index, sample_index] = points
            confidence[detector_index, sample_index] = np.clip(scores, 0.0, 1.0)
            detected[detector_index, sample_index] = True
        by_subject = {}
        for subject in sorted(set(index["subject"].tolist())):
            mask = index["subject"] == subject
            by_subject[str(subject)] = float(detected[detector_index, mask].mean())
        coverage[detector_id] = {
            "detected": int(detected[detector_index].sum()),
            "total": total,
            "success_rate": float(detected[detector_index].mean()),
            "by_subject": by_subject,
        }
    cache_path = output_root / "phase9c_r1f_official_detector_cache.npz"
    np.savez_compressed(cache_path, **index, detector_id=np.asarray(DETECTORS), prediction_c15_px=predictions, confidence_c15=confidence, detected=detected)
    report = {
        "status": "PASS",
        "format": "phase9c_r1f_detector_cache_report_v1",
        "phase": PHASE,
        "protocol_sha256": sha256_file(protocol_path),
        "cache_sha256": sha256_file(cache_path),
        "samples": total,
        "coverage": coverage,
        "ground_truth_assisted_person_selection": False,
        "missing_detection_policy": protocol["evaluation_scope"]["missing_detection_policy"],
        "scientific_3d_metrics_opened": False,
        "official_rgb_pixels_read": True,
        "official_ground_truth_3d_read": False,
        "runtime": runtime,
    }
    write_json(output_root / "phase9c_r1f_detector_cache_report.json", report)
    print("=" * 104)
    print("PHASE 9C-R1F OFFICIAL RGB DETECTOR CACHE")
    print("=" * 104)
    print("Status              : PASS")
    for detector in DETECTORS:
        print(f"{detector:20s}: {100.0 * coverage[detector]['success_rate']:.2f}%")
    print("Official 3D metrics : SEALED")
    print("=" * 104)
    return 0


def load_official_targets(root: Path, protocol: dict[str, Any], cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    mapping = np.asarray(protocol["evaluation_scope"]["official17_to_canonical15_zero_based"], dtype=np.int64)
    targets: list[np.ndarray] = []
    activities: list[np.ndarray] = []
    for item in protocol["subjects"]:
        path = root / item["annotation_relpath"]
        with h5py.File(path, "r") as handle:
            valid_raw = np.asarray(handle["valid_frame"]).reshape(-1)
            activity_raw = np.asarray(handle["activity_annotation"]).reshape(-1).astype(np.int16)
            frames = valid_raw.size
            universal = _reshape_pose(np.asarray(handle["univ_annot3"]), frames, 17, 3, "univ_annot3")
        valid = valid_raw.astype(np.int64) == 1
        target = universal[valid][:, mapping]
        target = target - target[:, :1]
        target = target[:, 1:]
        subject_mask = cache["subject"] == item["subject"]
        expected_frames = np.flatnonzero(valid).astype(np.int32)
        if not np.array_equal(cache["frame_index_zero_based"][subject_mask], expected_frames):
            raise ValueError(f"Official frame index mismatch: {item['subject']}")
        targets.append(target.astype(np.float32))
        activities.append(activity_raw[valid])
    result = np.concatenate(targets)
    activity = np.concatenate(activities)
    if result.shape != (len(cache["subject"]), 14, 3) or not np.isfinite(result).all():
        raise ValueError(f"Official target shape/integrity mismatch: {result.shape}")
    return result, activity


def risk_joint_scale(residual_risk: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    if residual_risk.shape != (1, 2, 15, 10):
        raise ValueError(f"Unexpected residual-risk shape: {residual_risk.shape}")
    result = np.empty((2, 15), dtype=np.float32)
    for detector in range(2):
        for joint in range(15):
            values = residual_risk[0, detector, joint]
            positive = values[np.isfinite(values) & (values > 0)]
            result[detector, joint] = max(float(np.median(positive)) if positive.size else floor, floor)
    return result


def locked_artifact(record: dict[str, Any]) -> Path:
    if not record.get("exists", False):
        raise FileNotFoundError(f"Locked artifact absent: {record.get('role')}")
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if record.get("sha256") and sha256_file(path) != record["sha256"]:
        raise ValueError(f"Locked artifact hash mismatch: {path}")
    return path


def find_system(registry: dict[str, Any], split: str) -> dict[str, Any]:
    matches = [item for item in registry.get("systems", []) if item.get("split") == split]
    if len(matches) != 1:
        raise ValueError(f"Expected one locked generator system for {split}")
    return matches[0]


def make_generator(torch: Any, checkpoint: dict[str, Any], device: Any):
    nn = torch.nn

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout))

        def forward(self, value):
            return value + self.net(value)

    class Generator(nn.Module):
        def __init__(self, hidden: int, blocks: int, dropout: float) -> None:
            super().__init__()
            self.input = nn.Sequential(nn.Linear(40, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
            self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(blocks)])
            self.output = nn.Linear(hidden, 28)

        def forward(self, pose, rotation, translation):
            batch = pose.shape[0]
            identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand(batch, -1, -1)
            geometry = torch.cat([(rotation - identity).flatten(1), translation / 5.0], dim=1)
            features = torch.cat([pose.flatten(1), geometry], dim=1)
            return pose + self.output(self.blocks(self.input(features))).reshape(-1, 14, 2)

    config = checkpoint.get("config", {})
    model = Generator(int(config.get("hidden_width", 1024)), int(config.get("generator_blocks", 3)), float(config.get("generator_dropout", 0.10))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def load_generator_bundle(root: Path, split: str, device: Any, torch: Any) -> tuple[Any, Any, Any, Any, Any]:
    registry = load_json(root / "outputs/phase4e1_final_model_lock/final_model_registry.json")
    system = find_system(registry, split)
    manifest_path = locked_artifact(system["training_provenance"]["generator_manifest"])
    normalization_path = locked_artifact(system["components"]["normalization"])
    manifest = load_json(manifest_path)
    records = [item for item in manifest["models"] if item.get("name") == "full_train"]
    if len(records) != 1:
        raise ValueError(f"{split}: expected one full_train generator")
    checkpoint_path = Path(records[0]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("name") != "full_train":
        raise ValueError(f"{split}: generator checkpoint name mismatch")
    generator = make_generator(torch, checkpoint, device)
    normalization = torch.load(normalization_path, map_location="cpu", weights_only=False)
    input_mean = normalization["input_mean"].float().to(device)
    input_std = normalization["input_std"].float().to(device)
    candidates = load_json(root / "outputs/phase5_virtual_camera/phase5a_geometry_candidates/candidate_registry.json")
    selected = [item for item in candidates["candidates"] if item.get("candidate_id") == "rotation_yaw_m15"]
    if len(selected) != 1:
        raise ValueError("Locked rotation_yaw_m15 candidate missing")
    rotation = torch.tensor(selected[0]["rotation_observed_to_virtual"], dtype=torch.float32, device=device)
    translation = torch.tensor(selected[0]["translation_observed_to_virtual_m"], dtype=torch.float32, device=device)
    return generator, input_mean, input_std, rotation, translation


def official_intrinsics(cache: dict[str, np.ndarray]) -> np.ndarray:
    total = len(cache["subject"])
    intrinsic = np.zeros((total, 3, 3), dtype=np.float32)
    intrinsic[:, 0, 0] = cache["focal_length_px"]
    intrinsic[:, 1, 1] = cache["focal_length_px"]
    intrinsic[:, 0, 2] = cache["image_width"] / 2.0
    intrinsic[:, 1, 2] = cache["image_height"] / 2.0
    intrinsic[:, 2, 2] = 1.0
    return intrinsic


def synthesize_virtual(b1: Any, detector_px: np.ndarray, intrinsic: np.ndarray, bundle: tuple[Any, Any, Any, Any, Any], device: Any, batch_size: int, torch: Any) -> tuple[np.ndarray, np.ndarray]:
    generator, input_mean, input_std, rotation, translation = bundle
    total = detector_px.shape[0]
    observed_out = np.empty((total, 14, 2), dtype=np.float32)
    virtual_out = np.empty((total, 14, 2), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            pixels = torch.from_numpy(detector_px[start:end]).to(device)
            k = torch.from_numpy(intrinsic[start:end]).to(device)
            observed = b1.normalized_root_from_pixels_torch(pixels, k)[:, 1:]
            standardized = (observed - input_mean) / input_std
            count = end - start
            generated = generator(standardized, rotation.expand(count, -1, -1), translation.expand(count, -1))
            virtual = generated * input_std + input_mean
            observed_out[start:end] = observed.cpu().numpy()
            virtual_out[start:end] = virtual.cpu().numpy()
    if not np.isfinite(observed_out).all() or not np.isfinite(virtual_out).all():
        raise ValueError("Non-finite official observed/virtual features")
    return observed_out, virtual_out


def make_features(b1: Any, arm: str, observed: np.ndarray, virtual: np.ndarray, confidence: np.ndarray, split: str, detector: str, detector_index: int, calibration: dict[str, Any], residual_risk: np.ndarray, joint_scale: np.ndarray) -> np.ndarray:
    observed_flat = observed.reshape(len(observed), -1)
    virtual_flat = virtual.reshape(len(virtual), -1)
    if arm == "detector_aware_observed":
        result = observed_flat
    elif arm == "detector_conditioned_virtual_view_no_confidence":
        result = np.concatenate([observed_flat, virtual_flat], axis=1)
    else:
        calibrated = b1.calibration_probability(calibration, split, detector, confidence)[:, 1:]
        risk = b1.residual_risk_features(residual_risk, 0, detector_index, confidence)[:, 1:]
        scale = joint_scale[detector_index, 1:][None]
        quality = 1.0 / (1.0 + np.maximum(risk, 0.0) / np.maximum(scale, 1e-6))
        reliability = np.clip(calibrated * quality, 0.0, 1.0).astype(np.float32)
        reliability[np.all(confidence <= 0.0, axis=1)] = 0.0
        gate = reliability[..., None]
        if arm == "bounded_reliability_observed_modulation":
            result = np.concatenate([observed_flat, (observed * gate).reshape(len(observed), -1), reliability], axis=1)
        elif arm == "bounded_reliability_dual_modulation":
            result = np.concatenate([observed_flat, virtual_flat, (observed * gate).reshape(len(observed), -1), (virtual * gate).reshape(len(observed), -1), reliability], axis=1)
        else:
            raise ValueError(arm)
    if result.shape[1] != FEATURE_DIMS[arm] or not np.isfinite(result).all():
        raise ValueError(f"Feature integrity failed: {arm} {result.shape}")
    return result.astype(np.float32, copy=False)


def infer_checkpoint(b1: Any, checkpoint_path: Path, features: np.ndarray, device: Any, batch_size: int, torch: Any) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = b1.TinyPoseLifter(int(checkpoint["feature_dim"]), hidden=int(checkpoint["hidden_width"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            value = torch.from_numpy(features[start:start + batch_size]).to(device)
            parts.append((model(value) * 1000.0).cpu().numpy().astype(np.float32))
    result = np.concatenate(parts)
    if result.shape != (len(features), 14, 3) or not np.isfinite(result).all():
        raise ValueError(f"Invalid 3D prediction: {checkpoint_path}")
    return result


def procrustes_joint_errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = np.empty(prediction.shape[:2], dtype=np.float32)
    for index, (x, y) in enumerate(zip(prediction.astype(np.float64), target.astype(np.float64))):
        mean_x = x.mean(axis=0, keepdims=True)
        mean_y = y.mean(axis=0, keepdims=True)
        x0, y0 = x - mean_x, y - mean_y
        norm_x, norm_y = max(float(np.linalg.norm(x0)), 1e-12), max(float(np.linalg.norm(y0)), 1e-12)
        u, singular, vh = np.linalg.svd((x0 / norm_x).T @ (y0 / norm_y))
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            singular[-1] *= -1
            rotation = u @ vh
        scale = float(singular.sum()) * norm_y / norm_x
        aligned = scale * (x0 @ rotation) + mean_y
        result[index] = np.linalg.norm(aligned - y, axis=-1)
    return result


def metrics_from_joint_errors(raw: np.ndarray, pa: np.ndarray) -> dict[str, float]:
    curve = np.asarray([(raw < threshold).mean() for threshold in THRESHOLDS_MM])
    return {
        "MPJPE_mm": float(raw.mean()),
        "PA_MPJPE_mm": float(pa.mean()),
        "3DPCK_at_150mm_pct": float(100.0 * (raw < 150.0).mean()),
        "AUC_0_to_150mm_pct": float(100.0 * curve.mean()),
    }


def metric_delta(baseline: dict[str, float], challenger: dict[str, float]) -> dict[str, float]:
    return {
        "MPJPE_improvement_mm": baseline["MPJPE_mm"] - challenger["MPJPE_mm"],
        "PA_MPJPE_improvement_mm": baseline["PA_MPJPE_mm"] - challenger["PA_MPJPE_mm"],
        "3DPCK_improvement_pct_point": challenger["3DPCK_at_150mm_pct"] - baseline["3DPCK_at_150mm_pct"],
        "AUC_improvement_pct_point": challenger["AUC_0_to_150mm_pct"] - baseline["AUC_0_to_150mm_pct"],
        "MPJPE_relative_gain_pct": 100.0 * (baseline["MPJPE_mm"] - challenger["MPJPE_mm"]) / baseline["MPJPE_mm"],
    }


def macro_metrics(rows: list[dict[str, Any]], arm: str, detector: str | None = None) -> dict[str, float]:
    selected = [row for row in rows if row["arm"] == arm and (detector is None or row["detector"] == detector)]
    keys = ("MPJPE_mm", "PA_MPJPE_mm", "3DPCK_at_150mm_pct", "AUC_0_to_150mm_pct")
    return {key: float(np.mean([float(row[key]) for row in selected])) for key in keys}


def bootstrap(rows: list[dict[str, Any]], baseline_arm: str, challenger_arm: str, replicates: int, seed: int, interaction: bool = False) -> dict[str, Any]:
    subjects = sorted({row["subject"] for row in rows})
    lookup = {(row["subject"], row["split"], int(row["seed"]), row["detector"], row["arm"]): row for row in rows}
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        selected_subjects = rng.choice(subjects, len(subjects), replace=True)
        selected_splits = rng.choice(SPLITS, len(SPLITS), replace=True)
        selected_detectors = rng.choice(DETECTORS, len(DETECTORS), replace=True)
        values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for subject in selected_subjects:
            for split in selected_splits:
                for model_seed in rng.choice(SEEDS, len(SEEDS), replace=True):
                    for detector in selected_detectors:
                        for arm in ARMS:
                            values[arm].append(lookup[(str(subject), str(split), int(model_seed), str(detector), arm)])
        means = {arm: {key: float(np.mean([float(row[key]) for row in arm_rows])) for key in ("MPJPE_mm", "PA_MPJPE_mm", "3DPCK_at_150mm_pct", "AUC_0_to_150mm_pct")} for arm, arm_rows in values.items()}
        if interaction:
            conditional = metric_delta(means["bounded_reliability_observed_modulation"], means["bounded_reliability_dual_modulation"])
            ungated = metric_delta(means["detector_aware_observed"], means["detector_conditioned_virtual_view_no_confidence"])
            delta = {key: conditional[key] - ungated[key] for key in conditional}
        else:
            delta = metric_delta(means[baseline_arm], means[challenger_arm])
        for key, value in delta.items():
            distributions[key].append(float(value))
    return {
        "replicates": replicates,
        "seed": seed,
        "hierarchy": ["subject", "split", "seed_within_split", "detector"],
        "confidence_intervals": {
            key: {"mean": float(np.mean(values)), "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))], "probability_positive": float(np.mean(np.asarray(values) > 0))}
            for key, values in distributions.items()
        },
    }


def run_evaluation(root: Path, protocol: dict[str, Any], protocol_path: Path) -> int:
    import torch

    output_root = root / OUTPUT_RELATIVE
    summary_path = output_root / "phase9c_r1f_official_test_summary.json"
    lock_path = output_root / "phase9c_r1f_official_result_lock.json"
    if summary_path.is_file() and lock_path.is_file():
        result_lock = load_json(lock_path)
        if result_lock.get("summary_sha256") == sha256_file(summary_path) and result_lock.get("protocol_sha256") == sha256_file(protocol_path):
            print(f"[REUSE] Completed locked Official Test result: {summary_path}")
            return 0
        raise ValueError("Existing Official result lock does not match; refusing recomputation.")
    detector_report = load_json(output_root / "phase9c_r1f_detector_cache_report.json")
    cache_path = output_root / "phase9c_r1f_official_detector_cache.npz"
    if detector_report.get("status") != "PASS" or detector_report.get("cache_sha256") != sha256_file(cache_path):
        raise ValueError("Official detector cache is not a locked PASS.")
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {name: handle[name] for name in handle.files}
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 9C-R1F evaluation requires CUDA:0")
    torch.cuda.set_device(0)
    b1 = load_b1(root)
    target, activities = load_official_targets(root, protocol, cache)
    intrinsic = official_intrinsics(cache)
    total = len(target)
    batch_size = int(protocol["compute"]["lifter_batch_size"])
    raw_frame_errors = np.empty((4, 3, 3, 2, total), dtype=np.float32)
    pa_frame_errors = np.empty_like(raw_frame_errors)
    ensemble_sum = np.zeros((2, 2, total, 14, 3), dtype=np.float64)
    subject_rows: list[dict[str, Any]] = []
    checkpoint_hashes: list[dict[str, Any]] = []
    subject_values = cache["subject"]
    for split_index, split in enumerate(SPLITS):
        print(f"[SPLIT] {split}")
        b1.SPLIT = split
        artifacts = b1.find_required_artifacts(root)
        calibration = load_json(artifacts["phase9c_a_models"])
        with np.load(artifacts["phase9c_a_residual_bank"], allow_pickle=False) as handle:
            residual_bank = {name: handle[name] for name in handle.files}
        all_risk = b1.residual_risk_table(residual_bank)
        residual_risk = all_risk[split_index:split_index + 1].copy()
        joint_scale = risk_joint_scale(residual_risk)
        bundle = load_generator_bundle(root, split, device, torch)
        features_by_detector: dict[str, dict[str, np.ndarray]] = {}
        for detector_index, detector in enumerate(DETECTORS):
            observed, virtual = synthesize_virtual(b1, cache["prediction_c15_px"][detector_index], intrinsic, bundle, device, batch_size, torch)
            missing = ~cache["detected"][detector_index]
            observed[missing] = 0.0
            virtual[missing] = 0.0
            features_by_detector[detector] = {
                arm: make_features(b1, arm, observed, virtual, cache["confidence_c15"][detector_index], split, detector, detector_index, calibration, residual_risk, joint_scale)
                for arm in ARMS
            }
        for seed_index, seed in enumerate(SEEDS):
            for detector_index, detector in enumerate(DETECTORS):
                for arm_index, arm in enumerate(ARMS):
                    checkpoint_path, _ = canonical_checkpoint(root, split, seed, detector, arm)
                    checkpoint_hashes.append({"split": split, "seed": seed, "detector": detector, "arm": arm, "sha256": sha256_file(checkpoint_path)})
                    prediction = infer_checkpoint(b1, checkpoint_path, features_by_detector[detector][arm], device, batch_size, torch)
                    raw_joint = np.linalg.norm(prediction - target, axis=-1).astype(np.float32)
                    pa_joint = procrustes_joint_errors(prediction, target)
                    raw_frame_errors[arm_index, split_index, seed_index, detector_index] = raw_joint.mean(axis=1)
                    pa_frame_errors[arm_index, split_index, seed_index, detector_index] = pa_joint.mean(axis=1)
                    if arm in ("detector_aware_observed", "bounded_reliability_dual_modulation"):
                        selected_index = 0 if arm == "detector_aware_observed" else 1
                        ensemble_sum[selected_index, detector_index] += prediction
                    for subject in sorted(set(subject_values.tolist())):
                        mask = subject_values == subject
                        subject_rows.append({
                            "subject": str(subject), "split": split, "seed": seed, "detector": detector, "arm": arm,
                            "valid_frames": int(mask.sum()), "detector_success_rate": float(cache["detected"][detector_index, mask].mean()),
                            **metrics_from_joint_errors(raw_joint[mask], pa_joint[mask]),
                        })
                    del prediction, raw_joint, pa_joint
                    torch.cuda.empty_cache()
    ensemble = (ensemble_sum / 9.0).astype(np.float32)
    aggregate = {arm: macro_metrics(subject_rows, arm) for arm in ARMS}
    comparisons = {
        "primary_full_vs_observed": ("detector_aware_observed", "bounded_reliability_dual_modulation", False),
        "conditional_virtual_full_vs_reliability": ("bounded_reliability_observed_modulation", "bounded_reliability_dual_modulation", False),
        "reliability_only_vs_observed": ("detector_aware_observed", "bounded_reliability_observed_modulation", False),
        "ungated_virtual_vs_observed": ("detector_aware_observed", "detector_conditioned_virtual_view_no_confidence", False),
        "virtual_reliability_interaction": ("detector_aware_observed", "bounded_reliability_dual_modulation", True),
    }
    bootstrap_results = {}
    for name, (baseline, challenger, interaction) in comparisons.items():
        print(f"[BOOTSTRAP] {name}")
        bootstrap_results[name] = bootstrap(subject_rows, baseline, challenger, int(protocol["bootstrap"]["replicates"]), int(protocol["bootstrap"]["seed"]), interaction)
    primary_delta = metric_delta(aggregate["detector_aware_observed"], aggregate["bounded_reliability_dual_modulation"])
    primary_ci = bootstrap_results["primary_full_vs_observed"]["confidence_intervals"]
    detector_deltas = {
        detector: metric_delta(macro_metrics(subject_rows, "detector_aware_observed", detector), macro_metrics(subject_rows, "bounded_reliability_dual_modulation", detector))
        for detector in DETECTORS
    }
    coverage = detector_report["coverage"]
    coverage_ok = all(float(coverage[detector]["success_rate"]) >= float(protocol["decision_rules"]["minimum_detector_coverage_for_publication_claim"]) for detector in DETECTORS)
    ci_ok = primary_ci["MPJPE_improvement_mm"]["ci95"][0] > 0 and primary_ci["PA_MPJPE_improvement_mm"]["ci95"][0] > 0
    detectors_ok = all(value["MPJPE_improvement_mm"] > 0 and value["PA_MPJPE_improvement_mm"] > 0 for value in detector_deltas.values())
    point_ok = primary_delta["MPJPE_improvement_mm"] > 0 and primary_delta["PA_MPJPE_improvement_mm"] > 0
    if ci_ok and detectors_ok and coverage_ok:
        decision = "OFFICIAL_CONFIRMED_Q1_Q2_CORE_EVIDENCE" if primary_delta["MPJPE_relative_gain_pct"] >= float(protocol["decision_rules"]["practical_mpjpe_gain_percent"]) else "OFFICIAL_CONFIRMED_Q2_MODEST"
    elif point_ok:
        decision = "OFFICIAL_TREND_ONLY_WRITE_Q2_CAUTIOUS"
    else:
        decision = "NOT_EXTERNALLY_CONFIRMED_LOCK_NEGATIVE_RESULT"
    activity_rows = []
    for arm_index, arm in enumerate(ARMS):
        for activity in sorted(set(activities.tolist())):
            mask = activities == activity
            activity_rows.append({
                "activity_id": int(activity), "arm": arm, "frames": int(mask.sum()),
                "MPJPE_mm": float(raw_frame_errors[arm_index, ..., mask].mean()),
                "PA_MPJPE_mm": float(pa_frame_errors[arm_index, ..., mask].mean()),
            })
    write_csv(output_root / "phase9c_r1f_subject_system_metrics.csv", subject_rows)
    write_csv(output_root / "phase9c_r1f_activity_metrics_descriptive.csv", activity_rows)
    errors_path = output_root / "phase9c_r1f_per_frame_errors.npz"
    np.savez_compressed(errors_path, arm=np.asarray(ARMS), split=np.asarray(SPLITS), seed=np.asarray(SEEDS), detector=np.asarray(DETECTORS), subject=cache["subject"], frame_index_zero_based=cache["frame_index_zero_based"], activity=activities, mpjpe_mm=raw_frame_errors, pa_mpjpe_mm=pa_frame_errors)
    ensemble_path = output_root / "phase9c_r1f_ensemble_predictions.npz"
    np.savez_compressed(ensemble_path, arm=np.asarray(("detector_aware_observed", "bounded_reliability_dual_modulation")), detector=np.asarray(DETECTORS), subject=cache["subject"], frame_index_zero_based=cache["frame_index_zero_based"], prediction3d_camera_root_nonroot_mm=ensemble, target3d_camera_root_nonroot_mm=target)
    summary = {
        "status": "PASS",
        "format": "phase9c_r1f_official_test_summary_v1",
        "phase": PHASE,
        "scientific_decision": decision,
        "protocol_sha256": sha256_file(protocol_path),
        "official_partition": ["TS1", "TS2", "TS3", "TS4", "TS5", "TS6"],
        "input_2d": "real RGB detector predictions; no GT2D model input",
        "valid_frames": total,
        "joint_layout": "canonical non-root 14 joints",
        "missing_detection_policy": protocol["evaluation_scope"]["missing_detection_policy"],
        "detector_coverage": coverage,
        "aggregate_subject_split_seed_detector_macro": aggregate,
        "primary_improvement_positive_is_better": primary_delta,
        "detector_specific_primary_improvement": detector_deltas,
        "bootstrap": bootstrap_results,
        "decision_checks": {"primary_ci_positive_both": ci_ok, "both_detector_means_positive_both": detectors_ok, "detector_coverage_threshold_met": coverage_ok, "primary_point_positive_both": point_ok},
        "official_test_used_for_selection": False,
        "retraining_on_official_test": False,
        "project_level_legacy_phase7_official_evaluation_exists": True,
        "r1f_detector_input_method_previously_evaluated_on_official_TS1_to_TS6": False,
        "completed_utc": utc_now(),
    }
    write_json(summary_path, summary)
    result_lock = {
        "status": "LOCKED",
        "format": "phase9c_r1f_official_result_lock_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "detector_cache_sha256": sha256_file(cache_path),
        "summary_sha256": sha256_file(summary_path),
        "per_frame_errors_sha256": sha256_file(errors_path),
        "ensemble_predictions_sha256": sha256_file(ensemble_path),
        "checkpoint_matrix_sha256": hashlib.sha256(json.dumps(checkpoint_hashes, sort_keys=True).encode()).hexdigest(),
        "checkpoint_count": len(checkpoint_hashes),
        "scientific_decision": decision,
        "rerun_for_selection_forbidden": True,
    }
    write_json(lock_path, result_lock)
    print("=" * 104)
    print("PHASE 9C-R1F ONE-TIME OFFICIAL TEST")
    print("=" * 104)
    print("Status              : PASS")
    print(f"Scientific decision : {decision}")
    print(f"MPJPE gain          : {primary_delta['MPJPE_improvement_mm']:.3f} mm ({primary_delta['MPJPE_relative_gain_pct']:.3f}%)")
    print(f"PA-MPJPE gain       : {primary_delta['PA_MPJPE_improvement_mm']:.3f} mm")
    print("Official result     : LOCKED; do not tune or rerun for selection")
    print("=" * 104)
    return 0


def manifest(output_root: Path, names: list[str]) -> dict[str, Any]:
    rows = []
    for name in names:
        path = output_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"relative_path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"format": "phase9c_r1f_output_manifest_v1", "files": rows}


def run_qc_pack(root: Path, protocol: dict[str, Any], protocol_path: Path) -> int:
    output_root = root / OUTPUT_RELATIVE
    summary = load_json(output_root / "phase9c_r1f_official_test_summary.json")
    result_lock = load_json(output_root / "phase9c_r1f_official_result_lock.json")
    required = [
        "phase9c_r1f_preflight_report.json",
        "phase9c_r1f_checkpoint_inventory.csv",
        "phase9c_r1f_detector_cache_report.json",
        "phase9c_r1f_official_detector_cache.npz",
        "phase9c_r1f_official_test_summary.json",
        "phase9c_r1f_official_result_lock.json",
        "phase9c_r1f_subject_system_metrics.csv",
        "phase9c_r1f_activity_metrics_descriptive.csv",
        "phase9c_r1f_per_frame_errors.npz",
        "phase9c_r1f_ensemble_predictions.npz",
    ]
    errors = []
    if summary.get("status") != "PASS":
        errors.append("summary_status")
    if result_lock.get("protocol_sha256") != sha256_file(protocol_path):
        errors.append("protocol_hash")
    if result_lock.get("summary_sha256") != sha256_file(output_root / "phase9c_r1f_official_test_summary.json"):
        errors.append("summary_hash")
    if int(result_lock.get("checkpoint_count", 0)) != 72:
        errors.append("checkpoint_count")
    output_manifest = manifest(output_root, required)
    write_json(output_root / "phase9c_r1f_output_manifest.json", output_manifest)
    qc = {
        "status": "PASS" if not errors else "FAIL",
        "format": "phase9c_r1f_qc_report_v1",
        "phase": PHASE,
        "scientific_decision": summary.get("scientific_decision"),
        "protocol_sha256": sha256_file(protocol_path),
        "required_artifacts": len(required),
        "official_result_locked": True,
        "errors": errors,
    }
    write_json(output_root / "phase9c_r1f_qc_report.json", qc)
    include = required + ["phase9c_r1f_output_manifest.json", "phase9c_r1f_qc_report.json"]
    output_zip = output_root / "phase9c_r1f_results.zip"
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(protocol_path, "phase9c_r1f_locked_official_test_protocol.json")
        for name in include:
            archive.write(output_root / name, name)
    with zipfile.ZipFile(output_zip) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure: {bad}")
    print("=" * 104)
    print("PHASE 9C-R1F QC AND PACK")
    print("=" * 104)
    print(f"Status              : {qc['status']}")
    print(f"Scientific decision : {qc['scientific_decision']}")
    print(f"Output ZIP          : {output_zip}")
    print("=" * 104)
    return 0 if not errors else 2


def self_test() -> None:
    points = np.arange(34, dtype=np.float32).reshape(17, 2)
    confidence = np.linspace(0.1, 0.9, 17, dtype=np.float32)
    c15, scores = coco17_to_c15(points, confidence)
    assert c15.shape == (15, 2) and scores.shape == (15,)
    raw = np.ones((2, 14), dtype=np.float32) * 100
    pa = raw * 0.8
    metrics = metrics_from_joint_errors(raw, pa)
    assert metrics["MPJPE_mm"] == 100.0 and metrics["3DPCK_at_150mm_pct"] == 100.0
    prediction = np.zeros((2, 14, 3), dtype=np.float32)
    target = prediction.copy()
    target[:, :, 0] = np.arange(14)
    errors = procrustes_joint_errors(target, target)
    assert float(errors.max()) < 1e-3
    rows = []
    arm_offsets = {
        "detector_aware_observed": 0.0,
        "detector_conditioned_virtual_view_no_confidence": -0.2,
        "bounded_reliability_observed_modulation": 2.0,
        "bounded_reliability_dual_modulation": 3.0,
    }
    for subject in ("TS1", "TS2"):
        for split in SPLITS:
            for model_seed in SEEDS:
                for detector in DETECTORS:
                    for arm in ARMS:
                        gain = arm_offsets[arm]
                        rows.append({
                            "subject": subject, "split": split, "seed": model_seed,
                            "detector": detector, "arm": arm,
                            "MPJPE_mm": 120.0 - gain,
                            "PA_MPJPE_mm": 80.0 - 0.5 * gain,
                            "3DPCK_at_150mm_pct": 80.0 + gain,
                            "AUC_0_to_150mm_pct": 50.0 + gain,
                        })
    boot = bootstrap(rows, "detector_aware_observed", "bounded_reliability_dual_modulation", 100, 7)
    assert boot["confidence_intervals"]["MPJPE_improvement_mm"]["ci95"][0] > 0
    print("SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9C-R1F locked Official Test")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("preflight", "detect", "evaluate", "qc", "self-test"), required=True)
    args = parser.parse_args()
    if args.mode == "self-test":
        self_test()
        return
    root, protocol, protocol_path = protocol_and_root(args.project_root)
    actions = {"preflight": run_preflight, "detect": run_detectors, "evaluate": run_evaluation, "qc": run_qc_pack}
    raise SystemExit(actions[args.mode](root, protocol, protocol_path))


if __name__ == "__main__":
    main()
