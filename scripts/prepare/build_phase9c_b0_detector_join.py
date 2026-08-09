from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


SPLITS = ("split_a", "split_b", "split_c")
PARTITIONS = ("train", "val", "test")
EXPECTED_DETECTORS = ("rtmpose_performance", "yolo11l_pose")


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
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_partition(value: str) -> str:
    value = value.lower()
    return {"validation": "val", "valid": "val"}.get(value, value)


def find_phase9ca_artifacts(project_root: Path) -> tuple[dict[str, bytes], str]:
    root = project_root / "outputs/phase9c_a_calibrated_error_bank"
    required = (
        "phase9c_a_protocol_lock.json",
        "phase9c_a_build_report.json",
        "phase9c_a_qc_report.json",
    )
    direct = {name: root / name for name in required}
    if all(path.is_file() for path in direct.values()):
        return ({name: path.read_bytes() for name, path in direct.items()}, str(root))

    zip_path = root / "phase9c_a_results.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(
            "Missing Phase 9C-A artifacts and phase9c_a_results.zip under "
            f"{root}"
        )
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        missing = [name for name in required if name not in names]
        if missing:
            raise ValueError(f"Phase 9C-A ZIP missing: {missing}")
        payload = {name: archive.read(name) for name in required}
    return payload, f"{zip_path}!/"


def find_phase9b_cache(project_root: Path, expected_hash: str) -> tuple[dict[str, np.ndarray], str, str]:
    root = project_root / "outputs/phase9b_rgb_detector_benchmark"
    candidates = (
        root / "full/phase9b_detector_cache.npz",
        root / "phase9b_detector_cache.npz",
    )
    mismatches: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        digest = sha256_file(path)
        if digest != expected_hash:
            mismatches.append(f"{path}: {digest}")
            continue
        with np.load(path, allow_pickle=False) as source:
            return ({key: source[key] for key in source.files}, str(path), digest)

    zip_path = root / "phase9b_rgb_detector_benchmark_results.zip"
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as archive:
            members = sorted(
                (name for name in archive.namelist() if name.endswith("phase9b_detector_cache.npz")),
                key=lambda name: (0 if "/full/" in f"/{name}" else 1, len(name)),
            )
            for member in members:
                data = archive.read(member)
                digest = sha256_bytes(data)
                if digest != expected_hash:
                    mismatches.append(f"{zip_path}!{member}: {digest}")
                    continue
                with np.load(io.BytesIO(data), allow_pickle=False) as source:
                    return (
                        {key: source[key] for key in source.files},
                        f"{zip_path}!{member}",
                        digest,
                    )
    detail = "\n".join(mismatches[:8])
    raise FileNotFoundError(
        "No Phase 9B detector cache matches the hash locked by Phase 9C-A.\n" + detail
    )


def detector_identity(cache: dict[str, np.ndarray], index: int) -> tuple[str, str, int, int]:
    return (
        str(cache["subject"][index]),
        str(cache["sequence"][index]),
        int(cache["camera_id"][index]),
        int(cache["frame_idx"][index]),
    )


def normalized_root_from_pixels(pixels: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    ones = np.ones((*pixels.shape[:-1], 1), dtype=np.float64)
    homogeneous = np.concatenate([pixels.astype(np.float64), ones], axis=-1)
    inverse = np.linalg.inv(intrinsic.astype(np.float64))
    normalized_h = np.einsum("nij,nkj->nki", inverse, homogeneous)
    normalized = normalized_h[..., :2] / normalized_h[..., 2:3]
    normalized -= normalized[:, 0:1, :]
    return normalized.astype(np.float32)


def validate_manifest(path: Path, split_name: str) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("format") != "phase5r1a_multigeometry_synthesized_manifest_v1":
        raise ValueError(f"Wrong cache manifest format: {path}")
    if manifest.get("status") != "PASS" or bool(manifest.get("partial", False)):
        raise ValueError(f"Cache manifest is not full PASS: {path}")
    if manifest.get("cross_subject_split") != split_name:
        raise ValueError(f"Cache manifest split mismatch: {path}")
    if int(manifest.get("candidate_count", -1)) != 8:
        raise ValueError(f"Expected eight virtual-view candidates: {path}")
    return manifest


def scan_split(
    manifest: dict[str, Any],
    detector_lookup: dict[tuple[str, str, int, int], int],
    detector_cache: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    joined_detector: list[np.ndarray] = []
    joined_dataset: list[np.ndarray] = []
    joined_partition: list[np.ndarray] = []
    coordinate_errors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    partition_cursor = {name: 0 for name in PARTITIONS}
    dataset_key_count: Counter[tuple[str, str, int, int]] = Counter()

    for record in manifest["shards"]:
        partition = canonical_partition(str(record["split"]))
        if partition not in PARTITIONS:
            raise ValueError(f"Unexpected partition {partition!r}")
        original_path = Path(record["original_path"])
        if not original_path.is_file():
            raise FileNotFoundError(original_path)
        payload = torch.load(original_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "phase4b_multisubject_paired_shard_v1":
            raise ValueError(f"Wrong original shard format: {original_path}")
        tensors = payload["tensors"]
        required = (
            "target_pose2d_camera_root",
            "target_pose3d_camera_root_mm",
            "target_intrinsic",
            "target_camera_id",
            "frame_idx",
        )
        missing = [key for key in required if key not in tensors]
        if missing:
            raise ValueError(f"{original_path} missing tensors: {missing}")
        count = int(record["sample_count"])
        if count != int(payload["sample_count"]):
            raise ValueError(f"Sample count mismatch: {original_path}")
        subject = str(record["subject"])
        sequence = str(record["sequence"])
        cameras = tensors["target_camera_id"][:count].cpu().numpy().astype(np.int16)
        frames = tensors["frame_idx"][:count].cpu().numpy().astype(np.int32)

        local_matches: list[int] = []
        detector_matches: list[int] = []
        for local_index, (camera, frame) in enumerate(zip(cameras, frames)):
            key = (subject, sequence, int(camera), int(frame))
            dataset_key_count[key] += 1
            detector_index = detector_lookup.get(key)
            if detector_index is not None:
                local_matches.append(local_index)
                detector_matches.append(detector_index)

        local_array = np.asarray(local_matches, dtype=np.int64)
        detector_array = np.asarray(detector_matches, dtype=np.int64)
        global_array = local_array + partition_cursor[partition]
        partition_array = np.full(local_array.shape, PARTITIONS.index(partition), dtype=np.int8)
        joined_detector.append(detector_array)
        joined_dataset.append(global_array)
        joined_partition.append(partition_array)

        if local_array.size:
            local_tensor = torch.from_numpy(local_array)
            gt_pixels = detector_cache["gt2d_px"][detector_array]
            intrinsic = tensors["target_intrinsic"][local_tensor].cpu().numpy()
            expected = tensors["target_pose2d_camera_root"][local_tensor].cpu().numpy()
            reconstructed = normalized_root_from_pixels(gt_pixels, intrinsic)
            coordinate_errors.append(
                np.linalg.vector_norm(reconstructed - expected, axis=-1).reshape(-1)
            )

        rows.append(
            {
                "partition": partition,
                "subject": subject,
                "sequence": sequence,
                "samples": count,
                "joined": int(local_array.size),
                "join_fraction": float(local_array.size / max(count, 1)),
                "original_path": str(original_path),
            }
        )
        partition_cursor[partition] += count

    detector_indices = np.concatenate(joined_detector) if joined_detector else np.empty(0, np.int64)
    dataset_indices = np.concatenate(joined_dataset) if joined_dataset else np.empty(0, np.int64)
    partition_codes = np.concatenate(joined_partition) if joined_partition else np.empty(0, np.int8)
    errors = np.concatenate(coordinate_errors) if coordinate_errors else np.asarray([np.inf])
    duplicate_dataset_keys = sum(value - 1 for value in dataset_key_count.values() if value > 1)
    summary = {
        "dataset_samples": {name: int(partition_cursor[name]) for name in PARTITIONS},
        "joined_samples": {
            name: int(np.sum(partition_codes == PARTITIONS.index(name))) for name in PARTITIONS
        },
        "joined_total": int(detector_indices.size),
        "unique_joined_detector_samples": int(np.unique(detector_indices).size),
        "duplicate_dataset_identity_count": int(duplicate_dataset_keys),
        "coordinate_alignment": {
            "joint_observations": int(errors.size),
            "mean": float(np.mean(errors)),
            "p95": float(np.quantile(errors, 0.95)),
            "maximum": float(np.max(errors)),
        },
    }
    arrays = {
        "detector_sample_index": detector_indices,
        "dataset_index": dataset_indices,
        "partition_code": partition_codes,
    }
    return arrays, summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 9C-B0 detector/training join lock.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol-lock", type=Path, default=Path("configs/phase9c_b_protocol_lock.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase9c_b_detector_join_preflight"))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    lock_path = args.protocol_lock if args.protocol_lock.is_absolute() else project_root / args.protocol_lock
    output = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    lock = load_json(lock_path)
    if lock.get("status") != "LOCKED" or lock.get("format") != "phase9c_b_protocol_lock_v1":
        raise ValueError("Phase 9C-B protocol lock is invalid.")
    if int(lock["training_budget"]["num_workers"]) != 8:
        raise ValueError("Phase 9C-B worker lock must be 8.")

    phase9ca_bytes, phase9ca_location = find_phase9ca_artifacts(project_root)
    phase9ca = {
        name: json.loads(data.decode("utf-8")) for name, data in phase9ca_bytes.items()
    }
    qc = phase9ca["phase9c_a_qc_report.json"]
    build = phase9ca["phase9c_a_build_report.json"]
    if qc.get("status") != "PASS" or qc.get("scientific_decision") != lock["source_gate"]:
        raise ValueError("Phase 9C-A gate is not PASS/READY.")
    if build.get("official_TS1_to_TS6_read") is not False:
        raise ValueError("Phase 9C-A reports Official TS1-TS6 access.")
    expected_cache_hash = str(build["source_hashes"]["phase9b_detector_cache.npz"])
    detector_cache, detector_location, detector_hash = find_phase9b_cache(
        project_root, expected_cache_hash
    )

    required_cache = (
        "detector_id", "subject", "sequence", "camera_id", "frame_idx",
        "gt2d_px", "prediction_c15_px", "confidence_c15", "detected",
    )
    missing = [key for key in required_cache if key not in detector_cache]
    if missing:
        raise ValueError(f"Phase 9B detector cache missing arrays: {missing}")
    detector_ids = tuple(str(value) for value in detector_cache["detector_id"].tolist())
    if detector_ids != EXPECTED_DETECTORS:
        raise ValueError(f"Detector order differs: {detector_ids}")
    detector_count = int(detector_cache["subject"].shape[0])
    if detector_cache["prediction_c15_px"].shape != (2, detector_count, 15, 2):
        raise ValueError("Detector prediction tensor shape mismatch.")

    detector_lookup: dict[tuple[str, str, int, int], int] = {}
    duplicate_detector_keys = 0
    for index in range(detector_count):
        key = detector_identity(detector_cache, index)
        if key in detector_lookup:
            duplicate_detector_keys += 1
        else:
            detector_lookup[key] = index

    output_arrays: dict[str, np.ndarray] = {
        "split_name": np.asarray(SPLITS),
        "partition_name": np.asarray(PARTITIONS),
        "detector_id": np.asarray(detector_ids),
    }
    split_summaries: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    for split_name in SPLITS:
        manifest_path = project_root / (
            "data/processed/caches/phase5r1a_multigeometry_oof_25fps_v1/"
            f"{split_name}/manifest.json"
        )
        manifest = validate_manifest(manifest_path, split_name)
        manifest_hashes[split_name] = sha256_file(manifest_path)
        arrays, summary, rows = scan_split(manifest, detector_lookup, detector_cache)
        for key, value in arrays.items():
            output_arrays[f"{split_name}_{key}"] = value
        split_summaries[split_name] = summary
        for row in rows:
            csv_rows.append({"split": split_name, **row})

    join_path = output / "phase9c_b0_detector_join_cache.npz"
    np.savez_compressed(join_path, **output_arrays)
    csv_path = output / "phase9c_b0_join_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    detected_rates = {
        detector_ids[index]: float(np.mean(detector_cache["detected"][index]))
        for index in range(len(detector_ids))
    }
    report = {
        "status": "PASS",
        "format": "phase9c_b0_join_build_report_v1",
        "phase": "9C-B0",
        "protocol_lock": str(lock_path),
        "phase9c_a_location": phase9ca_location,
        "phase9c_a_qc_status": qc.get("status"),
        "phase9c_a_scientific_decision": qc.get("scientific_decision"),
        "phase9b_detector_cache_location": detector_location,
        "phase9b_detector_cache_sha256": detector_hash,
        "phase9b_detector_cache_expected_sha256": expected_cache_hash,
        "phase9b_detector_samples": detector_count,
        "duplicate_detector_identity_count": duplicate_detector_keys,
        "detectors": list(detector_ids),
        "detected_rates": detected_rates,
        "manifest_hashes": manifest_hashes,
        "splits": split_summaries,
        "workers": {
            "this_phase": "not_applicable_no_DataLoader",
            "locked_for_phase9c_b1_training": 8,
        },
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
        "ground_truth_3d_used_for_join": False,
        "artifacts": {
            "join_cache": str(join_path),
            "join_summary_csv": str(csv_path),
        },
    }
    report_path = output / "phase9c_b0_join_build_report.json"
    write_json(report_path, report)
    print("=" * 112)
    print("PHASE 9C-B0 - DETECTOR TO TRAINING JOIN BUILD")
    print("=" * 112)
    print(f"Phase 9C-A gate       : {qc.get('scientific_decision')}")
    print(f"Detector samples      : {detector_count}")
    for split_name in SPLITS:
        item = split_summaries[split_name]
        print(
            f"{split_name:<22}: joined={item['joined_total']} "
            f"unique={item['unique_joined_detector_samples']} "
            f"p95={item['coordinate_alignment']['p95']:.9f}"
        )
    print("Workers next training : 8")
    print("Official TS1-TS6 read : NO")
    print(f"Output                : {output}")
    print("Status                : PASS")
    print("=" * 112)


if __name__ == "__main__":
    main()
