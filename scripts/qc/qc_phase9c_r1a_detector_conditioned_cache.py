from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("split_a", "split_b", "split_c")
PARTITIONS = ("train", "val", "test")
DETECTORS = ("rtmpose_performance", "yolo11l_pose")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def scalar_text(value: np.ndarray) -> str:
    return str(value.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="QC and pack Phase 9C-R1A.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs/phase9c_r1a_detector_conditioned_virtual_cache"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "configs/phase9c_r1a_detector_conditioned_cache_protocol.json"
    protocol = load_json(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    b2_qc = load_json(root / "outputs/phase9c_b2_full_multisplit_multiseed/phase9c_b2_qc_report.json")
    checks: list[dict[str, Any]] = []
    check(
        checks,
        "protocol_lock",
        protocol.get("status") == "LOCKED"
        and protocol.get("format") == "phase9c_r1a_detector_conditioned_cache_protocol_lock_v1",
        f"status={protocol.get('status')}; format={protocol.get('format')}",
    )
    check(
        checks,
        "b2_redesign_gate",
        b2_qc.get("status") == "PASS"
        and b2_qc.get("scientific_decision") == "REDESIGN_CONFIDENCE_FUSION_BEFORE_OFFICIAL_TEST",
        f"status={b2_qc.get('status')}; decision={b2_qc.get('scientific_decision')}",
    )
    rows: list[dict[str, Any]] = []
    report_paths: list[Path] = []
    cache_hash_rows: list[dict[str, str]] = []
    official_ok = b2_qc.get("official_TS1_to_TS6_read") is False
    all_cache_ok = True
    all_provenance_ok = True
    all_differences_ok = True
    all_oof_ok = True
    total_samples = 0
    for split in SPLITS:
        split_root = output / split
        cache_path = split_root / "phase9c_r1a_detector_virtual_cache.npz"
        report_path = cache_path.with_suffix(".report.json")
        report_paths.append(report_path)
        if not cache_path.is_file() or not report_path.is_file():
            all_cache_ok = False
            rows.append({"split": split, "partition": "MISSING", "detector": "MISSING"})
            continue
        report = load_json(report_path)
        actual_hash = sha256_file(cache_path)
        cache_hash_rows.append({"split": split, "path": str(cache_path), "sha256": actual_hash})
        report_ok = (
            report.get("status") == "PASS"
            and report.get("protocol_sha256") == protocol_sha
            and report.get("cache_sha256") == actual_hash
            and report.get("ground_truth_3d_metrics_computed") is False
        )
        provenance_ok = (
            report.get("detector_input_source") == "phase9b_prediction_c15_px_real_detector"
            and report.get("legacy_clean_conditioned_cache_used_as_model_input") is False
            and report.get("legacy_clean_conditioned_cache_used_for_difference_audit_only") is True
        )
        official_ok &= report.get("official_TS1_to_TS6_read") is False and report.get("official_TS1_to_TS6_used") is False
        with np.load(cache_path, allow_pickle=False) as data:
            required = {
                "format", "split", "detector_id", "candidate_id", "source_input",
                "dataset_index", "detector_sample_index", "partition_code", "virtual_pose_camera_root",
            }
            fields_ok = set(data.files) == required
            virtual = data["virtual_pose_camera_root"]
            count = int(virtual.shape[1]) if virtual.ndim == 4 else -1
            structural = (
                fields_ok
                and scalar_text(data["format"]) == "phase9c_r1a_detector_conditioned_virtual_cache_v1"
                and scalar_text(data["split"]) == split
                and tuple(str(item) for item in data["detector_id"].tolist()) == DETECTORS
                and scalar_text(data["candidate_id"]) == "rotation_yaw_m15"
                and scalar_text(data["source_input"]) == "phase9b_prediction_c15_px_real_detector"
                and virtual.ndim == 4
                and virtual.shape[0] == 2
                and tuple(virtual.shape[2:]) == (15, 2)
                and data["dataset_index"].shape == (count,)
                and data["detector_sample_index"].shape == (count,)
                and data["partition_code"].shape == (count,)
                and set(np.unique(data["partition_code"]).tolist()) == {0, 1, 2}
                and np.isfinite(virtual).all()
                and np.array_equal(virtual[:, :, 0, :], np.zeros_like(virtual[:, :, 0, :]))
            )
        partition_stats = report.get("partition_stats", {})
        partition_counts_ok = set(partition_stats) == set(PARTITIONS)
        split_difference_ok = True
        split_oof_ok = True
        summed = 0
        for partition in PARTITIONS:
            stats = partition_stats.get(partition, {})
            samples = int(stats.get("samples", -1))
            summed += max(samples, 0)
            usage = stats.get("generator_model_usage", {})
            if partition == "train":
                oof = bool(usage) and "full_train" not in usage and all(name.startswith("holdout_") for name in usage)
            else:
                oof = set(usage) == {"full_train"}
            split_oof_ok &= oof
            legacy_diff = stats.get("mean_abs_difference_vs_legacy_clean_conditioned", {})
            cross_diff = float(stats.get("mean_abs_cross_detector_virtual_difference", math.nan))
            detector_diffs_ok = all(float(legacy_diff.get(detector, 0.0)) > 1e-6 for detector in DETECTORS)
            difference_ok = detector_diffs_ok and math.isfinite(cross_diff) and cross_diff > 1e-6
            split_difference_ok &= difference_ok
            for detector in DETECTORS:
                rows.append({
                    "split": split,
                    "partition": partition,
                    "detector": detector,
                    "samples": samples,
                    "mean_abs_difference_vs_legacy_clean_conditioned": legacy_diff.get(detector),
                    "mean_abs_virtual_delta_from_detector_observed": stats.get("mean_abs_virtual_delta_from_detector_observed", {}).get(detector),
                    "mean_abs_cross_detector_virtual_difference": cross_diff,
                    "generator_policy_ok": oof,
                })
        total_samples += summed
        partition_counts_ok &= summed == count == int(report.get("total_samples", -1))
        all_cache_ok &= report_ok and structural and partition_counts_ok
        all_provenance_ok &= provenance_ok
        all_differences_ok &= split_difference_ok
        all_oof_ok &= split_oof_ok
    check(checks, "split_cache_integrity", all_cache_ok, f"splits={len(cache_hash_rows)}/3; total_samples={total_samples}")
    check(checks, "real_detector_input_provenance", all_provenance_ok, "source=phase9b prediction_c15_px; clean legacy audit-only")
    check(checks, "detector_conditioned_not_legacy", all_differences_ok, "all detector/partition differences > 1e-6")
    check(checks, "generator_oof_policy", all_oof_ok, "train=holdout_subject; val/test=full_train")
    check(checks, "no_3d_metric_selection", all_cache_ok, "cache build and provenance audit only; zero 3D metric rows")
    check(checks, "official_test_isolation", official_ok, "Official TS1-TS6 unread")
    summary_path = output / "phase9c_r1a_cache_summary.csv"
    fieldnames = [
        "split", "partition", "detector", "samples",
        "mean_abs_difference_vs_legacy_clean_conditioned",
        "mean_abs_virtual_delta_from_detector_observed",
        "mean_abs_cross_detector_virtual_difference", "generator_policy_ok",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    hash_path = output / "phase9c_r1a_cache_hashes.csv"
    with hash_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "path", "sha256"])
        writer.writeheader()
        writer.writerows(cache_hash_rows)
    errors = [item["name"] for item in checks if not item["passed"]]
    status = "PASS" if not errors else "FAIL"
    decision = (
        "READY_FOR_PHASE9C_R1B_DETECTOR_CONDITIONED_SMOKE"
        if status == "PASS"
        else "REPAIR_PHASE9C_R1A_BEFORE_ANY_FURTHER_TRAINING"
    )
    qc = {
        "status": status,
        "format": "phase9c_r1a_qc_report_v1",
        "phase": "9C-R1A",
        "scientific_decision": decision,
        "checks": checks,
        "errors": errors,
        "cache_artifact_hashes": cache_hash_rows,
        "protocol_sha256": protocol_sha,
        "ground_truth_3d_metric_rows": 0,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    qc_path = output / "phase9c_r1a_qc_report.json"
    write_json(qc_path, qc)
    result_zip = output / "phase9c_r1a_results.zip"
    members = [protocol_path, qc_path, summary_path, hash_path] + [path for path in report_paths if path.is_file()]
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            path = Path(path)
            if path == protocol_path:
                arcname = path.name
            elif path.parent == output:
                arcname = path.name
            else:
                arcname = f"split_reports/{path.parent.name}/{path.name}"
            archive.write(path, arcname)
    print("=" * 108)
    print("PHASE 9C-R1A - DETECTOR-CONDITIONED VIRTUAL CACHE QC")
    print("=" * 108)
    print(f"Status                : {status}")
    print(f"Scientific decision   : {decision}")
    print(f"Cache splits          : {len(cache_hash_rows)} / 3")
    print("3D metric rows        : 0")
    print("Official TS1-TS6 read: NO")
    print(f"Output ZIP            : {result_zip}")
    if errors:
        print(f"Errors                : {errors}")
    print("=" * 108)
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
