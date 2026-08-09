from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any


EXPECTED_ARMS = {
    "clean_observed_legacy",
    "detector_aware_observed",
    "virtual_view_no_confidence",
    "calibrated_confidence_virtual_view",
}
EXPECTED_DETECTORS = {"rtmpose_performance", "yolo11l_pose"}


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


def finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser(description="QC and pack Phase 9C-B1 smoke results.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase9c_b1_four_arm_smoke"))
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    build_path = output / "phase9c_b1_smoke_build_report.json"
    metrics_path = output / "phase9c_b1_smoke_metrics.csv"
    protocol_path = root / "configs/phase9c_b1_smoke_protocol.json"
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    build = load_json(build_path)
    protocol = load_json(protocol_path)
    add_check(
        checks,
        "protocol_lock",
        protocol.get("status") == "LOCKED"
        and protocol.get("format") == "phase9c_b1_smoke_protocol_lock_v3",
        f"status={protocol.get('status')}; format={protocol.get('format')}",
    )
    add_check(
        checks,
        "build_pass",
        build.get("status") == "PASS" and build.get("format") == "phase9c_b1_smoke_build_report_v3",
        f"status={build.get('status')}; format={build.get('format')}",
    )
    add_check(
        checks,
        "official_test_isolation",
        build.get("official_TS1_to_TS6_read") is False and build.get("official_TS1_to_TS6_used") is False,
        "Official TS1-TS6 unread and unused",
    )
    workers = build.get("workers", {})
    add_check(
        checks,
        "worker_lock_8",
        int(workers.get("requested", -1)) == 8
        and int(workers.get("effective", -1)) == 8
        and int(workers.get("peak_concurrent", -1)) == 8,
        str(workers),
    )
    loading = build.get("data_loading", {})
    add_check(
        checks,
        "windows_spawn_memory_lock",
        loading.get("materialized_before_worker_spawn") is True
        and int(loading.get("materialization_workers", -1)) == 0
        and loading.get("detector_cache_in_dataset_workers") is False
        and loading.get("training_shard_io_in_dataset_workers") is False
        and workers.get("persistent_workers") is False
        and int(workers.get("prefetch_factor", -1)) == 1,
        f"workers={workers}; data_loading={loading}",
    )
    missingness_policy = protocol.get("detector_missingness_policy", {})
    add_check(
        checks,
        "paired_common_detection_policy",
        missingness_policy.get("primary_metric_cohort") == "paired_common_detection"
        and missingness_policy.get("same_sample_cohort_across_arms_and_detectors") is True
        and missingness_policy.get("exclude_only_if_any_detector_failed_or_nonfinite") is True
        and missingness_policy.get("report_detection_coverage_separately") is True
        and missingness_policy.get("ground_truth_imputation_for_detector_failures") is False
        and loading.get("sample_cohort") == "paired_common_detection"
        and loading.get("same_sample_cohort_across_arms_and_detectors") is True,
        f"protocol={missingness_policy}; data_loading={loading.get('sample_cohort')}",
    )
    coverage = build.get("detector_coverage_audit", {})
    common = coverage.get("common_cohort", {})
    detector_coverage = coverage.get("detectors", {})
    joined = int(common.get("joined_samples", 0))
    eligible = int(common.get("eligible_samples", 0))
    excluded = int(common.get("excluded_any_detector_failure", -1))
    coverage_accounting = (
        coverage.get("policy") == "paired_common_detection"
        and coverage.get("same_sample_cohort_across_arms_and_detectors") is True
        and coverage.get("ground_truth_imputation_for_detector_failures") is False
        and set(detector_coverage) == EXPECTED_DETECTORS
        and joined > 0
        and eligible > 0
        and eligible + excluded == joined
    )
    for detector in EXPECTED_DETECTORS:
        stats = detector_coverage.get(detector, {})
        coverage_accounting = coverage_accounting and (
            int(stats.get("joined_samples", -1)) == joined
            and int(stats.get("detected_samples", -1)) + int(stats.get("missed_detections", -1)) == joined
            and int(stats.get("invalid_detected_samples", -1)) == 0
        )
    add_check(
        checks,
        "detector_coverage_accounting",
        coverage_accounting,
        f"joined={joined}; eligible={eligible}; excluded={excluded}; detectors={detector_coverage}",
    )
    partition_coverage = coverage.get("partitions", {})
    selection = build.get("dataset_selection", {})
    cohort_accounting = set(partition_coverage) == {"train", "val", "test"} and set(selection) == {"train", "val", "test"}
    for partition in ("train", "val", "test"):
        part = partition_coverage.get(partition, {})
        selected = selection.get(partition, {})
        part_joined = int(part.get("joined_samples", -1))
        part_eligible = int(part.get("paired_common_detected_samples", -1))
        part_excluded = int(part.get("excluded_any_detector_failure", -1))
        final_count = int(selected.get("selected_after_smoke_limit", -1))
        cohort_accounting = cohort_accounting and (
            part_joined >= 0
            and part_eligible > 0
            and part_eligible + part_excluded == part_joined
            and int(selected.get("joined_before_eligibility", -1)) == part_joined
            and int(selected.get("excluded_by_common_detection", -1)) == part_excluded
            and int(selected.get("eligible_before_smoke_limit", -1)) == part_eligible
            and 0 < final_count <= part_eligible
        )
    add_check(
        checks,
        "paired_cohort_partition_accounting",
        cohort_accounting,
        f"partitions={partition_coverage}; selection={selection}",
    )
    add_check(
        checks,
        "source_phase9c_a_gate",
        build.get("source_gates", {}).get("phase9c_a") == "READY_FOR_PHASE9C_B_FOUR_ARM_MULTISPLIT_TRAINING",
        str(build.get("source_gates", {}).get("phase9c_a")),
    )
    add_check(
        checks,
        "source_phase9c_b0_gate",
        build.get("source_gates", {}).get("phase9c_b0") == "READY_FOR_PHASE9C_B1_FOUR_ARM_SMOKE",
        str(build.get("source_gates", {}).get("phase9c_b0")),
    )
    observed_arms = {str(item.get("arm")) for item in build.get("run_reports", [])}
    add_check(checks, "four_arms_present", observed_arms == EXPECTED_ARMS, str(sorted(observed_arms)))
    run_count = int(build.get("run_count", 0))
    add_check(checks, "run_count", run_count == 7, f"runs={run_count}")
    detector_runs = {
        (str(item.get("arm")), str(item.get("train_detector")))
        for item in build.get("run_reports", [])
        if str(item.get("arm")) != "clean_observed_legacy"
    }
    required_detector_runs = {
        (arm, detector)
        for arm in EXPECTED_ARMS - {"clean_observed_legacy"}
        for detector in EXPECTED_DETECTORS
    }
    add_check(
        checks,
        "detector_train_matrix",
        detector_runs == required_detector_runs,
        f"runs={len(detector_runs)}",
    )
    checkpoint_paths: list[Path] = []
    nonfinite_loss_runs: list[str] = []
    for item in build.get("run_reports", []):
        checkpoint = Path(str(item.get("checkpoint", "")))
        checkpoint_paths.append(checkpoint)
        if not finite_number(item.get("train_loss_first")) or not finite_number(item.get("train_loss_last")):
            nonfinite_loss_runs.append(f"{item.get('arm')}/{item.get('train_detector')}")
    add_check(checks, "checkpoints_exist", all(path.is_file() for path in checkpoint_paths), f"count={len(checkpoint_paths)}")
    add_check(
        checks,
        "finite_training_losses",
        not nonfinite_loss_runs,
        f"nonfinite_runs={nonfinite_loss_runs}",
    )

    metric_rows: list[dict[str, str]] = []
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        metric_rows = list(reader)
    nonfinite_metric_rows = [
        f"{row.get('arm')}/{row.get('train_detector')}/{row.get('eval_detector')}/{row.get('partition')}"
        for row in metric_rows
        if not finite_number(row.get("mpjpe_mm"))
        or not finite_number(row.get("pa_mpjpe_mm"))
        or int(float(row.get("samples", "0"))) <= 0
    ]
    add_check(
        checks,
        "metrics_finite",
        not nonfinite_metric_rows,
        f"rows={len(metric_rows)}; nonfinite_rows={nonfinite_metric_rows}",
    )
    eval_pairs = {
        (row["arm"], row["train_detector"], row["eval_detector"], row["partition"])
        for row in metric_rows
    }
    cross_rows = [
        row for row in metric_rows
        if row["train_detector"] in EXPECTED_DETECTORS
        and row["eval_detector"] in EXPECTED_DETECTORS
        and row["train_detector"] != row["eval_detector"]
    ]
    matched_rows = [
        row for row in metric_rows
        if row["train_detector"] in EXPECTED_DETECTORS
        and row["eval_detector"] == row["train_detector"]
    ]
    add_check(checks, "matched_detector_eval_present", len(matched_rows) >= 12, f"rows={len(matched_rows)}")
    add_check(checks, "cross_detector_eval_present", len(cross_rows) >= 12, f"rows={len(cross_rows)}")
    add_check(checks, "val_and_test_present", {"val", "test"} <= {row["partition"] for row in metric_rows}, str(sorted({row["partition"] for row in metric_rows})))

    artifact_hashes = {
        "phase9c_b1_smoke_build_report.json": sha256_file(build_path),
        "phase9c_b1_smoke_metrics.csv": sha256_file(metrics_path),
    }
    for checkpoint in checkpoint_paths:
        artifact_hashes[str(checkpoint.relative_to(output))] = sha256_file(checkpoint)

    passed = all(check["passed"] for check in checks)
    if not passed:
        errors = [f"{check['name']}: {check['detail']}" for check in checks if not check["passed"]]
    qc = {
        "status": "PASS" if passed else "FAIL",
        "format": "phase9c_b1_smoke_qc_report_v3",
        "phase": "9C-B1",
        "scientific_decision": (
            "READY_FOR_PHASE9C_B2_FULL_MULTISPLIT_MULTISEED_TRAINING"
            if passed
            else "REPAIR_PHASE9C_B1_BEFORE_FULL_TRAINING"
        ),
        "scientific_role": "training_smoke_not_final_scientific_comparison",
        "checks": checks,
        "errors": errors,
        "metric_rows": len(metric_rows),
        "artifact_hashes": artifact_hashes,
        "official_TS1_to_TS6_read": False,
        "training_performed": True,
        "phase9c_b2_num_workers": 8,
    }
    qc_path = output / "phase9c_b1_smoke_qc_report.json"
    write_json(qc_path, qc)

    zip_path = output / "phase9c_b1_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(build_path, build_path.name)
        archive.write(metrics_path, metrics_path.name)
        archive.write(qc_path, qc_path.name)
        archive.write(protocol_path, protocol_path.name)
        for checkpoint in checkpoint_paths:
            archive.write(checkpoint, str(Path("checkpoints") / checkpoint.parent.parent.name / checkpoint.parent.name / checkpoint.name))
    print("=" * 112)
    print("PHASE 9C-B1 - QC AND PACK")
    print("=" * 112)
    print(f"Status                : {qc['status']}")
    print(f"Scientific decision   : {qc['scientific_decision']}")
    print(f"Metric rows           : {len(metric_rows)}")
    print(f"Output ZIP            : {zip_path}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    print("=" * 112)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
