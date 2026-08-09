from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any


ARMS = {
    "clean_observed_legacy",
    "detector_aware_observed",
    "detector_conditioned_virtual_view_no_confidence",
    "detector_conditioned_calibrated_confidence_virtual_view",
}
DETECTORS = {"rtmpose_performance", "yolo11l_pose"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="QC and pack Phase 9C-R1B.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs/phase9c_r1b_detector_conditioned_smoke"
    build_path = output / "phase9c_r1b_smoke_build_report.json"
    metrics_path = output / "phase9c_r1b_smoke_metrics.csv"
    protocol_path = root / "configs/phase9c_r1b_detector_conditioned_smoke_protocol.json"
    build = load_json(build_path)
    protocol = load_json(protocol_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("protocol_lock", protocol.get("status") == "LOCKED" and protocol.get("format") == "phase9c_r1b_detector_conditioned_smoke_protocol_lock_v1", str(protocol.get("format")))
    check("build_pass", build.get("status") == "PASS" and build.get("format") == "phase9c_r1b_detector_conditioned_smoke_build_report_v1", str(build.get("format")))
    check("r1a_gate", build.get("source_gate") == protocol.get("source_gate"), str(build.get("source_gate")))
    check("r1a_protocol_hash", build.get("source_r1a_protocol_sha256") == protocol.get("source_r1a_protocol_sha256"), str(build.get("source_r1a_protocol_sha256")))
    r1a_cache = Path(str(build.get("source_r1a_cache", "")))
    check("r1a_cache_hash", r1a_cache.is_file() and sha256_file(r1a_cache) == build.get("source_r1a_cache_sha256"), str(build.get("source_r1a_cache_sha256")))
    check("real_detector_virtual_provenance", build.get("virtual_view_input") == "phase9b_prediction_c15_px_real_detector" and build.get("legacy_clean_conditioned_virtual_view_used_as_input") is False, str(build.get("virtual_view_input")))
    check("detector_specific_virtual_selection", build.get("virtual_view_selected_by_eval_detector") is True and build.get("exact_cache_alignment_verified") is True, f"selected={build.get('virtual_view_selected_by_eval_detector')}; aligned={build.get('exact_cache_alignment_verified')}")
    check("strict_gpu", build.get("device") == "cuda:0" and bool(build.get("gpu_name")) and int(build.get("workers", -1)) == 8, f"device={build.get('device')}; gpu={build.get('gpu_name')}; workers={build.get('workers')}")
    spawn_preflight = build.get("dataloader_spawn_preflight", {})
    spawn_partitions = spawn_preflight.get("partitions", {}) if isinstance(spawn_preflight, dict) else {}
    check(
        "windows_spawn_safe_dataloader",
        isinstance(spawn_preflight, dict)
        and spawn_preflight.get("status") == "PASS"
        and spawn_preflight.get("worker_model") == "windows_spawn_safe_local_initializer"
        and int(spawn_preflight.get("requested_workers", -1)) == 8
        and set(spawn_partitions) == {"train", "val"}
        and all(int(spawn_partitions[name].get("batch_size", 0)) > 0 for name in ("train", "val"))
        and build.get("dynamic_b1_callable_passed_to_workers") is False,
        f"status={spawn_preflight.get('status') if isinstance(spawn_preflight, dict) else None}; "
        f"workers={spawn_preflight.get('requested_workers') if isinstance(spawn_preflight, dict) else None}; "
        f"partitions={sorted(spawn_partitions)}; dynamic_b1={build.get('dynamic_b1_callable_passed_to_workers')}",
    )
    check("validation_only", build.get("development_test_metrics_computed") is False, str(build.get("development_test_metrics_computed")))
    check("official_test_isolation", build.get("official_TS1_to_TS6_read") is False and build.get("official_TS1_to_TS6_used") is False, "Official TS1-TS6 unread and unused")
    check("four_arms", set(build.get("arms", [])) == ARMS, str(build.get("arms")))
    check("run_count", int(build.get("run_count", -1)) == 7, str(build.get("run_count")))
    run_matrix = {(str(item.get("arm")), str(item.get("train_detector"))) for item in build.get("run_reports", [])}
    expected_matrix = {("clean_observed_legacy", "clean")} | {(arm, detector) for arm in ARMS - {"clean_observed_legacy"} for detector in DETECTORS}
    check("run_matrix", run_matrix == expected_matrix, f"runs={len(run_matrix)}")
    checkpoints = [Path(str(item.get("checkpoint", ""))) for item in build.get("run_reports", [])]
    check("checkpoints", len(checkpoints) == 7 and all(path.is_file() for path in checkpoints), f"count={sum(path.is_file() for path in checkpoints)}/7")
    loss_failures = [f"{item.get('arm')}/{item.get('train_detector')}" for item in build.get("run_reports", []) if not finite(item.get("train_loss_first")) or not finite(item.get("train_loss_last"))]
    check("finite_losses", not loss_failures, str(loss_failures))

    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metric_failures = [f"{row.get('arm')}/{row.get('train_detector')}/{row.get('eval_detector')}" for row in rows if not finite(row.get("mpjpe_mm")) or not finite(row.get("pa_mpjpe_mm")) or int(float(row.get("samples", "0"))) <= 0]
    check("metric_rows", len(rows) == 13, f"rows={len(rows)}")
    check("finite_metrics", not metric_failures, str(metric_failures))
    check("val_only_rows", {row.get("partition") for row in rows} == {"val"}, str(sorted({row.get("partition") for row in rows})))
    matched = [row for row in rows if row.get("train_detector") in DETECTORS and row.get("eval_detector") == row.get("train_detector")]
    cross = [row for row in rows if row.get("train_detector") in DETECTORS and row.get("eval_detector") in DETECTORS and row.get("eval_detector") != row.get("train_detector")]
    check("matched_and_cross_eval", len(matched) == 6 and len(cross) == 6, f"matched={len(matched)}; cross={len(cross)}")

    coverage = build.get("detector_coverage_audit", {}).get("common_cohort", {})
    joined = int(coverage.get("joined_samples", 0))
    eligible = int(coverage.get("eligible_samples", 0))
    excluded = int(coverage.get("excluded_any_detector_failure", -1))
    check("paired_cohort_accounting", joined > 0 and eligible > 0 and eligible + excluded == joined, f"joined={joined}; eligible={eligible}; excluded={excluded}")

    passed = all(item["passed"] for item in checks)
    errors = [f"{item['name']}: {item['detail']}" for item in checks if not item["passed"]]
    hashes = {
        build_path.name: sha256_file(build_path),
        metrics_path.name: sha256_file(metrics_path),
        protocol_path.name: sha256_file(protocol_path),
    }
    for checkpoint in checkpoints:
        if checkpoint.is_file():
            hashes[str(checkpoint.relative_to(output))] = sha256_file(checkpoint)
    qc = {
        "status": "PASS" if passed else "FAIL",
        "format": "phase9c_r1b_detector_conditioned_smoke_qc_report_v1",
        "phase": "9C-R1B",
        "scientific_decision": "READY_FOR_PHASE9C_R1C_CONFIDENCE_MECHANISM_SCREEN" if passed else "REPAIR_PHASE9C_R1B_BEFORE_CONFIDENCE_SCREEN",
        "scientific_role": "pipeline_smoke_without_model_or_claim_selection",
        "checks": checks,
        "errors": errors,
        "metric_rows": len(rows),
        "artifact_hashes": hashes,
        "development_test_metrics_computed": False,
        "official_TS1_to_TS6_read": False,
    }
    qc_path = output / "phase9c_r1b_smoke_qc_report.json"
    write_json(qc_path, qc)
    zip_path = output / "phase9c_r1b_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (protocol_path, build_path, metrics_path, qc_path):
            archive.write(path, path.name)
        for checkpoint in checkpoints:
            if checkpoint.is_file():
                archive.write(checkpoint, str(Path("checkpoints") / checkpoint.relative_to(output)))
    print("=" * 108)
    print("PHASE 9C-R1B - QC AND PACK")
    print("=" * 108)
    print(f"Status                : {qc['status']}")
    print(f"Scientific decision   : {qc['scientific_decision']}")
    print(f"Metric rows           : {len(rows)}")
    print(f"Output ZIP            : {zip_path}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    print("=" * 108)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
