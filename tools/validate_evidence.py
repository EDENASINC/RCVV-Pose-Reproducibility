#!/usr/bin/env python3
"""Validate the small locked evidence package without dataset or ML dependencies."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"

REQUIRED = (
    "00_README_EVIDENCE_INDEX.md",
    "01_rgb_2d_detector_audit_protocol.json",
    "02_rgb_2d_detector_audit_summary.csv",
    "03_rgb_2d_detector_full_metrics.json",
    "04_rgb_2d_detector_runtime_environment.json",
    "05_leakage_safe_confidence_calibration_protocol.json",
    "06_leakage_safe_confidence_calibration_summary.csv",
    "07_leakage_safe_confidence_calibration_qc_report.json",
    "08_development_factorial_training_protocol.json",
    "09_development_factorial_training_qc_report.json",
    "10_development_factorial_attribution_protocol.json",
    "11_development_factorial_attribution_statistics.json",
    "12_development_factorial_attribution_qc_report.json",
    "13_locked_official_ts1_ts6_evaluation_protocol.json",
    "14_locked_official_ts1_ts6_evaluation_qc_report.json",
    "15_locked_official_ts1_ts6_factorial_results.json",
)


def load_json(name: str):
    with (EVIDENCE / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected}, got {actual}")


def main() -> int:
    missing = [name for name in REQUIRED if not (EVIDENCE / name).is_file()]
    if missing:
        print("FAIL: missing evidence files:", *missing, sep="\n  - ")
        return 1

    training = load_json("08_development_factorial_training_protocol.json")
    assert training["development_scope"]["splits"] == ["split_a", "split_b", "split_c"]
    assert training["development_scope"]["seeds"] == [42, 123, 2026]
    assert training["locked_selected_arm"] == "bounded_reliability_dual_modulation"
    assert training["virtual_view_policy"]["candidate_id"] == "rotation_yaw_m15"

    official = load_json("13_locked_official_ts1_ts6_evaluation_protocol.json")
    matrix = official["locked_model_matrix"]
    assert matrix["checkpoint_count"] == 72
    assert matrix["official_test_used_for_checkpoint_or_arm_selection"] is False
    assert official["evaluation_scope"]["input_2d"] == "RGB detector predictions only"
    assert sum(subject["valid_frames"] for subject in official["subjects"]) == 2875

    results = load_json("15_locked_official_ts1_ts6_factorial_results.json")
    aggregate = results["aggregate_subject_split_seed_detector_macro"]
    close(aggregate["detector_aware_observed"]["MPJPE_mm"], 127.29545543811939)
    close(aggregate["detector_aware_observed"]["PA_MPJPE_mm"], 84.17566776275635)
    close(aggregate["bounded_reliability_dual_modulation"]["MPJPE_mm"], 126.06470284638581)
    close(aggregate["bounded_reliability_dual_modulation"]["PA_MPJPE_mm"], 81.84858551731816)

    with (EVIDENCE / "06_leakage_safe_confidence_calibration_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        calibration = list(csv.DictReader(handle))
    assert len(calibration) == 6
    assert all(float(row["validation_ece_reduction"]) > 0 for row in calibration)
    assert all(float(row["validation_brier_reduction"]) > 0 for row in calibration)

    for qc_name in (
        "07_leakage_safe_confidence_calibration_qc_report.json",
        "09_development_factorial_training_qc_report.json",
        "12_development_factorial_attribution_qc_report.json",
        "14_locked_official_ts1_ts6_evaluation_qc_report.json",
    ):
        assert load_json(qc_name)["status"] == "PASS", qc_name

    print("PASS: locked evidence contract is internally consistent.")
    print("  checkpoints: 72")
    print("  official valid frames: 2875")
    print("  final arm: bounded_reliability_dual_modulation (OVR)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
