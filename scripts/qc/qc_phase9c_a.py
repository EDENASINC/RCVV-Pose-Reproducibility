from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def check(name: str, passed: bool, detail: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/phase9c_a_calibrated_error_bank"),
    )
    args = parser.parse_args()
    root = args.output_root if args.output_root.is_absolute() else args.project_root / args.output_root
    root = root.resolve()
    required = [
        "phase9c_a_protocol_lock.json",
        "phase9c_a_build_report.json",
        "phase9c_a_calibration_models.json",
        "phase9c_a_calibration_summary.csv",
        "phase9c_a_residual_bank.npz",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 9C-A artifacts: {missing}")

    protocol = load_json(root / required[0])
    build = load_json(root / required[1])
    models = load_json(root / required[2])
    rows = list(csv.DictReader((root / required[3]).open("r", encoding="utf-8")))
    bank = np.load(root / required[4], allow_pickle=False)
    checks: list[dict[str, Any]] = []
    check("protocol_lock", protocol.get("status") == "LOCKED", str(protocol.get("format")), checks)
    check("build_pass", build.get("status") == "PASS", str(build.get("status")), checks)
    check(
        "official_test_isolation",
        build.get("official_TS1_to_TS6_read") is False
        and build.get("official_TS1_to_TS6_used") is False,
        "Official TS1-TS6 unread and unused",
        checks,
    )
    check(
        "no_test_subject_calibration",
        build.get("test_subjects_used_for_calibration") is False
        and all(not item.get("test_subjects_used") for item in models.get("splits", {}).values()),
        "Calibration fit uses training subjects only",
        checks,
    )
    check("six_split_detector_pairs", len(rows) == 6, f"pairs={len(rows)}", checks)
    finite = True
    ece_better = True
    brier_better = True
    minimum_ece_reduction = float("inf")
    minimum_brier_reduction = float("inf")
    for row in rows:
        values = [
            float(row["validation_raw_ece"]),
            float(row["validation_calibrated_ece"]),
            float(row["validation_raw_brier"]),
            float(row["validation_calibrated_brier"]),
        ]
        finite = finite and bool(np.isfinite(values).all())
        ece_delta = values[0] - values[1]
        brier_delta = values[2] - values[3]
        ece_better = ece_better and ece_delta > 0.0
        brier_better = brier_better and brier_delta > 0.0
        minimum_ece_reduction = min(minimum_ece_reduction, ece_delta)
        minimum_brier_reduction = min(minimum_brier_reduction, brier_delta)
    check("finite_calibration_metrics", finite, "all six pairs finite", checks)
    check("validation_ece_improved_all_pairs", ece_better, f"minimum reduction={minimum_ece_reduction:.6f}", checks)
    check("validation_brier_improved_all_pairs", brier_better, f"minimum reduction={minimum_brier_reduction:.6f}", checks)

    expected_shape = (3, 2, 15, 10, 512, 2)
    actual_shape = tuple(bank["residual_xy_torso"].shape)
    counts = bank["reservoir_counts"]
    population = bank["population_counts"]
    bank_consistent = (
        actual_shape == expected_shape
        and counts.shape == (3, 2, 15, 10)
        and population.shape == counts.shape
        and np.all(counts >= 0)
        and np.all(counts <= 512)
        and np.all(population >= counts)
        and int(counts.sum()) > 0
    )
    valid_vectors_finite = True
    residual = bank["residual_xy_torso"]
    for index in np.ndindex(counts.shape):
        count_value = int(counts[index])
        if count_value and not np.isfinite(residual[index][:count_value]).all():
            valid_vectors_finite = False
            break
    check("residual_bank_shape_and_counts", bank_consistent, f"shape={actual_shape}, stored={int(counts.sum())}", checks)
    check("residual_bank_valid_vectors_finite", valid_vectors_finite, "all populated residuals finite", checks)

    hashes_match = True
    for name, expected in build.get("artifact_hashes", {}).items():
        hashes_match = hashes_match and (root / name).is_file() and sha256(root / name) == expected
    check("artifact_hashes", hashes_match, f"artifacts={len(build.get('artifact_hashes', {}))}", checks)
    passed = all(item["passed"] for item in checks)
    decision = (
        protocol["next_phase_gate"]["pass_decision"]
        if passed else protocol["next_phase_gate"]["fail_decision"]
    )
    report = {
        "status": "PASS" if passed else "BLOCKED",
        "format": "phase9c_a_qc_report_v1",
        "phase": "9C-A",
        "scientific_decision": decision,
        "checks": checks,
        "calibration_pairs": len(rows),
        "minimum_validation_ece_reduction": minimum_ece_reduction,
        "minimum_validation_brier_reduction": minimum_brier_reduction,
        "official_TS1_to_TS6_read": False,
        "3d_training_performed": False,
        "next_phase_if_pass": "Phase 9C-B: four-arm, three-split, three-seed confidence-aware V2 training and detector-matched/cross-detector evaluation.",
    }
    report_path = root / "phase9c_a_qc_report.json"
    write_json(report_path, report)
    zip_path = root / "phase9c_a_results.zip"
    files_to_pack = [root / name for name in required] + [report_path]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files_to_pack:
            archive.write(path, arcname=path.name)
    print("=" * 96)
    print("PHASE 9C-A QC + PACK")
    print("=" * 96)
    for item in checks:
        print(f"{item['name']:<44}: {'PASS' if item['passed'] else 'FAIL'}  {item['detail']}")
    print(f"Scientific decision : {decision}")
    print(f"RESULT_ZIP={zip_path}")
    print(f"Status              : {report['status']}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
