from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SPLITS = ("split_a", "split_b", "split_c")
EXPECTED_B1_SCRIPT_SHA256 = (
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


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight Phase 9C-B2 full study.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs/phase9c_b2_full_multisplit_multiseed"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "configs/phase9c_b2_full_protocol.json"
    b1_script = root / "scripts/train/run_phase9c_b1_four_arm_smoke.py"
    b1_qc_path = root / "outputs/phase9c_b1_four_arm_smoke/phase9c_b1_smoke_qc_report.json"
    b1_build_path = root / "outputs/phase9c_b1_four_arm_smoke/phase9c_b1_smoke_build_report.json"
    required = {
        "protocol": protocol_path,
        "b1_script": b1_script,
        "b1_qc": b1_qc_path,
        "b1_build": b1_build_path,
        "phase9c_a_qc": root / "outputs/phase9c_a_calibrated_error_bank/phase9c_a_qc_report.json",
        "phase9c_b0_qc": root / "outputs/phase9c_b_detector_join_preflight/phase9c_b0_qc_report.json",
        "join_cache": root / "outputs/phase9c_b_detector_join_preflight/phase9c_b0_detector_join_cache.npz",
    }
    checks: list[dict[str, Any]] = []
    missing = [name for name, path in required.items() if not path.is_file()]
    add(checks, "required_artifacts", not missing, f"missing={missing}")
    if missing:
        report = {
            "status": "FAIL",
            "format": "phase9c_b2_preflight_report_v1",
            "phase": "9C-B2",
            "checks": checks,
            "errors": [f"Missing required artifacts: {missing}"],
            "official_TS1_to_TS6_read": False,
        }
        write_json(output / "phase9c_b2_preflight_report.json", report)
        raise SystemExit(2)

    protocol = load_json(protocol_path)
    b1_qc = load_json(b1_qc_path)
    b1_build = load_json(b1_build_path)
    b1_digest = sha256_file(b1_script)
    add(
        checks,
        "protocol_lock",
        protocol.get("status") == "LOCKED"
        and protocol.get("format") == "phase9c_b2_full_protocol_lock_v1",
        f"status={protocol.get('status')}; format={protocol.get('format')}",
    )
    add(
        checks,
        "b1_gate",
        b1_qc.get("status") == "PASS"
        and b1_qc.get("scientific_decision")
        == "READY_FOR_PHASE9C_B2_FULL_MULTISPLIT_MULTISEED_TRAINING",
        f"status={b1_qc.get('status')}; decision={b1_qc.get('scientific_decision')}",
    )
    add(
        checks,
        "b1_v3_source_lock",
        b1_digest == EXPECTED_B1_SCRIPT_SHA256
        == protocol.get("source_b1_script_sha256"),
        f"sha256={b1_digest}",
    )
    official_isolated = (
        b1_qc.get("official_TS1_to_TS6_read") is False
        and b1_build.get("official_TS1_to_TS6_read") is False
        and protocol.get("official_test_policy", {}).get("official_TS1_to_TS6_read") is False
    )
    add(checks, "official_test_isolation", official_isolated, "Official TS1-TS6 unread")
    budget = protocol.get("training_budget", {})
    add(
        checks,
        "worker_lock_8",
        int(budget.get("num_workers", -1)) == 8,
        f"num_workers={budget.get('num_workers')}",
    )
    matrix = protocol.get("run_matrix", {})
    add(
        checks,
        "run_matrix_lock",
        int(matrix.get("total_training_runs", -1)) == 63
        and int(matrix.get("expected_metric_rows", -1)) == 234,
        str(matrix),
    )
    risk_mapping = protocol.get("calibration_and_risk_policy", {}).get(
        "split_to_residual_risk_index", {}
    )
    add(
        checks,
        "split_specific_calibration_and_risk_lock",
        risk_mapping == {"split_a": 0, "split_b": 1, "split_c": 2},
        str(risk_mapping),
    )
    manifest_detail: dict[str, Any] = {}
    manifests_ok = True
    for split in SPLITS:
        path = (
            root
            / "data/processed/caches/phase5r1a_multigeometry_oof_25fps_v1"
            / split
            / "manifest.json"
        )
        if not path.is_file():
            manifests_ok = False
            manifest_detail[split] = "MISSING"
            continue
        manifest = load_json(path)
        good = (
            manifest.get("status") == "PASS"
            and not bool(manifest.get("partial", False))
            and manifest.get("format") == "phase5r1a_multigeometry_synthesized_manifest_v1"
        )
        manifests_ok &= good
        manifest_detail[split] = {
            "status": manifest.get("status"),
            "partial": manifest.get("partial"),
            "candidate_count": manifest.get("candidate_count"),
        }
    add(checks, "full_split_manifests", manifests_ok, str(manifest_detail))
    errors = [item["name"] for item in checks if not item["passed"]]
    status = "PASS" if not errors else "FAIL"
    report = {
        "status": status,
        "format": "phase9c_b2_preflight_report_v1",
        "phase": "9C-B2",
        "scientific_decision": (
            "READY_FOR_PHASE9C_B2_FULL_RUNS" if status == "PASS"
            else "REPAIR_PHASE9C_B2_PREFLIGHT"
        ),
        "checks": checks,
        "errors": errors,
        "protocol_sha256": sha256_file(protocol_path),
        "b1_script_sha256": b1_digest,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    write_json(output / "phase9c_b2_preflight_report.json", report)
    print("=" * 104)
    print("PHASE 9C-B2 - PREFLIGHT")
    print("=" * 104)
    print(f"Status                : {status}")
    print(f"B1 gate               : {b1_qc.get('scientific_decision')}")
    print("Workers               : 8")
    print("Official TS1-TS6 read: NO")
    print(f"Scientific decision   : {report['scientific_decision']}")
    if errors:
        print(f"Errors                : {errors}")
    print("=" * 104)
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
