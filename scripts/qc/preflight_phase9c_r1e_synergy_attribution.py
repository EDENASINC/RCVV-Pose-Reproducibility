from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight Phase 9C-R1E synergy attribution.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    protocol_path = root / "configs/phase9c_r1e_synergy_attribution_protocol.json"
    protocol = load_json(protocol_path)
    r1d_root = root / "outputs/phase9c_r1d_locked_full_confirmation"
    r1d_qc = load_json(r1d_root / "phase9c_r1d_qc_report.json")
    r1d_stats = load_json(r1d_root / "phase9c_r1d_statistics.json")
    r1d_protocol_path = root / "configs/phase9c_r1d_locked_full_confirmation_protocol.json"
    train_script = root / "scripts/train/run_phase9c_r1e_synergy_attribution.py"
    errors: list[str] = []

    if protocol.get("status") != "LOCKED" or protocol.get("format") != "phase9c_r1e_synergy_attribution_protocol_v1":
        errors.append("invalid_r1e_protocol")
    if sha256_file(r1d_protocol_path) != protocol.get("source_r1d_protocol_sha256"):
        errors.append("r1d_protocol_hash_mismatch")
    if r1d_qc.get("status") != "PASS":
        errors.append("r1d_repaired_qc_not_pass")
    if r1d_qc.get("format") != "phase9c_r1d_qc_report_v2_evidence_contract_repair":
        errors.append("r1d_evidence_contract_repair_not_applied")
    if r1d_stats.get("scientific_decision") != protocol.get("source_r1d_required_decision"):
        errors.append("unexpected_r1d_scientific_decision")
    if r1d_qc.get("official_TS1_to_TS6_read") is not False or r1d_stats.get("official_TS1_to_TS6_read") is not False:
        errors.append("official_test_isolation_failed")
    if int(r1d_qc.get("run_reports", -1)) != 54 or int(r1d_qc.get("metric_rows", -1)) != 216:
        errors.append("r1d_matrix_incomplete")

    hashes_path = r1d_root / "phase9c_r1d_artifact_hashes.csv"
    with hashes_path.open("r", newline="", encoding="utf-8-sig") as handle:
        artifact_rows = list(csv.DictReader(handle))
    if len(artifact_rows) != 378 or any(row.get("verified") != "True" for row in artifact_rows):
        errors.append("r1d_artifact_hashes_not_378_verified")
    if not train_script.is_file():
        errors.append("missing_r1e_train_script")

    status = "PASS" if not errors else "FAIL"
    report = {
        "status": status,
        "format": "phase9c_r1e_preflight_report_v1",
        "phase": "9C-R1E",
        "scientific_decision": "READY_FOR_PHASE9C_R1E_FULL_RUNS" if status == "PASS" else "REPAIR_BEFORE_PHASE9C_R1E",
        "protocol_sha256": sha256_file(protocol_path),
        "train_script_sha256": sha256_file(train_script) if train_script.is_file() else None,
        "source_r1d_qc_format": r1d_qc.get("format"),
        "source_r1d_scientific_decision": r1d_stats.get("scientific_decision"),
        "source_r1d_primary_mpjpe_gain_percent": r1d_stats.get("gate_detail", {}).get("primary_macro_mpjpe_gain_percent"),
        "source_r1d_artifacts_verified": sum(row.get("verified") == "True" for row in artifact_rows),
        "planned_new_runs": 18,
        "workers": 8,
        "device": "cuda:0",
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
        "errors": errors,
    }
    output = root / "outputs/phase9c_r1e_synergy_attribution"
    write_json(output / "phase9c_r1e_preflight_report.json", report)
    print("=" * 104)
    print("PHASE 9C-R1E SYNERGY ATTRIBUTION PREFLIGHT")
    print("=" * 104)
    print(f"Status                : {status}")
    print("R1D artifacts verified: 378 / 378")
    print("New training runs     : 18")
    print("Official TS1-TS6 read : NO")
    if errors:
        print(f"Errors                : {errors}")
    print("=" * 104)
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
