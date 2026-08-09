from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("split_a", "split_b", "split_c")
PARTITIONS = ("train", "val", "test")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="QC and pack Phase 9C-B0.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol-lock", type=Path, default=Path("configs/phase9c_b_protocol_lock.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase9c_b_detector_join_preflight"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    lock_path = args.protocol_lock if args.protocol_lock.is_absolute() else root / args.protocol_lock
    lock = load_json(lock_path)
    report_path = output / "phase9c_b0_join_build_report.json"
    join_path = output / "phase9c_b0_detector_join_cache.npz"
    csv_path = output / "phase9c_b0_join_summary.csv"
    report = load_json(report_path)
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    check("protocol_lock", lock.get("status") == "LOCKED", str(lock.get("status")))
    check("build_pass", report.get("status") == "PASS", str(report.get("status")))
    check(
        "source_phase9c_a_gate",
        report.get("phase9c_a_scientific_decision") == lock.get("source_gate"),
        str(report.get("phase9c_a_scientific_decision")),
    )
    check(
        "phase9b_cache_hash_lock",
        report.get("phase9b_detector_cache_sha256")
        == report.get("phase9b_detector_cache_expected_sha256"),
        str(report.get("phase9b_detector_cache_sha256")),
    )
    check(
        "unique_detector_identity",
        int(report.get("duplicate_detector_identity_count", -1)) == 0,
        f"duplicates={report.get('duplicate_detector_identity_count')}",
    )
    check(
        "worker_lock",
        report.get("workers", {}).get("locked_for_phase9c_b1_training") == 8,
        str(report.get("workers")),
    )
    check(
        "official_test_isolation",
        report.get("official_TS1_to_TS6_read") is False
        and report.get("official_TS1_to_TS6_used") is False,
        "Official TS1-TS6 unread and unused",
    )
    check("join_cache_exists", join_path.is_file(), str(join_path))
    check("join_summary_exists", csv_path.is_file(), str(csv_path))

    detector_samples = int(report.get("phase9b_detector_samples", -1))
    max_p95 = float(lock["b0_gate"]["require_camera_normalized_gt_p95_error_max"])
    for split_name in SPLITS:
        item = report.get("splits", {}).get(split_name, {})
        check(
            f"{split_name}_complete_unique_join",
            int(item.get("joined_total", -1)) == detector_samples
            and int(item.get("unique_joined_detector_samples", -1)) == detector_samples,
            f"joined={item.get('joined_total')} unique={item.get('unique_joined_detector_samples')} expected={detector_samples}",
        )
        check(
            f"{split_name}_dataset_identity_unique",
            int(item.get("duplicate_dataset_identity_count", -1)) == 0,
            f"duplicates={item.get('duplicate_dataset_identity_count')}",
        )
        counts = item.get("joined_samples", {})
        check(
            f"{split_name}_partition_coverage",
            all(int(counts.get(partition, 0)) > 0 for partition in PARTITIONS),
            str(counts),
        )
        p95 = float(item.get("coordinate_alignment", {}).get("p95", np.inf))
        check(
            f"{split_name}_coordinate_alignment",
            np.isfinite(p95) and p95 <= max_p95,
            f"p95={p95:.9f} threshold={max_p95:.9f}",
        )

    if join_path.is_file():
        with np.load(join_path, allow_pickle=False) as data:
            expected_keys = {
                f"{split}_{suffix}"
                for split in SPLITS
                for suffix in ("detector_sample_index", "dataset_index", "partition_code")
            }
            check(
                "join_cache_schema",
                expected_keys.issubset(data.files),
                f"arrays={len(data.files)}",
            )
            finite = True
            for split in SPLITS:
                for suffix in ("detector_sample_index", "dataset_index", "partition_code"):
                    finite = finite and bool(np.isfinite(data[f"{split}_{suffix}"]).all())
            check("join_cache_finite", finite, "all join arrays finite")

    status = "PASS" if not errors else "FAIL"
    decision = (
        lock["b0_gate"]["pass_decision"]
        if status == "PASS"
        else lock["b0_gate"]["fail_decision"]
    )
    artifact_hashes = {}
    for path in (report_path, join_path, csv_path):
        if path.is_file():
            artifact_hashes[path.name] = sha256_file(path)
    qc = {
        "status": status,
        "format": "phase9c_b0_qc_report_v1",
        "phase": "9C-B0",
        "scientific_decision": decision,
        "checks": checks,
        "errors": errors,
        "artifact_hashes": artifact_hashes,
        "official_TS1_to_TS6_read": False,
        "training_performed": False,
        "phase9c_b1_num_workers": 8,
    }
    qc_path = output / "phase9c_b0_qc_report.json"
    write_json(qc_path, qc)

    zip_path = output / "phase9c_b0_results.zip"
    if zip_path.exists():
        zip_path.unlink()
    selected = (report_path, join_path, csv_path, qc_path, lock_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in selected:
            if path.is_file():
                archive.write(path, arcname=path.name)

    print("=" * 112)
    print("PHASE 9C-B0 - JOIN QC AND PACK")
    print("=" * 112)
    print(f"Checks                : {sum(item['passed'] for item in checks)}/{len(checks)}")
    print(f"Errors                : {len(errors)}")
    for error in errors:
        print("ERROR:", error)
    print("Workers next training : 8")
    print("Official TS1-TS6 read : NO")
    print(f"Scientific decision   : {decision}")
    print(f"RESULT_ZIP            : {zip_path}")
    print(f"Status                : {status}")
    print("=" * 112)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
