from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


DETECTORS = ("rtmpose_performance", "yolo11l_pose")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def add_check(
    checks: list[dict[str, Any]],
    blockers: list[str],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        blockers.append(f"{name}: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = root / "outputs" / "phase9b_rgb_detector_benchmark"
    full = out / "full"
    lock = load_json(root / "configs" / "phase9b_rgb_detector_protocol_lock.json")
    preflight = load_json(out / "phase9b_preflight_report.json")
    metrics = load_json(full / "phase9b_detector_metrics.json")
    runtime = load_json(full / "phase9b_detector_runtime.json")
    error_model = load_json(full / "phase9b_real_detector_error_model.json")
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    add_check(
        checks, blockers, "protocol_lock",
        lock.get("status") == "LOCKED"
        and lock.get("format") == "phase9b_rgb_detector_protocol_lock_v1",
        str(lock.get("format")),
    )
    add_check(
        checks, blockers, "preflight_pass",
        preflight.get("status") == "PASS",
        str(preflight.get("status")),
    )
    add_check(
        checks, blockers, "official_test_isolation",
        preflight.get("protocol", {}).get("official_TS1_to_TS6_read") is False
        and metrics.get("protocol", {}).get("official_TS1_to_TS6_read") is False
        and metrics.get("protocol", {}).get("model_training_performed") is False,
        "Official TS1-TS6 unread; no 3D training",
    )

    cache_path = full / "phase9b_detector_cache.npz"
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            detector_ids = tuple(str(value) for value in cache["detector_id"])
            sample_count = len(cache["sample_key"])
            shapes_ok = (
                cache["prediction_c15_px"].shape
                == (2, sample_count, 15, 2)
                and cache["confidence_c15"].shape
                == (2, sample_count, 15)
                and cache["detected"].shape == (2, sample_count)
                and cache["gt2d_px"].shape == (sample_count, 15, 2)
            )
            finite_gt = bool(np.isfinite(cache["gt2d_px"]).all())
    except Exception as error:
        detector_ids = ()
        sample_count = 0
        shapes_ok = False
        finite_gt = False
        blockers.append(f"cache_load: {error!r}")
    add_check(
        checks, blockers, "two_detector_cache",
        detector_ids == DETECTORS and shapes_ok and sample_count > 0,
        f"detectors={detector_ids}, samples={sample_count}, shapes_ok={shapes_ok}",
    )
    add_check(
        checks, blockers, "finite_ground_truth",
        finite_gt,
        str(finite_gt),
    )
    add_check(
        checks, blockers, "finite_reports",
        finite_tree(metrics) and finite_tree(error_model),
        "metrics/error model finite or explicit null",
    )

    gate = lock["scientific_gate"]
    detector_gate: dict[str, Any] = {}
    for detector_id in DETECTORS:
        item = metrics.get("detectors", {}).get(detector_id, {})
        success = float(item.get("detection_success_rate", 0.0))
        finite_fraction = float(
            item.get("common12", {}).get("finite_fraction", 0.0)
        )
        passed = (
            success >= float(gate["minimum_success_rate_each_detector"])
            and finite_fraction
            >= float(gate["minimum_finite_common12_fraction"])
        )
        detector_gate[detector_id] = {
            "detection_success_rate": success,
            "finite_common12_fraction": finite_fraction,
            "passed": passed,
        }
        add_check(
            checks, blockers, f"detector_gate:{detector_id}", passed,
            f"success={success:.4f}, finite_common12={finite_fraction:.4f}",
        )

    model_files = {
        detector_id: runtime.get("detectors", {})
        .get(detector_id, {})
        .get("model_files", [])
        for detector_id in DETECTORS
    }
    add_check(
        checks, blockers, "detector_runtime_and_weight_hashes_recorded",
        all(
            detector_id in runtime.get("detectors", {})
            and len(model_files[detector_id]) >= 1
            and all(
                len(str(item.get("sha256", ""))) == 64
                for item in model_files[detector_id]
            )
            for detector_id in DETECTORS
        ),
        str({key: len(value) for key, value in model_files.items()}),
    )

    decision = (
        gate["pass_decision"] if not blockers else gate["fail_decision"]
    )
    report = {
        "format": "phase9b_rgb_detector_qc_v1",
        "phase": "9B",
        "status": "PASS" if not blockers else "BLOCKED",
        "scientific_decision": decision,
        "checks": checks,
        "blockers": blockers,
        "detector_gate": detector_gate,
        "coverage_interpretation": preflight.get("coverage_interpretation"),
        "sample_count": sample_count,
        "official_TS1_to_TS6_read": False,
        "model_training_performed": False,
        "next_phase_if_pass": (
            "Phase 9C confidence-aware V2 training with real-detector cache, "
            "noise augmentation matched to Phase 9B error quantiles, "
            "observed-only and virtual-view ablations, 3 splits x 3 seeds."
        ),
    }
    write_json(out / "phase9b_qc_report.json", report)

    summary_path = full / "phase9b_detector_summary.csv"
    with summary_path.open("r", newline="", encoding="utf-8-sig") as file:
        summary_rows = list(csv.DictReader(file))
    if len(summary_rows) != 2:
        raise RuntimeError("Detector summary must contain two rows.")

    include = [
        root / "configs" / "phase9b_rgb_detector_protocol_lock.json",
        out / "phase9b_preflight_report.json",
        out / "phase9b_data_root_diagnostic.json",
        out / "phase9b_rgb_coverage.csv",
        out / "phase9b_rgb_index.csv",
        out / "phase9b_qc_report.json",
        full / "phase9b_detector_runtime.json",
        full / "phase9b_detector_metrics.json",
        full / "phase9b_real_detector_error_model.json",
        full / "phase9b_detector_summary.csv",
        full / "phase9b_detector_cache.npz",
    ]
    zip_path = out / "phase9b_rgb_detector_benchmark_results.zip"
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in include:
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, arcname=path.name)
    print(json.dumps(report, indent=2))
    print(f"RESULT_ZIP={zip_path}")
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
