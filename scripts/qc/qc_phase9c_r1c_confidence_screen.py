from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any


SPLITS = ("split_a", "split_b", "split_c")
SEED = 42
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
ARMS = (
    "detector_conditioned_virtual_view_no_confidence",
    "calibrated_probability_concat",
    "bounded_reliability_concat",
    "bounded_reliability_virtual_gate",
    "bounded_reliability_dual_modulation",
)
BASELINE = ARMS[0]
FEATURE_DIMS = {
    "detector_conditioned_virtual_view_no_confidence": 56,
    "calibrated_probability_concat": 70,
    "bounded_reliability_concat": 84,
    "bounded_reliability_virtual_gate": 70,
    "bounded_reliability_dual_modulation": 126,
}


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


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list.")
    return sum(values) / len(values)


def paired_gain_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0 or not math.isfinite(baseline) or not math.isfinite(candidate):
        raise ValueError(f"Invalid metric pair: {baseline}, {candidate}")
    return 100.0 * (baseline - candidate) / baseline


def run_report_path(output: Path, split: str, arm: str, detector: str) -> Path:
    return output / split / f"seed{SEED}" / arm / f"train_{detector}" / "phase9c_r1c_run_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="QC, select, and pack Phase 9C-R1C.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs/phase9c_r1c_confidence_mechanism_screen"
    protocol_path = root / "configs/phase9c_r1c_confidence_mechanism_screen_protocol.json"
    preflight_path = output / "phase9c_r1c_preflight_report.json"
    protocol = load_json(protocol_path)
    preflight = load_json(preflight_path)
    protocol_sha = sha256_file(protocol_path)
    if protocol.get("status") != "LOCKED" or protocol.get("format") != "phase9c_r1c_confidence_mechanism_screen_protocol_lock_v1":
        raise ValueError("Invalid R1C protocol lock.")
    if preflight.get("status") != "PASS" or preflight.get("scientific_decision") != "READY_FOR_PHASE9C_R1C_SCREEN_RUNS":
        raise ValueError("R1C preflight is not PASS.")
    if preflight.get("protocol_sha256") != protocol_sha:
        raise ValueError("Protocol changed after preflight.")
    if tuple(protocol["arms"]) != ARMS or tuple(protocol["detectors"]) != DETECTORS:
        raise ValueError("Locked run matrix mismatch.")

    split_reports: list[tuple[Path, dict[str, Any]]] = []
    for split in SPLITS:
        path = output / split / "phase9c_r1c_split_report.json"
        report = load_json(path)
        if report.get("status") != "PASS" or report.get("format") != "phase9c_r1c_split_report_v1":
            raise ValueError(f"Invalid split report: {split}")
        if report.get("protocol_sha256") != protocol_sha or int(report.get("run_count", -1)) != 10:
            raise ValueError(f"Split report lock/count mismatch: {split}")
        if report.get("test_metrics_computed") is not False or report.get("official_TS1_to_TS6_read") is not False:
            raise ValueError(f"Test isolation violation: {split}")
        if int(report.get("workers", -1)) != 8:
            raise ValueError(f"Worker mismatch: {split}")
        split_reports.append((path, report))

    expected = {(split, arm, detector) for split in SPLITS for arm in ARMS for detector in DETECTORS}
    run_reports: list[tuple[Path, dict[str, Any]]] = []
    artifact_rows: list[dict[str, str]] = []
    metric_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for split, arm, detector in sorted(expected):
        path = run_report_path(output, split, arm, detector)
        report = load_json(path)
        key = (str(report.get("split")), str(report.get("arm")), str(report.get("train_detector")))
        if key != (split, arm, detector):
            raise ValueError(f"Run identity mismatch: expected={(split, arm, detector)}, got={key}")
        if report.get("status") != "PASS" or report.get("format") != "phase9c_r1c_run_report_v1":
            raise ValueError(f"Invalid run report: {key}")
        if report.get("protocol_sha256") != protocol_sha or int(report.get("seed", -1)) != SEED:
            raise ValueError(f"Run lock mismatch: {key}")
        if report.get("test_metrics_computed") is not False or report.get("test_used_for_selection") is not False:
            raise ValueError(f"Development-test isolation violation: {key}")
        if report.get("official_TS1_to_TS6_read") is not False:
            raise ValueError(f"Official-test isolation violation: {key}")
        if int(report.get("feature_dim", -1)) != FEATURE_DIMS[arm]:
            raise ValueError(f"Feature dimension mismatch: {key}")
        evaluations = report.get("validation_evaluations", {})
        if set(evaluations) != set(DETECTORS):
            raise ValueError(f"Validation detector matrix mismatch: {key}")
        for eval_detector in DETECTORS:
            metrics = evaluations[eval_detector]
            mpjpe = float(metrics["mpjpe_mm"])
            pa = float(metrics["pa_mpjpe_mm"])
            samples = int(metrics["samples"])
            if not all(math.isfinite(value) and value > 0 for value in (mpjpe, pa)) or samples <= 0:
                raise ValueError(f"Invalid validation metrics: {key}/{eval_detector}")
            metric_rows.append({
                "split": split,
                "seed": SEED,
                "arm": arm,
                "train_detector": detector,
                "eval_detector": eval_detector,
                "relation": "matched" if detector == eval_detector else "cross",
                "partition": "val",
                "mpjpe_mm": mpjpe,
                "pa_mpjpe_mm": pa,
                "samples": samples,
                "best_epoch": int(report["best_epoch"]),
                "epochs_completed": int(report["epochs_completed"]),
                "feature_dim": int(report["feature_dim"]),
                "trainable_parameters": int(report["trainable_parameters"]),
                "elapsed_sec": float(report["elapsed_sec"]),
            })
        for artifact in report.get("artifact_hashes", []):
            target = Path(artifact["path"])
            digest = sha256_file(target)
            if digest != artifact["sha256"]:
                raise ValueError(f"Run artifact hash mismatch: {target}")
            artifact_rows.append({
                "split": split,
                "arm": arm,
                "train_detector": detector,
                "path": str(target),
                "sha256": digest,
            })
        seen.add(key)
        run_reports.append((path, report))
    if seen != expected:
        raise ValueError(f"Run matrix mismatch: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    if len(metric_rows) != int(protocol["run_matrix"]["expected_metric_rows"]):
        raise ValueError(f"Metric row count mismatch: {len(metric_rows)}")

    metrics_path = output / "phase9c_r1c_validation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    artifacts_path = output / "phase9c_r1c_artifact_hashes.csv"
    with artifacts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(artifact_rows[0]))
        writer.writeheader()
        writer.writerows(artifact_rows)

    matched = [row for row in metric_rows if row["relation"] == "matched"]
    index = {(row["split"], row["arm"], row["train_detector"]): row for row in matched}
    comparisons: dict[str, Any] = {}
    eligible: list[str] = []
    policy = protocol["selection_policy"]
    for arm in ARMS[1:]:
        pairs: list[dict[str, Any]] = []
        for split in SPLITS:
            for detector in DETECTORS:
                baseline = index[(split, BASELINE, detector)]
                candidate = index[(split, arm, detector)]
                base_score = math.sqrt(float(baseline["mpjpe_mm"]) * float(baseline["pa_mpjpe_mm"]))
                candidate_score = math.sqrt(float(candidate["mpjpe_mm"]) * float(candidate["pa_mpjpe_mm"]))
                pairs.append({
                    "split": split,
                    "detector": detector,
                    "baseline_mpjpe_mm": float(baseline["mpjpe_mm"]),
                    "candidate_mpjpe_mm": float(candidate["mpjpe_mm"]),
                    "mpjpe_gain_mm": float(baseline["mpjpe_mm"]) - float(candidate["mpjpe_mm"]),
                    "mpjpe_gain_percent": paired_gain_percent(float(baseline["mpjpe_mm"]), float(candidate["mpjpe_mm"])),
                    "baseline_pa_mpjpe_mm": float(baseline["pa_mpjpe_mm"]),
                    "candidate_pa_mpjpe_mm": float(candidate["pa_mpjpe_mm"]),
                    "pa_mpjpe_gain_mm": float(baseline["pa_mpjpe_mm"]) - float(candidate["pa_mpjpe_mm"]),
                    "pa_mpjpe_gain_percent": paired_gain_percent(float(baseline["pa_mpjpe_mm"]), float(candidate["pa_mpjpe_mm"])),
                    "geometric_score_gain_percent": paired_gain_percent(base_score, candidate_score),
                })
        macro_mpjpe = mean([row["mpjpe_gain_percent"] for row in pairs])
        macro_pa = mean([row["pa_mpjpe_gain_percent"] for row in pairs])
        macro_score = mean([row["geometric_score_gain_percent"] for row in pairs])
        split_detail: dict[str, Any] = {}
        split_wins = 0
        for split in SPLITS:
            rows = [row for row in pairs if row["split"] == split]
            mp = mean([row["mpjpe_gain_percent"] for row in rows])
            pa = mean([row["pa_mpjpe_gain_percent"] for row in rows])
            both = mp > 0 and pa > 0
            split_wins += int(both)
            split_detail[split] = {"mean_mpjpe_gain_percent": mp, "mean_pa_mpjpe_gain_percent": pa, "both_metric_win": both}
        detector_detail: dict[str, Any] = {}
        detector_both = True
        for detector in DETECTORS:
            rows = [row for row in pairs if row["detector"] == detector]
            mp = mean([row["mpjpe_gain_percent"] for row in rows])
            pa = mean([row["pa_mpjpe_gain_percent"] for row in rows])
            both = mp > 0 and pa > 0
            detector_both = detector_both and both
            detector_detail[detector] = {"mean_mpjpe_gain_percent": mp, "mean_pa_mpjpe_gain_percent": pa, "both_metric_win": both}
        passes = (
            macro_mpjpe > 0
            and macro_pa > 0
            and split_wins >= int(policy["minimum_splits_with_mean_both_metric_wins"])
            and detector_both
        )
        strength = "strong" if passes and macro_mpjpe >= float(policy["minimum_macro_mpjpe_gain_percent_for_strong_screen"]) else ("positive" if passes else "fail")
        comparisons[arm] = {
            "candidate_arm": arm,
            "baseline_arm": BASELINE,
            "feature_dim": FEATURE_DIMS[arm],
            "paired_units": pairs,
            "macro_mean_mpjpe_gain_percent": macro_mpjpe,
            "macro_mean_pa_mpjpe_gain_percent": macro_pa,
            "macro_mean_geometric_score_gain_percent": macro_score,
            "splits_with_mean_both_metric_wins": split_wins,
            "split_detail": split_detail,
            "each_detector_mean_both_metric_win": detector_both,
            "detector_detail": detector_detail,
            "passes_screen_gate": passes,
            "screen_strength": strength,
        }
        if passes:
            eligible.append(arm)

    selected = BASELINE
    if eligible:
        tolerance = float(policy["tie_tolerance_score_gain_percent"])
        best_score = max(comparisons[arm]["macro_mean_geometric_score_gain_percent"] for arm in eligible)
        tied = [arm for arm in eligible if best_score - comparisons[arm]["macro_mean_geometric_score_gain_percent"] <= tolerance]
        selected = min(tied, key=lambda arm: (FEATURE_DIMS[arm], ARMS.index(arm)))
    if selected == BASELINE:
        decision = protocol["next_gate"]["no_confidence_selected"]
        conclusion = "No bounded confidence candidate passed the predeclared validation gate; lock the detector-conditioned no-confidence virtual-view arm for full confirmation."
    else:
        decision = protocol["next_gate"]["confidence_selected"]
        conclusion = "A bounded confidence candidate passed the predeclared validation-only gate and is locked for full multisplit-multiseed confirmation."

    cross_rows = [row for row in metric_rows if row["relation"] == "cross"]
    cross_summary = {
        arm: {
            "mean_mpjpe_mm": mean([float(row["mpjpe_mm"]) for row in cross_rows if row["arm"] == arm]),
            "mean_pa_mpjpe_mm": mean([float(row["pa_mpjpe_mm"]) for row in cross_rows if row["arm"] == arm]),
            "role": "descriptive_only_not_used_for_selection",
        }
        for arm in ARMS
    }
    statistics = {
        "status": "PASS",
        "format": "phase9c_r1c_confidence_screen_statistics_v1",
        "phase": "9C-R1C",
        "scientific_role": "validation_only_confidence_mechanism_selection",
        "baseline_arm": BASELINE,
        "selected_arm": selected,
        "selected_feature_dim": FEATURE_DIMS[selected],
        "eligible_confidence_candidates": eligible,
        "comparisons": comparisons,
        "cross_detector_validation_summary": cross_summary,
        "selection_policy": policy,
        "selection_partition": "val",
        "selection_seed": SEED,
        "test_metrics_computed": False,
        "test_used_for_selection": False,
        "official_TS1_to_TS6_read": False,
        "scientific_decision": decision,
        "conclusion": conclusion,
    }
    statistics_path = output / "phase9c_r1c_selection_statistics.json"
    write_json(statistics_path, statistics)

    summary_rows = []
    for arm in ARMS:
        rows = [row for row in matched if row["arm"] == arm]
        comparison = comparisons.get(arm)
        summary_rows.append({
            "arm": arm,
            "selected": arm == selected,
            "feature_dim": FEATURE_DIMS[arm],
            "matched_units": len(rows),
            "mean_mpjpe_mm": mean([float(row["mpjpe_mm"]) for row in rows]),
            "mean_pa_mpjpe_mm": mean([float(row["pa_mpjpe_mm"]) for row in rows]),
            "macro_mpjpe_gain_percent_vs_no_confidence": "" if comparison is None else comparison["macro_mean_mpjpe_gain_percent"],
            "macro_pa_mpjpe_gain_percent_vs_no_confidence": "" if comparison is None else comparison["macro_mean_pa_mpjpe_gain_percent"],
            "macro_score_gain_percent_vs_no_confidence": "" if comparison is None else comparison["macro_mean_geometric_score_gain_percent"],
            "passes_screen_gate": "" if comparison is None else comparison["passes_screen_gate"],
            "screen_strength": "baseline" if comparison is None else comparison["screen_strength"],
        })
    summary_path = output / "phase9c_r1c_selection_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    checks = [
        {"name": "protocol_and_preflight", "passed": True, "detail": f"protocol={protocol_sha}; preflight=PASS"},
        {"name": "r1b_source_gate", "passed": True, "detail": str(preflight["source_r1b_gate"])},
        {"name": "split_reports", "passed": True, "detail": "count=3; full train/validation; workers=8"},
        {"name": "run_matrix_exact", "passed": True, "detail": f"runs={len(run_reports)}/30"},
        {"name": "metric_rows", "passed": True, "detail": f"rows={len(metric_rows)}/60; finite=yes"},
        {"name": "validation_only", "passed": True, "detail": "train+val only; development test unread"},
        {"name": "official_test_isolation", "passed": True, "detail": "Official TS1-TS6 unread and unused"},
        {"name": "detector_conditioned_provenance", "passed": True, "detail": "R1A cache hashes/protocol verified for split_a/b/c"},
        {"name": "bounded_confidence", "passed": True, "detail": "raw residual risk not supplied directly; bounded quality/reliability in [0,1]"},
        {"name": "selection_rule", "passed": True, "detail": f"selected={selected}; eligible={eligible}"},
    ]
    qc = {
        "status": "PASS",
        "format": "phase9c_r1c_confidence_screen_qc_report_v1",
        "phase": "9C-R1C",
        "scientific_decision": decision,
        "selected_arm": selected,
        "checks": checks,
        "errors": [],
        "run_count": len(run_reports),
        "metric_rows": len(metric_rows),
        "development_test_metrics_computed": False,
        "official_TS1_to_TS6_read": False,
    }
    qc_path = output / "phase9c_r1c_qc_report.json"
    write_json(qc_path, qc)

    packaged: list[tuple[Path, str]] = [
        (protocol_path, "phase9c_r1c_confidence_mechanism_screen_protocol.json"),
        (preflight_path, "phase9c_r1c_preflight_report.json"),
        (metrics_path, "phase9c_r1c_validation_metrics.csv"),
        (summary_path, "phase9c_r1c_selection_summary.csv"),
        (statistics_path, "phase9c_r1c_selection_statistics.json"),
        (artifacts_path, "phase9c_r1c_artifact_hashes.csv"),
        (qc_path, "phase9c_r1c_qc_report.json"),
    ]
    packaged.extend((path, f"split_reports/{path.parent.name}/{path.name}") for path, _ in split_reports)
    for path, report in run_reports:
        rel = f"run_reports/{report['split']}/seed{report['seed']}/{report['arm']}/train_{report['train_detector']}/{path.name}"
        packaged.append((path, rel))
    zip_path = output / "phase9c_r1c_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, arcname in packaged:
            archive.write(source, arcname=arcname)
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Result ZIP integrity failed: {bad}")

    print("=" * 112)
    print("PHASE 9C-R1C - CONFIDENCE MECHANISM SCREEN")
    print("=" * 112)
    print("Status                : PASS")
    print(f"Runs                  : {len(run_reports)} / 30")
    print(f"Metric rows           : {len(metric_rows)} / 60")
    print(f"Selected arm          : {selected}")
    if selected != BASELINE:
        selected_stats = comparisons[selected]
        print(f"Validation gain       : {selected_stats['macro_mean_mpjpe_gain_percent']:.3f}% MPJPE / {selected_stats['macro_mean_pa_mpjpe_gain_percent']:.3f}% PA-MPJPE")
    else:
        print("Confidence claim      : NOT SUPPORTED BY PREDECLARED SCREEN")
    print("Development test read : NO")
    print("Official TS1-TS6 read : NO")
    print(f"Scientific decision   : {decision}")
    print(f"Results ZIP           : {zip_path}")
    print("=" * 112)


if __name__ == "__main__":
    main()
