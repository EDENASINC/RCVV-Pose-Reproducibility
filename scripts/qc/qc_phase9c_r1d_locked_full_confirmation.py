from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("split_a", "split_b", "split_c")
SEEDS = (42, 123, 2026)
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
ARMS = (
    "detector_aware_observed",
    "detector_conditioned_virtual_view_no_confidence",
    "bounded_reliability_dual_modulation",
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


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def relation(row: dict[str, Any]) -> str:
    return "matched" if row["train_detector"] == row["eval_detector"] else "cross"


def hierarchical_bootstrap(
    pairs: list[dict[str, Any]],
    metric: str,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    nested: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in pairs:
        nested[str(row["split"])][int(row["seed"])][str(row["detector"])] = float(row[metric])
    split_names = sorted(nested)
    if split_names != list(SPLITS):
        raise ValueError(f"Bootstrap split coverage mismatch: {split_names}")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        values: list[float] = []
        for _ in range(len(split_names)):
            split = split_names[int(rng.integers(len(split_names)))]
            seed_names = sorted(nested[split])
            for _ in range(len(seed_names)):
                selected_seed = seed_names[int(rng.integers(len(seed_names)))]
                detectors = sorted(nested[split][selected_seed])
                for _ in range(len(detectors)):
                    detector = detectors[int(rng.integers(len(detectors)))]
                    values.append(nested[split][selected_seed][detector])
        draws[replicate] = float(np.mean(values))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def paired_comparison(
    metrics_index: dict[tuple[Any, ...], dict[str, Any]],
    *,
    challenger: str,
    baseline: str,
    mode: str,
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for split in SPLITS:
        for seed in SEEDS:
            for train_detector in DETECTORS:
                eval_detector = (
                    train_detector if mode == "matched"
                    else DETECTORS[1 - DETECTORS.index(train_detector)]
                )
                key_base = (split, seed, baseline, train_detector, eval_detector, "test")
                key_challenger = (split, seed, challenger, train_detector, eval_detector, "test")
                if key_base not in metrics_index or key_challenger not in metrics_index:
                    raise KeyError(f"Missing paired metric: {key_base} / {key_challenger}")
                base = metrics_index[key_base]
                chal = metrics_index[key_challenger]
                mp_gain = float(base["mpjpe_mm"]) - float(chal["mpjpe_mm"])
                pa_gain = float(base["pa_mpjpe_mm"]) - float(chal["pa_mpjpe_mm"])
                pairs.append(
                    {
                        "split": split,
                        "seed": seed,
                        "detector": train_detector,
                        "eval_detector": eval_detector,
                        "baseline_mpjpe_mm": float(base["mpjpe_mm"]),
                        "challenger_mpjpe_mm": float(chal["mpjpe_mm"]),
                        "baseline_pa_mpjpe_mm": float(base["pa_mpjpe_mm"]),
                        "challenger_pa_mpjpe_mm": float(chal["pa_mpjpe_mm"]),
                        "mpjpe_gain_mm": mp_gain,
                        "pa_mpjpe_gain_mm": pa_gain,
                        "mpjpe_gain_percent": 100.0 * mp_gain / float(base["mpjpe_mm"]),
                        "pa_mpjpe_gain_percent": 100.0 * pa_gain / float(base["pa_mpjpe_mm"]),
                    }
                )
    result: dict[str, Any] = {
        "challenger": challenger,
        "baseline": baseline,
        "evaluation_mode": mode,
        "partition": "development_test_post_selection",
        "pair_count": len(pairs),
        "pairs": pairs,
    }
    for metric in (
        "mpjpe_gain_mm",
        "pa_mpjpe_gain_mm",
        "mpjpe_gain_percent",
        "pa_mpjpe_gain_percent",
    ):
        values = np.asarray([float(row[metric]) for row in pairs], dtype=np.float64)
        low, high = hierarchical_bootstrap(
            pairs, metric, replicates, bootstrap_seed + len(metric) + (0 if mode == "matched" else 1000)
        )
        result[metric] = {
            "mean": float(values.mean()),
            "std_across_18_units": float(values.std(ddof=1)),
            "hierarchical_bootstrap_95ci": [low, high],
        }
    split_summary: dict[str, Any] = {}
    for split in SPLITS:
        rows = [item for item in pairs if item["split"] == split]
        mp = float(np.mean([item["mpjpe_gain_mm"] for item in rows]))
        pa = float(np.mean([item["pa_mpjpe_gain_mm"] for item in rows]))
        split_summary[split] = {
            "mpjpe_gain_mm": mp,
            "pa_mpjpe_gain_mm": pa,
            "both_metrics_improved": mp > 0 and pa > 0,
        }
    detector_summary: dict[str, Any] = {}
    for detector in DETECTORS:
        rows = [item for item in pairs if item["detector"] == detector]
        mp = float(np.mean([item["mpjpe_gain_mm"] for item in rows]))
        pa = float(np.mean([item["pa_mpjpe_gain_mm"] for item in rows]))
        detector_summary[detector] = {
            "mpjpe_gain_mm": mp,
            "pa_mpjpe_gain_mm": pa,
            "both_metrics_improved": mp > 0 and pa > 0,
        }
    result["split_summary"] = split_summary
    result["detector_summary"] = detector_summary
    result["splits_with_both_metric_wins"] = sum(
        int(item["both_metrics_improved"]) for item in split_summary.values()
    )
    result["detectors_with_both_metric_wins"] = sum(
        int(item["both_metrics_improved"]) for item in detector_summary.values()
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="QC and aggregate Phase 9C-R1D.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs/phase9c_r1d_locked_full_confirmation"
    protocol_path = root / "configs/phase9c_r1d_locked_full_confirmation_protocol.json"
    preflight_path = output / "phase9c_r1d_preflight_report.json"
    qc_path = output / "phase9c_r1d_qc_report.json"
    metrics_path = output / "phase9c_r1d_full_metrics.csv"
    summary_path = output / "phase9c_r1d_summary.csv"
    statistics_path = output / "phase9c_r1d_statistics.json"
    hashes_path = output / "phase9c_r1d_artifact_hashes.csv"
    zip_path = output / "phase9c_r1d_results.zip"
    checks: list[dict[str, Any]] = []
    protocol = load_json(protocol_path)
    preflight = load_json(preflight_path)
    protocol_sha = sha256_file(protocol_path)
    add(
        checks,
        "protocol_and_preflight",
        protocol.get("status") == "LOCKED"
        and protocol.get("format") == "phase9c_r1d_locked_full_confirmation_protocol_v1"
        and preflight.get("status") == "PASS"
        and preflight.get("protocol_sha256") == protocol_sha,
        f"protocol={protocol.get('status')}; preflight={preflight.get('status')}",
    )
    split_reports: list[tuple[Path, dict[str, Any]]] = []
    for split in SPLITS:
        path = output / split / "phase9c_r1d_split_report.json"
        if path.is_file():
            split_reports.append((path, load_json(path)))
    add(checks, "split_reports", len(split_reports) == 3, f"count={len(split_reports)}")
    split_integrity = len(split_reports) == 3
    risk_mapping = protocol.get("calibration_and_risk_policy", {}).get(
        "split_to_residual_risk_index", {}
    )
    for _, report in split_reports:
        split_name = str(report.get("split"))
        split_integrity &= (
            split_name in SPLITS
            and report.get("status") == "PASS"
            and report.get("format") == "phase9c_r1d_split_report_v1"
            and int(report.get("run_count", -1)) == 18
            and tuple(int(value) for value in report.get("seeds", ())) == SEEDS
            and int(report.get("workers", -1)) == 8
            and report.get("test_metrics_computed") is True
            and report.get("test_used_for_selection") is False
            and report.get("protocol_sha256") == protocol_sha
            and report.get("official_TS1_to_TS6_read") is False
            and report.get("official_TS1_to_TS6_used") is False
            and report.get("residual_risk_split") == split_name
            and int(report.get("residual_risk_source_index", -1))
            == int(risk_mapping.get(split_name, -2))
        )
        sizes = report.get("dataset_sizes", {})
        alignment = report.get("dataset_alignment", {})
        for partition in ("train", "val", "test"):
            split_integrity &= int(sizes.get(partition, -1)) == int(
                alignment.get(partition, {}).get("selected_samples", -2)
            )
    add(checks, "full_cohort_and_worker_integrity", split_integrity, "3 splits; workers=8; no sample limits")
    report_paths = sorted(output.glob("split_*/seed*/**/phase9c_r1d_run_report.json"))
    reports = [(path, load_json(path)) for path in report_paths]
    add(checks, "run_report_count", len(reports) == 54, f"count={len(reports)}")
    expected_specs = set()
    for split in SPLITS:
        for seed in SEEDS:
            for arm in ARMS:
                for detector in DETECTORS:
                    expected_specs.add((split, seed, arm, detector))
    actual_specs = {
        (str(item.get("split")), int(item.get("seed", -1)), str(item.get("arm")), str(item.get("train_detector")))
        for _, item in reports
    }
    add(
        checks,
        "run_matrix_exact",
        actual_specs == expected_specs,
        f"missing={len(expected_specs - actual_specs)}; extra={len(actual_specs - expected_specs)}",
    )
    reports_ok = len(reports) == 54
    artifact_rows: list[dict[str, str]] = []
    metric_rows: list[dict[str, Any]] = []
    index_digest_by_unit: dict[tuple[str, int, str, str, str], str] = {}
    per_sample_ok = True
    for report_path, report in reports:
        reports_ok &= (
            report.get("status") == "PASS"
            and report.get("format") == "phase9c_r1d_run_report_v1"
            and report.get("protocol_sha256") == protocol_sha
            and report.get("test_used_for_selection") is False
            and report.get("test_metrics_computed") is True
            and report.get("checkpoint_selection")
            == "matched_validation_geometric_mean_mpjpe_pa_mpjpe"
            and report.get("official_TS1_to_TS6_read") is False
            and report.get("official_TS1_to_TS6_used") is False
            and 1 <= int(report.get("best_epoch", 0)) <= 50
            and int(report.get("best_epoch", 0))
            <= int(report.get("epochs_completed", 0))
            and 1 <= int(report.get("epochs_completed", 0)) <= 50
        )
        run_artifacts = report.get("artifact_hashes", [])
        expected_artifact_count = 7
        reports_ok &= len(run_artifacts) == expected_artifact_count
        run_artifact_map = {
            str(item.get("path", "")): str(item.get("sha256", ""))
            for item in run_artifacts
        }
        # R1D writes epoch history as a separately hashed CSV artifact.  The v1
        # QC incorrectly expected the same rows to also be embedded in the run
        # report JSON, so every otherwise valid run failed integrity.  Validate
        # the authoritative CSV artifact instead; do not alter any scientific
        # metric, checkpoint choice, or gate.
        history_path_text = next(
            (
                path_text
                for path_text in run_artifact_map
                if Path(path_text).name.lower() == "training_history.csv"
            ),
            "",
        )
        history: list[dict[str, str]] = []
        try:
            with Path(history_path_text).open("r", newline="", encoding="utf-8-sig") as handle:
                history = list(csv.DictReader(handle))
        except Exception:
            reports_ok = False
        reports_ok &= len(history) == int(report.get("epochs_completed", -1))
        reports_ok &= all(
            finite(item.get("train_loss"))
            and finite(item.get("val_mpjpe_mm"))
            and finite(item.get("val_pa_mpjpe_mm"))
            and finite(item.get("val_selection_score"))
            for item in history
        )
        for artifact in run_artifacts:
            path = Path(str(artifact.get("path", "")))
            expected = str(artifact.get("sha256", ""))
            valid = path.is_file() and sha256_file(path) == expected
            reports_ok &= valid
            artifact_rows.append(
                {
                    "run_report": str(report_path),
                    "path": str(path),
                    "sha256": expected,
                    "verified": str(bool(valid)),
                }
            )
        evaluations = report.get("evaluations", {})
        expected_eval = set(DETECTORS)
        reports_ok &= set(evaluations) == expected_eval
        for eval_detector, partitions in evaluations.items():
            reports_ok &= set(partitions) == {"val", "test"}
            for partition, metrics in partitions.items():
                per_sample_path = str(metrics.get("per_sample_path", ""))
                reports_ok &= (
                    run_artifact_map.get(per_sample_path)
                    == str(metrics.get("per_sample_sha256", ""))
                )
                per_sample_file = Path(per_sample_path)
                try:
                    with np.load(per_sample_file, allow_pickle=False) as payload:
                        sample_index = payload["detector_sample_index"].astype(np.int64, copy=False)
                        sample_mpjpe = payload["mpjpe_mm"].astype(np.float64, copy=False)
                        sample_pa = payload["pa_mpjpe_mm"].astype(np.float64, copy=False)
                    per_sample_ok &= (
                        sample_index.ndim == 1
                        and sample_mpjpe.shape == sample_index.shape
                        and sample_pa.shape == sample_index.shape
                        and np.isfinite(sample_mpjpe).all()
                        and np.isfinite(sample_pa).all()
                        and int(sample_index.size) == int(metrics["samples"])
                        and abs(float(sample_mpjpe.mean()) - float(metrics["mpjpe_mm"])) <= 1e-4
                        and abs(float(sample_pa.mean()) - float(metrics["pa_mpjpe_mm"])) <= 1e-4
                    )
                    unit = (
                        str(report["split"]), int(report["seed"]), str(report["train_detector"]),
                        str(eval_detector), str(partition),
                    )
                    index_digest = hashlib.sha256(sample_index.tobytes()).hexdigest()
                    previous = index_digest_by_unit.setdefault(unit, index_digest)
                    per_sample_ok &= previous == index_digest
                except Exception:
                    per_sample_ok = False
                row = {
                    "split": report["split"],
                    "seed": int(report["seed"]),
                    "arm": report["arm"],
                    "train_detector": report["train_detector"],
                    "eval_detector": eval_detector,
                    "partition": partition,
                    "relation": "",
                    "mpjpe_mm": float(metrics["mpjpe_mm"]),
                    "pa_mpjpe_mm": float(metrics["pa_mpjpe_mm"]),
                    "samples": int(metrics["samples"]),
                    "best_epoch": int(report["best_epoch"]),
                    "epochs_completed": int(report["epochs_completed"]),
                }
                row["relation"] = relation(row)
                reports_ok &= finite(row["mpjpe_mm"]) and finite(row["pa_mpjpe_mm"])
                split_report = next(
                    (
                        item
                        for _, item in split_reports
                        if item.get("split") == report.get("split")
                    ),
                    {},
                )
                reports_ok &= row["samples"] == int(
                    split_report.get("dataset_sizes", {}).get(partition, -1)
                )
                metric_rows.append(row)
    add(checks, "run_and_artifact_integrity", reports_ok, f"artifacts={len(artifact_rows)}")
    add(checks, "metric_rows_finite", len(metric_rows) == 216, f"rows={len(metric_rows)}")
    add(
        checks,
        "per_sample_pairing_and_metric_reconstruction",
        per_sample_ok and len(index_digest_by_unit) == 72,
        f"paired_units={len(index_digest_by_unit)}/72",
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(metric_rows[0]) if metric_rows else ["error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    with hashes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_report", "path", "sha256", "verified"])
        writer.writeheader()
        writer.writerows(artifact_rows)
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(row["arm"], row["relation"], row["partition"], row["train_detector"], row["eval_detector"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        mp = np.asarray([item["mpjpe_mm"] for item in rows], dtype=np.float64)
        pa = np.asarray([item["pa_mpjpe_mm"] for item in rows], dtype=np.float64)
        summary_rows.append(
            {
                "arm": key[0],
                "relation": key[1],
                "partition": key[2],
                "train_detector": key[3],
                "eval_detector": key[4],
                "runs": len(rows),
                "mean_mpjpe_mm": float(mp.mean()),
                "std_mpjpe_mm": float(mp.std(ddof=1)) if len(mp) > 1 else 0.0,
                "mean_pa_mpjpe_mm": float(pa.mean()),
                "std_pa_mpjpe_mm": float(pa.std(ddof=1)) if len(pa) > 1 else 0.0,
            }
        )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]) if summary_rows else ["error"])
        writer.writeheader()
        writer.writerows(summary_rows)
    metrics_index = {
        (
            row["split"], row["seed"], row["arm"], row["train_detector"],
            row["eval_detector"], row["partition"],
        ): row
        for row in metric_rows
    }
    statistics_policy = protocol["statistical_policy"]
    replicates = int(statistics_policy["bootstrap_replicates"])
    bootstrap_seed = int(statistics_policy["bootstrap_seed"])
    comparisons: dict[str, Any] = {}
    specifications = (
        ("full_method_vs_observed", "bounded_reliability_dual_modulation", "detector_aware_observed"),
        ("confidence_vs_no_confidence", "bounded_reliability_dual_modulation", "detector_conditioned_virtual_view_no_confidence"),
        ("virtual_view_vs_observed", "detector_conditioned_virtual_view_no_confidence", "detector_aware_observed"),
    )
    stats_ok = len(metric_rows) == 216
    if stats_ok:
        for name, challenger, baseline in specifications:
            for mode in ("matched", "cross"):
                comparisons[f"{name}_{mode}"] = paired_comparison(
                    metrics_index,
                    challenger=challenger,
                    baseline=baseline,
                    mode=mode,
                    replicates=replicates,
                    bootstrap_seed=bootstrap_seed,
                )
    add(checks, "paired_statistics_complete", stats_ok and len(comparisons) == 6, f"comparisons={len(comparisons)}")
    decision = "STOP_AND_REASSESS_BEFORE_OFFICIAL_TEST"
    gate_detail: dict[str, Any] = {}
    if stats_ok:
        primary = comparisons["full_method_vs_observed_matched"]
        additive = comparisons["confidence_vs_no_confidence_matched"]
        virtual = comparisons["virtual_view_vs_observed_matched"]
        primary_mp = primary["mpjpe_gain_mm"]
        primary_pa = primary["pa_mpjpe_gain_mm"]
        primary_pct = primary["mpjpe_gain_percent"]["mean"]
        add_mp = additive["mpjpe_gain_mm"]
        add_pa = additive["pa_mpjpe_gain_mm"]
        primary_positive = (
            primary_mp["mean"] > 0
            and primary_pa["mean"] > 0
            and primary["splits_with_both_metric_wins"] >= 2
            and primary["detectors_with_both_metric_wins"] == 2
        )
        primary_ci_positive = (
            primary_mp["hierarchical_bootstrap_95ci"][0] > 0
            and primary_pa["hierarchical_bootstrap_95ci"][0] > 0
        )
        additive_positive = add_mp["mean"] > 0 and add_pa["mean"] > 0
        additive_ci_positive = (
            add_mp["hierarchical_bootstrap_95ci"][0] > 0
            and add_pa["hierarchical_bootstrap_95ci"][0] > 0
        )
        virtual_mp = virtual["mpjpe_gain_mm"]
        virtual_pa = virtual["pa_mpjpe_gain_mm"]
        virtual_positive = (
            virtual_mp["mean"] > 0
            and virtual_pa["mean"] > 0
            and virtual["splits_with_both_metric_wins"] >= 2
            and virtual["detectors_with_both_metric_wins"] == 2
        )
        virtual_ci_positive = (
            virtual_mp["hierarchical_bootstrap_95ci"][0] > 0
            and virtual_pa["hierarchical_bootstrap_95ci"][0] > 0
        )
        strong = (
            primary_positive
            and primary_ci_positive
            and primary_pct >= float(statistics_policy["minimum_primary_mpjpe_gain_percent"])
            and additive_positive
            and additive_ci_positive
            and virtual_positive
            and virtual_ci_positive
        )
        cautionary = primary_positive and additive_positive and virtual_positive
        if strong:
            decision = protocol["next_gate"]["strong"]
        elif cautionary:
            decision = protocol["next_gate"]["cautionary"]
        gate_detail = {
            "primary_positive": primary_positive,
            "primary_ci_positive_both_metrics": primary_ci_positive,
            "primary_macro_mpjpe_gain_percent": primary_pct,
            "primary_reaches_practical_target": primary_pct >= float(statistics_policy["minimum_primary_mpjpe_gain_percent"]),
            "confidence_ablation_positive": additive_positive,
            "confidence_ablation_ci_positive_both_metrics": additive_ci_positive,
            "virtual_view_ablation_positive": virtual_positive,
            "virtual_view_ablation_ci_positive_both_metrics": virtual_ci_positive,
            "strong_gate": strong,
            "cautionary_gate": cautionary,
        }
    statistics = {
        "status": "PASS" if stats_ok else "FAIL",
        "format": "phase9c_r1d_statistics_v1",
        "phase": "9C-R1D",
        "scientific_role": "development_post_selection_robustness_not_official_test",
        "bootstrap": {
            "replicates": replicates,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "hierarchy": ["split", "seed", "detector"],
        },
        "comparisons": comparisons,
        "gate_detail": gate_detail,
        "scientific_decision": decision,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    write_json(statistics_path, statistics)
    errors = [item["name"] for item in checks if not item["passed"]]
    status = "PASS" if not errors else "FAIL"
    qc = {
        "status": status,
        "format": "phase9c_r1d_qc_report_v2_evidence_contract_repair",
        "phase": "9C-R1D",
        "repair_note": "training_history.csv is the authoritative hashed epoch-history artifact; no training result or scientific gate was changed",
        "scientific_decision": decision if status == "PASS" else "REPAIR_PHASE9C_R1D_EVIDENCE",
        "checks": checks,
        "errors": errors,
        "run_reports": len(reports),
        "metric_rows": len(metric_rows),
        "artifact_hash_rows": len(artifact_rows),
        "protocol_sha256": protocol_sha,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    write_json(qc_path, qc)
    archive_inputs: list[tuple[Path, str]] = [
        (protocol_path, "phase9c_r1d_locked_full_confirmation_protocol.json"),
        (preflight_path, "phase9c_r1d_preflight_report.json"),
        (metrics_path, metrics_path.name),
        (summary_path, summary_path.name),
        (statistics_path, statistics_path.name),
        (hashes_path, hashes_path.name),
        (qc_path, qc_path.name),
    ]
    for path, _ in split_reports:
        archive_inputs.append((path, str(Path("split_reports") / path.parent.name / path.name)))
    for path, _ in reports:
        relative = path.relative_to(output)
        archive_inputs.append((path, str(Path("run_reports") / relative)))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, arcname in archive_inputs:
            archive.write(source, arcname)
    print("=" * 112)
    print("PHASE 9C-R1D - QC, STATISTICS AND PACK")
    print("=" * 112)
    print(f"Status                : {status}")
    print(f"Run reports           : {len(reports)} / 54")
    print(f"Metric rows           : {len(metric_rows)} / 216")
    print("Workers               : 8")
    print("Official TS1-TS6 read: NO")
    print(f"Scientific decision   : {qc['scientific_decision']}")
    print(f"Output ZIP            : {zip_path}")
    if gate_detail:
        print(
            "Primary matched gain  : "
            f"{gate_detail['primary_macro_mpjpe_gain_percent']:.3f}% MPJPE"
        )
    if errors:
        print(f"Errors                : {errors}")
    print("=" * 112)
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
