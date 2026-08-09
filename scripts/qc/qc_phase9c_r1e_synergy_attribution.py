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
NEW_ARM = "bounded_reliability_observed_modulation"
OBSERVED = "detector_aware_observed"
UNGATED_VIRTUAL = "detector_conditioned_virtual_view_no_confidence"
FULL = "bounded_reliability_dual_modulation"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def hierarchical_bootstrap(rows: list[dict[str, Any]], key: str, replicates: int, seed: int) -> list[float]:
    nested: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        nested[str(row["split"])][int(row["seed"])][str(row["detector"])] = float(row[key])
    if sorted(nested) != list(SPLITS):
        raise ValueError("Bootstrap split coverage mismatch.")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    split_names = list(SPLITS)
    for replicate in range(replicates):
        values: list[float] = []
        for _ in SPLITS:
            split = split_names[int(rng.integers(len(split_names)))]
            seed_names = sorted(nested[split])
            for _ in seed_names:
                selected_seed = seed_names[int(rng.integers(len(seed_names)))]
                detector_names = sorted(nested[split][selected_seed])
                for _ in detector_names:
                    detector = detector_names[int(rng.integers(len(detector_names)))]
                    values.append(nested[split][selected_seed][detector])
        draws[replicate] = float(np.mean(values))
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def summarize_effect(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pair_count": len(rows), "pairs": rows}
    for offset, key in enumerate(("mpjpe_gain_mm", "pa_mpjpe_gain_mm")):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "std_across_18_units": float(values.std(ddof=1)),
            "hierarchical_bootstrap_95ci": hierarchical_bootstrap(rows, key, replicates, seed + offset),
        }
    split_summary: dict[str, Any] = {}
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        mp = float(np.mean([row["mpjpe_gain_mm"] for row in subset]))
        pa = float(np.mean([row["pa_mpjpe_gain_mm"] for row in subset]))
        split_summary[split] = {"mpjpe_gain_mm": mp, "pa_mpjpe_gain_mm": pa, "both_metrics_improved": mp > 0 and pa > 0}
    detector_summary: dict[str, Any] = {}
    for detector in DETECTORS:
        subset = [row for row in rows if row["detector"] == detector]
        mp = float(np.mean([row["mpjpe_gain_mm"] for row in subset]))
        pa = float(np.mean([row["pa_mpjpe_gain_mm"] for row in subset]))
        detector_summary[detector] = {"mpjpe_gain_mm": mp, "pa_mpjpe_gain_mm": pa, "both_metrics_improved": mp > 0 and pa > 0}
    result["split_summary"] = split_summary
    result["detector_summary"] = detector_summary
    result["splits_with_both_metric_wins"] = sum(int(row["both_metrics_improved"]) for row in split_summary.values())
    result["detectors_with_both_metric_wins"] = sum(int(row["both_metrics_improved"]) for row in detector_summary.values())
    return result


def matched_test_metrics(report: dict[str, Any]) -> tuple[float, float, str]:
    detector = str(report["train_detector"])
    metrics = report["evaluations"][detector]["test"]
    return float(metrics["mpjpe_mm"]), float(metrics["pa_mpjpe_mm"]), str(metrics["per_sample_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="QC Phase 9C-R1E factorial synergy attribution.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs/phase9c_r1e_synergy_attribution"
    r1d_output = root / "outputs/phase9c_r1d_locked_full_confirmation"
    protocol_path = root / "configs/phase9c_r1e_synergy_attribution_protocol.json"
    protocol = load_json(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    preflight_path = output / "phase9c_r1e_preflight_report.json"
    preflight = load_json(preflight_path)
    r1d_qc_path = r1d_output / "phase9c_r1d_qc_report.json"
    r1d_stats_path = r1d_output / "phase9c_r1d_statistics.json"
    r1d_qc = load_json(r1d_qc_path)
    r1d_stats = load_json(r1d_stats_path)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(name)

    check(
        "protocol_preflight_and_r1d_evidence",
        protocol.get("status") == "LOCKED"
        and preflight.get("status") == "PASS"
        and preflight.get("protocol_sha256") == protocol_sha
        and r1d_qc.get("status") == "PASS"
        and r1d_qc.get("format") == "phase9c_r1d_qc_report_v2_evidence_contract_repair"
        and r1d_stats.get("scientific_decision") == "STOP_AND_REASSESS_BEFORE_OFFICIAL_TEST",
        f"preflight={preflight.get('status')}; r1d_qc={r1d_qc.get('status')}; r1d_gate={r1d_stats.get('scientific_decision')}",
    )

    split_reports = [output / split / "phase9c_r1e_split_report.json" for split in SPLITS]
    split_ok = all(path.is_file() for path in split_reports)
    if split_ok:
        for path in split_reports:
            report = load_json(path)
            split_ok &= (
                report.get("status") == "PASS"
                and report.get("format") == "phase9c_r1e_split_report_v1"
                and int(report.get("run_count", -1)) == 6
                and int(report.get("workers", -1)) == 8
                and report.get("protocol_sha256") == protocol_sha
                and report.get("test_used_for_selection") is False
                and report.get("official_TS1_to_TS6_read") is False
            )
    check("split_reports", split_ok, "expected=3 reports and 6 runs per split")

    new_paths = sorted(output.glob("split_*/seed*/**/phase9c_r1e_run_report.json"))
    new_reports = [load_json(path) for path in new_paths]
    expected = {(split, seed, NEW_ARM, detector) for split in SPLITS for seed in SEEDS for detector in DETECTORS}
    actual = {(str(report.get("split")), int(report.get("seed", -1)), str(report.get("arm")), str(report.get("train_detector"))) for report in new_reports}
    check("new_run_matrix_exact", len(new_reports) == 18 and actual == expected, f"reports={len(new_reports)} missing={len(expected-actual)} extra={len(actual-expected)}")

    artifact_rows: list[dict[str, Any]] = []
    new_integrity = len(new_reports) == 18
    new_index: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    sample_digest: dict[tuple[str, int, str], str] = {}
    for path, report in zip(new_paths, new_reports):
        new_integrity &= (
            report.get("status") == "PASS"
            and report.get("format") == "phase9c_r1e_run_report_v1"
            and report.get("protocol_sha256") == protocol_sha
            and report.get("checkpoint_selection") == "matched_validation_geometric_mean_mpjpe_pa_mpjpe"
            and report.get("test_used_for_selection") is False
            and report.get("official_TS1_to_TS6_read") is False
            and int(report.get("feature_dim", -1)) == 70
        )
        artifacts = report.get("artifact_hashes", [])
        new_integrity &= len(artifacts) == 7
        history_rows: list[dict[str, str]] = []
        for artifact in artifacts:
            artifact_path = Path(str(artifact.get("path", "")))
            expected_hash = str(artifact.get("sha256", ""))
            valid = artifact_path.is_file() and sha256_file(artifact_path) == expected_hash
            new_integrity &= valid
            artifact_rows.append({"run_report": str(path), "path": str(artifact_path), "sha256": expected_hash, "verified": str(bool(valid))})
            if artifact_path.name.lower() == "training_history.csv" and valid:
                with artifact_path.open("r", newline="", encoding="utf-8-sig") as handle:
                    history_rows = list(csv.DictReader(handle))
        new_integrity &= len(history_rows) == int(report.get("epochs_completed", -1))
        new_integrity &= all(finite(row.get(key)) for row in history_rows for key in ("train_loss", "val_mpjpe_mm", "val_pa_mpjpe_mm", "val_selection_score"))
        mp, pa, per_sample_path = matched_test_metrics(report)
        new_integrity &= finite(mp) and finite(pa)
        try:
            with np.load(Path(per_sample_path), allow_pickle=False) as payload:
                indices = payload["detector_sample_index"].astype(np.int64, copy=False)
                sample_mp = payload["mpjpe_mm"].astype(np.float64, copy=False)
                sample_pa = payload["pa_mpjpe_mm"].astype(np.float64, copy=False)
            new_integrity &= indices.shape == sample_mp.shape == sample_pa.shape
            new_integrity &= abs(float(sample_mp.mean()) - mp) <= 1e-4 and abs(float(sample_pa.mean()) - pa) <= 1e-4
            sample_digest[(str(report["split"]), int(report["seed"]), str(report["train_detector"]))] = hashlib.sha256(indices.tobytes()).hexdigest()
        except Exception:
            new_integrity = False
        new_index[(str(report["split"]), int(report["seed"]), NEW_ARM, str(report["train_detector"]))] = report
    check("new_run_and_artifact_integrity", new_integrity and len(artifact_rows) == 126, f"artifacts={len(artifact_rows)}/126")

    r1d_paths = sorted(r1d_output.glob("split_*/seed*/**/phase9c_r1d_run_report.json"))
    r1d_reports = [load_json(path) for path in r1d_paths]
    r1d_index = {(str(report["split"]), int(report["seed"]), str(report["arm"]), str(report["train_detector"])): report for report in r1d_reports}
    source_ok = len(r1d_reports) == 54
    pairing_ok = len(sample_digest) == 18
    for split in SPLITS:
        for seed in SEEDS:
            for detector in DETECTORS:
                unit = (split, seed, detector)
                for arm in (OBSERVED, UNGATED_VIRTUAL, FULL):
                    report = r1d_index.get((split, seed, arm, detector))
                    source_ok &= report is not None
                    if report is None:
                        continue
                    _, _, per_sample_path = matched_test_metrics(report)
                    try:
                        with np.load(Path(per_sample_path), allow_pickle=False) as payload:
                            indices = payload["detector_sample_index"].astype(np.int64, copy=False)
                        pairing_ok &= hashlib.sha256(indices.tobytes()).hexdigest() == sample_digest.get(unit)
                    except Exception:
                        pairing_ok = False
    check("r1d_source_matrix", source_ok, f"reports={len(r1d_reports)}/54")
    check("four_arm_sample_pairing", pairing_ok, "paired_units=18/18 across O, OV, OR and OVR")

    conditional_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    if source_ok and len(new_index) == 18:
        for split in SPLITS:
            for seed in SEEDS:
                for detector in DETECTORS:
                    observed = matched_test_metrics(r1d_index[(split, seed, OBSERVED, detector)])
                    ungated = matched_test_metrics(r1d_index[(split, seed, UNGATED_VIRTUAL, detector)])
                    reliability = matched_test_metrics(new_index[(split, seed, NEW_ARM, detector)])
                    full = matched_test_metrics(r1d_index[(split, seed, FULL, detector)])
                    base = {"split": split, "seed": seed, "detector": detector}
                    conditional = {**base, "mpjpe_gain_mm": reliability[0]-full[0], "pa_mpjpe_gain_mm": reliability[1]-full[1]}
                    reliability_effect = {**base, "mpjpe_gain_mm": observed[0]-reliability[0], "pa_mpjpe_gain_mm": observed[1]-reliability[1]}
                    ungated_virtual_mp = observed[0]-ungated[0]
                    ungated_virtual_pa = observed[1]-ungated[1]
                    interaction = {
                        **base,
                        "mpjpe_gain_mm": conditional["mpjpe_gain_mm"]-ungated_virtual_mp,
                        "pa_mpjpe_gain_mm": conditional["pa_mpjpe_gain_mm"]-ungated_virtual_pa,
                        "conditional_virtual_mpjpe_gain_mm": conditional["mpjpe_gain_mm"],
                        "ungated_virtual_mpjpe_gain_mm": ungated_virtual_mp,
                    }
                    conditional_rows.append(conditional)
                    reliability_rows.append(reliability_effect)
                    interaction_rows.append(interaction)

    replicates = int(protocol["statistical_policy"]["bootstrap_replicates"])
    bootstrap_seed = int(protocol["statistical_policy"]["bootstrap_seed"])
    statistics_ok = len(conditional_rows) == len(reliability_rows) == len(interaction_rows) == 18
    comparisons: dict[str, Any] = {}
    decision = protocol["next_gate"]["redesign"]
    gate: dict[str, Any] = {}
    if statistics_ok:
        comparisons["conditional_virtual_contribution_full_vs_reliability_observed"] = summarize_effect(conditional_rows, replicates, bootstrap_seed)
        comparisons["reliability_only_contribution_vs_observed"] = summarize_effect(reliability_rows, replicates, bootstrap_seed + 100)
        comparisons["virtual_reliability_interaction_difference_in_differences"] = summarize_effect(interaction_rows, replicates, bootstrap_seed + 200)
        primary = r1d_stats["comparisons"]["full_method_vs_observed_matched"]
        conditional = comparisons["conditional_virtual_contribution_full_vs_reliability_observed"]
        interaction = comparisons["virtual_reliability_interaction_difference_in_differences"]
        primary_ci = primary["mpjpe_gain_mm"]["hierarchical_bootstrap_95ci"][0] > 0 and primary["pa_mpjpe_gain_mm"]["hierarchical_bootstrap_95ci"][0] > 0
        conditional_positive = conditional["mpjpe_gain_mm"]["mean"] > 0 and conditional["pa_mpjpe_gain_mm"]["mean"] > 0
        conditional_ci = conditional["mpjpe_gain_mm"]["hierarchical_bootstrap_95ci"][0] > 0 and conditional["pa_mpjpe_gain_mm"]["hierarchical_bootstrap_95ci"][0] > 0
        interaction_positive = interaction["mpjpe_gain_mm"]["mean"] > 0 and interaction["pa_mpjpe_gain_mm"]["mean"] > 0
        interaction_ci = interaction["mpjpe_gain_mm"]["hierarchical_bootstrap_95ci"][0] > 0 and interaction["pa_mpjpe_gain_mm"]["hierarchical_bootstrap_95ci"][0] > 0
        distribution_ok = conditional["splits_with_both_metric_wins"] >= 2 and conditional["detectors_with_both_metric_wins"] == 2
        strong = primary_ci and conditional_ci and interaction_ci and distribution_ok
        cautious = primary_ci and conditional_positive and interaction_positive and distribution_ok
        if strong:
            decision = protocol["next_gate"]["strong"]
        elif cautious:
            decision = protocol["next_gate"]["cautionary"]
        gate = {
            "source_full_vs_observed_ci_positive_both_metrics": primary_ci,
            "conditional_virtual_positive_both_metrics": conditional_positive,
            "conditional_virtual_ci_positive_both_metrics": conditional_ci,
            "interaction_positive_both_metrics": interaction_positive,
            "interaction_ci_positive_both_metrics": interaction_ci,
            "conditional_virtual_distribution_ok": distribution_ok,
            "strong_gate": strong,
            "cautionary_gate": cautious,
        }
    check("paired_statistics_complete", statistics_ok, f"paired_units={len(conditional_rows)}/18")

    statistics = {
        "status": "PASS" if statistics_ok else "FAIL",
        "format": "phase9c_r1e_synergy_statistics_v1",
        "phase": "9C-R1E",
        "scientific_role": "factorial attribution of reliability-conditioned virtual-view synergy",
        "bootstrap": {"replicates": replicates, "seed": bootstrap_seed, "confidence_level": 0.95, "hierarchy": ["split", "seed", "detector"]},
        "source_r1d_primary": r1d_stats.get("comparisons", {}).get("full_method_vs_observed_matched"),
        "comparisons": comparisons,
        "gate_detail": gate,
        "scientific_decision": decision,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    statistics_path = output / "phase9c_r1e_statistics.json"
    write_json(statistics_path, statistics)
    hashes_path = output / "phase9c_r1e_artifact_hashes.csv"
    with hashes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_report", "path", "sha256", "verified"])
        writer.writeheader()
        writer.writerows(artifact_rows)
    status = "PASS" if not errors else "FAIL"
    qc = {
        "status": status,
        "format": "phase9c_r1e_qc_report_v1",
        "phase": "9C-R1E",
        "scientific_decision": decision if status == "PASS" else "REPAIR_PHASE9C_R1E_EVIDENCE",
        "checks": checks,
        "errors": errors,
        "new_run_reports": len(new_reports),
        "reused_r1d_run_reports": len(r1d_reports),
        "new_artifact_hash_rows": len(artifact_rows),
        "protocol_sha256": protocol_sha,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    qc_path = output / "phase9c_r1e_qc_report.json"
    write_json(qc_path, qc)
    zip_path = output / "phase9c_r1e_results.zip"
    archive_inputs: list[tuple[Path, str]] = [
        (protocol_path, protocol_path.name), (preflight_path, preflight_path.name),
        (r1d_qc_path, "source_r1d/phase9c_r1d_qc_report.json"),
        (r1d_stats_path, "source_r1d/phase9c_r1d_statistics.json"),
        (hashes_path, hashes_path.name), (statistics_path, statistics_path.name), (qc_path, qc_path.name),
    ]
    archive_inputs.extend((path, str(Path("split_reports") / path.parent.name / path.name)) for path in split_reports if path.is_file())
    archive_inputs.extend((path, str(Path("run_reports") / path.relative_to(output))) for path in new_paths)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, arcname in archive_inputs:
            archive.write(source, arcname)
    print("=" * 112)
    print("PHASE 9C-R1E - SYNERGY ATTRIBUTION QC, STATISTICS AND PACK")
    print("=" * 112)
    print(f"Status                : {status}")
    print(f"New run reports       : {len(new_reports)} / 18")
    print(f"Reused R1D reports    : {len(r1d_reports)} / 54")
    print("Official TS1-TS6 read : NO")
    print(f"Scientific decision   : {qc['scientific_decision']}")
    print(f"Output ZIP            : {zip_path}")
    if errors:
        print(f"Errors                : {errors}")
    print("=" * 112)
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
