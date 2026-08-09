#!/usr/bin/env python3
"""Phase 9C-R1G: reporting-only Q2 evidence freeze and claim lock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PHASE = "9C-R1G"
OUT_REL = Path("outputs/phase9c_r1g_q2_evidence_freeze")
DECISION = "READY_FOR_PHASE9C_R1H_Q2_MANUSCRIPT_UPDATE"
R1E_REQUIRED_DECISION = "READY_FOR_PHASE9C_R1F_OFFICIAL_TEST_Q2_CAUTIOUS_SYNERGY"
R1F_REQUIRED_DECISION = "OFFICIAL_TREND_ONLY_WRITE_Q2_CAUTIOUS"

ARM_LABELS = {
    "detector_aware_observed": "Observed only",
    "detector_conditioned_virtual_view_no_confidence": "Observed + virtual view (ungated)",
    "bounded_reliability_observed_modulation": "Observed + reliability",
    "bounded_reliability_dual_modulation": "Full: reliability-conditioned fusion",
}

CONTRAST_LABELS = {
    "primary_full_vs_observed": "Full vs observed-only",
    "conditional_virtual_full_vs_reliability": "Conditional virtual view: full vs reliability-only",
    "reliability_only_vs_observed": "Reliability-only vs observed-only",
    "ungated_virtual_vs_observed": "Ungated virtual view vs observed-only",
    "virtual_reliability_interaction": "Virtual-view × reliability interaction",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def read_csv_bytes(data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def project_root_from_script() -> Path:
    # <root>/scripts/report/this_file.py
    return Path(__file__).resolve().parents[2]


def locate_unique(root: Path, exact_rel: str, patterns: list[str]) -> Path:
    exact = root / exact_rel
    if exact.is_file():
        return exact
    found: list[Path] = []
    for pattern in patterns:
        found.extend(p for p in root.glob(pattern) if p.is_file())
    unique = sorted(set(p.resolve() for p in found))
    if not unique:
        raise FileNotFoundError(f"Required input not found: {exact_rel}; patterns={patterns}")
    if len(unique) > 1:
        rendered = "\n  ".join(str(p) for p in unique)
        raise RuntimeError(f"Ambiguous evidence input for {exact_rel}:\n  {rendered}")
    return unique[0]


def locate_inputs(root: Path) -> dict[str, Path]:
    return {
        "r1e": locate_unique(
            root,
            "outputs/phase9c_r1e_synergy_attribution/phase9c_r1e_results.zip",
            ["outputs/**/phase9c_r1e_results.zip"],
        ),
        "r1f": locate_unique(
            root,
            "outputs/phase9c_r1f_locked_official_test/phase9c_r1f_results.zip",
            ["outputs/**/phase9c_r1f_results.zip"],
        ),
    }


class EvidenceZip:
    def __init__(self, path: Path):
        self.path = path
        self.zf = zipfile.ZipFile(path, "r")
        bad = self.zf.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP member in {path}: {bad}")
        self.names = set(self.zf.namelist())

    def close(self) -> None:
        self.zf.close()

    def bytes(self, name: str) -> bytes:
        if name not in self.names:
            raise KeyError(f"Missing {name} in {self.path}")
        return self.zf.read(name)

    def json(self, name: str) -> dict[str, Any]:
        return json.loads(self.bytes(name).decode("utf-8-sig"))


def check(condition: bool, name: str, detail: str, checks: list[dict[str, Any]], errors: list[str]) -> None:
    checks.append({"name": name, "passed": bool(condition), "detail": detail})
    if not condition:
        errors.append(name)


def load_evidence(root: Path) -> tuple[dict[str, Path], EvidenceZip, EvidenceZip, dict[str, Any]]:
    paths = locate_inputs(root)
    r1e = EvidenceZip(paths["r1e"])
    r1f = EvidenceZip(paths["r1f"])

    required_r1e = [
        "phase9c_r1e_qc_report.json",
        "phase9c_r1e_statistics.json",
        "phase9c_r1e_synergy_attribution_protocol.json",
        "source_r1d/phase9c_r1d_qc_report.json",
        "source_r1d/phase9c_r1d_statistics.json",
    ]
    required_r1f = [
        "phase9c_r1f_locked_official_test_protocol.json",
        "phase9c_r1f_detector_cache_report.json",
        "phase9c_r1f_official_test_summary.json",
        "phase9c_r1f_official_result_lock.json",
        "phase9c_r1f_subject_system_metrics.csv",
        "phase9c_r1f_activity_metrics_descriptive.csv",
        "phase9c_r1f_output_manifest.json",
        "phase9c_r1f_qc_report.json",
    ]
    for name in required_r1e:
        r1e.bytes(name)
    for name in required_r1f:
        r1f.bytes(name)

    data = {
        "r1e_qc": r1e.json("phase9c_r1e_qc_report.json"),
        "r1e_stats": r1e.json("phase9c_r1e_statistics.json"),
        "r1e_protocol": r1e.json("phase9c_r1e_synergy_attribution_protocol.json"),
        "r1d_qc": r1e.json("source_r1d/phase9c_r1d_qc_report.json"),
        "r1d_stats": r1e.json("source_r1d/phase9c_r1d_statistics.json"),
        "r1f_protocol": r1f.json("phase9c_r1f_locked_official_test_protocol.json"),
        "r1f_detector": r1f.json("phase9c_r1f_detector_cache_report.json"),
        "r1f_summary": r1f.json("phase9c_r1f_official_test_summary.json"),
        "r1f_lock": r1f.json("phase9c_r1f_official_result_lock.json"),
        "r1f_manifest": r1f.json("phase9c_r1f_output_manifest.json"),
        "r1f_qc": r1f.json("phase9c_r1f_qc_report.json"),
        "subject_rows": read_csv_bytes(r1f.bytes("phase9c_r1f_subject_system_metrics.csv")),
        "activity_rows": read_csv_bytes(r1f.bytes("phase9c_r1f_activity_metrics_descriptive.csv")),
    }
    return paths, r1e, r1f, data


def evidence_checks(paths: dict[str, Path], r1e: EvidenceZip, r1f: EvidenceZip, data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    check(data["r1e_qc"].get("status") == "PASS", "r1e_qc_pass", str(data["r1e_qc"].get("status")), checks, errors)
    check(data["r1e_qc"].get("scientific_decision") == R1E_REQUIRED_DECISION, "r1e_decision_locked", str(data["r1e_qc"].get("scientific_decision")), checks, errors)
    check(data["r1e_qc"].get("official_TS1_to_TS6_read") is False, "r1e_official_isolation", "official read must be false", checks, errors)
    check(data["r1d_qc"].get("status") == "PASS" and data["r1d_qc"].get("format") == "phase9c_r1d_qc_report_v2_evidence_contract_repair", "r1d_repaired_evidence_pass", str(data["r1d_qc"].get("format")), checks, errors)
    check(data["r1f_qc"].get("status") == "PASS", "r1f_qc_pass", str(data["r1f_qc"].get("status")), checks, errors)
    check(data["r1f_summary"].get("scientific_decision") == R1F_REQUIRED_DECISION, "r1f_decision_locked", str(data["r1f_summary"].get("scientific_decision")), checks, errors)
    check(data["r1f_lock"].get("status") == "LOCKED" and data["r1f_lock"].get("rerun_for_selection_forbidden") is True, "r1f_result_lock", str(data["r1f_lock"].get("status")), checks, errors)
    check(data["r1f_summary"].get("official_test_used_for_selection") is False and data["r1f_summary"].get("retraining_on_official_test") is False, "r1f_no_selection_or_retraining", "selection=false; retraining=false", checks, errors)
    check(data["r1f_protocol"].get("locked_model_matrix", {}).get("checkpoint_count") == 72, "r1f_checkpoint_matrix", "checkpoint_count=72", checks, errors)
    check(data["r1f_summary"].get("valid_frames") == 2875, "r1f_frame_cohort", f"valid_frames={data['r1f_summary'].get('valid_frames')}", checks, errors)

    manifest_failures = []
    for entry in data["r1f_manifest"].get("files", []):
        name = entry["relative_path"]
        try:
            actual = sha256_bytes(r1f.bytes(name))
            if actual != entry["sha256"]:
                manifest_failures.append(name)
        except Exception:
            manifest_failures.append(name)
    check(not manifest_failures, "r1f_manifest_hashes", f"verified={len(data['r1f_manifest'].get('files', []))}; failures={manifest_failures}", checks, errors)
    check(sha256_bytes(r1f.bytes("phase9c_r1f_locked_official_test_protocol.json")) == data["r1f_lock"].get("protocol_sha256"), "r1f_protocol_hash_lock", data["r1f_lock"].get("protocol_sha256", ""), checks, errors)
    check(sha256_bytes(r1f.bytes("phase9c_r1f_official_test_summary.json")) == data["r1f_lock"].get("summary_sha256"), "r1f_summary_hash_lock", data["r1f_lock"].get("summary_sha256", ""), checks, errors)
    check(len(data["subject_rows"]) == 432, "r1f_subject_metric_rows", f"rows={len(data['subject_rows'])}; expected=6*3*3*2*4=432", checks, errors)
    check(len(data["activity_rows"]) == 28, "r1f_activity_metric_rows", f"rows={len(data['activity_rows'])}; expected=7*4=28", checks, errors)
    coverage = data["r1f_summary"].get("detector_coverage", {})
    check(all(float(v.get("success_rate", 0)) >= 0.9 for v in coverage.values()) and len(coverage) == 2, "r1f_detector_coverage", str({k: v.get('success_rate') for k, v in coverage.items()}), checks, errors)
    return checks, errors


def run_preflight(root: Path) -> int:
    out = root / OUT_REL
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any]
    r1e = r1f = None
    try:
        paths, r1e, r1f, data = load_evidence(root)
        checks, errors = evidence_checks(paths, r1e, r1f, data)
        report = {
            "status": "PASS" if not errors else "FAIL",
            "format": "phase9c_r1g_preflight_report_v1",
            "phase": PHASE,
            "scientific_decision": "READY_TO_BUILD_Q2_EVIDENCE_PACK" if not errors else "REPAIR_INPUT_EVIDENCE_BEFORE_Q2_FREEZE",
            "created_utc": utc_now(),
            "project_root": str(root),
            "inputs": {k: {"path": str(v), "size_bytes": v.stat().st_size, "sha256": sha256_file(v)} for k, v in paths.items()},
            "checks": checks,
            "errors": errors,
            "environment": {"platform": platform.platform(), "python": sys.version},
            "training_performed": False,
            "model_inference_performed": False,
            "official_TS1_to_TS6_reopened": False,
        }
    except Exception as exc:
        report = {
            "status": "FAIL",
            "format": "phase9c_r1g_preflight_report_v1",
            "phase": PHASE,
            "scientific_decision": "REPAIR_INPUT_EVIDENCE_BEFORE_Q2_FREEZE",
            "created_utc": utc_now(),
            "project_root": str(root),
            "checks": [],
            "errors": [f"{type(exc).__name__}: {exc}"],
            "training_performed": False,
            "model_inference_performed": False,
            "official_TS1_to_TS6_reopened": False,
        }
    finally:
        if r1e is not None:
            r1e.close()
        if r1f is not None:
            r1f.close()
    write_json(out / "phase9c_r1g_preflight_report.json", report)
    print("=" * 112)
    print("PHASE 9C-R1G PREFLIGHT")
    print("=" * 112)
    print(f"Status                : {report['status']}")
    print(f"Scientific decision   : {report['scientific_decision']}")
    print("Training performed    : NO")
    print("Official test reopened: NO")
    if report.get("errors"):
        print(f"Errors                : {report['errors']}")
    print("=" * 112)
    return 0 if report["status"] == "PASS" else 1


def metric_ci(ci_block: dict[str, Any], key: str) -> tuple[float, float, float, float]:
    item = ci_block["confidence_intervals"][key]
    return float(item["mean"]), float(item["ci95"][0]), float(item["ci95"][1]), float(item["probability_positive"])


def development_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    primary = data["r1e_stats"]["source_r1d_primary"]
    comps = data["r1e_stats"]["comparisons"]
    mapping = [
        ("Full vs observed-only", primary),
        ("Reliability-only vs observed-only", comps["reliability_only_contribution_vs_observed"]),
        ("Conditional virtual view: full vs reliability-only", comps["conditional_virtual_contribution_full_vs_reliability_observed"]),
        ("Virtual-view × reliability interaction", comps["virtual_reliability_interaction_difference_in_differences"]),
    ]
    rows = []
    for label, item in mapping:
        for metric_key, metric_label in [("mpjpe_gain_mm", "MPJPE"), ("pa_mpjpe_gain_mm", "PA-MPJPE")]:
            metric = item[metric_key]
            lo, hi = metric["hierarchical_bootstrap_95ci"]
            rows.append({
                "partition": "development_test_post_selection",
                "contrast": label,
                "metric": metric_label,
                "gain_mm_positive_is_better": metric["mean"],
                "ci95_low_mm": lo,
                "ci95_high_mm": hi,
                "ci_excludes_zero_positive": lo > 0,
                "paired_units": item.get("pair_count", primary.get("pair_count", 18)),
            })
    return rows


def official_contrast_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, block in summary["bootstrap"].items():
        label = CONTRAST_LABELS[key]
        for metric_key, metric_label in [("MPJPE_improvement_mm", "MPJPE"), ("PA_MPJPE_improvement_mm", "PA-MPJPE")]:
            mean, lo, hi, prob = metric_ci(block, metric_key)
            rows.append({
                "partition": "Official TS1-TS6",
                "contrast": label,
                "metric": metric_label,
                "gain_mm_positive_is_better": mean,
                "ci95_low_mm": lo,
                "ci95_high_mm": hi,
                "probability_positive": prob,
                "ci_excludes_zero_positive": lo > 0,
                "bootstrap_replicates": block["replicates"],
                "bootstrap_seed": block["seed"],
            })
    return rows


def subject_gain_rows(subject_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in subject_rows:
        key = (row["subject"], row["arm"])
        for metric in ["MPJPE_mm", "PA_MPJPE_mm", "3DPCK_at_150mm_pct", "AUC_0_to_150mm_pct"]:
            grouped[key][metric].append(float(row[metric]))
    rows = []
    for subject in sorted({r["subject"] for r in subject_rows}):
        base = grouped[(subject, "detector_aware_observed")]
        full = grouped[(subject, "bounded_reliability_dual_modulation")]
        b_m = statistics.fmean(base["MPJPE_mm"])
        f_m = statistics.fmean(full["MPJPE_mm"])
        b_pa = statistics.fmean(base["PA_MPJPE_mm"])
        f_pa = statistics.fmean(full["PA_MPJPE_mm"])
        rows.append({
            "subject": subject,
            "macro_units": len(base["MPJPE_mm"]),
            "observed_MPJPE_mm": b_m,
            "full_MPJPE_mm": f_m,
            "MPJPE_gain_mm": b_m - f_m,
            "observed_PA_MPJPE_mm": b_pa,
            "full_PA_MPJPE_mm": f_pa,
            "PA_MPJPE_gain_mm": b_pa - f_pa,
            "both_primary_metrics_improve": (b_m > f_m and b_pa > f_pa),
        })
    return rows


def activity_gain_rows(activity_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key = {(r["activity_id"], r["arm"]): r for r in activity_rows}
    rows = []
    for activity in sorted({r["activity_id"] for r in activity_rows}, key=int):
        base = by_key[(activity, "detector_aware_observed")]
        full = by_key[(activity, "bounded_reliability_dual_modulation")]
        b_m, f_m = float(base["MPJPE_mm"]), float(full["MPJPE_mm"])
        b_pa, f_pa = float(base["PA_MPJPE_mm"]), float(full["PA_MPJPE_mm"])
        rows.append({
            "activity_id": activity,
            "frames": int(base["frames"]),
            "observed_MPJPE_mm": b_m,
            "full_MPJPE_mm": f_m,
            "MPJPE_gain_mm": b_m - f_m,
            "observed_PA_MPJPE_mm": b_pa,
            "full_PA_MPJPE_mm": f_pa,
            "PA_MPJPE_gain_mm": b_pa - f_pa,
            "descriptive_only_post_hoc": True,
        })
    return rows


def claim_rows() -> list[dict[str, Any]]:
    return [
        {"claim_id": "C01", "status": "SUPPORTED", "paper_location": "Results/Abstract", "claim": "On matched post-selection development evaluation, the full method improves MPJPE and PA-MPJPE over observed-only with paired hierarchical-bootstrap 95% CIs above zero.", "required_wording": "State the partition and that checkpoint selection used matched validation only."},
        {"claim_id": "C02", "status": "SUPPORTED_CAUTIOUS", "paper_location": "Results/Abstract", "claim": "On locked Official TS1-TS6 RGB-detector input, the full method shows positive point improvements for both primary errors and for both detectors.", "required_wording": "Call this a positive external trend; report the CIs and do not call it statistically confirmed."},
        {"claim_id": "C03", "status": "FORBIDDEN", "paper_location": "All", "claim": "The Phase9C method is statistically superior on Official TS1-TS6.", "required_wording": "Do not write; both primary 95% CIs include zero."},
        {"claim_id": "C04", "status": "SUPPORTED_CAUTIOUS", "paper_location": "Ablation/Discussion", "claim": "Reliability conditioning accounts for most of the development improvement, while the conditional virtual-view increment is positive but unresolved.", "required_wording": "Report reliability-only and conditional-virtual contrasts with their CIs."},
        {"claim_id": "C05", "status": "FORBIDDEN", "paper_location": "All", "claim": "Virtual-view synergy is statistically confirmed.", "required_wording": "Do not write; conditional and interaction CIs include zero."},
        {"claim_id": "C06", "status": "FORBIDDEN", "paper_location": "All", "claim": "Ungated virtual views consistently improve 3D pose estimation.", "required_wording": "Do not write; ungated virtual view is neutral/slightly worse in Official MPJPE and has unresolved CIs."},
        {"claim_id": "C07", "status": "SUPPORTED", "paper_location": "Protocol/Results", "claim": "Both RTMPose and YOLO11-L detected a person in all 2,875 valid Official frames under detector-native person selection.", "required_wording": "Clarify that detection coverage is not the same as keypoint accuracy."},
        {"claim_id": "C08", "status": "SUPPORTED", "paper_location": "Methods/Limitations", "claim": "Official evaluation was locked, used all valid frames, did not tune or retrain on Official Test, and cannot be rerun for model selection.", "required_wording": "Retain the transparency note that legacy Phase7 had previously evaluated TS1-TS6 with GT-2D input."},
        {"claim_id": "C09", "status": "FORBIDDEN", "paper_location": "All", "claim": "This is a pristine first look at TS1-TS6 for the entire project.", "required_wording": "Do not write; call it the first locked Official evaluation of the Phase9C real-RGB-detector method."},
        {"claim_id": "C10", "status": "FORBIDDEN_PENDING_AUDIT", "paper_location": "Related Work/Results", "claim": "The method is state of the art or directly better than published systems.", "required_wording": "Requires a protocol-, detector-, joint-layout-, and training-data-matched literature audit."},
        {"claim_id": "C11", "status": "SUPPORTED", "paper_location": "Limitations/Conclusion", "claim": "The study exposes a generalization gap between internal development confirmation and Official external evidence.", "required_wording": "Present the gap as a result and boundary condition, not as a hidden failure."},
    ]


def make_figures(out: Path, data: dict[str, Any], dev_rows: list[dict[str, Any]], off_rows: list[dict[str, Any]], subj_rows: list[dict[str, Any]]) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "figure.dpi": 150})
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    # Development vs Official primary contrast with locked CIs.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, metric in zip(axes, ["MPJPE", "PA-MPJPE"]):
        d = next(r for r in dev_rows if r["contrast"] == "Full vs observed-only" and r["metric"] == metric)
        o = next(r for r in off_rows if r["contrast"] == "Full vs observed-only" and r["metric"] == metric)
        vals = [float(d["gain_mm_positive_is_better"]), float(o["gain_mm_positive_is_better"])]
        los = [float(d["ci95_low_mm"]), float(o["ci95_low_mm"])]
        his = [float(d["ci95_high_mm"]), float(o["ci95_high_mm"])]
        y = [1, 0]
        ax.axvline(0, color="#6b7280", linewidth=1)
        ax.errorbar(vals, y, xerr=[[v-l for v, l in zip(vals, los)], [h-v for h, v in zip(his, vals)]], fmt="o", color="#1f4e79", ecolor="#4f81bd", capsize=4)
        ax.set_yticks(y, ["Development", "Official TS1–TS6"])
        ax.set_xlabel("Improvement (mm; positive is better)")
        ax.set_title(metric)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Full reliability-conditioned fusion vs observed-only")
    fig.tight_layout()
    p = figure_dir / "figure_q2_development_vs_official_ci.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    files.append(str(p.relative_to(out)))

    # Official arm metrics.
    aggregate = data["r1f_summary"]["aggregate_subject_split_seed_detector_macro"]
    arms = list(ARM_LABELS)
    x = list(range(len(arms)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in [(axes[0], "MPJPE_mm", "MPJPE"), (axes[1], "PA_MPJPE_mm", "PA-MPJPE")]:
        vals = [aggregate[a][metric] for a in arms]
        colors = ["#9ca3af", "#c97a40", "#5b8c5a", "#1f4e79"]
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(x, ["Observed", "Ungated VV", "Reliability", "Full"], rotation=15)
        ax.set_ylabel("Error (mm; lower is better)")
        ax.set_title(f"Official {title}")
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_ylim(min(vals) - max(2.0, (max(vals)-min(vals))*0.8), max(vals) + max(2.0, (max(vals)-min(vals))*0.8))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Locked Official TS1–TS6 arm comparison")
    fig.tight_layout()
    p = figure_dir / "figure_q2_official_arm_metrics.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    files.append(str(p.relative_to(out)))

    # Detector-specific gains.
    det = data["r1f_summary"]["detector_specific_primary_improvement"]
    detectors = list(det)
    labels = ["RTMPose", "YOLO11-L"]
    mp = [det[d]["MPJPE_improvement_mm"] for d in detectors]
    pa = [det[d]["PA_MPJPE_improvement_mm"] for d in detectors]
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    width = 0.34
    xx = list(range(len(detectors)))
    b1 = ax.bar([v-width/2 for v in xx], mp, width, label="MPJPE", color="#1f4e79")
    b2 = ax.bar([v+width/2 for v in xx], pa, width, label="PA-MPJPE", color="#82a6c9")
    ax.axhline(0, color="#6b7280", linewidth=1)
    ax.set_xticks(xx, labels)
    ax.set_ylabel("Improvement (mm; positive is better)")
    ax.set_title("Official full-vs-observed gain by detector")
    ax.legend()
    ax.bar_label(b1, fmt="%.2f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize=9, padding=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    p = figure_dir / "figure_q2_official_detector_gains.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    files.append(str(p.relative_to(out)))

    # Descriptive subject gains.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    subjects = [r["subject"] for r in subj_rows]
    for ax, key, title in [(axes[0], "MPJPE_gain_mm", "MPJPE"), (axes[1], "PA_MPJPE_gain_mm", "PA-MPJPE")]:
        vals = [float(r[key]) for r in subj_rows]
        colors = ["#2f7d32" if v >= 0 else "#b23a3a" for v in vals]
        ax.barh(subjects, vals, color=colors)
        ax.axvline(0, color="#6b7280", linewidth=1)
        ax.set_xlabel("Improvement (mm)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Official subject-macro full-vs-observed gains (descriptive)")
    fig.tight_layout()
    p = figure_dir / "figure_q2_official_subject_gains_descriptive.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    files.append(str(p.relative_to(out)))
    return files


def build_narratives(data: dict[str, Any]) -> tuple[str, str, str]:
    s = data["r1f_summary"]
    dev = data["r1e_stats"]["source_r1d_primary"]
    cond = data["r1e_stats"]["comparisons"]["conditional_virtual_contribution_full_vs_reliability_observed"]
    rel = data["r1e_stats"]["comparisons"]["reliability_only_contribution_vs_observed"]
    off = s["bootstrap"]["primary_full_vs_observed"]["confidence_intervals"]
    results = f"""# Phase 9C-R1G — Q2 Results Draft

## Locked development confirmation

Across three subject splits, three seeds, and two real 2D pose detectors (18 paired units), the full reliability-conditioned fusion model reduced MPJPE by {dev['mpjpe_gain_mm']['mean']:.3f} mm (95% CI {dev['mpjpe_gain_mm']['hierarchical_bootstrap_95ci'][0]:.3f} to {dev['mpjpe_gain_mm']['hierarchical_bootstrap_95ci'][1]:.3f}) and PA-MPJPE by {dev['pa_mpjpe_gain_mm']['mean']:.3f} mm (95% CI {dev['pa_mpjpe_gain_mm']['hierarchical_bootstrap_95ci'][0]:.3f} to {dev['pa_mpjpe_gain_mm']['hierarchical_bootstrap_95ci'][1]:.3f}) relative to the detector-aware observed-only baseline. The relative MPJPE gain was {dev['mpjpe_gain_percent']['mean']:.3f}% (95% CI {dev['mpjpe_gain_percent']['hierarchical_bootstrap_95ci'][0]:.3f}% to {dev['mpjpe_gain_percent']['hierarchical_bootstrap_95ci'][1]:.3f}%). Mean gains were positive for all three subject splits and both detector families.

## Factorial attribution

Reliability-only modulation accounted for most of the development improvement: {rel['mpjpe_gain_mm']['mean']:.3f} mm MPJPE (95% CI {rel['mpjpe_gain_mm']['hierarchical_bootstrap_95ci'][0]:.3f} to {rel['mpjpe_gain_mm']['hierarchical_bootstrap_95ci'][1]:.3f}) and {rel['pa_mpjpe_gain_mm']['mean']:.3f} mm PA-MPJPE (95% CI {rel['pa_mpjpe_gain_mm']['hierarchical_bootstrap_95ci'][0]:.3f} to {rel['pa_mpjpe_gain_mm']['hierarchical_bootstrap_95ci'][1]:.3f}). Adding the reliability-conditioned virtual view above reliability-only yielded positive point increments of {cond['mpjpe_gain_mm']['mean']:.3f} mm MPJPE and {cond['pa_mpjpe_gain_mm']['mean']:.3f} mm PA-MPJPE, but the corresponding confidence intervals included zero. These results support reliability conditioning as the stable component and treat the additional virtual-view contribution as an unresolved trend rather than confirmed synergy.

## Locked Official TS1–TS6 evaluation

The Official evaluation used 2,875 valid RGB frames, detector-native person selection, two detectors, and a fixed matrix of 72 validation-selected checkpoints. No checkpoint, arm, detector, or hyperparameter was selected from Official results. Relative to observed-only, the full method improved subject–split–seed–detector macro MPJPE from {s['aggregate_subject_split_seed_detector_macro']['detector_aware_observed']['MPJPE_mm']:.3f} to {s['aggregate_subject_split_seed_detector_macro']['bounded_reliability_dual_modulation']['MPJPE_mm']:.3f} mm and PA-MPJPE from {s['aggregate_subject_split_seed_detector_macro']['detector_aware_observed']['PA_MPJPE_mm']:.3f} to {s['aggregate_subject_split_seed_detector_macro']['bounded_reliability_dual_modulation']['PA_MPJPE_mm']:.3f} mm. The point improvements were {s['primary_improvement_positive_is_better']['MPJPE_improvement_mm']:.3f} mm ({s['primary_improvement_positive_is_better']['MPJPE_relative_gain_pct']:.3f}%) and {s['primary_improvement_positive_is_better']['PA_MPJPE_improvement_mm']:.3f} mm, respectively. The locked hierarchical-bootstrap intervals nevertheless included zero for MPJPE ({off['MPJPE_improvement_mm']['ci95'][0]:.3f} to {off['MPJPE_improvement_mm']['ci95'][1]:.3f} mm) and PA-MPJPE ({off['PA_MPJPE_improvement_mm']['ci95'][0]:.3f} to {off['PA_MPJPE_improvement_mm']['ci95'][1]:.3f} mm). Both detector-specific means improved both primary metrics, but the Official result is reported as a positive external trend, not confirmatory superiority.

Secondary Official metrics were mixed: 3DPCK improved by {s['primary_improvement_positive_is_better']['3DPCK_improvement_pct_point']:.3f} percentage points, whereas AUC changed by {s['primary_improvement_positive_is_better']['AUC_improvement_pct_point']:.3f} percentage points. Both detectors achieved 100% person-detection coverage on the locked valid-frame cohort; this coverage result does not imply perfect 2D keypoint accuracy.

## Interpretation for a Q2 submission

The defensible contribution is a locked multi-subject, multi-seed, real-detector study showing that reliability conditioning stabilizes pose fusion internally while external subject/domain shift reduces the gain to a modest, statistically unresolved trend. The paper should emphasize protocol discipline, factorial attribution, and robustness boundaries. It should not claim Official statistical superiority, statistically confirmed virtual-view synergy, or state-of-the-art accuracy.
"""
    limitations = """# Phase 9C-R1G — Mandatory Limitations Text

1. The locked Official TS1–TS6 primary confidence intervals include zero, so external statistical superiority was not established despite positive point estimates for both primary metrics and both detectors.
2. The conditional virtual-view increment above reliability-only and the virtual-view × reliability interaction were not statistically resolved. The results therefore do not isolate a confirmed causal gain from synthesized virtual views.
3. Official AUC decreased slightly even though MPJPE, PA-MPJPE, and PCK point estimates improved; secondary metrics are mixed and must be reported together.
4. Performance depends on matched detector conditioning. Cross-detector development results show a substantial detector-domain shift and should not be generalized away.
5. The study evaluates one dataset family for the locked Phase9C Official result. Independent-dataset confirmation is reserved for a separately pre-registered Q1 extension.
6. TS1–TS6 had been evaluated earlier by a legacy Phase7 GT-2D system. R1F is the first locked Official evaluation of the Phase9C real-RGB-detector method, not a pristine project-level first look.
7. Detector coverage was 100% on the valid-frame cohort, but coverage is not a measure of 2D keypoint accuracy or calibration.
8. Subject and activity breakdowns are descriptive, post-hoc analyses and must not be presented as confirmatory subgroup findings.
"""
    extension = """# Q1 Extension Registry — Isolated from the Q2 Evidence Freeze

The Q2 paper should be completed from the locked R1E/R1F evidence before beginning this extension.

## Preferred Q1 extension

Run an independent-dataset confirmation with the Phase9C architecture, detector-conditioning rule, arms, seeds, metrics, and statistical hierarchy pre-registered before any test labels are opened.

### Stage Q1-A: feasibility and protocol lock

- HumanEva-I is the smaller first choice for feasibility and fast pipeline verification.
- Human3.6M is the stronger but larger standard-protocol extension.
- Define joint mapping, root alignment, camera split, subject split, detector versions, missing-detection policy, and checkpoint-selection partition before test evaluation.
- Do not use MPI-INF-3DHP Official TS1–TS6 to choose any Q1-extension model or rule.

### Stage Q1-B: external confirmation

- Freeze observed-only, reliability-only, ungated virtual-view, and full arms.
- Use at least three seeds and two detector families if the dataset license and compute budget permit.
- Make the full-vs-observed and full-vs-reliability contrasts confirmatory.
- Report all valid frames, detector failures, efficiency, and calibration.

### Optional Q1 strengthening

- Add temporal reliability using a pre-registered short-window mechanism.
- Add an occlusion-specific benchmark whose groups are defined without inspecting 3D model errors.
- Report parameter count, FLOPs, detector and lifter latency, and memory.
- Release protocol locks, hashes, split definitions, and reporting scripts.

These studies are new work. They must not alter the frozen Q2 Official result or be described as a favorable rerun of R1F.
"""
    return results, limitations, extension


def run_build(root: Path) -> int:
    out = root / OUT_REL
    preflight_path = out / "phase9c_r1g_preflight_report.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(encoding="utf-8"))["status"] != "PASS":
        if run_preflight(root) != 0:
            return 1

    paths, r1e, r1f, data = load_evidence(root)
    try:
        checks, errors = evidence_checks(paths, r1e, r1f, data)
        if errors:
            raise RuntimeError(f"Evidence checks failed during build: {errors}")
        table_dir = out / "tables"
        source_dir = out / "source_evidence"
        table_dir.mkdir(parents=True, exist_ok=True)
        source_dir.mkdir(parents=True, exist_ok=True)

        # Copy only small, paper-relevant locked evidence. NPZ inference outputs remain in R1F.
        copies = {
            "source_evidence/phase9c_r1e_qc_report.json": r1e.bytes("phase9c_r1e_qc_report.json"),
            "source_evidence/phase9c_r1e_statistics.json": r1e.bytes("phase9c_r1e_statistics.json"),
            "source_evidence/phase9c_r1e_protocol.json": r1e.bytes("phase9c_r1e_synergy_attribution_protocol.json"),
            "source_evidence/phase9c_r1d_repaired_qc_report.json": r1e.bytes("source_r1d/phase9c_r1d_qc_report.json"),
            "source_evidence/phase9c_r1d_statistics.json": r1e.bytes("source_r1d/phase9c_r1d_statistics.json"),
            "source_evidence/phase9c_r1f_protocol.json": r1f.bytes("phase9c_r1f_locked_official_test_protocol.json"),
            "source_evidence/phase9c_r1f_detector_cache_report.json": r1f.bytes("phase9c_r1f_detector_cache_report.json"),
            "source_evidence/phase9c_r1f_official_test_summary.json": r1f.bytes("phase9c_r1f_official_test_summary.json"),
            "source_evidence/phase9c_r1f_official_result_lock.json": r1f.bytes("phase9c_r1f_official_result_lock.json"),
            "source_evidence/phase9c_r1f_subject_system_metrics.csv": r1f.bytes("phase9c_r1f_subject_system_metrics.csv"),
            "source_evidence/phase9c_r1f_activity_metrics_descriptive.csv": r1f.bytes("phase9c_r1f_activity_metrics_descriptive.csv"),
            "source_evidence/phase9c_r1f_output_manifest.json": r1f.bytes("phase9c_r1f_output_manifest.json"),
            "source_evidence/phase9c_r1f_qc_report.json": r1f.bytes("phase9c_r1f_qc_report.json"),
        }
        for rel, content in copies.items():
            p = out / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)

        dev_rows = development_rows(data)
        off_rows = official_contrast_rows(data["r1f_summary"])
        subj_rows = subject_gain_rows(data["subject_rows"])
        act_rows = activity_gain_rows(data["activity_rows"])
        claims = claim_rows()

        write_csv(table_dir / "table_q2_development_confirmation.csv", list(dev_rows[0]), dev_rows)
        write_csv(table_dir / "table_q2_official_contrasts.csv", list(off_rows[0]), off_rows)
        write_csv(table_dir / "table_q2_official_subject_gains_descriptive.csv", list(subj_rows[0]), subj_rows)
        write_csv(table_dir / "table_q2_official_activity_gains_descriptive.csv", list(act_rows[0]), act_rows)
        write_csv(table_dir / "table_q2_claim_matrix.csv", list(claims[0]), claims)

        aggregate_rows = []
        for arm, metrics in data["r1f_summary"]["aggregate_subject_split_seed_detector_macro"].items():
            aggregate_rows.append({"arm": arm, "paper_label": ARM_LABELS[arm], **metrics})
        write_csv(table_dir / "table_q2_official_aggregate_metrics.csv", list(aggregate_rows[0]), aggregate_rows)

        detector_rows = []
        for detector, metrics in data["r1f_summary"]["detector_specific_primary_improvement"].items():
            detector_rows.append({"detector": detector, **metrics, "coverage": data["r1f_summary"]["detector_coverage"][detector]["success_rate"]})
        write_csv(table_dir / "table_q2_official_detector_gains.csv", list(detector_rows[0]), detector_rows)

        runtime_rows = []
        for detector, runtime in data["r1f_detector"]["runtime"]["detectors"].items():
            elapsed = float(runtime["elapsed_sec"])
            n = int(runtime["new_samples"])
            runtime_rows.append({"detector": detector, "official_frames": n, "elapsed_sec": elapsed, "detector_throughput_fps": n / elapsed, "scope": "detector inference only; excludes lifter/fusion"})
        write_csv(table_dir / "table_q2_detector_runtime.csv", list(runtime_rows[0]), runtime_rows)

        results_md, limitations_md, extension_md = build_narratives(data)
        write_text(out / "PHASE9C_R1G_Q2_RESULTS_DRAFT.md", results_md)
        write_text(out / "PHASE9C_R1G_Q2_LIMITATIONS_DRAFT.md", limitations_md)
        write_text(out / "PHASE9C_R1G_Q1_EXTENSION_REGISTRY.md", extension_md)
        write_text(out / "PHASE9C_R1G_Q2_CLAIM_LOCK.md", "# Q2 Claim Lock\n\nThe authoritative claim statuses are in `tables/table_q2_claim_matrix.csv`. Any manuscript sentence that conflicts with a FORBIDDEN row must be revised. Official TS1–TS6 is frozen and may not be rerun for selection.\n")

        figure_files = make_figures(out, data, dev_rows, off_rows, subj_rows)
        protocol_path = root / "configs/phase9c_r1g_q2_publication_lock_protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(f"Missing installed R1G protocol: {protocol_path}")
        protocol_copy = out / "phase9c_r1g_q2_publication_lock_protocol.json"
        protocol_copy.write_bytes(protocol_path.read_bytes())

        source_ledger = [
            {"source": "phase9c_r1e_results.zip", "path": str(paths["r1e"]), "sha256": sha256_file(paths["r1e"]), "role": "locked development confirmation and factorial attribution"},
            {"source": "phase9c_r1f_results.zip", "path": str(paths["r1f"]), "sha256": sha256_file(paths["r1f"]), "role": "locked Official TS1-TS6 RGB-detector evaluation"},
        ]
        write_csv(out / "phase9c_r1g_source_evidence_ledger.csv", list(source_ledger[0]), source_ledger)

        abstract_numbers = {
            "format": "phase9c_r1g_q2_abstract_numbers_v1",
            "development": {
                "paired_units": dev["pair_count"] if (dev := data["r1e_stats"]["source_r1d_primary"]) else 18,
                "MPJPE_gain_mm": dev["mpjpe_gain_mm"],
                "PA_MPJPE_gain_mm": dev["pa_mpjpe_gain_mm"],
                "MPJPE_gain_percent": dev["mpjpe_gain_percent"],
            },
            "official": {
                "valid_frames": data["r1f_summary"]["valid_frames"],
                "checkpoint_count": data["r1f_lock"]["checkpoint_count"],
                "primary_improvement": data["r1f_summary"]["primary_improvement_positive_is_better"],
                "primary_bootstrap": data["r1f_summary"]["bootstrap"]["primary_full_vs_observed"],
                "detector_specific": data["r1f_summary"]["detector_specific_primary_improvement"],
                "scientific_decision": data["r1f_summary"]["scientific_decision"],
            },
            "wording_lock": "Development improvement was confirmed; Official improvement remained a positive but statistically unresolved external trend.",
        }
        write_json(out / "phase9c_r1g_q2_abstract_numbers.json", abstract_numbers)

        built_files = sorted([
            *copies.keys(),
            "tables/table_q2_development_confirmation.csv",
            "tables/table_q2_official_contrasts.csv",
            "tables/table_q2_official_subject_gains_descriptive.csv",
            "tables/table_q2_official_activity_gains_descriptive.csv",
            "tables/table_q2_claim_matrix.csv",
            "tables/table_q2_official_aggregate_metrics.csv",
            "tables/table_q2_official_detector_gains.csv",
            "tables/table_q2_detector_runtime.csv",
            "PHASE9C_R1G_Q2_RESULTS_DRAFT.md",
            "PHASE9C_R1G_Q2_LIMITATIONS_DRAFT.md",
            "PHASE9C_R1G_Q1_EXTENSION_REGISTRY.md",
            "PHASE9C_R1G_Q2_CLAIM_LOCK.md",
            "phase9c_r1g_q2_publication_lock_protocol.json",
            "phase9c_r1g_source_evidence_ledger.csv",
            "phase9c_r1g_q2_abstract_numbers.json",
            *figure_files,
        ])
        summary = {
            "status": "PASS",
            "format": "phase9c_r1g_q2_evidence_freeze_summary_v1",
            "phase": PHASE,
            "scientific_decision": DECISION,
            "created_utc": utc_now(),
            "source_decisions": {"R1E": data["r1e_qc"]["scientific_decision"], "R1F": data["r1f_summary"]["scientific_decision"]},
            "locked_interpretation": {
                "development_full_vs_observed_confirmed": True,
                "official_full_vs_observed_positive_point_both_metrics": True,
                "official_full_vs_observed_ci_positive_both_metrics": False,
                "official_both_detector_means_positive_both_metrics": True,
                "conditional_virtual_synergy_confirmed": False,
                "q2_position": "cautious locked-study submission",
                "q1_extension": "independent dataset; separate protocol",
            },
            "built_files": built_files,
            "training_performed": False,
            "model_inference_performed": False,
            "official_TS1_to_TS6_reopened": False,
            "official_result_changed": False,
        }
        write_json(out / "phase9c_r1g_q2_evidence_freeze_summary.json", summary)
        print("=" * 112)
        print("PHASE 9C-R1G BUILD Q2 EVIDENCE ASSETS")
        print("=" * 112)
        print("Status                : PASS")
        print(f"Built files           : {len(built_files)}")
        print("Training performed    : NO")
        print("Official test reopened: NO")
        print(f"Scientific decision   : {DECISION}")
        print("=" * 112)
        return 0
    except Exception as exc:
        failure = {"status": "FAIL", "format": "phase9c_r1g_build_failure_v1", "phase": PHASE, "created_utc": utc_now(), "errors": [f"{type(exc).__name__}: {exc}"]}
        write_json(out / "phase9c_r1g_build_failure.json", failure)
        print(json.dumps(failure, indent=2))
        return 1
    finally:
        r1e.close()
        r1f.close()


def run_qc(root: Path) -> int:
    out = root / OUT_REL
    summary_path = out / "phase9c_r1g_q2_evidence_freeze_summary.json"
    preflight_path = out / "phase9c_r1g_preflight_report.json"
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        if not summary_path.is_file():
            raise FileNotFoundError(f"Run build first: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        check(preflight.get("status") == "PASS", "preflight_pass", str(preflight.get("status")), checks, errors)
        check(summary.get("status") == "PASS" and summary.get("scientific_decision") == DECISION, "build_summary_pass", str(summary.get("scientific_decision")), checks, errors)
        missing = [rel for rel in summary.get("built_files", []) if not (out / rel).is_file() or (out / rel).stat().st_size == 0]
        check(not missing, "built_files_complete", f"expected={len(summary.get('built_files', []))}; missing={missing}", checks, errors)

        claim_path = out / "tables/table_q2_claim_matrix.csv"
        claims = list(csv.DictReader(claim_path.open(encoding="utf-8-sig", newline="")))
        statuses = {r["claim_id"]: r["status"] for r in claims}
        check(len(claims) >= 10 and statuses.get("C03") == "FORBIDDEN" and statuses.get("C05") == "FORBIDDEN", "claim_lock_complete", f"rows={len(claims)}", checks, errors)

        abstract = json.loads((out / "phase9c_r1g_q2_abstract_numbers.json").read_text(encoding="utf-8"))
        off_decision = abstract["official"]["scientific_decision"]
        off_ci = abstract["official"]["primary_bootstrap"]["confidence_intervals"]
        ci_crosses = float(off_ci["MPJPE_improvement_mm"]["ci95"][0]) <= 0 and float(off_ci["PA_MPJPE_improvement_mm"]["ci95"][0]) <= 0
        check(off_decision == R1F_REQUIRED_DECISION and ci_crosses, "official_cautious_interpretation_preserved", f"decision={off_decision}; both_ci_cross_zero={ci_crosses}", checks, errors)
        check(summary.get("training_performed") is False and summary.get("model_inference_performed") is False and summary.get("official_TS1_to_TS6_reopened") is False, "reporting_only_protocol", "train=false; inference=false; official_reopened=false", checks, errors)

        qc = {
            "status": "PASS" if not errors else "FAIL",
            "format": "phase9c_r1g_qc_report_v1",
            "phase": PHASE,
            "scientific_decision": DECISION if not errors else "REPAIR_PHASE9C_R1G_REPORTING_ARTIFACTS",
            "created_utc": utc_now(),
            "checks": checks,
            "errors": errors,
            "training_performed": False,
            "model_inference_performed": False,
            "official_TS1_to_TS6_reopened": False,
        }
        write_json(out / "phase9c_r1g_qc_report.json", qc)
        if errors:
            raise RuntimeError(f"QC failed: {errors}")

        archive_members = sorted(set(summary["built_files"] + [
            "phase9c_r1g_preflight_report.json",
            "phase9c_r1g_q2_evidence_freeze_summary.json",
            "phase9c_r1g_qc_report.json",
        ]))
        manifest_rows = []
        for rel in archive_members:
            p = out / rel
            manifest_rows.append({"relative_path": rel, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
        manifest = {"format": "phase9c_r1g_output_manifest_v1", "phase": PHASE, "files": manifest_rows}
        write_json(out / "phase9c_r1g_output_manifest.json", manifest)
        archive_members.append("phase9c_r1g_output_manifest.json")

        zip_path = out / "phase9c_r1g_results.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for rel in sorted(archive_members):
                zf.write(out / rel, rel)
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"Output ZIP integrity failure: {bad}")
        print("=" * 112)
        print("PHASE 9C-R1G - QC AND PACK")
        print("=" * 112)
        print("Status                : PASS")
        print(f"Archived files        : {len(archive_members)}")
        print("Official result changed: NO")
        print(f"Scientific decision   : {DECISION}")
        print(f"Output ZIP            : {zip_path}")
        print("=" * 112)
        return 0
    except Exception as exc:
        if not (out / "phase9c_r1g_qc_report.json").is_file():
            write_json(out / "phase9c_r1g_qc_report.json", {"status": "FAIL", "format": "phase9c_r1g_qc_report_v1", "phase": PHASE, "scientific_decision": "REPAIR_PHASE9C_R1G_REPORTING_ARTIFACTS", "created_utc": utc_now(), "checks": checks, "errors": errors + [f"{type(exc).__name__}: {exc}"]})
        print(f"Phase 9C-R1G QC failed: {type(exc).__name__}: {exc}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight", "build", "qc", "all"], required=True)
    parser.add_argument("--project-root", type=Path, default=None, help="Testing/advanced override; BAT files use the installed project root.")
    args = parser.parse_args()
    root = (args.project_root or project_root_from_script()).resolve()
    if args.mode in ("preflight", "all") and run_preflight(root) != 0:
        return 1
    if args.mode in ("build", "all") and run_build(root) != 0:
        return 1
    if args.mode in ("qc", "all") and run_qc(root) != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
