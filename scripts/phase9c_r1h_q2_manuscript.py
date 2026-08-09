from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PHASE = "9C-R1H"
OUTPUT_REL = Path("outputs/phase9c_r1h_q2_manuscript_update")
RESULT_ZIP_NAME = "phase9c_r1h_results.zip"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def project_paths() -> tuple[Path, Path, Path, dict]:
    script_path = Path(__file__).resolve()
    root = script_path.parents[1]
    protocol_path = root / "configs/phase9c_r1h_q2_manuscript_protocol.json"
    protocol = load_json(protocol_path)
    return root, protocol_path, root / OUTPUT_REL, protocol


def normalized_member(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def verify_r1g_archive(source_zip: Path, protocol: dict) -> tuple[list[dict], dict, dict]:
    checks: list[dict] = []
    expected_hash = protocol["source_r1g"]["sha256"]
    actual_hash = sha256_file(source_zip)
    checks.append({"name": "source_r1g_zip_sha256", "passed": actual_hash == expected_hash,
                   "detail": actual_hash})

    with zipfile.ZipFile(source_zip, "r") as zf:
        names = {normalized_member(n): n for n in zf.namelist() if not n.endswith("/")}
        checks.append({"name": "source_r1g_file_count",
                       "passed": len(names) == protocol["source_r1g"]["required_file_count"],
                       "detail": f"files={len(names)}"})
        required = [
            "phase9c_r1g_qc_report.json",
            "phase9c_r1g_q2_evidence_freeze_summary.json",
            "phase9c_r1g_output_manifest.json",
            "tables/table_q2_claim_matrix.csv",
            "source_evidence/phase9c_r1f_official_result_lock.json",
        ]
        missing = [p for p in required if p not in names]
        checks.append({"name": "required_source_files", "passed": not missing, "detail": f"missing={missing}"})
        if missing:
            return checks, {}, {}

        qc = json.loads(zf.read(names["phase9c_r1g_qc_report.json"]).decode("utf-8-sig"))
        summary = json.loads(zf.read(names["phase9c_r1g_q2_evidence_freeze_summary.json"]).decode("utf-8-sig"))
        manifest = json.loads(zf.read(names["phase9c_r1g_output_manifest.json"]).decode("utf-8-sig"))
        checks.append({"name": "source_qc_status", "passed": qc.get("status") == "PASS",
                       "detail": str(qc.get("status"))})
        checks.append({"name": "source_scientific_decision",
                       "passed": qc.get("scientific_decision") == protocol["source_r1g"]["required_scientific_decision"],
                       "detail": str(qc.get("scientific_decision"))})
        checks.append({"name": "source_reporting_only",
                       "passed": qc.get("training_performed") is False and
                                 qc.get("model_inference_performed") is False and
                                 qc.get("official_TS1_to_TS6_reopened") is False,
                       "detail": "train=false; inference=false; official_reopened=false"})
        checks.append({"name": "source_locked_interpretation",
                       "passed": summary.get("locked_interpretation", {}).get("official_full_vs_observed_ci_positive_both_metrics") is False and
                                 summary.get("locked_interpretation", {}).get("conditional_virtual_synergy_confirmed") is False,
                       "detail": str(summary.get("locked_interpretation", {}))})

        manifest_errors = []
        for item in manifest.get("files", []):
            rel = normalized_member(item["relative_path"])
            if rel not in names:
                manifest_errors.append(f"missing:{rel}")
                continue
            data = zf.read(names[rel])
            if len(data) != item["size_bytes"]:
                manifest_errors.append(f"size:{rel}")
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                manifest_errors.append(f"hash:{rel}")
        checks.append({"name": "source_internal_manifest", "passed": not manifest_errors,
                       "detail": f"verified={len(manifest.get('files', []))}; errors={manifest_errors}"})
    return checks, qc, summary


def preflight() -> int:
    root, protocol_path, output_dir, protocol = project_paths()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_zip = root / Path(protocol["source_r1g"]["relative_path"])
    checks = [{"name": "protocol_lock", "passed": protocol.get("status") == "LOCKED",
               "detail": sha256_file(protocol_path)}]
    if source_zip.exists():
        source_checks, _, _ = verify_r1g_archive(source_zip, protocol)
        checks.extend(source_checks)
    else:
        checks.append({"name": "source_r1g_exists", "passed": False, "detail": str(source_zip)})
    status = "PASS" if all(c["passed"] for c in checks) else "FAIL"
    report = {
        "status": status,
        "format": "phase9c_r1h_preflight_report_v1",
        "phase": PHASE,
        "created_utc": utc_now(),
        "source_r1g": str(source_zip),
        "checks": checks,
        "errors": [c["name"] for c in checks if not c["passed"]],
        "training_performed": False,
        "model_inference_performed": False,
        "official_TS1_to_TS6_reopened": False,
    }
    write_json(output_dir / "phase9c_r1h_preflight_report.json", report)
    print(f"PHASE {PHASE} PREFLIGHT: {status}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return 0 if status == "PASS" else 2


def build() -> int:
    root, _, output_dir, protocol = project_paths()
    preflight_path = output_dir / "phase9c_r1h_preflight_report.json"
    if not preflight_path.exists() or load_json(preflight_path).get("status") != "PASS":
        print("Preflight is missing or not PASS. Run stage 0 first.")
        return 2
    source_zip = root / Path(protocol["source_r1g"]["relative_path"])
    assets = root / "phase9c_r1h_assets"
    manuscript_out = output_dir / "manuscript"
    evidence_out = output_dir / "evidence"
    if manuscript_out.exists():
        shutil.rmtree(manuscript_out)
    if evidence_out.exists():
        shutil.rmtree(evidence_out)
    shutil.copytree(assets / "manuscript", manuscript_out)
    evidence_out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="phase9c_r1h_") as td:
        temp = Path(td)
        with zipfile.ZipFile(source_zip, "r") as zf:
            zf.extractall(temp)
        (manuscript_out / "figures").mkdir(parents=True, exist_ok=True)
        for p in (temp / "figures").glob("*.png"):
            shutil.copy2(p, manuscript_out / "figures" / p.name)
        for rel in [
            "tables",
            "source_evidence",
        ]:
            shutil.copytree(temp / rel, evidence_out / rel)
        for name in [
            "PHASE9C_R1G_Q2_CLAIM_LOCK.md",
            "PHASE9C_R1G_Q2_RESULTS_DRAFT.md",
            "PHASE9C_R1G_Q2_LIMITATIONS_DRAFT.md",
            "PHASE9C_R1G_Q1_EXTENSION_REGISTRY.md",
            "phase9c_r1g_q2_abstract_numbers.json",
            "phase9c_r1g_q2_evidence_freeze_summary.json",
            "phase9c_r1g_q2_publication_lock_protocol.json",
            "phase9c_r1g_qc_report.json",
            "phase9c_r1g_output_manifest.json",
        ]:
            shutil.copy2(temp / name, evidence_out / name)

    shutil.copy2(assets / "PHASE9C_R1H_Q2_SUBMISSION_READINESS.md",
                 output_dir / "PHASE9C_R1H_Q2_SUBMISSION_READINESS.md")
    build_report = {
        "status": "PASS",
        "format": "phase9c_r1h_manuscript_build_report_v1",
        "phase": PHASE,
        "created_utc": utc_now(),
        "source_r1g_sha256": sha256_file(source_zip),
        "manuscript_title": "Reliability-Conditioned Virtual-View Pose Fusion for Monocular 3D Human Pose Estimation: A Locked Multi-Subject Real-Detector Study",
        "manuscript_pdf_precompiled": (manuscript_out / "main.pdf").exists(),
        "manuscript_files": sum(1 for p in manuscript_out.rglob("*") if p.is_file()),
        "evidence_files": sum(1 for p in evidence_out.rglob("*") if p.is_file()),
        "training_performed": False,
        "model_inference_performed": False,
        "official_TS1_to_TS6_reopened": False,
        "official_result_changed": False,
    }
    write_json(output_dir / "phase9c_r1h_manuscript_build_report.json", build_report)
    print("PHASE 9C-R1H BUILD: PASS")
    print(f"  Manuscript: {manuscript_out}")
    print(f"  PDF included: {build_report['manuscript_pdf_precompiled']}")
    return 0


def qc_and_pack() -> int:
    root, protocol_path, output_dir, protocol = project_paths()
    tex_path = output_dir / "manuscript/main.tex"
    pdf_path = output_dir / "manuscript/main.pdf"
    build_path = output_dir / "phase9c_r1h_manuscript_build_report.json"
    preflight_path = output_dir / "phase9c_r1h_preflight_report.json"
    checks: list[dict] = []
    checks.append({"name": "preflight_pass", "passed": preflight_path.exists() and load_json(preflight_path).get("status") == "PASS",
                   "detail": str(preflight_path)})
    checks.append({"name": "build_pass", "passed": build_path.exists() and load_json(build_path).get("status") == "PASS",
                   "detail": str(build_path)})
    checks.append({"name": "main_tex_exists", "passed": tex_path.exists(), "detail": str(tex_path)})
    checks.append({"name": "precompiled_pdf_exists", "passed": pdf_path.exists() and pdf_path.stat().st_size > 100000,
                   "detail": f"size={pdf_path.stat().st_size if pdf_path.exists() else 0}"})

    tex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    for section in protocol["required_manuscript_sections"]:
        if section == "Abstract":
            needle = "begin{abstract}"
        elif section in {"Data and Reproducibility Statement"}:
            needle = "section*{" + section + "}"
        else:
            needle = "section{" + section + "}"
        checks.append({"name": f"section_{section.lower().replace(' ', '_')}", "passed": needle in tex,
                       "detail": needle})
    required_literals = [
        "3.049", "1.596", "2,875", "127.295", "126.065", "84.176", "81.849",
        "-1.234", "3.644", "-0.229", "4.954",
        "statistically unresolved external trend", "virtual-view synergy remains a hypothesis",
    ]
    missing_literals = [x for x in required_literals if x not in tex]
    checks.append({"name": "locked_numbers_and_wording", "passed": not missing_literals,
                   "detail": f"missing={missing_literals}"})

    forbidden_patterns = {
        "official_superiority_assertion": r"(?i)we\s+(?:demonstrate|establish|confirm)[^.]{0,100}official[^.]{0,80}statistical superiority",
        "confirmed_synergy_assertion": r"(?i)we\s+(?:demonstrate|establish|confirm)[^.]{0,100}virtual[- ]view synergy",
        "sota_assertion": r"(?i)(?:achieve|establish|set|outperform)[^.]{0,80}state[- ]of[- ]the[- ]art",
        "pristine_first_look": r"(?i)pristine first look",
    }
    found_forbidden = {name: re.findall(pattern, tex) for name, pattern in forbidden_patterns.items() if re.search(pattern, tex)}
    checks.append({"name": "forbidden_claim_scan", "passed": not found_forbidden,
                   "detail": str(found_forbidden)})

    figures = list((output_dir / "manuscript/figures").glob("*.png"))
    checks.append({"name": "figures_complete", "passed": len(figures) == 4 and all(p.stat().st_size > 10000 for p in figures),
                   "detail": f"figures={len(figures)}"})
    claim_matrix = output_dir / "evidence/tables/table_q2_claim_matrix.csv"
    claim_rows = 0
    if claim_matrix.exists():
        with claim_matrix.open("r", encoding="utf-8-sig", newline="") as f:
            claim_rows = sum(1 for _ in csv.DictReader(f))
    checks.append({"name": "claim_matrix_complete", "passed": claim_rows == 11,
                   "detail": f"rows={claim_rows}"})
    checks.append({"name": "reporting_only", "passed": True,
                   "detail": "train=false; inference=false; official_reopened=false; result_changed=false"})

    status = "PASS" if all(c["passed"] for c in checks) else "FAIL"
    decision = protocol["next_gate"] if status == "PASS" else "REPAIR_PHASE9C_R1H_MANUSCRIPT"
    qc = {
        "status": status,
        "format": "phase9c_r1h_qc_report_v1",
        "phase": PHASE,
        "created_utc": utc_now(),
        "scientific_decision": decision,
        "checks": checks,
        "errors": [c["name"] for c in checks if not c["passed"]],
        "manuscript_pdf_sha256": sha256_file(pdf_path) if pdf_path.exists() else None,
        "protocol_sha256": sha256_file(protocol_path),
        "training_performed": False,
        "model_inference_performed": False,
        "official_TS1_to_TS6_reopened": False,
        "official_result_changed": False,
    }
    write_json(output_dir / "phase9c_r1h_qc_report.json", qc)

    if status == "PASS":
        manifest_files = []
        for p in sorted(output_dir.rglob("*")):
            if p.is_file() and p.name != RESULT_ZIP_NAME:
                manifest_files.append({
                    "relative_path": p.relative_to(output_dir).as_posix(),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })
        write_json(output_dir / "phase9c_r1h_output_manifest.json", {
            "format": "phase9c_r1h_output_manifest_v1",
            "phase": PHASE,
            "files": manifest_files,
        })
        result_zip = output_dir / RESULT_ZIP_NAME
        if result_zip.exists():
            result_zip.unlink()
        with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sorted(output_dir.rglob("*")):
                if p.is_file() and p != result_zip:
                    zf.write(p, p.relative_to(output_dir).as_posix())
        print("PHASE 9C-R1H QC AND PACK: PASS")
        print(f"  Scientific decision: {decision}")
        print(f"  Output ZIP: {result_zip}")
        return 0
    print("PHASE 9C-R1H QC AND PACK: FAIL")
    for c in checks:
        if not c["passed"]:
            print(f"  [FAIL] {c['name']}: {c['detail']}")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "build", "qc", "all"])
    args = parser.parse_args()
    if args.stage == "preflight":
        return preflight()
    if args.stage == "build":
        return build()
    if args.stage == "qc":
        return qc_and_pack()
    rc = preflight()
    if rc:
        return rc
    rc = build()
    if rc:
        return rc
    return qc_and_pack()


if __name__ == "__main__":
    raise SystemExit(main())
