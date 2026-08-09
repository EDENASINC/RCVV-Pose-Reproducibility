#!/usr/bin/env python3
"""Collect the locked RCVV-Pose artifacts from the original Windows project.

This script is intentionally conservative. It copies code/config/lightweight
provenance into this repository, places model binaries in Git-ignored release
input folders, and never copies the MPI-INF-3DHP dataset or detector caches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("split_a", "split_b", "split_c")
SEEDS = (42, 123, 2026)
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
ARM_PHASE = {
    "detector_aware_observed": "phase9c_r1d_locked_full_confirmation",
    "detector_conditioned_virtual_view_no_confidence": "phase9c_r1d_locked_full_confirmation",
    "bounded_reliability_observed_modulation": "phase9c_r1e_synergy_attribution",
    "bounded_reliability_dual_modulation": "phase9c_r1d_locked_full_confirmation",
}
ARM_SHORT = {
    "detector_aware_observed": "O",
    "detector_conditioned_virtual_view_no_confidence": "OV",
    "bounded_reliability_observed_modulation": "OR",
    "bounded_reliability_dual_modulation": "OVR",
}


@dataclass
class Record:
    category: str
    status: str
    source: str
    destination: str
    size_bytes: int | str = ""
    sha256: str = ""
    note: str = ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Collector:
    def __init__(self, project: Path, repo: Path, dry_run: bool, replace: bool) -> None:
        self.project = project.resolve()
        self.repo = repo.resolve()
        self.dry_run = dry_run
        self.replace = replace
        self.records: list[Record] = []

    def add_missing(self, category: str, source: Path, destination: Path, note: str = "") -> None:
        self.records.append(
            Record(category, "MISSING", str(source), str(destination), note=note)
        )

    def copy(self, source: Path, destination: Path, category: str, required: bool = True) -> bool:
        if not source.is_file():
            if required:
                self.add_missing(category, source, destination)
            return False

        size = source.stat().st_size
        source_hash = sha256(source)
        if destination.exists() and not self.replace:
            destination_hash = sha256(destination)
            if destination_hash == source_hash:
                self.records.append(
                    Record(category, "UNCHANGED", str(source), str(destination), size, source_hash)
                )
                return True
            self.records.append(
                Record(
                    category,
                    "CONFLICT",
                    str(source),
                    str(destination),
                    size,
                    source_hash,
                    "Use --replace-existing only after reviewing the existing file.",
                )
            )
            return False

        if not self.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        self.records.append(
            Record(category, "DRY-RUN" if self.dry_run else "COPIED", str(source), str(destination), size, source_hash)
        )
        return True

    def copy_tree_files(self, source_root: Path, destination_root: Path, category: str, suffixes: tuple[str, ...]) -> None:
        if not source_root.is_dir():
            self.add_missing(category, source_root, destination_root, "Required directory not found.")
            return
        for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
            if source.suffix.lower() in suffixes and "__pycache__" not in source.parts:
                self.copy(source, destination_root / source.relative_to(source_root), category)

    def collect_code_and_configs(self) -> None:
        self.copy_tree_files(self.project / "src", self.repo / "src", "git_source", (".py",))

        script_patterns = (
            "scripts/phase9b/*.py",
            "scripts/prepare/*phase9c*.py",
            "scripts/train/*phase9c*.py",
            "scripts/qc/*phase9c*.py",
            "scripts/eval/*phase9c*.py",
            "scripts/report/*phase9c*.py",
            "scripts/reporting/*phase9c*.py",
            "scripts/phase9c*.py",
        )
        selected: set[Path] = set()
        for pattern in script_patterns:
            selected.update(path for path in self.project.glob(pattern) if path.is_file())
        for source in sorted(selected):
            self.copy(source, self.repo / source.relative_to(self.project), "git_script")

        config_patterns = (
            "configs/phase9*.json",
            "configs/datasets/mpi_inf_3dhp.yaml",
            "configs/skeletons/canonical_15j.yaml",
        )
        for pattern in config_patterns:
            matches = sorted(path for path in self.project.glob(pattern) if path.is_file())
            if not matches:
                self.add_missing("git_config", self.project / pattern, self.repo / pattern)
            for source in matches:
                self.copy(source, self.repo / source.relative_to(self.project), "git_config")

        mapping_root = self.project / "data" / "annotations" / "mappings"
        if mapping_root.is_dir():
            for source in sorted(mapping_root.glob("*.json")):
                destination = self.repo / "configs" / "joint_mappings" / source.name
                self.copy(source, destination, "git_joint_mapping")

        root_files = (
            "requirements_phase9b.txt",
            "README_PHASE9B_TH.md",
            "README_PHASE9C_A_TH.md",
            "README_PHASE9C_B0_TH.md",
            "README_PHASE9C_B1_TH.md",
            "README_PHASE9C_B2_TH.md",
            "README_PHASE9C_R1A_TH.md",
            "README_PHASE9C_R1B_TH.md",
            "README_PHASE9C_R1C_TH.md",
            "README_PHASE9C_R1D_TH.md",
            "README_PHASE9C_R1E_TH.md",
            "README_PHASE9C_R1F_TH.md",
            "README_PHASE9C_R1G_TH.md",
            "README_PHASE9C_R1H_TH.md",
            "PHASE9C_R1H_Q2_SUBMISSION_READINESS.md",
        )
        for name in root_files:
            source = self.project / name
            self.copy(source, self.repo / "docs" / "original" / name, "git_documentation", required=False)

    def collect_environment_and_evidence(self) -> None:
        environment = self.project / "outputs" / "environment"
        if environment.is_dir():
            for source in sorted(path for path in environment.iterdir() if path.is_file()):
                if source.suffix.lower() in (".txt", ".json", ".csv"):
                    self.copy(source, self.repo / "provenance" / "environment" / source.name, "git_environment")

        evidence = self.project / "deliverables" / "vv_pose_paper_mdpi_information_v1" / "evidence"
        if evidence.is_dir():
            for source in sorted(path for path in evidence.iterdir() if path.is_file()):
                if source.suffix.lower() in (".md", ".json", ".csv", ".txt"):
                    self.copy(source, self.repo / "evidence" / source.name, "git_evidence")
        else:
            self.add_missing("git_evidence", evidence, self.repo / "evidence", "The package already contains evidence 00-15; compare it with the original later.")

    def collect_locked_runs(self) -> None:
        manifest_rows: list[dict[str, str | int]] = []
        for split in SPLITS:
            for seed in SEEDS:
                for detector in DETECTORS:
                    for arm, phase in ARM_PHASE.items():
                        run = (
                            self.project
                            / "outputs"
                            / phase
                            / split
                            / f"seed{seed}"
                            / arm
                            / f"train_{detector}"
                        )
                        checkpoint_destination = (
                            self.repo
                            / "checkpoints"
                            / split
                            / f"seed_{seed}"
                            / detector
                            / arm
                            / "best.pt"
                        )
                        checkpoint = run / "best.pt"
                        checkpoint_ok = self.copy(checkpoint, checkpoint_destination, "release_checkpoint")

                        report_name = "phase9c_r1d_run_report.json" if "r1d" in phase else "phase9c_r1e_run_report.json"
                        provenance_destination = (
                            self.repo
                            / "provenance"
                            / "locked_runs"
                            / split
                            / f"seed_{seed}"
                            / detector
                            / arm
                        )
                        self.copy(run / report_name, provenance_destination / "run_report.json", "git_run_report")
                        self.copy(run / "training_history.csv", provenance_destination / "training_history.csv", "git_training_history")

                        manifest_rows.append(
                            {
                                "logical_id": f"{split}__seed{seed}__{detector}__{ARM_SHORT[arm]}",
                                "split": split,
                                "seed": seed,
                                "detector": detector,
                                "arm_short": ARM_SHORT[arm],
                                "arm": arm,
                                "source_phase": phase,
                                "source_path": str(checkpoint),
                                "destination_path": checkpoint_destination.relative_to(self.repo).as_posix(),
                                "sha256": sha256(checkpoint) if checkpoint_ok and checkpoint.is_file() else "",
                                "size_bytes": checkpoint.stat().st_size if checkpoint_ok and checkpoint.is_file() else "",
                            }
                        )

        if not self.dry_run:
            manifest_path = self.repo / "models" / "checkpoint_manifest.collected.csv"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
                writer.writeheader()
                writer.writerows(manifest_rows)

        for phase in ("phase9c_r1d_locked_full_confirmation", "phase9c_r1e_synergy_attribution"):
            phase_root = self.project / "outputs" / phase
            prefix = "phase9c_r1d" if "r1d" in phase else "phase9c_r1e"
            names = (
                f"{prefix}_artifact_hashes.csv",
                f"{prefix}_preflight_report.json",
                f"{prefix}_qc_report.json",
                f"{prefix}_statistics.json",
                f"{prefix}_summary.csv",
                f"{prefix}_full_metrics.csv",
            )
            for name in names:
                self.copy(
                    phase_root / name,
                    self.repo / "evidence" / "training_generated" / phase / name,
                    "git_training_summary",
                    required=False,
                )

    def collect_learned_artifacts(self) -> None:
        source_root = self.project / "outputs" / "phase9c_a_calibrated_error_bank"
        for name in (
            "phase9c_a_build_report.json",
            "phase9c_a_calibration_models.json",
            "phase9c_a_calibration_summary.csv",
            "phase9c_a_protocol_lock.json",
            "phase9c_a_qc_report.json",
        ):
            self.copy(source_root / name, self.repo / "artifacts" / "calibration" / name, "git_calibration")
        self.copy(
            source_root / "phase9c_a_residual_bank.npz",
            self.repo / "local_release_inputs" / "learned_artifacts" / "phase9c_a_residual_bank.npz",
            "release_learned_artifact",
        )

    def collect_detector_provenance(self) -> None:
        phase9b = self.project / "outputs" / "phase9b_rgb_detector_benchmark"
        for subdir in ("full", ""):
            root = phase9b / subdir if subdir else phase9b
            if not root.is_dir():
                continue
            for source in sorted(path for path in root.iterdir() if path.is_file()):
                if source.suffix.lower() in (".json", ".csv", ".txt"):
                    self.copy(
                        source,
                        self.repo / "provenance" / "detectors" / (subdir or "root") / source.name,
                        "git_detector_provenance",
                    )

        yolo_candidates: list[Path] = []
        for root, dirs, files in os.walk(self.project):
            dirs[:] = [d for d in dirs if d.lower() not in {".venv", "venv", "data", "__pycache__"}]
            for filename in files:
                if filename.lower() == "yolo11l-pose.pt":
                    yolo_candidates.append(Path(root) / filename)
        note_path = self.repo / "local_release_inputs" / "detectors" / "DETECTOR_WEIGHTS_STATUS.txt"
        expected = "61921abe1f2ed930bf28328a16b3162278cdb239a1e8280b6ac827119f13cee0"
        lines = [
            "Detector weights are third-party artifacts and must not be committed to Git.\n",
            f"Expected yolo11l-pose.pt SHA-256: {expected}\n",
        ]
        if yolo_candidates:
            for candidate in yolo_candidates:
                lines.append(f"FOUND: {candidate} SHA-256={sha256(candidate)}\n")
        else:
            lines.append("NOT FOUND inside the project. Use the official provider download and verify the SHA-256.\n")
        if not self.dry_run:
            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_text("".join(lines), encoding="utf-8")

    def collect_official_lightweight(self, include_derived: bool) -> None:
        source_root = self.project / "outputs" / "phase9c_r1f_locked_official_test"
        lightweight_names = (
            "phase9c_r1f_activity_metrics_descriptive.csv",
            "phase9c_r1f_checkpoint_inventory.csv",
            "phase9c_r1f_detector_cache_report.json",
            "phase9c_r1f_official_result_lock.json",
            "phase9c_r1f_official_test_summary.json",
            "phase9c_r1f_output_manifest.json",
            "phase9c_r1f_preflight_report.json",
            "phase9c_r1f_qc_report.json",
            "phase9c_r1f_subject_system_metrics.csv",
        )
        for name in lightweight_names:
            self.copy(source_root / name, self.repo / "evidence" / "official_generated" / name, "git_official_summary")

        if include_derived:
            for name in (
                "phase9c_r1f_ensemble_predictions.npz",
                "phase9c_r1f_per_frame_errors.npz",
            ):
                self.copy(
                    source_root / name,
                    self.repo / "private_review_only" / "official_derived" / name,
                    "private_official_derived",
                )

    def write_report(self) -> Path:
        report_dir = self.repo / "collection_report"
        report_path = report_dir / "collected_files.csv"
        summary_path = report_dir / "COLLECTION_SUMMARY.json"
        if not self.dry_run:
            report_dir.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(Record.__annotations__))
                writer.writeheader()
                writer.writerows(record.__dict__ for record in self.records)
            counts: dict[str, int] = {}
            for record in self.records:
                counts[record.status] = counts.get(record.status, 0) + 1
            summary = {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "project_root": str(self.project),
                "repository_root": str(self.repo),
                "expected_locked_checkpoints": 72,
                "status_counts": counts,
                "dataset_copied": False,
                "notes": [
                    "checkpoints/ and local_release_inputs/ are intentionally ignored by Git",
                    "private_review_only/ must not be published without a dataset-license review",
                    "Review collected_files.csv and git status before committing",
                ],
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return report_path


def validate_roots(project: Path, repo: Path) -> None:
    if not project.is_dir():
        raise SystemExit(f"Project root not found: {project}")
    if not (project / "outputs").is_dir() or not (project / "scripts").is_dir():
        raise SystemExit(f"This does not look like vv_pose_project: {project}")
    try:
        repo.resolve().relative_to((project / "data").resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("Refusing to write the repository inside the dataset directory.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect locked RCVV-Pose GitHub/release artifacts")
    parser.add_argument("--project-root", type=Path, default=Path(r"D:\Research\vv_pose_project"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="List and hash files without copying")
    parser.add_argument("--replace-existing", action="store_true", help="Replace destination files after explicit review")
    parser.add_argument(
        "--include-private-official-derived",
        action="store_true",
        help="Copy selected Official NPZ files to private_review_only (never to Git)",
    )
    args = parser.parse_args()

    validate_roots(args.project_root, args.repo_root)
    collector = Collector(args.project_root, args.repo_root, args.dry_run, args.replace_existing)
    collector.collect_code_and_configs()
    collector.collect_environment_and_evidence()
    collector.collect_locked_runs()
    collector.collect_learned_artifacts()
    collector.collect_detector_provenance()
    collector.collect_official_lightweight(args.include_private_official_derived)
    report = collector.write_report()

    counts: dict[str, int] = {}
    for record in collector.records:
        counts[record.status] = counts.get(record.status, 0) + 1
    print(json.dumps(counts, indent=2))
    if not args.dry_run:
        print(f"Report: {report}")
        print("Next: review collection_report, then run python tools/release_preflight.py --mode release")
    return 1 if counts.get("MISSING", 0) or counts.get("CONFLICT", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
