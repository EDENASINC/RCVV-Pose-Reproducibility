#!/usr/bin/env python3
"""Check evidence-only or complete release readiness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"MISSING: {message}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("evidence", "release", "published"), default="release"
    )
    args = parser.parse_args()

    validation = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "validate_evidence.py")], check=False
    )
    if validation.returncode:
        return validation.returncode
    if args.mode == "evidence":
        print("PASS: evidence-only package is ready.")
        return 0

    errors: list[str] = []
    source_requirements = (
        ROOT / "src" / "datasets" / "mpi_inf_3dhp_dataset.py",
        ROOT / "src" / "metrics" / "pose_metrics.py",
        ROOT / "scripts" / "train" / "run_phase9c_r1d_locked_full_confirmation.py",
        ROOT / "scripts" / "train" / "run_phase9c_r1e_synergy_attribution.py",
        ROOT / "scripts" / "eval" / "run_phase9c_r1f_locked_official_test.py",
    )
    for path in source_requirements:
        if not path.is_file():
            fail(f"original source {path.relative_to(ROOT)}", errors)

    manifest_path = ROOT / "models" / "checkpoint_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 72:
        fail(f"exactly 72 checkpoint manifest rows (found {len(rows)})", errors)
    for row in rows:
        path = ROOT / row["destination_path"]
        if not path.is_file():
            fail(f"checkpoint {row['logical_id']} at {row['destination_path']}", errors)

    for name in (
        "phase9c_a_protocol_lock.json",
        "phase9c_r1d_locked_full_confirmation_protocol.json",
        "phase9c_r1e_synergy_attribution_protocol.json",
        "phase9c_r1f_locked_official_test_protocol.json",
    ):
        if not (ROOT / "configs" / name).is_file():
            fail(f"executable config configs/{name}", errors)

    release_manifest_path = ROOT / "models" / "release_manifest.json"
    try:
        release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"valid model release manifest ({exc})", errors)
        release_manifest = {}

    expected_statuses = {"READY_FOR_UPLOAD", "PUBLISHED"}
    if release_manifest.get("status") not in expected_statuses:
        fail("model release manifest generated from the eight local assets", errors)
    assets = release_manifest.get("assets")
    if not isinstance(assets, list) or len(assets) != 8:
        fail("exactly 8 model release assets in models/release_manifest.json", errors)
        assets = []
    for asset in assets:
        name = asset.get("file", "") if isinstance(asset, dict) else ""
        path = ROOT / "release_assets" / name
        if not name or not path.is_file():
            fail(f"local release asset release_assets/{name or '<unnamed>'}", errors)
            continue
        if path.stat().st_size != asset.get("size_bytes"):
            fail(f"matching size for release asset {name}", errors)
        if sha256(path) != asset.get("sha256"):
            fail(f"matching SHA-256 for release asset {name}", errors)

    if args.mode == "published":
        if release_manifest.get("status") != "PUBLISHED":
            fail("release manifest status PUBLISHED", errors)
        if not str(release_manifest.get("release_url", "")).startswith("https://github.com/"):
            fail("GitHub release URL in models/release_manifest.json", errors)

    forbidden_roots = (ROOT / "data", ROOT / "datasets")
    for forbidden in forbidden_roots:
        if forbidden.exists() and any(forbidden.rglob("*")):
            errors.append(f"FORBIDDEN: dataset content present under {forbidden.relative_to(ROOT)}")

    if errors:
        print(f"FAIL: release is not ready ({len(errors)} issue(s)).")
        return 1
    if args.mode == "published":
        print("PASS: published release contract is complete.")
    else:
        print("PASS: release assets are ready for GitHub upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
