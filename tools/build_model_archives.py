#!/usr/bin/env python3
"""Build eight checkpoint ZIPs and one learned-artifact ZIP for release."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import zipfile
from pathlib import Path


CHECKPOINT_ASSET_NAMES = tuple(
    f"checkpoints_{arm}_{detector}.zip"
    for arm in ("O", "OV", "OR", "OVR")
    for detector in ("rtmpose", "yolo11l")
)
LEARNED_ASSET_NAME = "learned_calibration_artifacts.zip"
EXPECTED_RELEASE_ASSET_NAMES = CHECKPOINT_ASSET_NAMES + (LEARNED_ASSET_NAME,)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_record(path: Path) -> dict[str, str | int]:
    return {
        "file": path.name,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def add_tree(archive: zipfile.ZipFile, source_root: Path, archive_root: Path) -> int:
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    for path in files:
        archive.write(path, arcname=(archive_root / path.relative_to(source_root)).as_posix())
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("artifacts/calibration"),
        help="Directory containing the released calibration model files.",
    )
    parser.add_argument(
        "--residual-risk-bank",
        type=Path,
        default=Path("local_release_inputs/learned_artifacts/phase9c_a_residual_bank.npz"),
        help="Locked residual-risk bank to include in the learned-artifact ZIP.",
    )
    parser.add_argument(
        "--release-url",
        default="",
        help="GitHub Release URL. Leave empty while preparing assets locally.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    checkpoint_root = args.checkpoint_root if args.checkpoint_root.is_absolute() else root / args.checkpoint_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    calibration_root = args.calibration_root if args.calibration_root.is_absolute() else root / args.calibration_root
    residual_bank = args.residual_risk_bank if args.residual_risk_bank.is_absolute() else root / args.residual_risk_bank

    manifest_path = root / "models" / "checkpoint_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 72:
        raise SystemExit(f"Expected 72 manifest rows; found {len(records)}")

    calibration_files = sorted(path for path in calibration_root.rglob("*") if path.is_file()) if calibration_root.is_dir() else []
    if not calibration_files:
        raise SystemExit(f"Missing calibration files under {calibration_root}")
    if not residual_bank.is_file():
        raise SystemExit(f"Missing residual-risk bank {residual_bank}")

    output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, str | int]] = []
    for arm in ("O", "OV", "OR", "OVR"):
        for detector in ("rtmpose_performance", "yolo11l_pose"):
            selected = [row for row in records if row["arm_short"] == arm and row["detector"] == detector]
            if len(selected) != 9:
                raise SystemExit(f"Expected 9 rows for {arm}/{detector}; found {len(selected)}")
            short_detector = "rtmpose" if detector == "rtmpose_performance" else "yolo11l"
            archive_path = output_dir / f"checkpoints_{arm}_{short_detector}.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for row in selected:
                    relative = Path(row["destination_path"])
                    source = checkpoint_root / relative.relative_to("checkpoints")
                    if not source.is_file():
                        raise SystemExit(f"Missing {source}")
                    archive.write(source, arcname=relative.as_posix())
            assets.append(asset_record(archive_path))
            print(f"Built {archive_path} ({len(selected)} checkpoints)")

    learned_archive = output_dir / LEARNED_ASSET_NAME
    with zipfile.ZipFile(learned_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        calibration_count = add_tree(archive, calibration_root, Path("artifacts/calibration"))
        archive.write(
            residual_bank,
            arcname="artifacts/learned/phase9c_a_residual_bank.npz",
        )
    assets.append(asset_record(learned_archive))
    print(f"Built {learned_archive} ({calibration_count} calibration file(s) + residual-risk bank)")

    names = tuple(asset["file"] for asset in assets)
    if names != EXPECTED_RELEASE_ASSET_NAMES:
        raise SystemExit(f"Unexpected release asset order: {names}")

    generated = {
        "paper_manuscript_id": "information-4512788",
        "release_tag": "v1.0.0-paper-information-4512788",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "assets": assets,
    }
    (output_dir / "release_assets.generated.json").write_text(
        json.dumps(generated, indent=2) + "\n", encoding="utf-8"
    )

    release_manifest = {
        "status": "PUBLISHED" if args.release_url else "READY_FOR_UPLOAD",
        "paper_manuscript_id": generated["paper_manuscript_id"],
        "release_tag": generated["release_tag"],
        "release_url": args.release_url,
        "assets": assets,
    }
    release_manifest_path = root / "models" / "release_manifest.json"
    release_manifest_path.write_text(
        json.dumps(release_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {release_manifest_path}")
    print("Release status: " + release_manifest["status"])


if __name__ == "__main__":
    main()
