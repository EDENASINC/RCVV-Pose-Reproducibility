#!/usr/bin/env python3
"""Bundle 72 checkpoints into eight reviewer-friendly ZIP release assets."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--release-url",
        default="",
        help="GitHub Release URL. Leave empty while preparing assets locally.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "models" / "checkpoint_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 72:
        raise SystemExit(f"Expected 72 manifest rows; found {len(records)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for arm in ("O", "OV", "OR", "OVR"):
        for detector in ("rtmpose_performance", "yolo11l_pose"):
            selected = [r for r in records if r["arm_short"] == arm and r["detector"] == detector]
            if len(selected) != 9:
                raise SystemExit(f"Expected 9 rows for {arm}/{detector}; found {len(selected)}")
            short_detector = "rtmpose" if detector == "rtmpose_performance" else "yolo11l"
            archive = args.output_dir / f"checkpoints_{arm}_{short_detector}.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for row in selected:
                    relative = Path(row["destination_path"])
                    source = args.checkpoint_root / relative.relative_to("checkpoints")
                    if not source.is_file():
                        raise SystemExit(f"Missing {source}")
                    zf.write(source, arcname=relative.as_posix())
            assets.append({"file": archive.name, "sha256": sha256(archive), "size_bytes": archive.stat().st_size})
            print(f"Built {archive} ({len(selected)} checkpoints)")
    generated = {
        "paper_manuscript_id": "information-4512788",
        "release_tag": "v1.0.0-paper-information-4512788",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "assets": assets,
    }
    (args.output_dir / "release_assets.generated.json").write_text(
        json.dumps(generated, indent=2) + "\n", encoding="utf-8"
    )

    # Write the repository-side manifest as part of the same deterministic
    # preparation step.  A URL is intentionally optional before the GitHub
    # Release exists; it can be supplied by rerunning this command afterwards.
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
    print(
        "Release status: "
        + ("PUBLISHED" if args.release_url else "READY_FOR_UPLOAD")
    )


if __name__ == "__main__":
    main()
