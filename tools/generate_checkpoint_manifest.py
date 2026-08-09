#!/usr/bin/env python3
"""Generate the locked 72-checkpoint inventory and local import map."""

from __future__ import annotations

import csv
from pathlib import Path

SPLITS = ("split_a", "split_b", "split_c")
SEEDS = (42, 123, 2026)
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
ARMS = (
    ("O", "detector_aware_observed"),
    ("OV", "detector_conditioned_virtual_view_no_confidence"),
    ("OR", "bounded_reliability_observed_modulation"),
    ("OVR", "bounded_reliability_dual_modulation"),
)


def rows():
    for split in SPLITS:
        for seed in SEEDS:
            for detector in DETECTORS:
                for short_arm, arm in ARMS:
                    logical_id = f"{split}__seed{seed}__{detector}__{short_arm}"
                    destination = f"checkpoints/{split}/seed_{seed}/{detector}/{arm}/best.pt"
                    yield {
                        "logical_id": logical_id,
                        "split": split,
                        "seed": seed,
                        "detector": detector,
                        "arm_short": short_arm,
                        "arm": arm,
                        "destination_path": destination,
                        "source_path": "",
                        "sha256": "",
                        "size_bytes": "",
                        "selected_epoch": "",
                        "validation_selection_score": "",
                    }


def write_csv(path: Path, records) -> None:
    records = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    records = list(rows())
    write_csv(root / "models" / "checkpoint_manifest.csv", records)
    write_csv(root / "models" / "local_checkpoint_map.csv", records)
    print(f"Wrote {len(records)} locked checkpoint rows.")


if __name__ == "__main__":
    main()
