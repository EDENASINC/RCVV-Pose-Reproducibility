#!/usr/bin/env python3
"""Print deterministic SHA256SUMS lines for supplied files or trees."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    files = set()
    for path in args.paths:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            files.update(item.resolve() for item in path.rglob("*") if item.is_file())
    for path in sorted(files, key=str):
        print(f"{sha256(path)}  {path}")


if __name__ == "__main__":
    main()
