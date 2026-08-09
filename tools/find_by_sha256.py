#!/usr/bin/env python3
"""Locate files under a directory by SHA-256 without relying on filenames."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("hashes", nargs="+", type=str.lower)
    args = parser.parse_args()
    wanted = set(args.hashes)
    found = set()
    for path in args.root.rglob("*"):
        if not path.is_file():
            continue
        try:
            value = digest(path)
        except (OSError, PermissionError):
            continue
        if value in wanted:
            print(f"{value}  {path}")
            found.add(value)
    for value in sorted(wanted - found):
        print(f"NOT_FOUND  {value}")


if __name__ == "__main__":
    main()
