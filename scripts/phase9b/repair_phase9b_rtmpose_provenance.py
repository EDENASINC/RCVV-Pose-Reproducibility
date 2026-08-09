from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_RTMLIB_VERSION = "0.0.15"
EXPECTED_MODE = "performance"
EXPECTED_URLS = {
    "person_detector": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
    ),
    "pose_estimator": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-"
        "71d7b7e9_20230629.zip"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def download_reusable(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            with zipfile.ZipFile(destination) as archive:
                if archive.testzip() is None:
                    print(f"[REUSE] {destination}")
                    return destination
        except zipfile.BadZipFile:
            pass
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "vv-pose-phase9b-provenance/1.0"},
    )
    print(f"[DOWNLOAD] {url}")
    with urllib.request.urlopen(request, timeout=120) as response:
        with partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    with zipfile.ZipFile(partial) as archive:
        failed_member = archive.testzip()
        if failed_member is not None:
            raise RuntimeError(f"Corrupt ZIP member: {failed_member}")
    os.replace(partial, destination)
    return destination


def extract_onnx_and_hashes(
    archive_path: Path,
    destination: Path,
    role: str,
    source_url: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    models: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            info for info in archive.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() == ".onnx"
        )
        if not members:
            raise RuntimeError(f"No ONNX model found in {archive_path}")
        used_names: set[str] = set()
        for info in members:
            output_name = Path(info.filename).name
            if output_name in used_names:
                raise RuntimeError(f"Duplicate ONNX basename in archive: {output_name}")
            used_names.add(output_name)
            output_path = destination / output_name
            with archive.open(info) as source, output_path.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            models.append(
                {
                    "role": role,
                    "path": str(output_path.resolve()),
                    "source_url": source_url,
                    "archive_name": archive_path.name,
                    "archive_member": info.filename,
                    "size_bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                }
            )
    archive_record = {
        "role": role,
        "path": str(archive_path.resolve()),
        "source_url": source_url,
        "size_bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }
    return models, archive_record


def validate_locked_configuration() -> tuple[type, Path, str]:
    version = importlib.metadata.version("rtmlib")
    if version != EXPECTED_RTMLIB_VERSION:
        raise RuntimeError(
            f"rtmlib version changed: expected {EXPECTED_RTMLIB_VERSION}, got {version}"
        )
    from rtmlib import Body

    mode = Body.MODE.get(EXPECTED_MODE, {})
    observed = {
        "person_detector": str(mode.get("det", "")),
        "pose_estimator": str(mode.get("pose", "")),
    }
    if observed != EXPECTED_URLS:
        raise RuntimeError(
            "rtmlib Body.MODE['performance'] no longer matches the Phase 9B lock: "
            + json.dumps(observed, indent=2)
        )
    source_path = Path(inspect.getsourcefile(Body) or "")
    if not source_path.is_file():
        raise RuntimeError("Cannot resolve the installed rtmlib Body source file.")
    return Body, source_path.resolve(), version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs" / "phase9b_rgb_detector_benchmark"
    full = output / "full"
    runtime_path = full / "phase9b_detector_runtime.json"
    qc_path = output / "phase9b_qc_report.json"
    report_path = output / "phase9b_r3_provenance_repair_report.json"
    model_root = output / "provenance_models" / "rtmpose_performance"

    report: dict[str, Any] = {
        "format": "phase9b_r3_provenance_repair_v1",
        "phase": "9B-R3",
        "status": "BLOCKED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_repeated": False,
        "official_TS1_to_TS6_read": False,
        "checks": [],
        "error": None,
    }
    try:
        runtime = load_json(runtime_path)
        qc_before = load_json(qc_path)
        report["checks"].append(
            {
                "name": "previous_blocker_is_provenance_only",
                "passed": qc_before.get("blockers") == [
                    "detector_runtime_and_weight_hashes_recorded: "
                    "{'rtmpose_performance': 0, 'yolo11l_pose': 1}"
                ],
                "detail": qc_before.get("blockers"),
            }
        )
        if not report["checks"][-1]["passed"]:
            raise RuntimeError(
                "The previous QC blocker is not the expected RTMPose provenance-only blocker."
            )
        if runtime.get("detectors", {}).get("rtmpose_performance", {}).get(
            "status"
        ) not in {"completed_on_gpu", "complete_cache_reused"}:
            raise RuntimeError("RTMPose full detector run is not marked complete.")
        if runtime.get("detectors", {}).get("yolo11l_pose", {}).get(
            "status"
        ) not in {"completed_on_gpu", "complete_cache_reused"}:
            raise RuntimeError("YOLO full detector run is not marked complete.")

        _body, source_path, version = validate_locked_configuration()
        archives_dir = model_root / "archives"
        extracted_dir = model_root / "onnx"
        model_files: list[dict[str, Any]] = []
        source_archives: list[dict[str, Any]] = []
        for role, url in EXPECTED_URLS.items():
            archive_path = download_reusable(url, archives_dir / Path(url).name)
            models, archive_record = extract_onnx_and_hashes(
                archive_path, extracted_dir, role, url
            )
            model_files.extend(models)
            source_archives.append(archive_record)
        roles_found = {item["role"] for item in model_files}
        if roles_found != set(EXPECTED_URLS):
            raise RuntimeError(
                f"Expected one or more ONNX files for each locked role; got {roles_found}"
            )

        detector_runtime = runtime["detectors"]["rtmpose_performance"]
        detector_runtime["model_files"] = model_files
        detector_runtime["source_archives"] = source_archives
        detector_runtime["implementation"] = {
            "package": "rtmlib",
            "version": version,
            "mode": EXPECTED_MODE,
            "source_file": str(source_path),
            "source_file_sha256": sha256_file(source_path),
            "configuration_urls_verified": True,
        }
        detector_runtime["provenance_repair"] = {
            "phase": "9B-R3",
            "timestamp_utc": report["timestamp_utc"],
            "inference_repeated": False,
            "prediction_cache_changed": False,
        }
        runtime["format"] = "phase9b_detector_runtime_v3"
        write_json_atomic(runtime_path, runtime)

        cache_path = full / "phase9b_detector_cache.npz"
        report.update(
            {
                "status": "PASS",
                "rtmlib_version": version,
                "rtmlib_source_file_sha256": sha256_file(source_path),
                "model_files": model_files,
                "source_archives": source_archives,
                "detector_cache": {
                    "path": str(cache_path),
                    "size_bytes": cache_path.stat().st_size,
                    "sha256": sha256_file(cache_path),
                    "changed": False,
                },
                "checks": report["checks"]
                + [
                    {
                        "name": "two_locked_rtmpose_onnx_models_hashed",
                        "passed": roles_found == set(EXPECTED_URLS)
                        and all(len(item["sha256"]) == 64 for item in model_files),
                        "detail": [item["role"] for item in model_files],
                    },
                    {
                        "name": "no_inference_repeated",
                        "passed": True,
                        "detail": "Runtime provenance JSON only; predictions/cache untouched",
                    },
                ],
            }
        )
        write_json_atomic(report_path, report)
        print(json.dumps(report, indent=2))
        print(f"PROVENANCE_REPORT={report_path}")
    except Exception as error:
        report["error"] = repr(error)
        write_json_atomic(report_path, report)
        print(json.dumps(report, indent=2), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
