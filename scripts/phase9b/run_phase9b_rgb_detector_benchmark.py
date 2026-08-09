from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SUBJECTS = tuple(f"S{i}" for i in range(1, 9))
SEQUENCES = ("Seq1", "Seq2")
JOINT_NAMES = (
    "pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee",
    "l_ankle", "neck", "head", "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
)
COMMON12 = np.asarray((1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14))
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
ARCHIVE_SUFFIXES = {".zip", ".7z", ".rar", ".tar", ".gz"}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}
DATASET_DIR_NAMES = (
    "MPI-INF-3DHP",
    "MPI-INF-3DHP-master",
    "mpi_inf_3dhp",
    "mpi-inf-3dhp",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def output_root(project_root: Path) -> Path:
    return project_root / "outputs" / "phase9b_rgb_detector_benchmark"


def path_key(path: Path) -> str:
    try:
        return os.path.normcase(str(path.resolve()))
    except OSError:
        return os.path.normcase(str(path.absolute()))


def unique_existing_directories(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path_key(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir():
            result.append(path.resolve())
    return result


def phase9a_inventory_roots(project_root: Path) -> list[Path]:
    inventory = (
        project_root
        / "outputs"
        / "phase9a_publication_strengthening_preflight"
        / "phase9a_dataset_inventory.csv"
    )
    if not inventory.is_file():
        return []
    result: list[Path] = []
    try:
        with inventory.open("r", newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                if row.get("dataset") != "mpi_inf_3dhp":
                    continue
                value = str(row.get("candidate_path", "")).strip()
                if value:
                    result.append(Path(value))
    except (OSError, csv.Error):
        return []
    return result


def initial_raw_root_candidates(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for variable in ("MPI_INF_3DHP_ROOT", "MPI3DHP_ROOT"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value))
    candidates.extend(phase9a_inventory_roots(project_root))
    bases = (
        project_root,
        project_root / "data",
        project_root / "data" / "raw",
        project_root.parent,
        project_root.parent / "Datasets",
        project_root.parent / "datasets",
    )
    for base in bases:
        candidates.append(base)
        candidates.extend(base / name for name in DATASET_DIR_NAMES)
    return unique_existing_directories(candidates)


def subject_sequence_count(root: Path) -> int:
    count = 0
    for subject in SUBJECTS:
        subject_dir = root / subject
        if not subject_dir.is_dir():
            subject_dir = root / subject.lower()
        if not subject_dir.is_dir():
            continue
        for sequence in SEQUENCES:
            if (subject_dir / sequence).is_dir() or (
                subject_dir / sequence.lower()
            ).is_dir():
                count += 1
    return count


def discover_training_layout_roots(
    initial_roots: Iterable[Path],
    maximum_depth: int = 3,
) -> list[Path]:
    layouts: list[Path] = []
    seen_depth: dict[str, int] = {}
    for initial in initial_roots:
        queue: list[tuple[Path, int]] = [(initial, 0)]
        while queue:
            current, depth = queue.pop(0)
            key = path_key(current)
            previous_depth = seen_depth.get(key)
            if previous_depth is not None and previous_depth <= depth:
                continue
            seen_depth[key] = depth
            if subject_sequence_count(current) > 0:
                layouts.append(current.resolve())
                continue
            if depth >= maximum_depth:
                continue
            try:
                children = sorted(
                    (
                        Path(entry.path)
                        for entry in os.scandir(current)
                        if entry.is_dir(follow_symlinks=False)
                    ),
                    key=lambda value: value.name.lower(),
                )
            except (OSError, PermissionError):
                continue
            for child in children:
                if re.fullmatch(r"TS[1-6]", child.name, re.I):
                    continue
                queue.append((child, depth + 1))
    return unique_existing_directories(layouts)


def count_named_test_directories(root: Path) -> int:
    count = 0
    queue: list[tuple[Path, int]] = [(root, 0)]
    visited: set[str] = set()
    while queue:
        current, depth = queue.pop(0)
        key = path_key(current)
        if key in visited:
            continue
        visited.add(key)
        try:
            children = [
                Path(entry.path)
                for entry in os.scandir(current)
                if entry.is_dir(follow_symlinks=False)
            ]
        except (OSError, PermissionError):
            continue
        for child in children:
            if re.fullmatch(r"TS[1-6]", child.name, re.I):
                count += 1
            elif depth < 2:
                queue.append((child, depth + 1))
    return count


def extension_inventory(root: Path, maximum_files: int = 250_000) -> dict[str, int]:
    counts: Counter[str] = Counter()
    visited = 0
    for _, directory_names, file_names in os.walk(root):
        directory_names[:] = [
            name
            for name in directory_names
            if not re.fullmatch(r"TS[1-6]", name, re.I)
        ]
        for name in file_names:
            suffix = Path(name).suffix.lower() or "[no_extension]"
            counts[suffix] += 1
            visited += 1
            if visited >= maximum_files:
                return dict(counts.most_common(30))
    return dict(counts.most_common(30))


def resolve_processed_root(project_root: Path) -> Path:
    canonical = (
        project_root / "data" / "processed" / "per_dataset" / "mpi_inf_3dhp"
    )
    candidates = [
        canonical,
        project_root / "data" / "processed" / "mpi_inf_3dhp",
    ]
    for candidate in candidates:
        if all(
            (
                candidate
                / f"mpi_inf_3dhp_{subject}_{sequence}_c15.pkl"
            ).is_file()
            for subject in SUBJECTS
            for sequence in SEQUENCES
        ):
            return candidate.resolve()
    return canonical.resolve()


def portable_image_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def write_blocked_diagnostic_zip(out: Path) -> Path:
    zip_path = out / "phase9b_r2_video_media_results.zip"
    include = (
        out / "phase9b_preflight_report.json",
        out / "phase9b_data_root_diagnostic.json",
        out / "phase9b_rgb_coverage.csv",
        out / "phase9b_rgb_index.csv",
    )
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in include:
            if path.is_file():
                archive.write(path, arcname=path.name)
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Diagnostic ZIP integrity failure: {bad}")
    return zip_path


def find_identity(path: Path, raw_root: Path) -> tuple[str, str, int, int] | None:
    try:
        relative = path.resolve().relative_to(raw_root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    subject = next((part for part in parts if re.fullmatch(r"S[1-8]", part, re.I)), None)
    sequence = next((part for part in parts if re.fullmatch(r"Seq[12]", part, re.I)), None)
    if subject is None or sequence is None:
        return None
    subject = subject.upper()
    sequence = f"Seq{int(sequence[-1])}"

    camera: int | None = None
    for part in reversed(parts[:-1]):
        match = re.search(r"(?:video|camera|cam|imageSequence)[_-]?(\d+)$", part, re.I)
        if match:
            camera = int(match.group(1))
            break
    stem = path.stem
    numbers = [int(value) for value in re.findall(r"\d+", stem)]
    if not numbers:
        return None
    frame_number = numbers[-1]
    if camera is None and len(numbers) >= 2:
        camera = numbers[-2]
    if camera is None:
        camera = 0
    return subject, sequence, camera, frame_number


def choose_offset(
    image_numbers: Iterable[int],
    annotation_numbers: Iterable[int],
    allowed: Iterable[int],
) -> tuple[int, int]:
    image_set = set(int(value) for value in image_numbers)
    annotation_set = set(int(value) for value in annotation_numbers)
    scores = {
        int(offset): sum(
            (number + int(offset)) in annotation_set
            for number in image_set
        )
        for offset in allowed
    }
    best_score = max(scores.values(), default=0)
    best_offsets = [offset for offset, score in scores.items() if score == best_score]
    return min(best_offsets, key=lambda value: (abs(value), value)), best_score


def coco17_to_c15(
    keypoints: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(confidence, dtype=np.float32)
    if points.shape != (17, 2) or scores.shape != (17,):
        raise ValueError(f"Expected COCO17 [17,2]/[17], got {points.shape}/{scores.shape}")
    result = np.zeros((15, 2), dtype=np.float32)
    result_conf = np.zeros((15,), dtype=np.float32)

    direct = {
        1: 12, 2: 14, 3: 16,
        4: 11, 5: 13, 6: 15,
        9: 5, 10: 7, 11: 9,
        12: 6, 13: 8, 14: 10,
    }
    for target, source in direct.items():
        result[target] = points[source]
        result_conf[target] = scores[source]
    result[0] = 0.5 * (points[11] + points[12])
    result_conf[0] = min(scores[11], scores[12])
    result[7] = 0.5 * (points[5] + points[6])
    result_conf[7] = min(scores[5], scores[6])
    result[8] = points[0]
    result_conf[8] = scores[0]
    return result, result_conf


def self_test() -> None:
    raw = Path("D:/data/mpi_inf_3dhp")
    samples = {
        raw / "S1/Seq1/imageSequence/video_0/img_000001.jpg": ("S1", "Seq1", 0, 1),
        raw / "S8/Seq2/cam_13/img_13_006054.jpg": ("S8", "Seq2", 13, 6054),
    }
    for path, expected in samples.items():
        actual = find_identity(path, raw)
        if actual != expected:
            raise AssertionError((path, actual, expected))
    offset, score = choose_offset([1, 2, 3], [0, 1, 2], range(-3, 4))
    if (offset, score) != (-1, 3):
        raise AssertionError((offset, score))
    points = np.arange(34, dtype=np.float32).reshape(17, 2)
    scores = np.linspace(0.1, 0.9, 17, dtype=np.float32)
    mapped, mapped_scores = coco17_to_c15(points, scores)
    if mapped.shape != (15, 2) or mapped_scores.shape != (15,):
        raise AssertionError("COCO mapping shape")
    if not np.allclose(mapped[0], 0.5 * (points[11] + points[12])):
        raise AssertionError("pelvis midpoint")
    print("SELF_TEST_PASS")


def load_bundle(processed_root: Path, subject: str, sequence: str) -> dict[str, Any]:
    path = processed_root / f"mpi_inf_3dhp_{subject}_{sequence}_c15.pkl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as file:
        return pickle.load(file)


def scan_rgb(raw_root: Path) -> dict[tuple[str, str, int], list[tuple[int, Path]]]:
    groups: dict[tuple[str, str, int], list[tuple[int, Path]]] = defaultdict(list)
    for path in raw_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        identity = find_identity(path, raw_root)
        if identity is None:
            continue
        subject, sequence, camera, frame_number = identity
        groups[(subject, sequence, camera)].append((frame_number, path))
    for values in groups.values():
        values.sort(key=lambda item: (item[0], str(item[1])))
    return groups


def find_video_identity(
    path: Path,
    raw_root: Path,
) -> tuple[str, str, int] | None:
    try:
        relative = path.resolve().relative_to(raw_root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    subject = next(
        (part for part in parts if re.fullmatch(r"S[1-8]", part, re.I)),
        None,
    )
    sequence = next(
        (part for part in parts if re.fullmatch(r"Seq[12]", part, re.I)),
        None,
    )
    camera_match = re.search(
        r"(?:video|camera|cam)[_-]?(\d+)$",
        path.stem,
        re.I,
    )
    if subject is None or sequence is None or camera_match is None:
        return None
    return (
        subject.upper(),
        f"Seq{int(sequence[-1])}",
        int(camera_match.group(1)),
    )


def scan_videos(raw_root: Path) -> dict[tuple[str, str, int], Path]:
    groups: dict[tuple[str, str, int], Path] = {}
    duplicates: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for path in raw_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if any(re.fullmatch(r"TS[1-6]", part, re.I) for part in path.parts):
            continue
        identity = find_video_identity(path, raw_root)
        if identity is None:
            continue
        duplicates[identity].append(path.resolve())
    for identity, paths in duplicates.items():
        if len(paths) == 1:
            groups[identity] = paths[0]
    return groups


def inspect_video(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {
            "opened": False,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "probe_decode_ok": False,
        }
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    probes = sorted(set((0, max(frame_count // 2, 0), max(frame_count - 1, 0))))
    probe_ok = True
    for frame_index in probes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if (
            not ok
            or frame is None
            or frame.ndim != 3
            or frame.shape[0] != height
            or frame.shape[1] != width
        ):
            probe_ok = False
            break
    capture.release()
    return {
        "opened": True,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "probe_decode_ok": probe_ok,
    }


def resolve_raw_root(
    project_root: Path,
) -> tuple[
    Path | None,
    dict[tuple[str, str, int], list[tuple[int, Path]]],
    dict[str, Any],
]:
    initial = initial_raw_root_candidates(project_root)
    layouts = discover_training_layout_roots(initial)
    evaluated: list[
        tuple[int, int, Path, dict[tuple[str, str, int], list[tuple[int, Path]]]]
    ] = []
    rows: list[dict[str, Any]] = []
    for layout in layouts:
        groups = scan_rgb(layout)
        image_count = sum(len(values) for values in groups.values())
        subject_sequences = subject_sequence_count(layout)
        evaluated.append((image_count, subject_sequences, layout, groups))
        extensions = extension_inventory(layout)
        rows.append({
            "path": str(layout),
            "subject_sequence_count": subject_sequences,
            "image_count": image_count,
            "camera_group_count": len(groups),
            "archive_count": sum(
                count for suffix, count in extensions.items()
                if suffix in ARCHIVE_SUFFIXES
            ),
            "video_count": sum(
                count for suffix, count in extensions.items()
                if suffix in VIDEO_SUFFIXES
            ),
            "extension_counts": extensions,
            "eligible_training_layout": True,
        })
    evaluated.sort(
        key=lambda item: (item[0], item[1], str(item[2])),
        reverse=True,
    )
    selected_root: Path | None = None
    selected_groups: dict[
        tuple[str, str, int], list[tuple[int, Path]]
    ] = {}
    if evaluated:
        _, _, selected_root, selected_groups = evaluated[0]

    official_test_candidates: list[dict[str, Any]] = []
    for root in initial:
        test_count = count_named_test_directories(root)
        if test_count:
            official_test_candidates.append({
                "path": str(root),
                "named_TS1_to_TS6_directories": test_count,
            })
    diagnostic = {
        "format": "phase9b_r1_data_root_diagnostic_v1",
        "phase": "9B-R1",
        "project_root": str(project_root),
        "environment_overrides": {
            name: os.environ.get(name)
            for name in ("MPI_INF_3DHP_ROOT", "MPI3DHP_ROOT")
            if os.environ.get(name)
        },
        "initial_existing_candidates": [str(path) for path in initial],
        "eligible_training_layouts": rows,
        "selected_training_root": str(selected_root) if selected_root else None,
        "official_test_candidates_excluded": official_test_candidates,
        "official_TS1_to_TS6_sample_content_read": False,
        "selection_rule": (
            "highest extracted-image count, then highest S1-S8/Seq1-Seq2 "
            "coverage; TS1-TS6 directories are excluded"
        ),
    }
    return selected_root, selected_groups, diagnostic


def run_preflight(project_root: Path) -> None:
    root = project_root.resolve()
    out = output_root(root)
    out.mkdir(parents=True, exist_ok=True)
    lock = load_json(root / "configs" / "phase9b_rgb_detector_protocol_lock.json")
    raw_root, image_groups, discovery = resolve_raw_root(root)
    processed_root = resolve_processed_root(root)
    discovery["selected_processed_root"] = str(processed_root)
    allowed_offsets = lock["alignment_policy"]["allowed_filename_to_annotation_frame_offsets"]
    video_policy = lock.get("video_media_policy", {})
    target_hz = float(video_policy.get("deterministic_sampling_hz", 5.0))
    maximum_frame_count_delta = int(
        video_policy.get("maximum_video_annotation_frame_count_delta", 3)
    )

    rows: list[dict[str, Any]] = []
    gt2d: list[np.ndarray] = []
    image_relpaths: list[str] = []
    media_kinds: list[str] = []
    video_relpaths: list[str] = []
    video_frame_indices: list[int] = []
    keys: list[str] = []
    subjects: list[str] = []
    sequences: list[str] = []
    cameras: list[int] = []
    frame_indices: list[int] = []
    frame_numbers: list[int] = []
    coverage_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    duplicate_image_identities = 0
    total_media_frames = 0
    total_selected_frames = 0
    video_groups: dict[tuple[str, str, int], Path] = {}
    media_mode = "extracted_images" if image_groups else "decoded_video"

    if raw_root is None:
        errors.append(
            "No eligible MPI-INF-3DHP training layout with S1-S8/Seq1-Seq2 "
            "was found. Official TS1-TS6 is intentionally excluded."
        )
    elif not image_groups:
        video_groups = scan_videos(raw_root)
        if not video_groups:
            errors.append(
                "Training layout was found but neither extracted RGB images "
                "nor uniquely identified video_<camera>.avi media were found."
            )
    discovery["phase9b_r2"] = {
        "media_mode": media_mode,
        "video_groups_found": len(video_groups),
        "video_identities": [
            {
                "subject": key[0],
                "sequence": key[1],
                "camera_id": key[2],
                "path": str(path),
            }
            for key, path in sorted(video_groups.items())
        ],
        "official_TS1_to_TS6_sample_content_read": False,
    }
    write_json(out / "phase9b_data_root_diagnostic.json", discovery)

    missing_bundles = [
        f"mpi_inf_3dhp_{subject}_{sequence}_c15.pkl"
        for subject in SUBJECTS
        for sequence in SEQUENCES
        if not (
            processed_root
            / f"mpi_inf_3dhp_{subject}_{sequence}_c15.pkl"
        ).is_file()
    ]
    if missing_bundles:
        errors.append(
            f"Processed C15 annotation bundles missing: {len(missing_bundles)}/16 "
            f"under {processed_root}"
        )

    for subject in SUBJECTS:
        for sequence in SEQUENCES:
            try:
                bundle = load_bundle(processed_root, subject, sequence)
            except Exception as error:
                errors.append(f"{subject}/{sequence}: {error!r}")
                continue
            annotation_numbers = [int(value) for value in bundle["frame_numbers"]]
            frame_to_index = {value: index for index, value in enumerate(annotation_numbers)}
            if image_groups:
                for camera in sorted(
                    key[2]
                    for key in image_groups
                    if key[:2] == (subject, sequence)
                ):
                    images = image_groups[(subject, sequence, camera)]
                    offset, overlap = choose_offset(
                        (item[0] for item in images),
                        annotation_numbers,
                        allowed_offsets,
                    )
                    seen: set[tuple[str, str, int, int]] = set()
                    matched = 0
                    total_media_frames += len(images)
                    total_selected_frames += len(images)
                    for image_number, image_path in images:
                        frame_no = image_number + offset
                        if frame_no not in frame_to_index:
                            continue
                        identity = (subject, sequence, camera, frame_no)
                        if identity in seen:
                            duplicate_image_identities += 1
                            continue
                        seen.add(identity)
                        if camera not in bundle["annot2_c15"]:
                            errors.append(
                                f"{subject}/{sequence}: camera {camera} "
                                "absent from annotation"
                            )
                            continue
                        frame_idx = frame_to_index[frame_no]
                        pose = np.asarray(
                            bundle["annot2_c15"][camera][frame_idx],
                            dtype=np.float32,
                        )
                        if pose.shape != (15, 2) or not np.isfinite(pose).all():
                            errors.append(f"{identity}: invalid GT 2D shape/content")
                            continue
                        relpath = portable_image_path(image_path, root)
                        sample_key = (
                            f"mpi_inf_3dhp:{subject}:{sequence}:"
                            f"c{camera:02d}:f{frame_no:06d}"
                        )
                        rows.append({
                            "sample_index": len(rows),
                            "sample_key": sample_key,
                            "subject": subject,
                            "sequence": sequence,
                            "camera_id": camera,
                            "media_kind": "image",
                            "image_frame_number": image_number,
                            "annotation_frame_number": frame_no,
                            "frame_idx": frame_idx,
                            "selected_offset": offset,
                            "image_relpath": relpath,
                            "video_relpath": "",
                            "video_frame_index": -1,
                        })
                        gt2d.append(pose)
                        image_relpaths.append(relpath)
                        media_kinds.append("image")
                        video_relpaths.append("")
                        video_frame_indices.append(-1)
                        keys.append(sample_key)
                        subjects.append(subject)
                        sequences.append(sequence)
                        cameras.append(camera)
                        frame_indices.append(frame_idx)
                        frame_numbers.append(frame_no)
                        matched += 1
                    coverage_rows.append({
                        "subject": subject,
                        "sequence": sequence,
                        "camera_id": camera,
                        "media_kind": "image",
                        "media_frames": len(images),
                        "sampling_step": 1,
                        "selected_frames": len(images),
                        "selected_offset": offset,
                        "matched_frames": matched,
                        "alignment_fraction": matched / max(len(images), 1),
                        "width": None,
                        "height": None,
                        "reported_fps": None,
                        "decode_probe_ok": None,
                    })
            else:
                for camera in sorted(
                    key[2]
                    for key in video_groups
                    if key[:2] == (subject, sequence)
                ):
                    video_path = video_groups[(subject, sequence, camera)]
                    metadata = inspect_video(video_path)
                    annotation_count = len(annotation_numbers)
                    video_count = int(metadata["frame_count"])
                    delta = abs(video_count - annotation_count)
                    if not metadata["opened"] or not metadata["probe_decode_ok"]:
                        errors.append(
                            f"{subject}/{sequence}/camera{camera}: video decode "
                            f"probe failed for {video_path}"
                        )
                        continue
                    if delta > maximum_frame_count_delta:
                        errors.append(
                            f"{subject}/{sequence}/camera{camera}: video frame "
                            f"count={video_count}, annotation count="
                            f"{annotation_count}, delta={delta} exceeds "
                            f"{maximum_frame_count_delta}"
                        )
                        continue
                    if camera not in bundle["annot2_c15"]:
                        errors.append(
                            f"{subject}/{sequence}: camera {camera} absent "
                            "from annotation"
                        )
                        continue
                    effective_count = min(video_count, annotation_count)
                    fps = float(bundle.get("fps") or metadata["fps"] or 25.0)
                    sampling_step = max(1, int(round(fps / target_hz)))
                    selected_indices = np.arange(
                        0, effective_count, sampling_step, dtype=np.int64
                    )
                    matched = 0
                    rel_video = portable_image_path(video_path, root)
                    total_media_frames += video_count
                    total_selected_frames += len(selected_indices)
                    poses = np.asarray(bundle["annot2_c15"][camera])
                    for frame_idx_value in selected_indices:
                        frame_idx = int(frame_idx_value)
                        frame_no = int(annotation_numbers[frame_idx])
                        pose = np.asarray(poses[frame_idx], dtype=np.float32)
                        identity = (subject, sequence, camera, frame_no)
                        if pose.shape != (15, 2) or not np.isfinite(pose).all():
                            errors.append(f"{identity}: invalid GT 2D shape/content")
                            continue
                        sample_key = (
                            f"mpi_inf_3dhp:{subject}:{sequence}:"
                            f"c{camera:02d}:f{frame_no:06d}"
                        )
                        rows.append({
                            "sample_index": len(rows),
                            "sample_key": sample_key,
                            "subject": subject,
                            "sequence": sequence,
                            "camera_id": camera,
                            "media_kind": "video",
                            "image_frame_number": "",
                            "annotation_frame_number": frame_no,
                            "frame_idx": frame_idx,
                            "selected_offset": 0,
                            "image_relpath": "",
                            "video_relpath": rel_video,
                            "video_frame_index": frame_idx,
                        })
                        gt2d.append(pose)
                        image_relpaths.append("")
                        media_kinds.append("video")
                        video_relpaths.append(rel_video)
                        video_frame_indices.append(frame_idx)
                        keys.append(sample_key)
                        subjects.append(subject)
                        sequences.append(sequence)
                        cameras.append(camera)
                        frame_indices.append(frame_idx)
                        frame_numbers.append(frame_no)
                        matched += 1
                    coverage_rows.append({
                        "subject": subject,
                        "sequence": sequence,
                        "camera_id": camera,
                        "media_kind": "video",
                        "media_frames": video_count,
                        "sampling_step": sampling_step,
                        "selected_frames": len(selected_indices),
                        "selected_offset": 0,
                        "matched_frames": matched,
                        "alignment_fraction": (
                            matched / max(len(selected_indices), 1)
                        ),
                        "width": int(metadata["width"]),
                        "height": int(metadata["height"]),
                        "reported_fps": float(metadata["fps"]),
                        "decode_probe_ok": bool(metadata["probe_decode_ok"]),
                    })

    if gt2d:
        gt_array = np.stack(gt2d).astype(np.float32)
        torso = np.linalg.norm(gt_array[:, 7] - gt_array[:, 0], axis=1)
        finite_torso_fraction = float(np.mean(np.isfinite(torso) & (torso > 1.0)))
    else:
        gt_array = np.empty((0, 15, 2), dtype=np.float32)
        finite_torso_fraction = 0.0

    fieldnames = list(rows[0]) if rows else [
        "sample_index", "sample_key", "subject", "sequence", "camera_id",
        "media_kind", "image_frame_number", "annotation_frame_number",
        "frame_idx", "selected_offset", "image_relpath", "video_relpath",
        "video_frame_index",
    ]
    with (out / "phase9b_rgb_index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    coverage_fields = list(coverage_rows[0]) if coverage_rows else [
        "subject", "sequence", "camera_id", "media_kind", "media_frames",
        "sampling_step", "selected_frames", "selected_offset",
        "matched_frames", "alignment_fraction", "width", "height",
        "reported_fps", "decode_probe_ok",
    ]
    with (out / "phase9b_rgb_coverage.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=coverage_fields)
        writer.writeheader()
        writer.writerows(coverage_rows)

    np.savez_compressed(
        out / "phase9b_alignment_cache.npz",
        sample_key=np.asarray(keys),
        image_relpath=np.asarray(image_relpaths),
        media_kind=np.asarray(media_kinds),
        video_relpath=np.asarray(video_relpaths),
        video_frame_index=np.asarray(video_frame_indices, dtype=np.int32),
        subject=np.asarray(subjects),
        sequence=np.asarray(sequences),
        camera_id=np.asarray(cameras, dtype=np.int16),
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        frame_number=np.asarray(frame_numbers, dtype=np.int32),
        gt2d_px=gt_array,
    )

    matched = len(rows)
    alignment_fraction = matched / max(total_selected_frames, 1)
    matched_subjects = sorted(set(subjects))
    matched_cameras = sorted(set(cameras))
    matched_sequences = sorted(set(zip(subjects, sequences)))
    in_frame_values: list[float] = []
    if media_mode == "decoded_video" and gt2d:
        dimension_by_group = {
            (row["subject"], row["sequence"], int(row["camera_id"])): (
                float(row["width"]),
                float(row["height"]),
            )
            for row in coverage_rows
            if row["media_kind"] == "video"
        }
        for pose, subject, sequence, camera in zip(
            gt_array, subjects, sequences, cameras
        ):
            width, height = dimension_by_group[(subject, sequence, camera)]
            inside = (
                np.isfinite(pose).all(axis=1)
                & (pose[:, 0] >= 0.0)
                & (pose[:, 0] < width)
                & (pose[:, 1] >= 0.0)
                & (pose[:, 1] < height)
            )
            in_frame_values.append(float(np.mean(inside)))
    in_frame_joint_fraction = (
        float(np.mean(in_frame_values))
        if in_frame_values
        else (1.0 if media_mode == "extracted_images" and gt2d else 0.0)
    )
    status = (
        "PASS"
        if (
            matched >= int(lock["alignment_policy"]["minimum_matched_images"])
            and alignment_fraction >= float(lock["alignment_policy"]["minimum_alignment_fraction"])
            and finite_torso_fraction >= 0.95
            and in_frame_joint_fraction
            >= float(lock["alignment_policy"]["minimum_in_frame_joint_fraction"])
            and len(matched_subjects)
            >= int(video_policy.get("minimum_subjects", 8))
            and len(matched_sequences)
            >= int(video_policy.get("minimum_subject_sequences", 16))
            and not errors
        )
        else "BLOCKED"
    )
    report = {
        "format": "phase9b_rgb_alignment_preflight_v1",
        "phase": "9B",
        "status": status,
        "project_root": str(root),
        "resolved_roots": {
            "raw_training_rgb": str(raw_root) if raw_root else None,
            "processed_c15": str(processed_root),
        },
        "media_mode": media_mode,
        "sampling_policy": {
            "deterministic_sampling_hz": target_hz,
            "selection": (
                "frame indices 0, step, 2*step, ... independently per "
                "subject/sequence/camera; step=round(sequence_fps/target_hz)"
            ),
            "total_media_frames": total_media_frames,
            "total_selected_frames": total_selected_frames,
            "rgb_recompression_performed": False,
        },
        "protocol": {
            "official_TS1_to_TS6_read": False,
            "model_training_performed": False,
            "ground_truth_assisted_person_selection": False,
        },
        "counts": {
            "images_found": sum(
                len(values) for values in image_groups.values()
            ),
            "videos_found": len(video_groups),
            "media_frames": total_media_frames,
            "selected_frames": total_selected_frames,
            "matched_images": matched,
            "alignment_fraction": alignment_fraction,
            "subjects": matched_subjects,
            "subject_count": len(matched_subjects),
            "subject_sequence_count": len(matched_sequences),
            "cameras": matched_cameras,
            "camera_count": len(matched_cameras),
            "duplicate_image_identities_skipped": duplicate_image_identities,
        },
        "coverage_interpretation": (
            "no_aligned_rgb"
            if len(matched_cameras) == 0
            else (
                "single_camera_rgb_subset"
                if len(matched_cameras) == 1
                else "multi_camera_rgb_subset"
            )
        ),
        "finite_torso_fraction": finite_torso_fraction,
        "in_frame_joint_fraction": in_frame_joint_fraction,
        "errors": errors,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": package_version("torch"),
            "rtmlib": package_version("rtmlib"),
            "ultralytics": package_version("ultralytics"),
            "onnxruntime_gpu": package_version("onnxruntime-gpu"),
            "opencv_python": package_version("opencv-python"),
            "opencv_contrib_python": package_version("opencv-contrib-python"),
            "opencv_python_headless": package_version("opencv-python-headless"),
        },
    }
    write_json(out / "phase9b_preflight_report.json", report)
    print(json.dumps(report, indent=2))
    if status != "PASS":
        diagnostic_zip = write_blocked_diagnostic_zip(out)
        print(f"DIAGNOSTIC_ZIP={diagnostic_zip}")
        raise SystemExit(2)


def read_alignment_cache(project_root: Path) -> dict[str, np.ndarray]:
    path = output_root(project_root) / "phase9b_alignment_cache.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run preflight first."
        )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def normalize_pose_output(
    keypoints: Any,
    scores: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(keypoints, dtype=np.float32)
    confidence = np.asarray(scores, dtype=np.float32)
    if points.ndim == 2:
        points = points[None]
    if confidence.ndim == 1:
        confidence = confidence[None]
    if points.ndim != 3 or points.shape[-2:] != (17, 2):
        return None
    if confidence.shape != points.shape[:2]:
        confidence = np.squeeze(confidence)
        if confidence.ndim == 1:
            confidence = confidence[None]
    if confidence.shape != points.shape[:2] or points.shape[0] == 0:
        return None
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    index = int(np.argmax(confidence.mean(axis=1)))
    return coco17_to_c15(points[index], confidence[index])


class RTMPoseDetector:
    detector_id = "rtmpose_performance"

    def __init__(self) -> None:
        # Importing torch first preloads the CUDA/cuDNN DLLs bundled with the
        # installed PyTorch wheel.  ORT >=1.21 can also locate those DLLs.
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RTMPose GPU is required, but torch.cuda.is_available() is False."
            )
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        from rtmlib import Body
        device = "cuda"
        try:
            self.model = Body(
                to_openpose=False,
                mode="performance",
                backend="onnxruntime",
                device=device,
            )
        except TypeError:
            self.model = Body(
                to_openpose=False,
                mode="performance",
                backend="onnxruntime",
                device=device,
            )
        self.execution_providers: dict[str, list[str]] = {}
        for name in ("det_model", "pose_model"):
            component = getattr(self.model, name, None)
            session = getattr(component, "session", None)
            if session is None or not hasattr(session, "get_providers"):
                raise RuntimeError(
                    f"Cannot verify RTMPose {name} ONNX Runtime session."
                )
            providers = list(session.get_providers())
            self.execution_providers[name] = providers
            if not providers or providers[0] != "CUDAExecutionProvider":
                raise RuntimeError(
                    "RTMPose GPU initialization failed: "
                    f"{name} providers={providers}. CPU fallback is disabled."
                )
        self.device = "cuda:0"

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        keypoints, scores = self.model(image)
        return normalize_pose_output(keypoints, scores)


class YOLOPoseDetector:
    detector_id = "yolo11l_pose"

    def __init__(self) -> None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "YOLO GPU is required, but torch.cuda.is_available() is False."
            )
        from ultralytics import YOLO
        self.model = YOLO("yolo11l-pose.pt")
        self.device = 0
        self.cuda_device_name = torch.cuda.get_device_name(0)
        self._gpu_verified = False

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        result = self.model.predict(
            source=image,
            imgsz=640,
            conf=0.05,
            iou=0.7,
            device=self.device,
            verbose=False,
        )[0]
        if not self._gpu_verified:
            backend = getattr(getattr(self.model, "predictor", None), "model", None)
            backend_device = getattr(backend, "device", None)
            if backend_device is not None and "cuda" not in str(backend_device).lower():
                raise RuntimeError(
                    "YOLO GPU initialization failed: "
                    f"Ultralytics backend device={backend_device!s}."
                )
            self._gpu_verified = True
        if result.keypoints is None or result.keypoints.xy is None:
            return None
        points = result.keypoints.xy.detach().cpu().numpy()
        if result.keypoints.conf is None:
            confidence = np.ones(points.shape[:2], dtype=np.float32)
        else:
            confidence = result.keypoints.conf.detach().cpu().numpy()
        if points.shape[0] == 0:
            return None
        if result.boxes is not None and result.boxes.conf is not None:
            box_conf = result.boxes.conf.detach().cpu().numpy()
            index = int(np.argmax(box_conf))
            return coco17_to_c15(points[index], confidence[index])
        return normalize_pose_output(points, confidence)


def run_gpu_check(project_root: Path) -> None:
    import torch
    import onnxruntime as ort

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable.")
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    rtmpose = RTMPoseDetector()
    rtmpose.predict(image)
    yolo = YOLOPoseDetector()
    yolo.predict(image)
    torch.cuda.synchronize(0)

    report = {
        "format": "phase9b_strict_gpu_check_v1",
        "status": "PASS",
        "cpu_fallback_allowed": False,
        "official_TS1_to_TS6_read": False,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(0),
        "onnxruntime_version": str(ort.__version__),
        "rtmpose_device": rtmpose.device,
        "rtmpose_execution_providers": rtmpose.execution_providers,
        "yolo_device": yolo.device,
        "yolo_cuda_device_name": yolo.cuda_device_name,
    }
    path = output_root(project_root.resolve()) / "phase9b_strict_gpu_check.json"
    write_json(path, report)
    print(json.dumps(report, indent=2))


def model_file_hashes(model: Any) -> list[dict[str, str]]:
    candidates: set[Path] = set()

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 4:
            return
        if isinstance(value, (str, os.PathLike)):
            path = Path(value)
            if path.suffix.lower() in {".onnx", ".pt", ".pth"} and path.is_file():
                candidates.add(path.resolve())
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, depth + 1)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, depth + 1)
        elif hasattr(value, "__dict__"):
            visit(vars(value), depth + 1)

    visit(model)
    return [
        {"path": str(path), "sha256": sha256(path)}
        for path in sorted(candidates)
    ]


def select_indices(total: int, max_samples: int | None) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)
    return np.unique(
        np.linspace(0, total - 1, num=max_samples, dtype=np.int64)
    )


def load_done_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    lines = path.read_text(encoding="utf-8").splitlines()
    valid_lines: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            completed[int(item["sample_index"])] = item
            valid_lines.append(line)
        except Exception as error:
            if line_number == len(lines):
                warning = path.with_suffix(path.suffix + ".ignored_incomplete_tail.txt")
                warning.write_text(line + "\n", encoding="utf-8")
                repaired = path.with_suffix(path.suffix + ".repairing")
                repaired.write_text(
                    "\n".join(valid_lines) + ("\n" if valid_lines else ""),
                    encoding="utf-8",
                )
                os.replace(repaired, path)
                print(
                    f"[RESUME] Removed incomplete final JSONL line {line_number}; "
                    f"saved the rejected tail as {warning.name}."
                )
                break
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return completed


class MediaFrameReader:
    def __init__(self, project_root: Path) -> None:
        import cv2

        self.cv2 = cv2
        self.project_root = project_root
        self.capture: Any = None
        self.capture_path: Path | None = None
        self.next_frame_index = 0

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.capture_path = None
        self.next_frame_index = 0

    def _open_video(self, path: Path) -> None:
        self.close()
        capture = self.cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cv2_video_open_failed:{path}")
        self.capture = capture
        self.capture_path = path
        self.next_frame_index = 0

    def read(self, cache: dict[str, np.ndarray], index: int) -> np.ndarray | None:
        media_kind = (
            str(cache["media_kind"][index])
            if "media_kind" in cache
            else "image"
        )
        if media_kind == "image":
            image_path = self.project_root / str(cache["image_relpath"][index])
            return self.cv2.imread(str(image_path), self.cv2.IMREAD_COLOR)
        if media_kind != "video":
            raise ValueError(f"Unsupported media kind: {media_kind!r}")
        video_path = self.project_root / str(cache["video_relpath"][index])
        target = int(cache["video_frame_index"][index])
        if self.capture_path != video_path:
            self._open_video(video_path)
        if target < self.next_frame_index:
            self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, target)
            self.next_frame_index = target
        frame = None
        ok = False
        while self.next_frame_index <= target:
            ok, frame = self.capture.read()
            if not ok:
                return None
            self.next_frame_index += 1
        return frame


def run_detection(
    project_root: Path,
    run_id: str,
    max_samples: int | None,
) -> None:
    import cv2

    root = project_root.resolve()
    out = output_root(root)
    preflight = load_json(out / "phase9b_preflight_report.json")
    if preflight.get("status") != "PASS":
        raise RuntimeError("Phase 9B preflight is not PASS.")
    cache = read_alignment_cache(root)
    indices = select_indices(len(cache["sample_key"]), max_samples)
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase 9B requires GPU inference, but PyTorch CUDA is unavailable. "
            "Run run_phase9b_gpu_repair_and_resume.bat."
        )
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    ort_available = list(ort.get_available_providers())
    if "CUDAExecutionProvider" not in ort_available:
        raise RuntimeError(
            "Phase 9B requires CUDAExecutionProvider, but ONNX Runtime reports "
            f"{ort_available}. Run run_phase9b_gpu_repair_and_resume.bat."
        )

    detector_builders = (
        ("rtmpose_performance", RTMPoseDetector),
        ("yolo11l_pose", YOLOPoseDetector),
    )
    runtime: dict[str, Any] = {
        "format": "phase9b_detector_runtime_v2",
        "run_id": run_id,
        "sample_count_requested": int(len(indices)),
        "cuda": {
            "required": True,
            "cpu_fallback_allowed": False,
            "torch_available": True,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "device_name": torch.cuda.get_device_name(0),
            "onnxruntime_version": str(ort.__version__),
            "onnxruntime_available_providers": ort_available,
        },
        "detectors": {},
    }
    for detector_id, builder in detector_builders:
        jsonl = run_dir / f"{detector_id}_predictions.jsonl"
        completed = load_done_jsonl(jsonl)
        completed_requested = sum(int(index) in completed for index in indices)
        remaining = len(indices) - completed_requested
        print(
            f"[RESUME] {detector_id}: completed={completed_requested} "
            f"remaining={remaining} total={len(indices)}"
        )
        if remaining == 0:
            runtime["detectors"][detector_id] = {
                "status": "complete_cache_reused",
                "device": "recorded_in_existing_cache",
                "cached_samples_reused": completed_requested,
                "new_samples": 0,
                "elapsed_seconds": 0.0,
                "new_samples_per_second": None,
                "model_files": [],
            }
            write_json(run_dir / "phase9b_detector_runtime.json", runtime)
            print(f"[SKIP] {detector_id} is already complete.")
            continue
        detector = builder()
        reader = MediaFrameReader(root)
        started = time.perf_counter()
        new_count = 0
        failures = 0
        try:
            with jsonl.open("a", encoding="utf-8") as file:
                for order, sample_index in enumerate(indices, start=1):
                    index = int(sample_index)
                    if index in completed:
                        continue
                    try:
                        image = reader.read(cache, index)
                    except Exception as error:
                        image = None
                        media_error = repr(error)
                    else:
                        media_error = "media_decode_failed"
                    if image is None:
                        result = {
                            "sample_index": index,
                            "sample_key": str(cache["sample_key"][index]),
                            "detected": False,
                            "error": media_error,
                        }
                        failures += 1
                    else:
                        try:
                            prediction = detector.predict(image)
                            if prediction is None:
                                result = {
                                    "sample_index": index,
                                    "sample_key": str(cache["sample_key"][index]),
                                    "detected": False,
                                    "error": "no_person_pose",
                                }
                                failures += 1
                            else:
                                keypoints, confidence = prediction
                                result = {
                                    "sample_index": index,
                                    "sample_key": str(cache["sample_key"][index]),
                                    "detected": True,
                                    "keypoints_c15_px": keypoints.tolist(),
                                    "confidence_c15": confidence.tolist(),
                                }
                        except Exception as error:
                            result = {
                                "sample_index": index,
                                "sample_key": str(cache["sample_key"][index]),
                                "detected": False,
                                "error": repr(error),
                            }
                            failures += 1
                    file.write(json.dumps(result, separators=(",", ":")) + "\n")
                    file.flush()
                    new_count += 1
                    done_count = completed_requested + new_count
                    if new_count % 100 == 0 or done_count == len(indices):
                        print(
                            f"[{detector_id}] {done_count}/{len(indices)} "
                            f"cached={completed_requested} new={new_count} "
                            f"failures_new={failures}"
                        )
        finally:
            reader.close()
        elapsed = time.perf_counter() - started
        runtime["detectors"][detector_id] = {
            "status": "completed_on_gpu",
            "device": getattr(detector, "device", None),
            "cached_samples_reused": completed_requested,
            "new_samples": new_count,
            "elapsed_seconds": elapsed,
            "new_samples_per_second": new_count / max(elapsed, 1e-9),
            "model_files": model_file_hashes(detector),
        }
        if hasattr(detector, "execution_providers"):
            runtime["detectors"][detector_id]["execution_providers"] = (
                detector.execution_providers
            )
        if hasattr(detector, "cuda_device_name"):
            runtime["detectors"][detector_id]["cuda_device_name"] = (
                detector.cuda_device_name
            )
        write_json(run_dir / "phase9b_detector_runtime.json", runtime)
    print(json.dumps(runtime, indent=2))


def percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def detector_metrics(
    detector_id: str,
    prediction: np.ndarray,
    confidence: np.ndarray,
    detected: np.ndarray,
    gt: np.ndarray,
    cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    torso = np.linalg.norm(gt[:, 7] - gt[:, 0], axis=1)
    valid_torso = np.isfinite(torso) & (torso > 1.0)
    error_px = np.linalg.norm(prediction - gt, axis=-1)
    normalized = error_px / np.maximum(torso[:, None], 1.0)
    valid = detected[:, None] & valid_torso[:, None] & np.isfinite(normalized)

    def summarize_subset(indices: np.ndarray) -> dict[str, Any]:
        subset = normalized[:, indices]
        subset_valid = valid[:, indices]
        values = subset[subset_valid]
        total = int(subset_valid.size)
        finite = int(subset_valid.sum())
        return {
            "finite_fraction": finite / max(total, 1),
            "mean_nme": float(values.mean()) if values.size else None,
            "median_nme": float(np.median(values)) if values.size else None,
            "pck2d_at_0_05_torso_pct": (
                float(100.0 * np.mean(values <= 0.05)) if values.size else None
            ),
            "pck2d_at_0_10_torso_pct": (
                float(100.0 * np.mean(values <= 0.10)) if values.size else None
            ),
            "p90_nme": percentile(values, 90) if values.size else None,
            "p95_nme": percentile(values, 95) if values.size else None,
        }

    by_joint: dict[str, Any] = {}
    for index, name in enumerate(JOINT_NAMES):
        values = normalized[:, index][valid[:, index]]
        by_joint[name] = {
            "count": int(values.size),
            "mean_nme": float(values.mean()) if values.size else None,
            "median_nme": float(np.median(values)) if values.size else None,
            "p90_nme": percentile(values, 90) if values.size else None,
        }

    def grouped(values: np.ndarray, label_name: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for label in sorted(set(str(item) for item in values)):
            mask = np.asarray([str(item) == label for item in values])
            joint_mask = valid[mask][:, COMMON12]
            subset = normalized[mask][:, COMMON12][joint_mask]
            result[label] = {
                "samples": int(mask.sum()),
                "detection_success_rate": float(detected[mask].mean()),
                "mean_common12_nme": float(subset.mean()) if subset.size else None,
                "median_common12_nme": float(np.median(subset)) if subset.size else None,
            }
        return result

    common_errors = normalized[:, COMMON12]
    common_conf = confidence[:, COMMON12]
    common_valid = valid[:, COMMON12]
    error_values = common_errors[common_valid]
    conf_values = common_conf[common_valid]
    if error_values.size >= 2 and np.std(conf_values) > 0 and np.std(error_values) > 0:
        confidence_error_correlation = float(
            np.corrcoef(conf_values, -error_values)[0, 1]
        )
    else:
        confidence_error_correlation = None

    bins: list[dict[str, Any]] = []
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        bin_mask = (
            common_valid
            & (common_conf >= lower)
            & (common_conf < upper if upper < 1.0 else common_conf <= upper)
        )
        values = common_errors[bin_mask]
        scores = common_conf[bin_mask]
        if values.size:
            empirical = float(np.mean(values <= 0.10))
            mean_conf = float(np.mean(scores))
            ece += values.size / max(error_values.size, 1) * abs(mean_conf - empirical)
            bins.append({
                "lower": float(lower),
                "upper": float(upper),
                "count": int(values.size),
                "mean_confidence": mean_conf,
                "mean_nme": float(values.mean()),
                "median_nme": float(np.median(values)),
                "p90_nme": percentile(values, 90),
                "p95_nme": percentile(values, 95),
                "pck2d_at_0_10_torso": empirical,
            })
        else:
            bins.append({
                "lower": float(lower), "upper": float(upper), "count": 0,
            })

    return {
        "detector_id": detector_id,
        "samples": int(len(gt)),
        "detected_samples": int(detected.sum()),
        "detection_success_rate": float(detected.mean()),
        "common12": summarize_subset(COMMON12),
        "full15": summarize_subset(np.arange(15)),
        "confidence_error_correlation": confidence_error_correlation,
        "confidence_calibration_ece_at_0_10_torso": float(ece),
        "confidence_bins": bins,
        "by_joint": by_joint,
        "by_subject": grouped(cache["subject"], "subject"),
        "by_camera": grouped(cache["camera_id"], "camera"),
    }


def run_aggregate(project_root: Path, run_id: str) -> None:
    root = project_root.resolve()
    out = output_root(root)
    run_dir = out / run_id
    cache = read_alignment_cache(root)
    total = len(cache["sample_key"])
    selected: set[int] = set()
    records_by_detector: dict[str, dict[int, dict[str, Any]]] = {}
    for detector_id in DETECTORS:
        records = load_done_jsonl(run_dir / f"{detector_id}_predictions.jsonl")
        records_by_detector[detector_id] = records
        selected.update(records)
    indices = np.asarray(sorted(selected), dtype=np.int64)
    if not indices.size:
        raise RuntimeError("No detector records found.")
    if run_id == "full" and len(indices) != total:
        raise RuntimeError(
            f"Full aggregate requires {total} samples, found {len(indices)}."
        )

    detector_count = len(DETECTORS)
    prediction = np.full(
        (detector_count, len(indices), 15, 2),
        np.nan,
        dtype=np.float32,
    )
    confidence = np.zeros(
        (detector_count, len(indices), 15),
        dtype=np.float32,
    )
    detected = np.zeros((detector_count, len(indices)), dtype=bool)
    for detector_index, detector_id in enumerate(DETECTORS):
        records = records_by_detector[detector_id]
        for local_index, global_index in enumerate(indices):
            item = records.get(int(global_index))
            if item is None or not item.get("detected"):
                continue
            points = np.asarray(item["keypoints_c15_px"], dtype=np.float32)
            scores = np.asarray(item["confidence_c15"], dtype=np.float32)
            if points.shape != (15, 2) or scores.shape != (15,):
                continue
            if not np.isfinite(points).all() or not np.isfinite(scores).all():
                continue
            prediction[detector_index, local_index] = points
            confidence[detector_index, local_index] = np.clip(scores, 0.0, 1.0)
            detected[detector_index, local_index] = True

    subset_cache = {key: value[indices] for key, value in cache.items()}
    gt = subset_cache["gt2d_px"].astype(np.float32)
    metrics = {
        "format": "phase9b_rgb_detector_metrics_v1",
        "phase": "9B",
        "run_id": run_id,
        "protocol": {
            "official_TS1_to_TS6_read": False,
            "model_training_performed": False,
            "ground_truth_assisted_person_selection": False,
            "primary_subset": "common12",
        },
        "detectors": {},
    }
    for detector_index, detector_id in enumerate(DETECTORS):
        metrics["detectors"][detector_id] = detector_metrics(
            detector_id,
            prediction[detector_index],
            confidence[detector_index],
            detected[detector_index],
            gt,
            subset_cache,
        )

    np.savez_compressed(
        run_dir / "phase9b_detector_cache.npz",
        detector_id=np.asarray(DETECTORS),
        sample_index=indices,
        sample_key=subset_cache["sample_key"],
        image_relpath=subset_cache["image_relpath"],
        media_kind=subset_cache["media_kind"],
        video_relpath=subset_cache["video_relpath"],
        video_frame_index=subset_cache["video_frame_index"],
        subject=subset_cache["subject"],
        sequence=subset_cache["sequence"],
        camera_id=subset_cache["camera_id"],
        frame_idx=subset_cache["frame_idx"],
        frame_number=subset_cache["frame_number"],
        gt2d_px=gt,
        prediction_c15_px=prediction,
        confidence_c15=confidence,
        detected=detected,
    )
    write_json(run_dir / "phase9b_detector_metrics.json", metrics)
    error_model = {
        "format": "phase9b_real_detector_error_model_v1",
        "phase": "9B",
        "run_id": run_id,
        "units": "normalized by GT pelvis-to-neck torso length",
        "primary_subset": "common12",
        "detectors": {
            detector_id: {
                "detection_success_rate": metrics["detectors"][detector_id][
                    "detection_success_rate"
                ],
                "confidence_error_correlation": metrics["detectors"][detector_id][
                    "confidence_error_correlation"
                ],
                "confidence_bins": metrics["detectors"][detector_id][
                    "confidence_bins"
                ],
                "by_joint": metrics["detectors"][detector_id]["by_joint"],
            }
            for detector_id in DETECTORS
        },
    }
    write_json(run_dir / "phase9b_real_detector_error_model.json", error_model)

    summary_rows: list[dict[str, Any]] = []
    for detector_id in DETECTORS:
        item = metrics["detectors"][detector_id]
        summary_rows.append({
            "detector_id": detector_id,
            "samples": item["samples"],
            "detected_samples": item["detected_samples"],
            "detection_success_rate": item["detection_success_rate"],
            "common12_mean_nme": item["common12"]["mean_nme"],
            "common12_median_nme": item["common12"]["median_nme"],
            "common12_pck_0_05_pct": item["common12"]["pck2d_at_0_05_torso_pct"],
            "common12_pck_0_10_pct": item["common12"]["pck2d_at_0_10_torso_pct"],
            "full15_mean_nme": item["full15"]["mean_nme"],
            "confidence_error_correlation": item["confidence_error_correlation"],
            "confidence_calibration_ece": item[
                "confidence_calibration_ece_at_0_10_torso"
            ],
        })
    with (run_dir / "phase9b_detector_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "self-test", "gpu-check"):
        item = subparsers.add_parser(command)
        item.add_argument(
            "--project-root",
            type=Path,
            default=Path.cwd(),
        )
    detect = subparsers.add_parser("detect")
    detect.add_argument(
        "--project-root", type=Path, default=Path.cwd()
    )
    detect.add_argument("--run-id", choices=("smoke", "full"), required=True)
    detect.add_argument("--max-samples", type=int)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument(
        "--project-root", type=Path, default=Path.cwd()
    )
    aggregate.add_argument("--run-id", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "gpu-check":
        run_gpu_check(args.project_root)
    elif args.command == "preflight":
        run_preflight(args.project_root)
    elif args.command == "detect":
        run_detection(args.project_root, args.run_id, args.max_samples)
    elif args.command == "aggregate":
        run_aggregate(args.project_root, args.run_id)


if __name__ == "__main__":
    main()
