from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


SPLITS = ("split_a", "split_b", "split_c")
PARTITIONS = ("train", "val", "test")
PARTITION_CODE = {name: index for index, name in enumerate(PARTITIONS)}
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
CANDIDATE_ID = "rotation_yaw_m15"
EXPECTED_B1_SHA256 = "fc1ae8f11ec6fb1b75c8d8c730e61a67f1b1aa45422227364e4bce71eb72b673"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(value: str) -> torch.device:
    value = value.strip().lower()
    if value in {"cuda", "cuda:0"}:
        value = "cuda:0"
    device = torch.device(value)
    if device.type != "cuda":
        raise RuntimeError("Phase 9C-R1A is strict-GPU and requires --device cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. CPU fallback is forbidden by the locked protocol.")
    index = 0 if device.index is None else int(device.index)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {index} is unavailable.")
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def load_b1(root: Path, expected: str):
    path = root / "scripts/train/run_phase9c_b1_four_arm_smoke.py"
    actual = sha256_file(path)
    if actual != expected or actual != EXPECTED_B1_SHA256:
        raise ValueError(f"Phase 9C-B1 V3 source hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("phase9c_b1_v3_r1a", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Block(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class GeometryConditionedVirtualViewGenerator(nn.Module):
    def __init__(
        self,
        hidden_width: int = 1024,
        residual_blocks: int = 3,
        dropout: float = 0.10,
        translation_scale_m: float = 5.0,
    ) -> None:
        super().__init__()
        self.translation_scale_m = float(translation_scale_m)
        self.input = nn.Sequential(
            nn.Linear(40, hidden_width), nn.LayerNorm(hidden_width), nn.GELU(), nn.Dropout(dropout)
        )
        self.blocks = nn.Sequential(*[Block(hidden_width, dropout) for _ in range(residual_blocks)])
        self.output = nn.Linear(hidden_width, 28)

    def forward(self, pose: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        batch = pose.shape[0]
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand(batch, -1, -1)
        geometry = torch.cat(
            [(rotation - identity).flatten(1), translation / self.translation_scale_m], dim=1
        )
        features = torch.cat([pose.flatten(1), geometry], dim=1)
        return pose + self.output(self.blocks(self.input(features))).reshape(-1, 14, 2)


def find_system(registry: dict[str, Any], split: str) -> dict[str, Any]:
    matches = [item for item in registry.get("systems", []) if item.get("split") == split]
    if len(matches) != 1:
        raise ValueError(f"{split}: expected one locked system, found {len(matches)}")
    return matches[0]


def locked_artifact(record: dict[str, Any]) -> Path:
    if not record.get("exists", False):
        raise FileNotFoundError(f"Locked artifact absent: {record.get('role')}")
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_normalization(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "phase4c1_observed_only_normalization_v1":
        raise ValueError(f"Unexpected normalization format: {path}")
    result = {name: payload[name].float() for name in ("input_mean", "input_std")}
    for name, tensor in result.items():
        if tuple(tensor.shape) != (14, 2) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid normalization tensor {name}: {tuple(tensor.shape)}")
    if bool((result["input_std"] <= 0).any()):
        raise ValueError("Input normalization std contains non-positive values.")
    return result


def load_generator_models(manifest: dict[str, Any], device: torch.device) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    supported = {"phase4c4_generator_checkpoint_v1", "phase4d2c_generator_checkpoint_v1"}
    models: dict[str, nn.Module] = {}
    metadata: dict[str, Any] = {}
    for record in manifest["models"]:
        name = str(record["name"])
        checkpoint_path = Path(record["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("format") not in supported or checkpoint.get("name") != name:
            raise ValueError(f"Unsupported or mismatched generator checkpoint: {checkpoint_path}")
        config = checkpoint.get("config", {})
        model = GeometryConditionedVirtualViewGenerator(
            hidden_width=int(config.get("hidden_width", 1024)),
            residual_blocks=int(config.get("generator_blocks", 3)),
            dropout=float(config.get("generator_dropout", 0.10)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[name] = model
        metadata[name] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_subjects": list(checkpoint.get("training_subjects", record.get("training_subjects", []))),
            "validation_subjects": list(checkpoint.get("validation_subjects", record.get("validation_subjects", []))),
        }
    return models, metadata


def load_candidate(root: Path, split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = root / "outputs/phase5_virtual_camera/phase5a_geometry_candidates/candidate_registry.json"
    registry = load_json(path)
    candidates = [item for item in registry.get("candidates", []) if item.get("candidate_id") == CANDIDATE_ID]
    if registry.get("status") != "PASS" or len(candidates) != 1:
        raise ValueError(f"Missing locked candidate {CANDIDATE_ID}")
    candidate = candidates[0]
    if candidate.get("candidate_type") != "rotation_only" or split not in candidate.get("support_by_split", {}):
        raise ValueError(f"Candidate {CANDIDATE_ID} is not supported for {split}")
    rotation = torch.tensor(candidate["rotation_observed_to_virtual"], dtype=torch.float32, device=device)
    translation = torch.tensor(candidate["translation_observed_to_virtual_m"], dtype=torch.float32, device=device)
    if tuple(rotation.shape) != (3, 3) or tuple(translation.shape) != (3,):
        raise ValueError("Invalid candidate geometry shape.")
    return rotation, translation, {"registry": str(path.resolve()), "registry_sha256": sha256_file(path)}


def model_names_for_source(source: Any, partition: str) -> np.ndarray:
    shard_ids = np.searchsorted(np.asarray(source.ends, dtype=np.int64), source.dataset_index, side="right")
    if int(shard_ids.max(initial=0)) >= len(source.shards):
        raise ValueError("Dataset index falls outside source shards.")
    subjects = np.asarray([str(source.shards[int(index)]["subject"]) for index in shard_ids])
    if partition == "train":
        return np.asarray([f"holdout_{subject}" for subject in subjects])
    return np.full(subjects.shape, "full_train", dtype="U64")


def synthesize_partition(
    *,
    b1: Any,
    source: Any,
    partition: str,
    detector_cache: dict[str, np.ndarray],
    models: dict[str, nn.Module],
    normalization: dict[str, torch.Tensor],
    rotation: torch.Tensor,
    translation: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    size = len(source)
    output = np.zeros((len(DETECTORS), size, 15, 2), dtype=np.float16)
    names = model_names_for_source(source, partition)
    unknown = sorted(set(names.tolist()) - set(models))
    if unknown:
        raise ValueError(f"Missing generator model(s): {unknown}")
    usage = {name: int((names == name).sum()) for name in sorted(set(names.tolist()))}
    input_mean = normalization["input_mean"].to(device)
    input_std = normalization["input_std"].to(device)
    legacy_sum = np.zeros(len(DETECTORS), dtype=np.float64)
    legacy_count = np.zeros(len(DETECTORS), dtype=np.int64)
    observed_delta_sum = np.zeros(len(DETECTORS), dtype=np.float64)
    observed_delta_count = np.zeros(len(DETECTORS), dtype=np.int64)
    for start in range(0, size, batch_size):
        end = min(start + batch_size, size)
        items = [source[index] for index in range(start, end)]
        intrinsic = torch.stack([item["intrinsic"] for item in items]).to(device)
        legacy = torch.stack([item["teacher_virtual"] for item in items]).float().numpy()
        detector_indices = source.detector_index[start:end]
        batch_names = names[start:end]
        for detector_index, detector in enumerate(DETECTORS):
            prediction_px = torch.from_numpy(
                detector_cache["prediction_c15_px"][detector_index, detector_indices]
            ).to(device=device, dtype=torch.float32)
            observed = b1.normalized_root_from_pixels_torch(prediction_px, intrinsic)
            predicted = torch.empty((end - start, 15, 2), dtype=torch.float32, device=device)
            predicted[:, 0, :] = 0.0
            for model_name in sorted(set(batch_names.tolist())):
                select_np = np.flatnonzero(batch_names == model_name)
                select = torch.from_numpy(select_np).to(device=device, dtype=torch.long)
                pose = observed.index_select(0, select)[:, 1:, :]
                standardized = (pose - input_mean) / input_std
                count = int(select.shape[0])
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    generated = models[model_name](
                        standardized,
                        rotation.expand(count, -1, -1),
                        translation.expand(count, -1),
                    )
                generated = generated.float() * input_std + input_mean
                predicted.index_copy_(0, select, torch.cat([torch.zeros((count, 1, 2), device=device), generated], dim=1))
            if not bool(torch.isfinite(predicted).all()):
                raise RuntimeError(f"Non-finite detector-conditioned virtual pose: {partition}/{detector}")
            predicted_np = predicted.cpu().numpy()
            output[detector_index, start:end] = predicted_np.astype(np.float16)
            legacy_sum[detector_index] += float(np.abs(predicted_np - legacy).sum())
            legacy_count[detector_index] += int(predicted_np.size)
            observed_np = observed.cpu().numpy()
            observed_delta_sum[detector_index] += float(np.abs(predicted_np - observed_np).sum())
            observed_delta_count[detector_index] += int(predicted_np.size)
        if end % 8192 == 0 or end == size:
            print(f"[BUILD] partition={partition} progress={end}/{size}")
    if not np.isfinite(output).all() or not np.array_equal(output[:, :, 0, :], np.zeros_like(output[:, :, 0, :])):
        raise RuntimeError(f"Cache tensor integrity failed: {partition}")
    cross_detector_difference = float(np.mean(np.abs(output[0].astype(np.float32) - output[1].astype(np.float32))))
    stats = {
        "samples": size,
        "generator_model_usage": usage,
        "mean_abs_difference_vs_legacy_clean_conditioned": {
            detector: float(legacy_sum[index] / legacy_count[index]) for index, detector in enumerate(DETECTORS)
        },
        "mean_abs_virtual_delta_from_detector_observed": {
            detector: float(observed_delta_sum[index] / observed_delta_count[index]) for index, detector in enumerate(DETECTORS)
        },
        "mean_abs_cross_detector_virtual_difference": cross_detector_difference,
    }
    return output, stats


def existing_cache_valid(path: Path, protocol_sha: str) -> bool:
    report = path.with_suffix(".report.json")
    if not path.is_file() or not report.is_file():
        return False
    try:
        payload = load_json(report)
        return (
            payload.get("status") == "PASS"
            and payload.get("protocol_sha256") == protocol_sha
            and payload.get("cache_sha256") == sha256_file(path)
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build detector-conditioned virtual-view caches.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-name", choices=SPLITS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.project_root.resolve()
    split = args.split_name
    output_root = root / "outputs/phase9c_r1a_detector_conditioned_virtual_cache"
    output_root.mkdir(parents=True, exist_ok=True)
    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    cache_path = split_root / "phase9c_r1a_detector_virtual_cache.npz"
    protocol_path = root / "configs/phase9c_r1a_detector_conditioned_cache_protocol.json"
    protocol = load_json(protocol_path)
    if protocol.get("status") != "LOCKED" or protocol.get("format") != "phase9c_r1a_detector_conditioned_cache_protocol_lock_v1":
        raise ValueError("Invalid Phase 9C-R1A protocol lock.")
    protocol_sha = sha256_file(protocol_path)
    if existing_cache_valid(cache_path, protocol_sha):
        print(f"[REUSE] Valid Phase 9C-R1A cache: {cache_path}")
        return
    b2_qc_path = root / "outputs/phase9c_b2_full_multisplit_multiseed/phase9c_b2_qc_report.json"
    b2_stats_path = root / "outputs/phase9c_b2_full_multisplit_multiseed/phase9c_b2_statistics.json"
    b2_qc = load_json(b2_qc_path)
    b2_stats = load_json(b2_stats_path)
    if b2_qc.get("status") != "PASS" or b2_qc.get("scientific_decision") != protocol["source_phase9c_b2_decision"]:
        raise ValueError("Phase 9C-B2 redesign gate is absent.")
    if b2_qc.get("official_TS1_to_TS6_read") is not False or b2_stats.get("official_TS1_to_TS6_read") is not False:
        raise ValueError("Official-test isolation was violated upstream.")
    device = resolve_device(args.device)
    b1 = load_b1(root, str(protocol["source_b1_script_sha256"]))
    b1.SPLIT = split
    artifacts = b1.find_required_artifacts(root)
    with np.load(artifacts["phase9c_b0_join_cache"], allow_pickle=False) as handle:
        join_cache = {name: handle[name] for name in handle.files}
    phase9ca_build = load_json(artifacts["phase9c_a_build"])
    detector_cache, detector_location, detector_hash = b1.load_phase9b_cache(
        root, str(phase9ca_build["source_hashes"]["phase9b_detector_cache.npz"])
    )
    audit, common_valid = b1.build_common_detection_audit(detector_cache, join_cache)
    final_registry_path = root / "outputs/phase4e1_final_model_lock/final_model_registry.json"
    final_registry = load_json(final_registry_path)
    if final_registry.get("status") != "LOCKED":
        raise ValueError("Phase 4E.1 system registry is not LOCKED.")
    system = find_system(final_registry, split)
    generator_manifest_path = locked_artifact(system["training_provenance"]["generator_manifest"])
    normalization_path = locked_artifact(system["components"]["normalization"])
    generator_manifest = load_json(generator_manifest_path)
    normalization = load_normalization(normalization_path)
    models, model_metadata = load_generator_models(generator_manifest, device)
    rotation, translation, candidate_source = load_candidate(root, split, device)
    manifest = root / "data/processed/caches/phase5r1a_multigeometry_oof_25fps_v1" / split / "manifest.json"
    batch_size = int(protocol["compute_policy"]["source_batch_size"])
    arrays: dict[str, list[np.ndarray]] = {
        "dataset_index": [], "detector_sample_index": [], "partition_code": [], "virtual_pose_camera_root": []
    }
    partition_stats: dict[str, Any] = {}
    started = time.time()
    for partition in PARTITIONS:
        source = b1.JoinedDetectorDataset(
            manifest_path=manifest,
            join_cache=join_cache,
            common_valid_detector_sample=common_valid,
            partition=partition,
            max_samples=0,
            seed=42,
        )
        print(f"[BUILD] split={split} partition={partition} samples={len(source)} detector_input=REAL")
        virtual, stats = synthesize_partition(
            b1=b1, source=source, partition=partition, detector_cache=detector_cache,
            models=models, normalization=normalization, rotation=rotation, translation=translation,
            device=device, batch_size=batch_size,
        )
        arrays["dataset_index"].append(source.dataset_index.astype(np.int64))
        arrays["detector_sample_index"].append(source.detector_index.astype(np.int64))
        arrays["partition_code"].append(np.full(len(source), PARTITION_CODE[partition], dtype=np.int8))
        arrays["virtual_pose_camera_root"].append(virtual)
        stats["joined_before_eligibility"] = source.joined_before_eligibility
        stats["excluded_by_common_detection"] = source.excluded_by_common_detection
        partition_stats[partition] = stats
        del source, virtual
        gc.collect()
        torch.cuda.empty_cache()
    virtual_all = np.concatenate(arrays["virtual_pose_camera_root"], axis=1)
    np.savez_compressed(
        cache_path,
        format=np.asarray("phase9c_r1a_detector_conditioned_virtual_cache_v1"),
        split=np.asarray(split),
        detector_id=np.asarray(DETECTORS),
        candidate_id=np.asarray(CANDIDATE_ID),
        source_input=np.asarray("phase9b_prediction_c15_px_real_detector"),
        dataset_index=np.concatenate(arrays["dataset_index"]),
        detector_sample_index=np.concatenate(arrays["detector_sample_index"]),
        partition_code=np.concatenate(arrays["partition_code"]),
        virtual_pose_camera_root=virtual_all,
    )
    cache_sha = sha256_file(cache_path)
    report = {
        "status": "PASS",
        "format": "phase9c_r1a_split_cache_report_v1",
        "phase": "9C-R1A",
        "split": split,
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha,
        "protocol_sha256": protocol_sha,
        "detector_input_source": "phase9b_prediction_c15_px_real_detector",
        "legacy_clean_conditioned_cache_used_as_model_input": False,
        "legacy_clean_conditioned_cache_used_for_difference_audit_only": True,
        "candidate_id": CANDIDATE_ID,
        "partition_stats": partition_stats,
        "total_samples": int(virtual_all.shape[1]),
        "cache_shape": list(virtual_all.shape),
        "detector_coverage_audit": audit,
        "generator_manifest": str(generator_manifest_path),
        "generator_manifest_sha256": sha256_file(generator_manifest_path),
        "normalization": str(normalization_path),
        "normalization_sha256": sha256_file(normalization_path),
        "generator_models": model_metadata,
        "candidate_source": candidate_source,
        "detector_cache_location": detector_location,
        "detector_cache_sha256": detector_hash,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device.index or 0),
        "ground_truth_3d_metrics_computed": False,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
        "elapsed_sec": time.time() - started,
    }
    write_json(cache_path.with_suffix(".report.json"), report)
    print("=" * 108)
    print(f"PHASE 9C-R1A - {split.upper()} DETECTOR-CONDITIONED VIRTUAL CACHE")
    print("=" * 108)
    print("Status                 : PASS")
    print(f"Samples                : {report['total_samples']}")
    print(f"Device                 : {report['device']} / {report['gpu_name']}")
    print("Virtual input          : REAL DETECTOR KEYPOINTS")
    print("3D metrics computed    : NO")
    print("Official TS1-TS6 read : NO")
    print(f"Cache                  : {cache_path}")
    print("=" * 108)


if __name__ == "__main__":
    main()
