from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SPLITS = ("split_a", "split_b", "split_c")
DETECTORS = ("rtmpose_performance", "yolo11l_pose")
EXPECTED_ARMS = (
    "detector_conditioned_virtual_view_no_confidence",
    "calibrated_probability_concat",
    "bounded_reliability_concat",
    "bounded_reliability_virtual_gate",
    "bounded_reliability_dual_modulation",
)


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


def resolve_strict_gpu(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value in {"cuda", "cuda:0"}:
        value = "cuda:0"
    device = torch.device(value)
    if device.type != "cuda" or device.index not in {None, 0}:
        raise RuntimeError("Phase 9C-R1C preflight is locked to cuda:0.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden.")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def load_b1(root: Path, expected_hash: str):
    path = root / "scripts/train/run_phase9c_b1_four_arm_smoke.py"
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(f"B1 source hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("phase9c_b1_v3_r1c_preflight", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def init_worker_preflight(_: int) -> None:
    torch.set_num_threads(1)


class SpawnProbeDataset(Dataset):
    def __len__(self) -> int:
        return 32

    def __getitem__(self, item: int) -> torch.Tensor:
        return torch.tensor([item, item + 1], dtype=torch.float32)


def spawn_probe(workers: int) -> dict[str, Any]:
    loader = DataLoader(
        SpawnProbeDataset(),
        batch_size=8,
        shuffle=False,
        num_workers=workers,
        persistent_workers=False,
        prefetch_factor=1,
        worker_init_fn=init_worker_preflight,
    )
    iterator = iter(loader)
    try:
        batch = next(iterator)
        if tuple(batch.shape) != (8, 2) or not bool(torch.isfinite(batch).all()):
            raise RuntimeError(f"Unexpected spawn probe batch: {tuple(batch.shape)}")
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
    return {
        "status": "PASS",
        "workers": workers,
        "initializer": "preflight_real_script_module",
        "batch_shape": list(batch.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight Phase 9C-R1C confidence screen.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs/phase9c_r1c_confidence_mechanism_screen"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "configs/phase9c_r1c_confidence_mechanism_screen_protocol.json"
    protocol = load_json(protocol_path)
    if protocol.get("status") != "LOCKED" or protocol.get("format") != "phase9c_r1c_confidence_mechanism_screen_protocol_lock_v1":
        raise ValueError("Invalid Phase 9C-R1C protocol lock.")
    if tuple(protocol["development_scope"]["splits"]) != SPLITS:
        raise ValueError("Split matrix mismatch.")
    if tuple(protocol["detectors"]) != DETECTORS or tuple(protocol["arms"]) != EXPECTED_ARMS:
        raise ValueError("Detector/arm matrix mismatch.")
    if protocol["development_scope"].get("development_test_metrics_computed") is not False:
        raise ValueError("R1C must not compute development-test metrics.")
    if protocol["official_test_policy"].get("official_TS1_to_TS6_read") is not False:
        raise ValueError("Official-test lock is invalid.")
    budget = protocol["training_budget"]
    if int(budget["num_workers"]) != 8 or int(budget["max_train_samples"]) != 0 or int(budget["max_eval_samples"]) != 0:
        raise ValueError("R1C is locked to workers=8 and full train/validation cohorts.")
    if int(protocol["run_matrix"]["total_training_runs"]) != 30:
        raise ValueError("R1C run count must be 30.")

    device = resolve_strict_gpu(args.device)
    b1_path = root / "scripts/train/run_phase9c_b1_four_arm_smoke.py"
    r1b_path = root / "scripts/train/run_phase9c_r1b_detector_conditioned_smoke.py"
    b1_hash = sha256_file(b1_path)
    r1b_hash = sha256_file(r1b_path)
    if b1_hash != protocol["source_b1_script_sha256"]:
        raise ValueError(f"B1 source hash mismatch: {b1_hash}")
    if r1b_hash != protocol["source_r1b_script_sha256"]:
        raise ValueError(f"R1B v2 source hash mismatch: {r1b_hash}")

    r1b_root = root / "outputs/phase9c_r1b_detector_conditioned_smoke"
    r1b_qc = load_json(r1b_root / "phase9c_r1b_smoke_qc_report.json")
    r1b_build = load_json(r1b_root / "phase9c_r1b_smoke_build_report.json")
    if r1b_qc.get("status") != "PASS" or r1b_qc.get("scientific_decision") != protocol["source_gate"]:
        raise ValueError("Phase 9C-R1B pass gate is absent.")
    if r1b_build.get("dataloader_spawn_preflight", {}).get("status") != "PASS":
        raise ValueError("R1B Windows spawn evidence is absent.")
    if r1b_build.get("exact_cache_alignment_verified") is not True:
        raise ValueError("R1B exact cache alignment was not verified.")
    if r1b_build.get("virtual_view_input") != "phase9b_prediction_c15_px_real_detector":
        raise ValueError("R1B virtual-view provenance is not detector-conditioned.")
    if r1b_build.get("development_test_metrics_computed") is not False:
        raise ValueError("R1B development-test isolation was violated.")
    if r1b_qc.get("official_TS1_to_TS6_read") is not False:
        raise ValueError("R1B official-test isolation was violated.")

    b1 = load_b1(root, str(protocol["source_b1_script_sha256"]))
    artifacts = b1.find_required_artifacts(root)
    phase9ca_qc = load_json(artifacts["phase9c_a_qc"])
    phase9ca_build = load_json(artifacts["phase9c_a_build"])
    phase9cb0_qc = load_json(artifacts["phase9c_b0_qc"])
    phase9cb0_build = load_json(artifacts["phase9c_b0_build"])
    if phase9ca_qc.get("status") != "PASS" or phase9cb0_qc.get("status") != "PASS":
        raise ValueError("Phase 9C-A/B0 prerequisite gate failed.")
    if phase9ca_build.get("official_TS1_to_TS6_read") is not False or phase9cb0_build.get("official_TS1_to_TS6_read") is not False:
        raise ValueError("Upstream official-test isolation was violated.")
    source_hashes = {
        "phase9c_a_residual_bank.npz": sha256_file(artifacts["phase9c_a_residual_bank"]),
        "phase9c_a_calibration_models.json": sha256_file(artifacts["phase9c_a_models"]),
        "phase9c_b0_detector_join_cache.npz": sha256_file(artifacts["phase9c_b0_join_cache"]),
    }
    expected_join = str(phase9cb0_qc["artifact_hashes"]["phase9c_b0_detector_join_cache.npz"])
    if source_hashes["phase9c_b0_detector_join_cache.npz"] != expected_join:
        raise ValueError("Phase 9C-B0 join cache hash mismatch.")

    with np.load(artifacts["phase9c_a_residual_bank"], allow_pickle=False) as handle:
        residual_bank = {name: handle[name] for name in handle.files}
    risk = b1.residual_risk_table(residual_bank)
    if risk.shape != (3, 2, 15, 10) or not np.isfinite(risk).all() or bool((risk < 0).any()):
        raise ValueError(f"Invalid residual-risk table: {risk.shape}")
    positive = risk[risk > 0]
    if positive.size == 0:
        raise ValueError("Residual-risk table has no positive values.")
    scale = np.empty((3, 2, 15), dtype=np.float32)
    for split_index in range(3):
        for detector_index in range(2):
            for joint in range(15):
                values = risk[split_index, detector_index, joint]
                values = values[np.isfinite(values) & (values > 0)]
                scale[split_index, detector_index, joint] = max(
                    float(np.median(values)) if values.size else 1e-6,
                    1e-6,
                )
    quality = 1.0 / (1.0 + risk / scale[..., None])
    if not np.isfinite(quality).all() or bool((quality < 0).any()) or bool((quality > 1).any()):
        raise ValueError("Bounded risk transform is invalid.")
    risk_audit = {
        "raw_positive_quantiles": {
            name: float(value) for name, value in zip(
                ("min", "p10", "p25", "median", "p75", "p90", "max"),
                np.quantile(positive, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)),
            )
        },
        "bounded_quality_min": float(quality.min()),
        "bounded_quality_max": float(quality.max()),
        "joint_scale_min": float(scale.min()),
        "joint_scale_max": float(scale.max()),
        "raw_risk_used_directly_as_model_input": False,
    }

    r1a_root = root / "outputs/phase9c_r1a_detector_conditioned_virtual_cache"
    r1a_qc = load_json(r1a_root / "phase9c_r1a_qc_report.json")
    if r1a_qc.get("status") != "PASS":
        raise ValueError("Phase 9C-R1A QC is not PASS.")
    if r1a_qc.get("official_TS1_to_TS6_read") is not False:
        raise ValueError("R1A official-test isolation was violated.")
    r1a_caches: dict[str, Any] = {}
    for split in SPLITS:
        cache = r1a_root / split / "phase9c_r1a_detector_virtual_cache.npz"
        report = load_json(cache.with_suffix(".report.json"))
        digest = sha256_file(cache)
        if report.get("cache_sha256") != digest:
            raise ValueError(f"R1A cache hash mismatch: {split}")
        if report.get("protocol_sha256") != protocol["source_r1a_protocol_sha256"]:
            raise ValueError(f"R1A protocol hash mismatch: {split}")
        if report.get("detector_input_source") != "phase9b_prediction_c15_px_real_detector":
            raise ValueError(f"R1A provenance mismatch: {split}")
        if report.get("official_TS1_to_TS6_read") is not False:
            raise ValueError(f"R1A official-test isolation mismatch: {split}")
        r1a_caches[split] = {
            "path": str(cache),
            "sha256": digest,
            "samples": int(report["total_samples"]),
        }

    spawn = spawn_probe(int(budget["num_workers"]))
    report = {
        "status": "PASS",
        "format": "phase9c_r1c_preflight_report_v1",
        "phase": "9C-R1C",
        "scientific_role": "validation_only_mechanism_selection_without_test_or_official_access",
        "scientific_decision": "READY_FOR_PHASE9C_R1C_SCREEN_RUNS",
        "protocol_sha256": sha256_file(protocol_path),
        "source_b1_script_sha256": b1_hash,
        "source_r1b_script_sha256": r1b_hash,
        "source_r1b_gate": r1b_qc.get("scientific_decision"),
        "source_hashes": source_hashes,
        "r1a_caches": r1a_caches,
        "risk_bounding_audit": risk_audit,
        "windows_spawn_probe": spawn,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "workers": int(budget["num_workers"]),
        "planned_runs": int(protocol["run_matrix"]["total_training_runs"]),
        "development_test_metrics_computed": False,
        "official_TS1_to_TS6_read": False,
        "official_TS1_to_TS6_used": False,
    }
    write_json(output / "phase9c_r1c_preflight_report.json", report)
    print("=" * 108)
    print("PHASE 9C-R1C - CONFIDENCE MECHANISM SCREEN PREFLIGHT")
    print("=" * 108)
    print("Status                : PASS")
    print(f"Device                : {device} / {report['gpu_name']}")
    print(f"Workers               : {report['workers']} (spawn probe PASS)")
    print(f"Planned training runs : {report['planned_runs']}")
    print("Development test read : NO")
    print("Official TS1-TS6 read : NO")
    print("Scientific decision   : READY_FOR_PHASE9C_R1C_SCREEN_RUNS")
    print("=" * 108)


if __name__ == "__main__":
    main()
