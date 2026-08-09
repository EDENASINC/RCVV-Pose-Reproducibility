from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GpuStack:
    name: str
    minimum_driver: int
    torch_version: str
    torchvision_version: str
    torch_index: str
    expected_cuda_major: int
    onnxruntime_version: str


CUDA13 = GpuStack(
    name="cuda13",
    minimum_driver=580,
    torch_version="2.13.0",
    torchvision_version="0.28.0",
    torch_index="https://download.pytorch.org/whl/cu130",
    expected_cuda_major=13,
    onnxruntime_version="1.28.0",
)

CUDA12 = GpuStack(
    name="cuda12",
    minimum_driver=525,
    torch_version="2.11.0",
    torchvision_version="0.26.0",
    torch_index="https://download.pytorch.org/whl/cu128",
    expected_cuda_major=12,
    onnxruntime_version="1.26.0",
)


def fail(message: str) -> int:
    print(f"[ERROR] {message}")
    return 2


def find_nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if found:
        return found
    candidates: list[Path] = []
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidates.append(Path(system_root) / "System32" / "nvidia-smi.exe")
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(
            Path(program_files)
            / "NVIDIA Corporation"
            / "NVSMI"
            / "nvidia-smi.exe"
        )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def query_nvidia() -> tuple[str, str, int]:
    executable = find_nvidia_smi()
    if executable is None:
        raise RuntimeError(
            "nvidia-smi.exe was not found. Install/update the NVIDIA display "
            "driver, restart Windows, then run this BAT again."
        )
    command = [
        executable,
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"nvidia-smi failed with code {completed.returncode}: {detail}"
        )
    first_line = next(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()),
        "",
    )
    parts = [part.strip() for part in first_line.rsplit(",", 1)]
    if len(parts) != 2:
        raise RuntimeError(f"Cannot parse nvidia-smi output: {first_line!r}")
    gpu_name, driver_version = parts
    match = re.match(r"^(\d+)", driver_version)
    if match is None:
        raise RuntimeError(f"Cannot parse NVIDIA driver: {driver_version!r}")
    return gpu_name, driver_version, int(match.group(1))


def select_stack(driver_major: int) -> GpuStack:
    if driver_major >= CUDA13.minimum_driver:
        return CUDA13
    if driver_major >= CUDA12.minimum_driver:
        return CUDA12
    raise RuntimeError(
        f"NVIDIA driver {driver_major} is too old. Phase 9B requires driver "
        f"{CUDA12.minimum_driver}+ for CUDA 12.8 or {CUDA13.minimum_driver}+ "
        "for CUDA 13. Update the NVIDIA driver, restart Windows, and rerun."
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_probe() -> dict[str, Any]:
    code = (
        "import json, torch, torchvision;"
        "print(json.dumps({"
        "'torch_version':str(torch.__version__),"
        "'torchvision_version':str(torchvision.__version__),"
        "'cuda_version':str(torch.version.cuda or ''),"
        "'cuda_available':bool(torch.cuda.is_available()),"
        "'device_name':torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else None"
        "}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        return {
            "probe_error": (completed.stderr or completed.stdout).strip(),
            "cuda_available": False,
        }
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as error:
        return {
            "probe_error": f"Invalid torch probe output: {error!r}",
            "probe_stdout": completed.stdout.strip(),
            "cuda_available": False,
        }


def torch_matches(probe: dict[str, Any], stack: GpuStack) -> bool:
    cuda_version = str(probe.get("cuda_version", ""))
    try:
        cuda_major = int(cuda_version.split(".", 1)[0])
    except (TypeError, ValueError):
        return False
    return (
        bool(probe.get("cuda_available"))
        and cuda_major == stack.expected_cuda_major
        and str(probe.get("torch_version", "")).startswith(stack.torch_version)
        and str(probe.get("torchvision_version", "")).startswith(
            stack.torchvision_version
        )
    )


def run_pip(arguments: list[str], label: str) -> None:
    print(f"[REPAIR] {label}")
    completed = subprocess.run(
        [sys.executable, "-m", "pip", *arguments],
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"{label} failed with code {completed.returncode}.")


def install_torch(stack: GpuStack) -> None:
    run_pip(
        [
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            f"torch=={stack.torch_version}",
            f"torchvision=={stack.torchvision_version}",
            "--index-url",
            stack.torch_index,
        ],
        (
            f"Installing PyTorch {stack.torch_version} and torchvision "
            f"{stack.torchvision_version} with {stack.name.upper()} support"
        ),
    )


def install_onnxruntime(stack: GpuStack) -> None:
    expected = stack.onnxruntime_version
    current = package_version("onnxruntime-gpu")
    if current == expected:
        print(f"[KEEP] onnxruntime-gpu {current} already matches {stack.name}.")
        return
    run_pip(
        [
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            f"onnxruntime-gpu=={expected}",
        ],
        f"Installing onnxruntime-gpu {expected}",
    )


def main() -> int:
    print("============================================================")
    print("PHASE 9B GPU RUNTIME REPAIR V2")
    print("============================================================")
    print("Existing JSONL detector results will not be modified.")
    try:
        gpu_name, driver_version, driver_major = query_nvidia()
        stack = select_stack(driver_major)
    except Exception as error:
        return fail(str(error))

    before = torch_probe()
    print(f"GPU                  : {gpu_name}")
    print(f"NVIDIA driver        : {driver_version}")
    print(f"Selected stack       : {stack.name}")
    print(f"PyTorch before repair: {before.get('torch_version', 'unavailable')}")

    try:
        if torch_matches(before, stack):
            print("[KEEP] PyTorch CUDA stack is already correct.")
        else:
            if str(before.get("torch_version", "")).endswith("+cpu"):
                print("[FOUND] CPU-only PyTorch; replacing it with a CUDA wheel.")
            install_torch(stack)

        after = torch_probe()
        if not torch_matches(after, stack):
            detail = after.get("probe_error") or json.dumps(after, sort_keys=True)
            raise RuntimeError(
                "PyTorch CUDA verification failed after installation. "
                f"Details: {detail}"
            )
        install_onnxruntime(stack)
    except Exception as error:
        return fail(str(error))

    report = {
        "format": "phase9b_gpu_runtime_repair_v2",
        "gpu_name_from_nvidia_smi": gpu_name,
        "nvidia_driver_version": driver_version,
        "selected_stack": asdict(stack),
        "torch_before": before,
        "torch_after": after,
        "selected_onnxruntime": f"onnxruntime-gpu=={stack.onnxruntime_version}",
        "jsonl_cache_modified": False,
    }
    report_path = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "phase9b_rgb_detector_benchmark"
        / "phase9b_gpu_runtime_repair.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"PyTorch after repair : {after['torch_version']}")
    print(f"Bundled CUDA         : {after['cuda_version']}")
    print(f"CUDA device          : {after['device_name']}")
    print(
        f"ONNX Runtime         : onnxruntime-gpu "
        f"{stack.onnxruntime_version}"
    )
    print("[PASS] CUDA runtime repaired. Starting strict model GPU check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
