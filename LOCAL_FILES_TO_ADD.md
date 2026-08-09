# Local files still required before public release

The attached evidence does not expose the original source filenames or exact checkpoint paths. Copy the original artifacts; do not regenerate them merely to fill this repository.

## Required source code

Copy the Python implementation that produced the locked hashes into `src/rcvv_pose/`, preserving history where possible. It must include:

1. MPI-INF-3DHP training and Official TS1--TS6 dataset adapters.
2. Official adapter using `annot2`, focal-length normalization, `univ_annot3`, `valid_frame`, activity labels, and the official-17 to canonical-15 joint-name mapping.
3. RTMPose and YOLO11L-Pose RGB inference, detector-native person selection, confidence extraction, and missing-detection handling.
4. Leakage-safe monotonic confidence calibration fitted only on training subjects.
5. Split-matched residual-risk bank construction and reliability computation.
6. Detector-conditioned virtual-view generation for `rotation_yaw_m15`.
7. Feature construction for O (28-D), OV (56-D), OR, and OVR (126-D).
8. Compact 2D-to-3D lifter architecture, training loop, deterministic seed setup, validation-only checkpoint selection, and early stopping.
9. MPJPE, PA-MPJPE, 3DPCK@150 mm, AUC 0--150 mm, subject/split/seed/detector macro aggregation, and hierarchical paired bootstrap.
10. Entrypoints for evidence audit, single-run inference, 72-checkpoint Official evaluation, and full retraining.

Also copy the original scripts whose recorded SHA-256 values are:

- `fc1ae8f11ec6fb1b75c8d8c730e61a67f1b1aa45422227364e4bce71eb72b673`
- `0bcfee2535017abc6d4648b62b1b8e358a0251fe25a6a7e07304499eb73b7dfc`

Use `python tools/find_by_sha256.py D:\Research\vv_pose_project HASH...` to locate them without guessing filenames.

## Required learned artifacts

- All 72 validation-selected `best.pt`/`best.pth` lifter checkpoints in the paths listed by `models/checkpoint_manifest.csv`.
- Six split-detector monotonic confidence calibrators (3 splits × 2 detectors).
- Six split-detector residual-risk banks, including bin definitions and metadata.
- Exact executable configs used for each run, or a deterministic config generator that recreates all 72 configurations.
- Checkpoint architecture/config metadata sufficient to load each state dict without the original run directory.
- Lightweight provenance from every locked run: `run_report.json`, authoritative `training_history.csv`, selected epoch/validation score, final metric rows, and the original artifact-hash manifest. The supplied QC reports refer to 54 R1D runs plus 18 R1E runs and 378 + 126 hashed artifacts; retain these records even when bulky per-sample tensors are released separately.

## Required detector artifacts

- YOLO detector weight `yolo11l-pose.pt`, recorded SHA-256 `61921abe1f2ed930bf28328a16b3162278cdb239a1e8280b6ac827119f13cee0`, or an official download script that verifies this hash.
- Exact RTMPose detector and pose model identifiers/URLs/cache filenames used by `rtmlib==0.0.15`, including SHA-256 where available.
- Detector preprocessing, model input size, coordinate transform, and person-selection settings.

Third-party detector weights should be downloaded from their official providers when redistribution is not clearly permitted.

## Strongly recommended derived artifacts

The evidence index describes five Official artifacts that were not included in the files supplied for this package. Add them under `evidence/official/` if their redistribution is compatible with the dataset terms:

- `16_locked_official_ts1_ts6_subject_split_seed_detector_arm_metrics.csv`
- `17_locked_official_ts1_ts6_activity_arm_metrics.csv`
- `18_locked_official_ts1_ts6_ensemble_3d_predictions.npz`
- `19_locked_official_ts1_ts6_rgb_detector_2d_keypoints_cache.npz`
- `20_locked_official_ts5_median_case_selection.json`

Because NPZ files may contain derived annotations or coordinates tied to restricted data, verify the MPI-INF-3DHP terms before public redistribution. If uncertain, publish scripts and hashes, and provide those files on reasonable request.

## Environment and provenance

- Export the actual Python environment (`pip freeze`, Conda YAML, or lock file).
- Record OS, Python, CUDA, cuDNN, GPU, PyTorch, ONNX Runtime, detector packages, and exact command lines.
- Export Git commit identifiers from the original code repository, if it already existed.
- Generate `SHA256SUMS.txt` only after the release contents are frozen.

## Files that must stay out of GitHub

- Original MPI-INF-3DHP RGB, videos, annotations, ZIP files, or extracted dataset tree.
- Credentials, private URLs, email tokens, `.env`, SSH keys, or absolute user-profile paths.
- Unlocked exploratory runs that could confuse the paper's frozen protocol.
- Build caches, virtual environments, logs, TensorBoard runs, and temporary shards.
