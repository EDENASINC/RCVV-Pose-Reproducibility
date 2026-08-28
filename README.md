# RCVV-Pose: Reviewer Reproducibility Package

Companion repository for:

> **Reliability-Conditioned Virtual-View Pose Fusion for Monocular 3D Human Pose Estimation with Real 2D Detector Inputs**  
> Phalakron Nilkhet and Thanaruk Theeramunkong  
> *Information*, manuscript `information-4512788`

This repository contains the source code, locked configurations, reviewer evidence, run provenance, and a 72-checkpoint release manifest for the paper. Model checkpoints and learned calibration artifacts are distributed separately as versioned GitHub Release assets so that large binaries do not enter ordinary Git history.

## Reproducibility levels

| Level | Requirements | Command |
|---|---|---|
| Evidence audit | Python 3.12; no dataset or model downloads | `python tools/validate_evidence.py` |
| Release audit | Local checkpoints and all nine generated release ZIP files | `python tools/release_preflight.py --mode release` |
| Official TS1--TS6 evaluation | MPI-INF-3DHP test set, 72 checkpoints, detector cache or detector weights, and learned artifacts | `python -m rcvv_pose.evaluate --config configs/official_ts1_ts6.yaml` |
| Full retraining | MPI-INF-3DHP training set and detector weights | `python -m rcvv_pose.train_matrix --config configs/development_factorial.yaml` |

## Locked experimental design

- Dataset: MPI-INF-3DHP only.
- Development splits: `split_a`, `split_b`, `split_c` (subject-disjoint).
- Seeds: `42`, `123`, `2026`.
- Real RGB detectors: `rtmpose_performance` and `yolo11l_pose`.
- Factorial arms: observed-only (`O`), observed + virtual (`OV`), observed + reliability (`OR`), and observed + virtual + reliability (`OVR`).
- Selected arm: `bounded_reliability_dual_modulation` (`OVR`).
- Virtual view: detector-conditioned fixed yaw `-15°` (`rotation_yaw_m15`).
- Model selection: matched-detector validation only; Official Test was not used for selection.
- Official evaluation: TS1--TS6, 2,875 valid frames, real detector predictions only, canonical 15-joint mapping with 14 non-root joints scored.
- Locked matrix: 3 splits × 3 seeds × 2 detectors × 4 arms = **72 checkpoints**.

## Quick start

```bash
git clone https://github.com/EDENASINC/RCVV-Pose-Reproducibility.git
cd RCVV-Pose-Reproducibility
python tools/validate_evidence.py
python -m unittest discover -s tests -v
python tools/release_preflight.py --mode evidence
```

The evidence-only audit uses the Python standard library and does not recompute inference.

## Dataset setup

Obtain MPI-INF-3DHP from the [official dataset provider](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) and follow its access conditions. The dataset, RGB frames, videos, and annotations are not redistributed here.

The Official adapter uses `annot2`, official focal-length normalization, `univ_annot3`, `valid_frame` and activity labels, and the locked official-17 to canonical-15 mapping.

## Release assets

Generate the eight checkpoint archives plus the learned calibration archive with:

```bash
python tools/build_model_archives.py --checkpoint-root checkpoints --output-dir release_assets
python tools/release_preflight.py --mode release
```

Expected ZIP assets:

- `checkpoints_O_rtmpose.zip`
- `checkpoints_O_yolo11l.zip`
- `checkpoints_OV_rtmpose.zip`
- `checkpoints_OV_yolo11l.zip`
- `checkpoints_OR_rtmpose.zip`
- `checkpoints_OR_yolo11l.zip`
- `checkpoints_OVR_rtmpose.zip`
- `checkpoints_OVR_yolo11l.zip`
- `learned_calibration_artifacts.zip`

The builder records the exact size and SHA-256 of every archive in `models/release_manifest.json`. After the GitHub Release exists, rerun it with `--release-url` and verify `--mode published`.

Third-party detector weights are not redistributed; obtain them from their official providers and verify the recorded hashes.

<!-- Q2A_RELEASE_METADATA_BEGIN -->
### Public release status

The frozen checkpoint and calibration archives are published at [v1.0.0-paper-information-4512788](https://github.com/EDENASINC/RCVV-Pose-Reproducibility/releases/tag/v1.0.0-paper-information-4512788).
The associated manuscript, `information-4512788`, is currently under review. This
software release does not claim that the article has been accepted or published.
<!-- Q2A_RELEASE_METADATA_END -->

## Repository map

```text
configs/       locked executable protocols and configurations
evidence/      reviewer-readable locked JSON/CSV evidence
models/        checkpoint inventory, release manifest, and model documentation
src/           reusable datasets, metrics, models, and pipeline code
scripts/       experiment preparation, training, QC, and evaluation scripts
tools/         validation, preflight, hashing, import, and release tooling
tests/         lightweight release-contract tests
```

## Interpretation boundary

On the locked Official TS1--TS6 evaluation, OVR has better MPJPE and PA-MPJPE point estimates for both detectors, but the hierarchical 95% confidence intervals cross zero. The evidence supports a **positive but unconfirmed trend**, not superiority or a state-of-the-art claim.

## Citation and licensing

See [`CITATION.cff`](CITATION.cff). The MIT License applies to original source code and documentation in this repository. It does not relicense MPI-INF-3DHP, third-party detector weights, or other externally licensed components.
