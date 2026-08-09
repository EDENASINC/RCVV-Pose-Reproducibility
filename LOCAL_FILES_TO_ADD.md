# Local-only inputs and excluded artifacts

The original source code, executable configurations, and lightweight training provenance have already been collected into this repository. This file now documents inputs that intentionally remain outside ordinary Git history.

## Inputs used to build GitHub Release assets

- `checkpoints/`: 72 validation-selected checkpoint files in the paths listed by `models/checkpoint_manifest.csv`.
- `artifacts/calibration/`: calibration model files required by the reliability arms.
- `local_release_inputs/learned_artifacts/phase9c_a_residual_bank.npz`: the locked residual-risk bank.

Run `tools/build_model_archives.py` to package these inputs into eight checkpoint ZIP files and one learned-artifact ZIP. The builder records actual sizes and SHA-256 values in `models/release_manifest.json`.

## External inputs not redistributed

- MPI-INF-3DHP dataset files, RGB frames, videos, annotations, and archives.
- YOLO11L-Pose and RTMPose third-party weights; obtain these from their official providers and verify the recorded hashes.
- Official-derived prediction and keypoint caches numbered `18` and `19` in the evidence narrative.

## Files that must remain outside GitHub

- Credentials, private URLs, `.env`, SSH keys, tokens, and API keys.
- Absolute user-profile or machine-specific paths.
- Virtual environments, build caches, logs, temporary shards, and exploratory runs outside the locked protocol.
- Dataset-derived caches unless their redistribution rights have been verified explicitly.
