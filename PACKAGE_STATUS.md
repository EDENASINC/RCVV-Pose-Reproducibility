# Package status

## Included and validated

- Locked reviewer evidence files `00` through `17` and `20`.
- Source code and executable configurations for detector audit, calibration, factorial training, QC, and locked Official evaluation.
- Training histories and run reports for the locked development matrix.
- A 72-row checkpoint inventory covering 3 splits × 3 seeds × 2 detectors × 4 arms.
- Evidence validation, release preflight, checkpoint import, archive builder, tests, and GitHub Actions audit.
- Eight checkpoint archives prepared locally and recorded by SHA-256.

## Distributed separately as Release assets

- 72 validation-selected checkpoint files grouped into eight ZIP archives.
- Calibration models and the residual-risk bank grouped as `learned_calibration_artifacts.zip`.

## Deliberately excluded

- MPI-INF-3DHP RGB, videos, annotations, and original archives.
- Third-party detector weights.
- Official-derived prediction and 2D-keypoint NPZ files numbered `18` and `19`; these can be regenerated locally after obtaining the dataset.
- Machine-specific paths, local collection reports, caches, and credentials.

The evidence-only contract is directly auditable from Git. Full evaluation additionally requires the official dataset, third-party detector weights or an authorized cache, and the nine versioned Release ZIP files.
