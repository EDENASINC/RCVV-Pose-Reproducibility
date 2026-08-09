# Package status

## Included and validated

- Sixteen supplied evidence/index files, renamed canonically from `00` through `15`.
- Locked protocols for detector audit, leakage-safe calibration, development factorial training, attribution, and Official TS1--TS6 evaluation.
- Generated 72-row checkpoint inventory for 3 splits × 3 seeds × 2 detectors × 4 arms.
- Evidence audit, release preflight, SHA-256 locator, checkpoint import, archive builder, tests, and GitHub Actions workflow.
- Reviewer-facing English README and Thai author handoff instructions.

## Deliberately not claimed as complete

- Original Python implementation was not supplied.
- The 72 checkpoint files and their exact local paths/hashes were not supplied.
- Executable runtime configs, six calibrators, six residual-risk banks, and exact RTMPose model identities were not supplied.
- Evidence files numbered `16` through `20` in the evidence index were not supplied.
- The exact complete Python environment was not supplied; the requirements file is a staging reference only.

The package is therefore ready for **evidence review and local completion**, but must not yet be advertised as a fully reproducible public release.
