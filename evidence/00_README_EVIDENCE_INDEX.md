# Reviewer Evidence Index

This folder contains the locked evidence used in the manuscript **Reliability-Conditioned Virtual-View Pose Fusion for Monocular 3D Human Pose Estimation**. Files are numbered in the order of the paper's experimental narrative. Their internal phase identifiers are intentionally retained inside the JSON metadata and hashes for provenance.

## A. RGB-to-2D detector audit

| File | Meaning in the paper |
|---|---|
| `01_rgb_2d_detector_audit_protocol.json` | Locked protocol for the 160,728-frame RTMPose and YOLO11L-Pose audit |
| `02_rgb_2d_detector_audit_summary.csv` | Reviewer-readable detector summary reported in the detector table |
| `03_rgb_2d_detector_full_metrics.json` | Full detector metrics underlying the summary |
| `04_rgb_2d_detector_runtime_environment.json` | Runtime, CUDA, model, and software-environment record |

## B. Leakage-safe confidence calibration

| File | Meaning in the paper |
|---|---|
| `05_leakage_safe_confidence_calibration_protocol.json` | Protocol for fitting calibrators and the residual-risk bank using training subjects only |
| `06_leakage_safe_confidence_calibration_summary.csv` | Raw-versus-calibrated ECE and Brier results for all split-detector pairs |
| `07_leakage_safe_confidence_calibration_qc_report.json` | Quality-control report confirming the calibration gate |

## C. Development Test factorial experiment

| File | Meaning in the paper |
|---|---|
| `08_development_factorial_training_protocol.json` | Locked training setup for O, OV, OR, and OVR, including fixed yaw -15 degrees |
| `09_development_factorial_training_qc_report.json` | Development training evidence-integrity and QC report |
| `10_development_factorial_attribution_protocol.json` | Predeclared factorial comparisons and hierarchical bootstrap policy |
| `11_development_factorial_attribution_statistics.json` | Development OVR-versus-O results and component-attribution statistics |
| `12_development_factorial_attribution_qc_report.json` | QC report for the factorial attribution analysis |

## D. Locked Official TS1-TS6 Evaluation

| File | Meaning in the paper |
|---|---|
| `13_locked_official_ts1_ts6_evaluation_protocol.json` | Frozen Official evaluation protocol; no retraining and no ground-truth 2D model input |
| `14_locked_official_ts1_ts6_evaluation_qc_report.json` | QC and result-lock confirmation for 2,875 valid frames |
| `15_locked_official_ts1_ts6_factorial_results.json` | Aggregate O/OV/OR/OVR metrics, detector-specific gains, bootstrap intervals, and decision checks |
| `16_locked_official_ts1_ts6_subject_split_seed_detector_arm_metrics.csv` | Detailed subject/split/seed/detector/arm metrics supporting the aggregate results |
| `17_locked_official_ts1_ts6_activity_arm_metrics.csv` | Descriptive activity-level metrics by factorial arm |
| `18_locked_official_ts1_ts6_ensemble_3d_predictions.npz` | Saved MoCap targets and ensemble 3D predictions used for the qualitative figure |
| `19_locked_official_ts1_ts6_rgb_detector_2d_keypoints_cache.npz` | Actual cached RTMPose and YOLO11L-Pose 2D keypoints and confidence from Official RGB frames |
| `20_locked_official_ts5_median_case_selection.json` | Deterministic median-case selection record for the TS5 qualitative example |

## Interpretation boundary

The Development Test supports the full OVR representation and the reliability-only OR component. On the Locked Official TS1-TS6 Evaluation, OVR has better MPJPE and PA-MPJPE point estimates for both detectors, but the hierarchical 95% confidence intervals cross zero. These files therefore support a **positive but unconfirmed trend**, not a claim of superiority or state-of-the-art performance.
