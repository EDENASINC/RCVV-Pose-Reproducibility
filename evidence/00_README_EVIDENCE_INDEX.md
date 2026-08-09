# Reviewer Evidence Index

This folder contains the locked evidence used in the manuscript **Reliability-Conditioned Virtual-View Pose Fusion for Monocular 3D Human Pose Estimation**. Files are numbered in the paper's experimental order. Internal phase identifiers are retained for provenance.

## A. RGB-to-2D detector audit

| File | Meaning in the paper |
|---|---|
| `01_rgb_2d_detector_audit_protocol.json` | Locked protocol for the 160,728-frame RTMPose and YOLO11L-Pose audit |
| `02_rgb_2d_detector_audit_summary.csv` | Reviewer-readable detector summary |
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
| `08_development_factorial_training_protocol.json` | Locked O/OV/OR/OVR training setup, including fixed yaw -15 degrees |
| `09_development_factorial_training_qc_report.json` | Development training integrity and QC report |
| `10_development_factorial_attribution_protocol.json` | Predeclared factorial comparisons and hierarchical bootstrap policy |
| `11_development_factorial_attribution_statistics.json` | Development OVR-versus-O results and component-attribution statistics |
| `12_development_factorial_attribution_qc_report.json` | QC report for factorial attribution |

## D. Locked Official TS1-TS6 evaluation

| File | Availability | Meaning in the paper |
|---|---|---|
| `13_locked_official_ts1_ts6_evaluation_protocol.json` | Included | Frozen Official evaluation protocol; no retraining and no ground-truth 2D model input |
| `14_locked_official_ts1_ts6_evaluation_qc_report.json` | Included | QC and result-lock confirmation for 2,875 valid frames |
| `15_locked_official_ts1_ts6_factorial_results.json` | Included | Aggregate O/OV/OR/OVR metrics, detector-specific gains, bootstrap intervals, and decision checks |
| `16_locked_official_ts1_ts6_subject_split_seed_detector_arm_metrics.csv` | Included | Detailed subject/split/seed/detector/arm metrics |
| `17_locked_official_ts1_ts6_activity_arm_metrics.csv` | Included | Descriptive activity-level metrics by factorial arm |
| `18_locked_official_ts1_ts6_ensemble_3d_predictions.npz` | Not distributed | Dataset-derived MoCap targets and predictions; regenerate locally after obtaining MPI-INF-3DHP |
| `19_locked_official_ts1_ts6_rgb_detector_2d_keypoints_cache.npz` | Not distributed | Derived from Official RGB frames; regenerate with the supplied evaluation pipeline |
| `20_locked_official_ts5_median_case_selection.json` | Included | Deterministic TS5 median-case selection record |

## Interpretation boundary

The Development Test supports the full OVR representation and the reliability-only OR component. On the locked Official TS1-TS6 evaluation, OVR has better MPJPE and PA-MPJPE point estimates for both detectors, but the hierarchical 95% confidence intervals cross zero. The evidence therefore supports a **positive but unconfirmed trend**, not superiority or a state-of-the-art claim.
