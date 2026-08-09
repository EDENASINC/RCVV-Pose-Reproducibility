# RCVV-Pose model card

## Intended use

Research reproduction of the submitted article on monocular 3D human pose estimation with real 2D detector inputs. Not validated for safety-critical, clinical, biometric, or surveillance decisions.

## Locked variants

- O: `detector_aware_observed`
- OV: `detector_conditioned_virtual_view_no_confidence`
- OR: `bounded_reliability_observed_modulation`
- OVR: `bounded_reliability_dual_modulation` (final selected model)

Every variant is represented across three subject splits, three seeds, and two detectors. Checkpoint selection used matched-detector validation only. Official TS1--TS6 was not used for checkpoint or arm selection.

## Inputs and outputs

The OVR feature is `observed || virtual || reliability*observed || reliability*virtual || reliability` (126-D). The model predicts root-relative 3D pose. Official scoring uses the canonical non-root 14 joints.

## Limitations

Official aggregate point estimates favored OVR, but hierarchical 95% confidence intervals crossed zero. Results should be described as a positive but unconfirmed trend.

## Required metadata per checkpoint

Each checkpoint release must record logical ID, source path, SHA-256, file size, architecture/config identifier, split, seed, detector, arm, selected epoch, and validation score.
