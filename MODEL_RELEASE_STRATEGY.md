# Model release strategy

Keep source code, configs, small evidence, and manifests in ordinary Git. Keep checkpoint binaries out of ordinary Git history.

Recommended release assets:

| Asset | Contents | Count |
|---|---|---:|
| `checkpoints_O_rtmpose.zip` | O, RTMPose, all splits/seeds | 9 |
| `checkpoints_O_yolo11l.zip` | O, YOLO11L-Pose, all splits/seeds | 9 |
| `checkpoints_OV_rtmpose.zip` | OV, RTMPose, all splits/seeds | 9 |
| `checkpoints_OV_yolo11l.zip` | OV, YOLO11L-Pose, all splits/seeds | 9 |
| `checkpoints_OR_rtmpose.zip` | OR, RTMPose, all splits/seeds | 9 |
| `checkpoints_OR_yolo11l.zip` | OR, YOLO11L-Pose, all splits/seeds | 9 |
| `checkpoints_OVR_rtmpose.zip` | final OVR, RTMPose, all splits/seeds | 9 |
| `checkpoints_OVR_yolo11l.zip` | final OVR, YOLO11L-Pose, all splits/seeds | 9 |

This grouping lets a reviewer download only the comparison needed. Publish `SHA256SUMS.txt`, `checkpoint_manifest.csv`, and model/config metadata with the same release tag. Keep a frozen release such as `v1.0.0-paper-information-4512788`; do not silently replace assets after sharing the URL.

If a single archive approaches the host's per-asset limit, split it by split or seed. Do not compress the MPI-INF-3DHP dataset into any release asset.
