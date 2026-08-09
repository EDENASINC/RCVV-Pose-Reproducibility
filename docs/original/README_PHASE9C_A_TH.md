# Phase 9C-A V2 — Leakage-Safe Confidence Calibration + Real Residual Bank

Phase 9B-R3 ผ่านแล้วด้วย detector จริง 2 ตัว จำนวน 160,728 samples แต่ raw
confidence ยัง calibration ไม่ดี (`ECE 0.2846` สำหรับ RTMPose และ `0.4701`
สำหรับ YOLO11-L Pose) จึงยังไม่ควรส่ง confidence ดิบเข้า V2 model

Phase 9C-A ทำสองอย่างก่อนเริ่ม training ยาว:

1. Fit confidence calibration แยก detector และแยก split โดยใช้ training
   subjects เท่านั้น
2. สร้าง nonparametric residual bank ของ `(prediction - GT) / torso` แยก
   detector, joint และ confidence bin สำหรับ detector-matched augmentation

Validation subject ใช้ QC calibration เท่านั้น ไม่เลือกวิธี calibration
และ test subject ไม่ถูกใช้ fit calibration หรือเลือก threshold

## วิธีรัน

แตก ZIP แล้ววางทับที่:

```text
D:\Research\vv_pose_project
```

จากนั้นรัน:

```cmd
cd /d D:\Research\vv_pose_project
run_phase9c_a_0-1_everything.bat
```

Phase นี้ไม่รัน detector ซ้ำ ไม่ train 3D model และไม่อ่าน Official TS1-TS6
เวลาหลักจะใช้ในการอ่าน cache 63 MB, fit calibration และ pack residual bank

รุ่น V2 รองรับโครงผล Phase 9B-R3 ทั้งสามแบบโดยอัตโนมัติ:

- ไฟล์ authoritative ใน `outputs\\phase9b_rgb_detector_benchmark\\full\\`
- ไฟล์ flattened ในโฟลเดอร์ `outputs\\phase9b_rgb_detector_benchmark\\`
- ไฟล์ภายใน `phase9b_rgb_detector_benchmark_results.zip`

ก่อนใช้ข้อมูล สคริปต์ยังบังคับตรวจ Evidence Lock ของ QC/cache และตรวจว่า runtime
เป็น `phase9b_detector_runtime_v3` พร้อม SHA-256 ของ RTMPose person detector,
RTMPose pose estimator และ YOLO ครบ จึงไม่ยอมใช้ runtime เก่าหรือผลก่อน R3

## QC gate

ต้องผ่านครบ:

- Phase 9B-R3 QC/hash ตรงกับ Evidence Lock
- calibration 6 คู่ = 3 splits x 2 detectors
- held-out Validation ECE ดีขึ้นครบทั้ง 6 คู่
- held-out Validation Brier score ดีขึ้นครบทั้ง 6 คู่
- residual bank shape/count/hash ถูกต้อง
- Official TS1-TS6 ยังไม่ถูกอ่าน

ผลผ่านที่คาดหวัง:

```text
Scientific decision : READY_FOR_PHASE9C_B_FOUR_ARM_MULTISPLIT_TRAINING
Status              : PASS
```

## ส่งกลับไฟล์เดียว

```text
outputs\phase9c_a_calibrated_error_bank\phase9c_a_results.zip
```

เมื่อผ่าน จะเปิด Phase 9C-B ซึ่งเปรียบเทียบ 4 arms แบบ 3 splits x 3 seeds:

1. clean-2D observed-only baseline
2. detector-aware observed-only
3. virtual-view without confidence
4. full calibrated confidence-aware virtual-view + joint fusion

Phase 9C-B จะล็อก paired statistics, detector-matched และ cross-detector
evaluation ก่อนเปิด held-out Development Test และจะไม่แตะ Official TS1-TS6
