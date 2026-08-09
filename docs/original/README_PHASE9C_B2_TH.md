# Phase 9C-B2 — Full Multisplit-Multiseed Real-Detector Study

Phase นี้เปิดได้เมื่อ Phase 9C-B1 V3 ได้ `PASS` และคำตัดสิน
`READY_FOR_PHASE9C_B2_FULL_MULTISPLIT_MULTISEED_TRAINING` เท่านั้น

## งานที่รัน

- 4 arms
- 3 cross-subject splits: `split_a`, `split_b`, `split_c`
- 3 seeds: `42`, `123`, `2026`
- detector-dependent arms train แยกด้วย RTMPose และ YOLO11-L
- รวม 63 training runs และ 234 validation/test metric rows
- ประเมิน matched-detector และ cross-detector
- ใช้ paired common-detection cohort เดียวกันทุก arm
- ไม่เติม detector failure ด้วย ground truth
- ไม่อ่าน Official `TS1–TS6`

Clean observed legacy เป็น upper reference บน cohort เดียวกัน ไม่ใช่ primary baseline
สำหรับ detector claim. Primary baseline คือ `detector_aware_observed`.

## Fairness และ model selection

- architecture เหมือน B1 smoke
- budget, batch size, optimizer, early stopping และ workers เท่ากัน
- เลือก checkpoint จาก matched validation เท่านั้น
- Development test ใช้หลังเลือก checkpoint เพื่อ robustness เท่านั้น
- Official test ยังล็อก
- บันทึก per-sample error เพื่อ hierarchical bootstrap ตาม split/seed/detector

## วิธีรัน

แตก ZIP วางทับ `D:\Research\vv_pose_project` แล้วรัน:

```bat
cd /d D:\Research\vv_pose_project
run_phase9c_b2_0-4_everything.bat
```

สามารถรันแยกได้:

```text
run_phase9c_b2_0_preflight.bat
run_phase9c_b2_1_full_split_a.bat
run_phase9c_b2_2_full_split_b.bat
run_phase9c_b2_3_full_split_c.bat
run_phase9c_b2_4_qc_aggregate_pack.bat
```

รันซ้ำได้: run ที่มี report/checkpoint/per-sample hash ตรง protocol จะถูก reuse
โดยไม่ train ซ้ำ

## Output ที่ต้องส่งกลับ

```text
outputs\phase9c_b2_full_multisplit_multiseed\phase9c_b2_results.zip
```

QC ที่ถูกต้องต้องเป็น `Status: PASS` แต่ scientific decision อาจเป็นได้ 3 แบบ:

```text
READY_FOR_PHASE9C_B3_Q1_Q2_DETECTOR_CLAIM_LOCK
READY_FOR_PHASE9C_B3_Q2_CAUTIONARY_CLAIM_LOCK
REDESIGN_CONFIDENCE_FUSION_BEFORE_OFFICIAL_TEST
```

แบบที่สามไม่ใช่ความผิดพลาดของโปรแกรม แต่หมายถึงหลักฐานเชิงผลลัพธ์ยังไม่แข็งพอ
และต้อง redesign ก่อนแตะ Official test

หากหยุดกลางทาง ให้รัน BAT เดิมซ้ำ ระบบจะข้าม run ที่สมบูรณ์แล้ว หากเกิด Error
ให้ส่ง console ตั้งแต่บรรทัด `[RUN]` ล่าสุดและไฟล์ `phase9c_b2_run_report.json`
ของ run นั้นถ้ามี
