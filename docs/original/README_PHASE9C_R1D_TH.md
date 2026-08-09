# Phase 9C-R1D — Locked Full Confirmation

Phase นี้เปิดได้เมื่อ Phase 9C-R1C ได้ `PASS` และคำตัดสิน
`READY_FOR_PHASE9C_R1D_LOCKED_FULL_CONFIRMATION` โดยเลือกกลไก
`bounded_reliability_dual_modulation` จาก validation เท่านั้น

## วัตถุประสงค์

ยืนยันผลภายใต้ protocol ที่ล็อกไว้ก่อนเปิด Official `TS1–TS6` โดยตอบคำถามหลักครบ 3 ข้อ:

1. วิธีเต็มชนะ observed-only detector baseline หรือไม่
2. Confidence แบบ bounded dual modulation ช่วยมากกว่า Virtual View ที่ไม่ใช้ confidence หรือไม่
3. Detector-conditioned Virtual View เองช่วยมากกว่า observed-only หรือไม่

## งานที่รัน

- 3 cross-subject splits: `split_a`, `split_b`, `split_c`
- 3 seeds: `42`, `123`, `2026`
- 2 detectors: RTMPose Performance และ YOLO11-L Pose
- 3 arms: observed-only, Virtual View no-confidence และวิธีเต็มที่ล็อกแล้ว
- รวม `54 training runs`
- `216` validation/development-test metric rows
- checkpoint เลือกจาก matched-detector validation เท่านั้น
- development test ใช้หลังเลือก checkpoint เพื่อยืนยันผล ไม่ใช้เลือกโมเดล
- cross-detector เป็น descriptive robustness ไม่ใช้ตัดสิน primary claim
- เก็บ per-sample MPJPE/PA-MPJPE สำหรับตรวจ pairing และ hierarchical bootstrap
- บังคับ `cuda:0`, `workers=8`, full cohort และห้าม CPU fallback
- ไม่อ่านหรือทำ inference บน Official `TS1–TS6`

## วิธีรันทั้งหมด

แตก ZIP วางทับ `D:\Research\vv_pose_project` แล้วรัน:

```bat
cd /d D:\Research\vv_pose_project
run_phase9c_r1d_0-4_everything.bat
```

เวลาประมาณ `5–6 ชั่วโมง` จากความเร็วจริงของ R1C บน RTX 4060 Laptop GPU

## แบ่งรันตาม Split

```bat
run_phase9c_r1d_0_preflight.bat
run_phase9c_r1d_1_full_split_a.bat
run_phase9c_r1d_2_full_split_b.bat
run_phase9c_r1d_3_full_split_c.bat
run_phase9c_r1d_4_qc_statistics_pack.bat
```

แต่ละ Split คาดประมาณ `1 ชั่วโมง 40 นาที–2 ชั่วโมง` สามารถรัน BAT เดิมซ้ำได้;
run ที่มี report และ artifact hash ตรง protocol จะถูก reuse โดยไม่ Train ซ้ำ

## Output ที่ต้องส่งกลับ

ส่งไฟล์เดียว:

```text
outputs\phase9c_r1d_locked_full_confirmation\phase9c_r1d_results.zip
```

QC ที่สมบูรณ์ต้องเป็น `Status: PASS` ส่วนคำตัดสินทางวิทยาศาสตร์อาจเป็น:

```text
READY_FOR_PHASE9C_R1E_OFFICIAL_TEST_Q1_Q2_STRONG
READY_FOR_PHASE9C_R1E_OFFICIAL_TEST_Q2_CAUTIOUS
STOP_AND_REASSESS_BEFORE_OFFICIAL_TEST
```

ทุกผลมีความหมายทางวิทยาศาสตร์ ไม่ควรแก้ threshold หลังเห็น development-test
หากผลไม่ผ่านเกณฑ์ ให้ประเมิน claim/วิธีใหม่ก่อนเปิด Official Test

หากโปรแกรมหยุดกลางทาง ให้รัน BAT ของ Split เดิมซ้ำ หากเกิด Error ให้ส่ง console
ตั้งแต่บรรทัด `[RUN]` ล่าสุดและไฟล์ `phase9c_r1d_run_report.json` ของ run นั้นถ้ามี
