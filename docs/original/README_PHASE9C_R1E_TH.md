# Phase 9C-R1E: Reliability–Virtual-View Synergy Attribution

## เหตุผล

R1D ยืนยันว่า full method ชนะ observed-only แต่ virtual view แบบไม่ใช้ reliability แพ้ observed-only ดังนั้นยังเปิด Official Test ไม่ได้ ขั้นนี้เพิ่ม factorial arm ที่ตัด virtual view ออกแต่คง reliability modulation ไว้ เพื่อพิสูจน์ว่า virtual view มี contribution เมื่อทำงานร่วมกับ reliability จริงหรือไม่

R1D QC v1 มี evidence-contract bug: Train เขียน epoch history ลง `training_history.csv` แต่ QC ไปค้น field ที่ไม่ได้ฝังใน JSON ไฟล์ patch นี้แก้เฉพาะการอ่านหลักฐาน ไม่แก้ metric, checkpoint หรือ scientific gate และไม่ต้อง Train R1D ใหม่

## งานทดลองใหม่

- Arm: `bounded_reliability_observed_modulation`
- Feature: `observed || reliability*observed || reliability` (70 dimensions)
- 3 splits × 3 seeds × 2 detectors = 18 runs
- Checkpoint เลือกจาก matched Validation เท่านั้น
- Development Test ใช้ยืนยันกลไกหลังเลือก checkpoint
- Official TS1–TS6 ไม่ถูกอ่าน

## วิธีรัน

แตก ZIP วางทับ project root แล้วรันทั้งหมด:

```bat
cd /d D:\Research\vv_pose_project
run_phase9c_r1e_0-4_everything.bat
```

หรือแบ่งรันทีละขั้นตาม BAT หมายเลข 0–4 ระบบ reuse run ที่สมบูรณ์แล้วเมื่อรันซ้ำ

## ไฟล์ที่ส่งกลับ

```text
outputs\phase9c_r1e_synergy_attribution\phase9c_r1e_results.zip
```

## Gate

- `READY_FOR_PHASE9C_R1F_OFFICIAL_TEST_Q1_Q2_STRONG_SYNERGY`
- `READY_FOR_PHASE9C_R1F_OFFICIAL_TEST_Q2_CAUTIOUS_SYNERGY`
- `STOP_AND_REDESIGN_VIRTUAL_VIEW_CONTRIBUTION`

Strong gate ต้องยืนยันพร้อมกันว่า full method ชนะ observed-only, full method ชนะ reliability-observed arm และ interaction ระหว่าง virtual view กับ reliability มี 95% hierarchical bootstrap CI เป็นบวกทั้ง MPJPE และ PA-MPJPE
