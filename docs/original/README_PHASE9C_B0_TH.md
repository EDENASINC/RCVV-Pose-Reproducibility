# Phase 9C-B0 — Detector-to-Training Join Lock

Phase นี้ตรวจและล็อกการเชื่อมข้อมูล Real 2D Detector จาก Phase 9B เข้ากับ
training cache เดิม ก่อนเริ่ม four-arm training ของ Phase 9C-B1

## สิ่งที่ตรวจ

- Phase 9C-A ต้อง PASS และมี decision ที่อนุญาตให้เปิด 9C-B
- SHA-256 ของ detector cache ต้องตรงกับ cache ที่ใช้สร้าง calibration/residual bank
- join key คือ `subject / sequence / target_camera_id / frame_idx`
- detector sample ต้อง join ได้ครบและไม่ซ้ำใน split A/B/C
- พิกัด GT pixel จาก Phase 9B ต้องแปลงกลับเป็น camera-normalized pose ตรงกับ training cache
- Train/Validation/Test ต้องมี coverage โดยไม่ใช้ Development Test เลือกโมเดล
- Official TS1–TS6 ต้องไม่ถูกอ่าน

Phase นี้ไม่ train และไม่มี DataLoader ค่า worker จึงยังไม่ถูกใช้งาน แต่ protocol
ล็อก `num_workers=8` สำหรับ Phase 9C-B1 แล้ว

## วิธีรัน

วางไฟล์ทั้งหมดทับที่ `D:\Research\vv_pose_project` แล้วรัน:

```bat
cd /d D:\Research\vv_pose_project
run_phase9c_b0_0-1_everything.bat
```

## ผลที่ต้องส่งกลับ

```text
outputs\phase9c_b_detector_join_preflight\phase9c_b0_results.zip
```

ต้องเห็น:

```text
Scientific decision   : READY_FOR_PHASE9C_B1_FOUR_ARM_SMOKE
Status                : PASS
```

หาก FAIL ให้ส่ง console output ทั้งหมดและ ZIP (ถ้ามี) ห้ามเริ่ม 9C-B1 เอง
