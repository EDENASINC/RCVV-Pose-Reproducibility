# Phase 9C-R1A — Detector-Conditioned Virtual-View Cache Repair

## เหตุผลที่ต้องทำ Phase นี้

ผล Phase 9C-B2 ผ่าน QC เชิงการรันครบ 63 runs / 234 metric rows แต่พบข้อจำกัดด้าน provenance ที่สำคัญต่อการตีพิมพ์:

- observed branch ใช้ keypoints จาก RTMPose หรือ YOLO11-L จริง
- แต่ virtual branch เดิมอ่าน `teacher_virtual` จาก Phase 5R.1A
- cache ดังกล่าวสร้างโดยป้อน `target_pose2d_camera_root` (2D ground truth) เข้า Virtual-View Generator

ดังนั้น B2 ยังไม่ใช่ end-to-end detector-conditioned virtual-view experiment ตามแผนภาพระบบ และห้ามใช้ผล B2 อ้างว่า virtual view จาก detector จริงช่วยเพิ่มความแม่นยำ

Phase 9C-R1A จะแก้ให้ Virtual-View Generator รับ input จาก:

`phase9b_detector_cache.prediction_c15_px -> inverse intrinsic -> root-centered camera coordinates`

แยกสำหรับ RTMPose และ YOLO11-L พร้อมรักษา subject-OOF generator policy เดิม

## วิธีรัน

แตก ZIP วางทับที่:

```text
D:\Research\vv_pose_project
```

จากนั้นรัน:

```bat
cd /d D:\Research\vv_pose_project
run_phase9c_r1a_0-4_everything.bat
```

Phase นี้บังคับใช้ `cuda:0` และไม่มี CPU fallback

## สิ่งที่ Phase นี้ทำ

1. ตรวจ B2 redesign gate และ Official-Test isolation
2. สร้าง detector-conditioned virtual-view cache สำหรับ split A/B/C
3. ใช้ paired common-detection cohort เดียวกันระหว่าง detector
4. ใช้ `holdout_<subject>` generator ใน train และ `full_train` ใน val/test
5. ตรวจว่า cache ใหม่ต่างจาก legacy clean-conditioned cache จริง
6. ตรวจว่า RTMPose และ YOLO ให้ virtual views ที่ไม่เหมือนกัน
7. บันทึก hash และ provenance แล้วแพ็กผล QC

Phase นี้ไม่ train 3D lifter, ไม่คำนวณ MPJPE/PA-MPJPE และไม่อ่าน Official TS1–TS6

## ไฟล์ที่ต้องส่งกลับ

เมื่อเสร็จ ส่งไฟล์เดียว:

```text
outputs\phase9c_r1a_detector_conditioned_virtual_cache\phase9c_r1a_results.zip
```

ผลที่ต้องได้:

```text
Status                : PASS
Scientific decision   : READY_FOR_PHASE9C_R1B_DETECTOR_CONDITIONED_SMOKE
```

หากหยุดกลางทาง รัน BAT เดิมซ้ำได้ split cache ที่ผ่าน hash แล้วจะถูก reuse

## แนวทางหลัง R1A

หาก R1A ผ่าน จะเปิด R1B smoke study โดยใช้ detector-conditioned virtual views เท่านั้น และเปรียบเทียบ:

- observed detector baseline
- detector-conditioned virtual view แบบไม่ใช้ confidence
- calibrated confidence แบบตัด residual-risk feature ที่มีปัญหา
- reliability-gated fusion ที่กำหนดไว้ล่วงหน้า

การเลือก candidate จะใช้ validation เท่านั้น ส่วน Development test ของ B2 ถือว่าเปิดดูแล้วและจะไม่ถูกใช้เป็นหลักฐานยืนยันอิสระสำหรับวิธีที่ redesign ภายหลัง
