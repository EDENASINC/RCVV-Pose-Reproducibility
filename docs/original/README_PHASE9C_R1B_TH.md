# Phase 9C-R1B — Detector-Conditioned Virtual-View Training Smoke

## จุดประสงค์

ตรวจว่า Virtual View cache ที่ซ่อมใน Phase 9C-R1A สามารถเข้าสู่ 3D lifter ได้ถูกต้องก่อนทำการทดลองเต็ม โดย Phase นี้เป็น pipeline smoke เท่านั้น ไม่ใช้เลือกโมเดลหรือสรุป claim ทางวิทยาศาสตร์

สิ่งที่ล็อกเพื่อความน่าเชื่อถือระดับ Q1/Q2:

- Virtual View ต้องมาจาก keypoints ของ RTMPose/YOLO11-L จริง
- cache row ต้องตรงกับ `dataset_index` และ `detector_sample_index` ทุกตัวอย่าง
- ตอนประเมิน detector ใด ต้องใช้ Virtual View ที่สร้างจาก detector เดียวกัน
- ใช้ paired common-detection cohort เดียวกันทุก arm
- ห้ามเติม detector failure ด้วย Ground Truth
- ประเมินเฉพาะ validation ใน smoke นี้ ไม่อ่าน development test
- ไม่อ่านหรือ infer Official TS1–TS6
- บังคับ GPU `cuda:0` และ DataLoader workers = 8
- ทดสอบ Windows spawn ด้วย workers = 8 จริงก่อนเริ่ม train; ถ้า worker import/collate ไม่ได้จะหยุดทันที

## การแก้ไขในแพ็กเกจ v2

แพ็กเกจ v1 ส่ง `worker_init_fn` มาจากโมดูล B1 ที่โหลดด้วยชื่อชั่วคราว
`phase9c_b1_v3_r1b` ทำให้ Windows worker import ชื่อนี้ไม่ได้และจบด้วย
`ModuleNotFoundError` แพ็กเกจ v2 เปลี่ยนเป็น initializer ที่อยู่ในสคริปต์ R1B
โดยตรง จึง pickle/import ได้ภายใต้ Windows spawn โดยไม่ลด workers และไม่เปลี่ยน
protocol ทางวิทยาศาสตร์

## วิธีรัน

แตก ZIP แล้ววางทับที่:

```text
D:\Research\vv_pose_project
```

จากนั้นรัน:

```bat
cd /d D:\Research\vv_pose_project
run_phase9c_r1b_0-1_everything.bat
```

หาก v1 เคยหยุดด้วย `ModuleNotFoundError: phase9c_b1_v3_r1b` ให้แตก v2 วางทับ
แล้วรันคำสั่งเดิมได้เลย ขั้น materialize จะทำใหม่ แต่ยังไม่มี training run ที่ต้องกู้คืน

## สิ่งที่ Phase นี้รัน

- Split A, seed 42
- 4 arms
- 7 training runs
- 13 validation metric rows
- matched-detector และ cross-detector evaluation
- 2 epochs แบบจำกัด batches เพื่อทดสอบ pipeline

อย่าใช้ตัวเลข smoke ในบทความ เพราะ training budget ตั้งใจให้เล็กมาก

## ผลลัพธ์ที่ต้องส่งกลับ

```text
outputs\phase9c_r1b_detector_conditioned_smoke\phase9c_r1b_results.zip
```

ผลผ่านที่คาดหวัง:

```text
Status                : PASS
Scientific decision   : READY_FOR_PHASE9C_R1C_CONFIDENCE_MECHANISM_SCREEN
```

ยังไม่ต้องเปิด Official Test และยังไม่ต้องรัน full multisplit–multiseed จนกว่าจะตรวจ R1B และล็อกแบบ Confidence ใน R1C
