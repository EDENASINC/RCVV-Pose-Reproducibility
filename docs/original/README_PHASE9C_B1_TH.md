# Phase 9C-B1 - Four-Arm Smoke Training

Phase นี้เป็น smoke gate ก่อนเปิด training เต็มของ Phase 9C-B2

สิ่งที่ทำ:

- ใช้ผล `Phase 9C-A` และ `Phase 9C-B0` ที่ผ่าน QC แล้ว
- ล็อก `num_workers=8`
- รันบน `split_a`, `seed=42`
- ตรวจ 4 arms:
  - `clean_observed_legacy`
  - `detector_aware_observed`
  - `virtual_view_no_confidence`
  - `calibrated_confidence_virtual_view`
- สำหรับ detector-dependent arms จะ train แยกจาก `rtmpose_performance` และ `yolo11l_pose`
- ประเมินทั้ง matched-detector และ cross-detector บน validation/test
- ไม่อ่าน Official `TS1-TS6`
- V2 แก้ Windows DataLoader worker crash โดย materialize เฉพาะ smoke subset
  ใน main process ก่อน แล้วจึงใช้ workers=8 กับ compact tensor cache
- detector cache และ training shards จะไม่ถูก copy/load ซ้ำใน worker
- ปิด persistent workers, ใช้ prefetch factor 1 และจำกัด 1 CPU thread ต่อ worker
- V3 แก้ detector missed frames ที่เก็บ keypoint เป็น NaN โดยไม่เติมค่าจาก ground truth
- metric หลักใช้ paired common-detection cohort: ทุก arm และ detector ใช้ sample ชุดเดียวกัน
- รายงาน detection coverage/missed detections แยกตาม detector และ partition เพื่อรักษาความโปร่งใส
- fail-fast ทันทีหาก feature, target, loss หรือ metric ยังมี NaN/Inf หลังใช้ cohort lock

วิธีรัน:

```bat
cd /d D:\Research\vv_pose_project
run_phase9c_b1_0-1_everything.bat
```

ไฟล์ที่ต้องส่งกลับ:

```text
outputs\phase9c_b1_four_arm_smoke\phase9c_b1_results.zip
```

ควรเห็น:

```text
Scientific decision   : READY_FOR_PHASE9C_B2_FULL_MULTISPLIT_MULTISEED_TRAINING
Status                : PASS
```

หมายเหตุ: ผล metric ของ B1 ยังไม่ใช่ผลสำหรับตีพิมพ์โดยตรง หน้าที่ของ Phase นี้คือพิสูจน์ว่า data join, detector cache, calibration, residual-bank feature, training, checkpoint, matched/cross evaluation และ QC ทำงานครบก่อนเปิด B2 full training

เหตุผลของ paired common-detection cohort: detector ที่พลาดทั้งเฟรมไม่มี 2D pose สำหรับคำนวณ MPJPE อย่างเป็นธรรม จึงไม่นำ GT มาแทนและไม่เลือก sample แยกตาม detector แต่ใช้จุดตัดของเฟรมที่ detector ทั้งสองตรวจพบสำหรับ paired comparison พร้อมรายงาน coverage แยกต่างหาก

ช่วง `[MATERIALIZE]` ใช้ workers=0 โดยตั้งใจเพื่อ diagnostic และลด RAM เท่านั้น
ช่วง DataLoader training/evaluation ยังคงใช้ workers=8 ตาม protocol lock
