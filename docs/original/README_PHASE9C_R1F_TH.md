# Phase 9C-R1F — One-Time Detector-Input Official TS1–TS6 Confirmation

Phase นี้ประเมินวิธี detector-input ที่ล็อกแล้วบน Official Test หลัง Phase 9C-R1E ผ่านสถานะ
`READY_FOR_PHASE9C_R1F_OFFICIAL_TEST_Q2_CAUTIOUS_SYNERGY`

## จุดประสงค์

- ประเมินโมเดลด้วย RGB detector จริง ไม่ใช้ GT 2D เป็น model input
- เปรียบเทียบ 4 arms ที่ล็อกแล้วด้วย checkpoint เดิม 72 ชุด
- ใช้ทุก official valid frame; detection failure ไม่ถูกตัดทิ้ง
- รายงาน MPJPE, PA-MPJPE, 3DPCK และ AUC
- ใช้ paired hierarchical bootstrap 20,000 ครั้ง
- ห้ามใช้ Official Test เลือก checkpoint, ปรับโมเดล หรือ retrain

## วิธีรัน

แตก ZIP วางทับ project root แล้วรัน:

```bat
cd /d D:\Research\vv_pose_project
run_phase9c_r1f_0-3_everything.bat
```

หรือรันแยกขั้น:

```bat
run_phase9c_r1f_0_preflight.bat
run_phase9c_r1f_1_official_rgb_detectors.bat
run_phase9c_r1f_2_unseal_official_3d_metrics.bat
run_phase9c_r1f_3_qc_pack.bat
```

ขั้น detector resume ได้จาก JSONL หากหยุดกลางทาง ส่วนผล 3D ที่เสร็จและถูกล็อกแล้วจะ reuse โดยไม่คำนวณใหม่

## Output ที่ต้องส่งกลับ

เมื่อจบครบ ส่งไฟล์เดียว:

```text
outputs\phase9c_r1f_locked_official_test\phase9c_r1f_results.zip
```

หาก preflight ถูก BLOCKED ส่ง:

```text
outputs\phase9c_r1f_locked_official_test\phase9c_r1f_preflight_diagnostic.zip
```

## การตีความ

- `OFFICIAL_CONFIRMED_Q1_Q2_CORE_EVIDENCE`: หลักฐานแกนกลางเหมาะสำหรับเดินหน้าสู่ paper Q1/Q2
- `OFFICIAL_CONFIRMED_Q2_MODEST`: ยืนยันผลภายนอกแต่ขนาดผลต่ำกว่าเป้าหมาย 2%
- `OFFICIAL_TREND_ONLY_WRITE_Q2_CAUTIOUS`: ผลเฉลี่ยเป็นบวกแต่เงื่อนไข confirmatory ยังไม่ครบ
- `NOT_EXTERNALLY_CONFIRMED_LOCK_NEGATIVE_RESULT`: ต้องล็อกผลลบ ห้าม rerun เพื่อหาผลที่ดีกว่า

หมายเหตุ: ผล R1E แสดงว่า full method ชนะ observed-only อย่างมี CI เป็นบวกบน Development Test แต่ incremental virtual-view และ interaction CI ยังคร่อมศูนย์ ดังนั้น claim ก่อน R1F คือ reliability-conditioned fusion เป็นแกนหลัก และ virtual view เป็น incremental trend เท่านั้น

เพื่อความโปร่งใสระดับวารสาร: โครงการเคยประเมิน TS1–TS6 ใน Phase 7 ด้วยระบบเก่าที่ใช้ GT 2D แล้ว R1F จึงเป็น “การประเมินครั้งแรกของวิธี Phase 9C ที่รับ RGB detector จริง” ไม่ใช่การเห็น Official partition ครั้งแรกของทั้งโครงการ การพัฒนา R1A–R1E ยังคงไม่อ่านหรือเลือกวิธีจาก TS1–TS6 และรายงานจะบันทึกข้อเท็จจริงนี้ตรงไปตรงมา
