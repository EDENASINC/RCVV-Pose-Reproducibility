# Phase 9C-R1G — Q2 Publication Evidence Freeze & Claim Lock

Phase นี้ไม่ Train และไม่เปิด Official TS1–TS6 ใหม่ หน้าที่คือรวมหลักฐาน R1E และ R1F ที่ล็อกแล้วให้เป็นตาราง รูป และข้อความพร้อมใช้เขียน Paper Q2 โดยรักษาผลลบและข้อจำกัดไว้ครบ

## Input ที่ต้องมี

วางแพ็กเกจนี้ทับ project root เดิม โดยต้องมีไฟล์จาก Phase ก่อนหน้า:

```text
outputs\phase9c_r1e_synergy_attribution\phase9c_r1e_results.zip
outputs\phase9c_r1f_locked_official_test\phase9c_r1f_results.zip
```

สคริปต์ค้นหาใน `outputs` แบบ recursive ด้วย จึงรองรับชื่อโฟลเดอร์ย่อยที่ต่างเล็กน้อย แต่หากพบหลายไฟล์จะหยุดเพื่อไม่เลือกหลักฐานผิดชุด

## วิธีรัน

```bat
cd /d D:\Research\vv_pose_project
run_phase9c_r1g_0-2_everything.bat
```

หรือแยกทีละขั้น:

```bat
run_phase9c_r1g_0_preflight.bat
run_phase9c_r1g_1_build_assets.bat
run_phase9c_r1g_2_qc_pack.bat
```

ใช้ CPU เท่านั้น คาดประมาณ 1–3 นาที ไม่ต้องใช้ GPU

## สิ่งที่ Phase สร้าง

- ตารางผล Development และ Official พร้อม CI
- ตาราง factorial attribution: reliability-only, conditional virtual view และ interaction
- ตารางแยก detector, subject และ activity
- รูป Development-vs-Official CI, arm metrics, detector gains และ subject gains
- Evidence ledger และ hash manifest
- Claim matrix ระบุข้อความที่เขียนได้/ต้องเขียนแบบระวัง/ห้ามอ้าง
- Results draft, Limitations draft และ Q1 extension registry
- สรุปตัวเลขสำหรับนำไปอัปเดต Abstract/Results/Discussion

## Output ที่ส่งกลับ

```text
outputs\phase9c_r1g_q2_evidence_freeze\phase9c_r1g_results.zip
```

ผลผ่านที่คาด:

```text
READY_FOR_PHASE9C_R1H_Q2_MANUSCRIPT_UPDATE
```

## ข้อห้าม

- ห้ามรัน Official inference ใหม่เพื่อหาผลที่ดีขึ้น
- ห้ามเปลี่ยน bootstrap seed/hierarchy, arm, checkpoint, detector, metric หรือ frame cohort
- ห้ามเขียนว่า Official superiority หรือ Virtual-View synergy ได้รับการยืนยันทางสถิติ
- ห้ามอ้าง SOTA จนกว่าจะมี protocol-matched literature audit แยกต่างหาก

งานต่อยอด Q1 ถูกแยกไว้ใน registry และต้องใช้ independent dataset protocol ใหม่ ไม่ย้อนใช้ TS1–TS6 สำหรับเลือกโมเดล
