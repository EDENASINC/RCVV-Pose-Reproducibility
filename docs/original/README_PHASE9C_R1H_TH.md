# Phase 9C-R1H — Q2 Manuscript Update

Phase นี้สร้างบทความฉบับใหม่จากหลักฐาน R1G ที่ล็อกแล้ว ไม่มีการ Train, ไม่มี inference และไม่เปิด Official TS1–TS6 ซ้ำ

## วิธีรัน

แตก ZIP วางทับ project root:

```text
D:\Research\vv_pose_project
```

แล้วรัน:

```bat
run_phase9c_r1h_0-2_everything.bat
```

ใช้เวลาประมาณ 10–30 วินาที ไม่ใช้ GPU และไม่จำเป็นต้องติดตั้ง LaTeX เพราะมี PDF ที่ compile และตรวจแล้วรวมอยู่ในแพ็ก

## Input ที่ต้องมี

```text
outputs\phase9c_r1g_q2_evidence_freeze\phase9c_r1g_results.zip
```

## Output ที่ส่งกลับ

```text
outputs\phase9c_r1h_q2_manuscript_update\phase9c_r1h_results.zip
```

ภายในมี `manuscript/main.tex`, `manuscript/main.pdf`, references, figures, evidence, build/QC reports และรายการข้อมูลผู้เขียนที่ต้องเติมก่อนส่งวารสาร

ผลผ่านที่คาด:

```text
READY_FOR_PHASE9C_R1I_Q2_VENUE_TEMPLATE_AND_SUBMISSION_AUDIT
```
