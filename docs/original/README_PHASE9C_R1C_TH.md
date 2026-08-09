# Phase 9C-R1C — Detector-Conditioned Bounded Confidence Mechanism Screen

## จุดประสงค์

Phase นี้คัดเลือกกลไก Confidence ที่เหมาะสมกับ Virtual View ซึ่งสร้างจาก output ของ RTMPose/YOLO11-L จริง โดยใช้เฉพาะ train และ validation ก่อนล็อกวิธีสำหรับการทดลองเต็ม 3 splits × 3 seeds

R1C ไม่ใช้ผล Smoke ของ R1B เลือกโมเดล และไม่อ่าน development-test หรือ Official TS1–TS6

## สิ่งที่ทดสอบ

ใช้ Split A/B/C, seed 42, detector 2 ตัว และ 5 arms รวม 30 training runs:

1. `detector_conditioned_virtual_view_no_confidence` — baseline
2. `calibrated_probability_concat` — ต่อ calibrated probability แบบ bounded
3. `bounded_reliability_concat` — ต่อ calibrated probability และ bounded residual quality
4. `bounded_reliability_virtual_gate` — ใช้ reliability gate ลดน้ำหนัก virtual joints ที่ไม่น่าเชื่อถือ
5. `bounded_reliability_dual_modulation` — เก็บ pose เดิมและเพิ่ม observed/virtual ที่ถูก reliability modulate

Residual risk ดิบจะไม่ถูกป้อนเข้าโมเดลโดยตรง เพราะสเกลกว้างและเป็นสาเหตุเชิงกลไกที่อาจรบกวน optimization ในรุ่นก่อน ค่า risk ถูกแปลงเป็นช่วง `[0,1]` ด้วยสถิติจาก training residual bank ของ split/detector/joint ที่ตรงกันเท่านั้น

## วิธีติดตั้ง

แตก ZIP วางทับ project root:

```text
D:\Research\vv_pose_project
```

ไฟล์จาก Phase 9C-R1A, R1B v2, B1 v3, 9C-A และ 9C-B0 ต้องยังอยู่ใน project เดิม

## วิธีรันทั้งหมด

```bat
cd /d D:\Research\vv_pose_project
run_phase9c_r1c_0-4_everything.bat
```

เวลาประมาณ 2.5–3.5 ชั่วโมงบน RTX 4060 Laptop GPU ตามความเร็ว Phase 9C-B2 เดิม

## วิธีแบ่งรันเพื่อไม่คร่อมการเดินทาง

รันตามลำดับและหยุดหลังจบแต่ละไฟล์ได้:

```bat
run_phase9c_r1c_0_preflight.bat
run_phase9c_r1c_1_screen_split_a.bat
run_phase9c_r1c_2_screen_split_b.bat
run_phase9c_r1c_3_screen_split_c.bat
run_phase9c_r1c_4_qc_select_pack.bat
```

คาดว่าแต่ละ Split ใช้ประมาณ 50–75 นาที การรันไฟล์เดิมซ้ำจะ reuse run ที่ผ่านและ hash ตรงแล้ว

## เกณฑ์เลือกที่ล็อกไว้

เลือกจาก matched-detector validation เท่านั้น Candidate ต้อง:

- Macro MPJPE และ PA-MPJPE ดีกว่า no-confidence ทั้งคู่
- อย่างน้อย 2 ใน 3 splits ชนะทั้งสอง metric เมื่อเฉลี่ย detector
- RTMPose และ YOLO11-L ต่างต้องชนะทั้งสอง metric เมื่อเฉลี่ย splits
- ถ้ามี candidate ใกล้กันภายใน 0.05 percentage point เลือก feature dimension ต่ำกว่า

Cross-detector validation รายงานเพื่อ robustness แต่ไม่ใช้เลือก Candidate

ถ้าไม่มี confidence arm ผ่าน จะล็อก `detector_conditioned_virtual_view_no_confidence` และเดินหน้าทำ full confirmation ต่อได้ ไม่บังคับสร้าง claim ว่า confidence ช่วย

## Output ที่ต้องส่งกลับ

เมื่อจบ ส่งไฟล์เดียว:

```text
outputs\phase9c_r1c_confidence_mechanism_screen\phase9c_r1c_results.zip
```

ผลที่ถูกต้องต้องขึ้น `Status: PASS` และ scientific decision อย่างใดอย่างหนึ่ง:

```text
READY_FOR_PHASE9C_R1D_LOCKED_FULL_CONFIRMATION
```

หรือ

```text
READY_FOR_PHASE9C_R1D_NO_CONFIDENCE_FULL_CONFIRMATION
```

ทั้งสองแบบเดินหน้าสู่ Paper ได้ แบบแรกอนุญาตให้ทดสอบ confidence claim ต่อ ส่วนแบบหลังล็อกผลเชิงลบของ confidence และโฟกัส claim หลักที่ detector-conditioned virtual-view fusion

## หลักฐานที่บรรจุในผลลัพธ์

- Protocol lock และ preflight
- 30 run reports และ 60 validation metric rows
- ตารางเปรียบเทียบ candidate กับ no-confidence แบบ paired
- ผล matched/cross detector
- Selected arm และเหตุผลตาม gate
- Artifact hashes
- หลักฐานว่า development-test และ Official TS1–TS6 ไม่ถูกอ่าน
