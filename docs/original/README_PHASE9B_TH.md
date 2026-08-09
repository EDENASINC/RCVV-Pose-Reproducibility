# Phase 9B-R2 — Direct Video RGB Detector Benchmark

Phase 9B-R1 ยืนยันแล้วว่า Training set มี `S1-S8/Seq1-Seq2`, annotation,
calibration และ video 16 ไฟล์ครบ แต่ยังไม่มี extracted image รุ่น R2 จึงอ่าน
frame จาก video โดยตรง ไม่สร้างภาพซ้ำและไม่ recompress RGB

## เหตุผลเชิงงานวิจัย

- ใช้ Training RGB จริงเพื่อสร้าง detector-error model สำหรับ Research Track V2
- ครอบคลุม 8 subjects และ 16 subject/sequence groups
- รายงานตามจริงว่าเป็น camera-0 subset หากมี video เพียงกล้องเดียว
- ใช้ deterministic temporal sampling 5 Hz ลด frame ซ้ำติดกัน
- จับคู่ video frame index กับ annotation row index และตรวจ frame-count
- ตรวจ GT joint อยู่ในขอบภาพก่อนอนุญาตให้ inference
- รัน detector อิสระ 2 ตัว: RTMPose และ YOLO11-L Pose
- ห้ามใช้ Ground Truth ช่วยเลือกคน
- ไม่อ่าน ไม่ infer และไม่เลือกโมเดลจาก Official TS1-TS6

## วิธีรัน

แตก ZIP วางทับ:

```text
D:\Research\vv_pose_project
```

รัน:

```cmd
cd /d D:\Research\vv_pose_project
run_phase9b_r2_video_repair_and_continue.bat
```

Dependency จาก Phase 9B เดิมใช้ต่อได้ ไม่ต้องติดตั้งซ้ำ ครั้งแรก detector
อาจดาวน์โหลด weights อัตโนมัติ หาก full inference หยุดกลางทาง ให้รัน BAT เดิม
ซ้ำ ระบบจะทำต่อจาก JSONL cache

## GPU repair / resume

หากพบ `cublasLt64_13.dll is missing` หรือ Task Manager แสดงว่า detector
ทำงานด้วย CPU ให้หยุดด้วย `Ctrl+C` แล้ววางแพ็กเกจ GPU patch ทับ project root
จากนั้นรัน:

```cmd
run_phase9b_gpu_repair_and_resume.bat
```

ตัวซ่อม V2 จะตรวจ NVIDIA Driver แล้วเปลี่ยน PyTorch แบบ CPU-only เป็น CUDA
wheel จาก PyTorch official index โดยอัตโนมัติ ไม่ต้องติดตั้ง CUDA Toolkit แยก:

- Driver 580 ขึ้นไป: PyTorch 2.13.0 CUDA 13.0 + torchvision 0.28.0 +
  ONNX Runtime GPU 1.28.0
- Driver 525-579: PyTorch 2.11.0 CUDA 12.8 + torchvision 0.26.0 +
  ONNX Runtime GPU 1.26.0
- Driver ต่ำกว่า 525: หยุดและให้ update NVIDIA Driver ก่อน

- RTMPose ตรวจ provider จาก ONNX session จริงทั้ง detector และ pose model
- รัน strict GPU check ด้วยภาพเปล่าก่อนกลับเข้าสู่ benchmark
- YOLO บังคับ `device=0` และตรวจ backend หลัง inference แรก
- ปิด CPU fallback; หาก GPU ใช้ไม่ได้ Phase จะหยุดทันที
- smoke/full ที่ครบแล้วจะข้ามโดยไม่โหลดโมเดล
- full ที่ทำไว้บางส่วนจะเก็บ JSONL เดิมและทำเฉพาะ sample index ที่ขาด
- ไม่ลบ ไม่คำนวณซ้ำ และไม่อ่าน Official Test `TS1-TS6`

การติดตั้ง PyTorch CUDA wheel ครั้งแรกเป็นไฟล์ขนาดใหญ่และอาจใช้เวลาหลายนาที
ห้ามกด `Ctrl+C` ระหว่างที่ pip กำลังแสดง Downloading/Installing หลังติดตั้ง
เสร็จ BAT จะตรวจ `torch.cuda.is_available()`, ชื่อ RTX 4060, provider ของ
RTMPose และ device ของ YOLO ก่อนทำเฉพาะ sample ที่เหลือ

## Output ที่ส่งกลับ

เมื่อจบ benchmark:

```text
D:\Research\vv_pose_project\outputs\
phase9b_rgb_detector_benchmark\
phase9b_rgb_detector_benchmark_results.zip
```

ถ้า preflight ยัง BLOCKED:

```text
D:\Research\vv_pose_project\outputs\
phase9b_rgb_detector_benchmark\
phase9b_r2_video_media_results.zip
```

ส่งกลับเพียงหนึ่งไฟล์ตามสถานะ ห้ามส่ง RGB, video, annotation หรือ checkpoint
เพิ่มเอง เพราะ ZIP ผลลัพธ์เก็บเฉพาะ index, cache detector, metrics, hashes และ
protocol evidence

## Scientific gate

- matched samples อย่างน้อย 1,000
- alignment อย่างน้อย 90%
- in-frame joint fraction อย่างน้อย 75%
- ครบ 8 subjects และ 16 subject/sequence groups
- detector 2 ตัวมี success rate อย่างน้อย 80% ต่อตัว
- weight hash และ runtime ถูกบันทึก

Phase นี้ยังไม่อ้างว่าค่า 3D ดีขึ้น หน้าที่ของมันคือสร้างหลักฐาน real-detector
ที่เชื่อถือได้สำหรับ Phase 9C เท่านั้น
