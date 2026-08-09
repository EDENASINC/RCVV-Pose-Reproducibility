# คู่มือเตรียม GitHub สำหรับงาน RCVV-Pose

แพ็กเกจนี้จัดโครงสร้างให้ Reviewer ตรวจได้ 3 ระดับ: ตรวจหลักฐานโดยไม่ใช้ dataset, รัน Official TS1--TS6 จาก checkpoint ที่ล็อกไว้ และ train ใหม่ครบทุก split/seed/detector/arm

ขณะนี้ใส่หลักฐานที่แนบมาแล้ว แต่ยัง **ไม่ใช่ repository ที่ reproduce ได้สมบูรณ์** เพราะยังขาด source code, config ที่โปรแกรมอ่านได้, calibration artifacts, detector cache และ checkpoint จากเครื่อง `D:\Research\vv_pose_project`

ขั้นตอนบนเครื่อง Windows:

1. แตก ZIP นี้ไว้นอกโฟลเดอร์ dataset
2. เปิด [`LOCAL_FILES_TO_ADD.md`](LOCAL_FILES_TO_ADD.md) และคัดลอกไฟล์ตามรายการ
3. เติม `source_path` ใน `models/local_checkpoint_map.csv`
4. รัน `powershell -ExecutionPolicy Bypass -File tools/import_checkpoints.ps1 -MapCsv models/local_checkpoint_map.csv`
5. รัน `python tools/release_preflight.py --mode release`
6. แก้ทุก `MISSING` หรือ `PLACEHOLDER` จนผ่าน
7. สร้าง GitHub repository แบบ Private ก่อน แล้วให้ผู้เขียนร่วมตรวจ
8. Commit source/evidence/config; checkpoint ให้เผยแพร่เป็น GitHub Release assets ไม่ใส่ Git history ปกติ

นอกจาก checkpoint ให้คัดลอก `run_report.json`, `training_history.csv`, metric rows และ artifact-hash manifest ของ locked run ทั้ง 72 ชุดด้วย เพราะเป็นหลักฐานการ train จริงที่ Reviewer ตรวจได้โดยไม่ต้อง train ใหม่ทั้งหมด

ห้ามอัปโหลด:

- MPI-INF-3DHP RGB, annotations หรือ archive ต้นฉบับ
- absolute path, username, token, API key หรือ credential
- `venv`, cache, log, temporary shards และไฟล์ทดลองที่ไม่อยู่ใน locked protocol
- checkpoint ที่ไม่ทราบ split/seed/detector/arm หรือไม่มี SHA-256

ไฟล์ `models/checkpoint_manifest.csv` ระบุปลายทางมาตรฐานของ checkpoint ทั้ง 72 ชุด ส่วน `models/local_checkpoint_map.csv` ใช้กรอกตำแหน่งต้นฉบับจริงบนเครื่องคุณ
