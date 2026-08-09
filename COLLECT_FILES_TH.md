# วิธีรวมไฟล์จริงจาก `D:\Research\vv_pose_project`

1. แตก ZIP นี้ไปไว้ที่อื่น เช่น `D:\Research\rcvv_pose_reproducibility_package` ห้ามแตกทับโครงการเดิม
2. ดับเบิลคลิก `RUN_COLLECT_FROM_D_DRIVE.bat`
3. เปิด `collection_report\COLLECTION_SUMMARY.json` และ `collection_report\collected_files.csv`
4. ถ้ามี `MISSING` ให้ส่งสองไฟล์รายงานกลับมาเพื่อตรวจตำแหน่งที่ขาด
5. ถ้าไม่มีไฟล์ขาด ให้รัน:

```powershell
python tools\validate_evidence.py
python -m unittest discover -s tests -v
python tools\release_preflight.py --mode release
```

## สิ่งที่ตัวช่วยคัดลอก

- เข้า Git: Python source, scripts Phase 9B/9C, locked configs, joint mappings, environment, evidence, run reports และ training histories
- ไม่เข้า Git แต่ใช้สร้าง GitHub Release: checkpoint `best.pt` 72 ชุด และ residual-risk bank
- เข้าโฟลเดอร์ private เฉพาะเมื่อใส่ option เพิ่ม: Official derived NPZ ซึ่งต้องตรวจสิทธิ์ก่อนเผยแพร่

ตัวช่วยจะไม่คัดลอก `.venv`, MPI-INF-3DHP RGB/video/annotation, cache, `last.pt`, per-sample NPZ หรือ detector weights

## รันทดสอบโดยไม่คัดลอก

```powershell
python tools\collect_local_artifacts.py `
  --project-root "D:\Research\vv_pose_project" `
  --repo-root . `
  --dry-run
```

## คัดลอก Official derived files ไว้ตรวจแบบ Private

```powershell
python tools\collect_local_artifacts.py `
  --project-root "D:\Research\vv_pose_project" `
  --repo-root . `
  --include-private-official-derived
```

ห้าม `git add -f checkpoints`, `local_release_inputs`, `private_review_only` หรือ detector weights
