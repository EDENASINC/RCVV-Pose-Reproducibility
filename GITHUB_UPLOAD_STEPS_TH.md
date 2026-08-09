# ขั้นตอนนำขึ้น GitHub

## 1. เตรียมในเครื่องก่อน

ใช้โฟลเดอร์โครงการนี้เป็น repository ใหม่ อย่า `git init` ที่ `D:\Research\vv_pose_project` ทันที เพราะอาจเผลอ commit dataset และ run artifacts ทั้งหมด

คัดลอกเฉพาะ source/config ตาม `LOCAL_FILES_TO_ADD.md` แล้วตรวจ:

```powershell
python tools\validate_evidence.py
python -m unittest discover -s tests -v
python tools\release_preflight.py --mode release
```

คำสั่งสุดท้ายต้องขึ้น `PASS` ก่อนเปิด Public

## 2. สร้าง Private repository

ชื่อที่แนะนำ: `rcvv-pose`

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Prepare reproducibility package for information-4512788"
git remote add origin https://github.com/YOUR_USERNAME/rcvv-pose.git
git push -u origin main
```

ตรวจ `git status` และ `git diff --cached --stat` ก่อน commit ทุกครั้ง ต้องไม่มี `data/`, MPI-INF-3DHP archive, `.env`, checkpoint binary หรือ absolute-path log

## 3. Version control ที่ควรใช้

- Tag ฉบับตรงกับ paper: `v1.0.0-paper-information-4512788`
- ใช้ branch/commit ปกติสำหรับ code, config, documentation และ evidence ขนาดเล็ก
- เมื่อแก้ตาม Reviewer ให้สร้าง commit ใหม่ เช่น `Address reviewer reproducibility request`
- อย่าแก้ย้อนหลังหรือ force-push tag ที่ Reviewer ได้รับไปแล้ว
- บันทึก commit hash ที่ใช้ตอบ Reviewer และใส่ URL แบบ tag/release ไม่ใช้ URL ที่ชี้เฉพาะ branch ซึ่งเปลี่ยนได้

```powershell
git tag -a v1.0.0-paper-information-4512788 -m "Frozen paper reproducibility release"
git push origin main --tags
```

## 4. อัปโหลด model เป็น GitHub Release assets

หลังคัดลอก checkpoint ทั้ง 72 ชุดตาม manifest ให้ดับเบิลคลิก
`RUN_PREPARE_MODEL_RELEASE.bat` หรือใช้คำสั่ง:

```powershell
python tools\build_model_archives.py --checkpoint-root checkpoints --output-dir release_assets
python tools\release_preflight.py --mode release
```

เมื่อขึ้น `PASS: release assets are ready for GitHub upload.` ให้สร้าง Release จาก tag
ข้างต้น แล้วอัปโหลด ZIP 8 ไฟล์ใน `release_assets/` พร้อม
`release_assets.generated.json` และ `models/checkpoint_manifest.csv`

หลังอัปโหลด ให้รันคำสั่งเดิมอีกครั้งโดยใส่ URL ของ Release เพื่อเปลี่ยนสถานะ
manifest เป็น `PUBLISHED` แล้วตรวจขั้นสุดท้าย:

```powershell
python tools\build_model_archives.py `
  --checkpoint-root checkpoints `
  --output-dir release_assets `
  --release-url "https://github.com/YOUR_USERNAME/rcvv-pose/releases/tag/v1.0.0-paper-information-4512788"
python tools\release_preflight.py --mode published
```

จากนั้น commit `models/release_manifest.json` และออก patch tag ใหม่หากจำเป็น เช่น
`v1.0.1-paper-information-4512788` ห้ามแทนที่ asset เดิมแบบเงียบ ๆ

## 5. เปิด Public หลังตรวจร่วมกัน

ก่อนเปลี่ยนจาก Private เป็น Public:

- ให้ผู้เขียนร่วมตรวจชื่อบทความ รายชื่อผู้เขียน และ interpretation boundary
- ตรวจ license ของ source code และ detector weights
- Clone ลงโฟลเดอร์ใหม่หรือเครื่องใหม่ แล้วทดสอบตาม README ตั้งแต่ต้น
- ตรวจว่า `pip install` และคำสั่ง evaluate/train ทำงานโดยไม่พึ่ง absolute path
- ตรวจ Release asset ทุกไฟล์ด้วย SHA-256

## 6. ข้อความที่ใช้ตอบ Reviewer ภายหลัง

เมื่อ repository สมบูรณ์แล้วจึงแจ้งว่า:

> The source code, locked configurations, model checkpoints, and reviewer evidence supporting the experiments are available in the versioned reproducibility release at [URL]. The MPI-INF-3DHP dataset is not redistributed and must be obtained from the original provider under its applicable access conditions.

อย่าใช้ข้อความนี้ก่อนที่ clone-and-run test จะผ่านจริง
