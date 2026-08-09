# แพตช์ขั้นสร้าง Model Release

แพตช์นี้แก้กรณี `release_preflight.py --mode release` แจ้งว่า
`published model release manifest without PLACEHOLDER` หลังรวบรวม checkpoint ครบแล้ว

## ติดตั้ง

แตก ZIP แล้วคัดลอกไฟล์ทั้งหมดไปวางทับที่:

`D:\Research\rcvv_pose_reproducibility_package`

โครงสร้าง `tools/` ต้องคงเดิม

## เตรียมไฟล์สำหรับ GitHub Release

ดับเบิลคลิก:

`RUN_PREPARE_MODEL_RELEASE.bat`

หรือรัน:

```powershell
python tools\build_model_archives.py --checkpoint-root checkpoints --output-dir release_assets
python tools\release_preflight.py --mode release
```

ผลที่ถูกต้องคือ:

`PASS: release assets are ready for GitHub upload.`

จากนั้นอัปโหลด ZIP ทั้ง 8 ไฟล์ใน `release_assets` พร้อม
`release_assets.generated.json` และ `models/checkpoint_manifest.csv` ไปยัง GitHub Release

## ยืนยันหลังเผยแพร่

แทน `YOUR_USERNAME` ด้วยชื่อบัญชีจริง แล้วรัน:

```powershell
python tools\build_model_archives.py `
  --checkpoint-root checkpoints `
  --output-dir release_assets `
  --release-url "https://github.com/YOUR_USERNAME/rcvv-pose/releases/tag/v1.0.0-paper-information-4512788"
python tools\release_preflight.py --mode published
```

ผลขั้นสุดท้ายต้องเป็น:

`PASS: published release contract is complete.`
