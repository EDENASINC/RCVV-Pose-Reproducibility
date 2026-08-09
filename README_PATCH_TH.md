# ขั้นสร้าง GitHub Release Assets

ตัวสร้าง Release จัดทำไฟล์ ZIP ทั้งหมด 9 ไฟล์และอัปเดต `models\release_manifest.json` อัตโนมัติ

## ไฟล์ต้นทางที่ต้องมี

- checkpoint 72 ชุดภายใต้ `checkpoints\` ตาม `models\checkpoint_manifest.csv`
- calibration files ภายใต้ `artifacts\calibration\`
- `local_release_inputs\learned_artifacts\phase9c_a_residual_bank.npz`

## สร้างและตรวจ

```powershell
python tools\build_model_archives.py `
  --checkpoint-root checkpoints `
  --output-dir release_assets

python tools\release_preflight.py --mode release
```

ต้องได้:

```text
PASS: release assets are ready for GitHub upload.
```

อัปโหลด ZIP 9 ไฟล์ใน `release_assets\` พร้อม `release_assets.generated.json` และ `models\checkpoint_manifest.csv` ไปยัง GitHub Release

## ยืนยันหลังเผยแพร่

```powershell
python tools\build_model_archives.py `
  --checkpoint-root checkpoints `
  --output-dir release_assets `
  --release-url "https://github.com/EDENASINC/RCVV-Pose-Reproducibility/releases/tag/v1.0.0-paper-information-4512788"

python tools\release_preflight.py --mode published
```

ผลขั้นสุดท้ายต้องเป็น `PASS: published release contract is complete.`
