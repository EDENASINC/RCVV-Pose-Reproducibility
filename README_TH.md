# คู่มือ RCVV-Pose Reproducibility Package

Repository นี้รวบรวม source code, locked config, reviewer evidence, run provenance และ manifest ของ checkpoint 72 ชุดสำหรับบทความ `information-4512788` แล้ว ส่วน checkpoint และ learned calibration artifacts จะแจกจ่ายเป็น GitHub Release assets เพื่อไม่ให้ binary ขนาดใหญ่เข้า Git history

## ตรวจแพ็กเกจแบบไม่ใช้ Dataset

```powershell
python tools\validate_evidence.py
python -m unittest discover -s tests -v
python tools\release_preflight.py --mode evidence
```

## สร้างไฟล์สำหรับ GitHub Release

วาง checkpoint 72 ชุดตาม `models\checkpoint_manifest.csv` และตรวจว่ามี:

- `artifacts\calibration\` สำหรับ calibration files
- `local_release_inputs\learned_artifacts\phase9c_a_residual_bank.npz`

จากนั้นรัน:

```powershell
python tools\build_model_archives.py `
  --checkpoint-root checkpoints `
  --output-dir release_assets

python tools\release_preflight.py --mode release
```

ผลที่ถูกต้องคือ:

```text
PASS: release assets are ready for GitHub upload.
```

ระบบต้องสร้าง ZIP 9 ไฟล์ ได้แก่ checkpoint 8 กลุ่ม และ `learned_calibration_artifacts.zip` พร้อมบันทึกขนาดและ SHA-256 จริงลงใน `models\release_manifest.json`

หลังสร้าง GitHub Release แล้ว ให้รัน:

```powershell
python tools\build_model_archives.py `
  --checkpoint-root checkpoints `
  --output-dir release_assets `
  --release-url "https://github.com/EDENASINC/RCVV-Pose-Reproducibility/releases/tag/v1.0.0-paper-information-4512788"

python tools\release_preflight.py --mode published
```

## สิ่งที่ไม่แจกจ่าย

- MPI-INF-3DHP dataset, RGB, video และ annotations
- detector weights ของบุคคลที่สาม
- cache และ derived Official NPZ ที่อาจอยู่ภายใต้เงื่อนไขของ dataset
- path เฉพาะเครื่อง, token, API key และ credential

ผู้ใช้ต้องดาวน์โหลด MPI-INF-3DHP และ detector weights จากผู้ให้บริการต้นทางและปฏิบัติตามเงื่อนไขของแต่ละรายการ
