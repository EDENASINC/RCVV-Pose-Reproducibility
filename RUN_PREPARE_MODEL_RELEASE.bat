@echo off
setlocal
cd /d "%~dp0"

echo Building eight checkpoint archives and generating the release manifest...
python tools\build_model_archives.py --checkpoint-root checkpoints --output-dir release_assets
if errorlevel 1 goto :failed

echo.
echo Running release preflight...
python tools\release_preflight.py --mode release
if errorlevel 1 goto :failed

echo.
echo PASS: Release assets are ready in release_assets\
echo Next: upload all eight ZIP files plus release_assets.generated.json.
exit /b 0

:failed
echo.
echo FAILED: Review the error above. No GitHub upload should be performed yet.
exit /b 1
