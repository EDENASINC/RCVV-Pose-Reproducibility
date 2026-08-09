@echo off
setlocal
cd /d "%~dp0"

set "VV_PROJECT_ROOT=D:\Research\vv_pose_project"

echo Collecting locked RCVV-Pose artifacts from:
echo   %VV_PROJECT_ROOT%
echo Into:
echo   %CD%
echo.

python tools\collect_local_artifacts.py --project-root "%VV_PROJECT_ROOT%" --repo-root "%CD%"
set "VV_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%VV_EXIT_CODE%"=="0" (
  echo Collection finished with missing or conflicting files.
  echo Open collection_report\collected_files.csv for the exact list.
) else (
  echo Collection completed successfully.
  echo Review collection_report\COLLECTION_SUMMARY.json before Git commit.
)
echo.
pause
exit /b %VV_EXIT_CODE%
