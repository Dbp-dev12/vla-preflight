@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create .venv and install .[train] first. See README.md.
  pause
  exit /b 1
)
if not exist "demo-output\learning-workbench\meta\info.json" (
  ".venv\Scripts\python.exe" -m vla_preflight learning-demo demo-output\learning-workbench
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m vla_preflight studio demo-output\learning-workbench --workspace workbench-output\learning --open --port 8766
pause
