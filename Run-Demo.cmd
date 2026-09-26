@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First install the project as explained in README.zh-CN.md or README.md.
  echo python -m venv .venv
  echo .venv\Scripts\python.exe -m pip install -e .
  pause
  exit /b 2
)
".venv\Scripts\python.exe" scripts\quick_demo.py --open
pause

