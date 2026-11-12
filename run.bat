@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m roadforge.main
) else (
  py -3.12 -m roadforge.main
)

if errorlevel 1 (
  echo.
  echo RoadForge stopped with an error. See TROUBLESHOOTING.md.
  pause
)
