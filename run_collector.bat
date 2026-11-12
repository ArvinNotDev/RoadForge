@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m datacollector.main
) else (
  py -3.12 -m datacollector.main
)

if errorlevel 1 (
  echo.
  echo RoadForge Data Collector stopped with an error.
  echo Check the CARLA Python API setup and TROUBLESHOOTING.md.
  pause
)
