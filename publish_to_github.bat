@echo off
setlocal
cd /d "%~dp0"

where gh >nul 2>nul
if errorlevel 1 (
  echo GitHub CLI is not installed.
  echo Install it from https://cli.github.com/ and run gh auth login.
  exit /b 1
)

gh auth status
if errorlevel 1 (
  echo Authenticate first by running: gh auth login
  exit /b 1
)

gh repo create ArvinNotDev/RoadForge --public --description "A CARLA-to-ONNX road-scene semantic perception workbench: RGB data collection, YOLOPv2 pseudo-labeling, YOLO26 fine-tuning, dataset auditing, evaluation, and deployment." --source . --remote origin --push
if errorlevel 1 (
  echo.
  echo Publishing failed. Check whether ArvinNotDev/RoadForge already exists.
  echo If it exists, see docs\PUBLISHING.md for the push-only command.
  exit /b 1
)

echo.
echo RoadForge was created and pushed successfully.
echo https://github.com/ArvinNotDev/RoadForge
