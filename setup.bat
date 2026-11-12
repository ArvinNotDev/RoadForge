@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo RoadForge dependencies are installed in .venv.
echo Run run.bat to start the application.
exit /b 0

:error
echo Setup failed. Review the error above and see TROUBLESHOOTING.md.
exit /b 1
