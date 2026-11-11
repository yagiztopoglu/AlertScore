@echo off
rem SKAB (endustriyel pompa/vana) verisini indirir. Internet gerekir. Veri GPL-3.0 lisanslidir, GitHub reponuza eklemeyin.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Once setup.bat dosyasini calistirin.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\download_skab.py %*
pause
