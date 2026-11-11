@echo off
rem Daha fazla NAB serisi indirir (internet gerekir). Daha fazla seri = test bolumunde daha fazla anomali olayi = daha guvenilir sonuc.
rem   download_data.bat          : onerilen 7 ek seri
rem   download_data.bat --all    : etiketli tum NAB serileri (yaklasik 50 dosya)
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Once setup.bat dosyasini calistirin.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\download_nab.py %*
pause
