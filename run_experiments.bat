@echo off
rem Tum deneyleri (benchmark, gurultu, kayma, gecikme, esik) calistirip results\ klasorune CSV yazar.
rem Hizli deneme icin:  run_experiments.bat --quick
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Once setup.bat dosyasini calistirin.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\run_experiments.py %*
pause
