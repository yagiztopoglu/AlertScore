@echo off
rem AlertScore kurulum: sanal ortam + paketler + hizli test. Cift tiklayarak calistirin.
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 goto nopython

echo [1/3] Sanal ortam olusturuluyor (.venv)...
if not exist .venv python -m venv .venv
if errorlevel 1 goto fail
call .venv\Scripts\activate.bat

echo [2/3] Paketler kuruluyor. PyTorch buyuk, birkac dakika surebilir...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto fail

echo [3/3] Testler calistiriliyor...
python -m unittest discover -s tests
if errorlevel 1 echo UYARI: Bazi testler basarisiz oldu. Paketler kuruldu, uygulama yine de acilabilir. Test ciktisini bana gonderin.

echo.
echo KURULUM TAMAM. Uygulamayi baslatmak icin run_app.bat dosyasina cift tiklayin.
pause
exit /b 0

:nopython
echo HATA: Python bulunamadi. python.org adresinden Python 3.11 veya 3.12 kurun.
echo Kurulumda "Add Python to PATH" kutusunu isaretleyin, sonra bu dosyayi tekrar calistirin.
pause
exit /b 1

:fail
echo.
echo HATA olustu. Yukaridaki kirmizi/son satirlari kopyalayip bana gonderin.
pause
exit /b 1
