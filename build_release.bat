@echo off
setlocal
cd /d "%~dp0"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

if exist dist\ArmorSim.exe del /q dist\ArmorSim.exe
if exist build rmdir /s /q build

python -m PyInstaller --noconfirm --clean ArmorSim.spec
if errorlevel 1 exit /b 1

if not exist dist\ArmorSim.exe (
  echo Build failed: dist\ArmorSim.exe not found
  exit /b 1
)

echo.
echo Built: %cd%\dist\ArmorSim.exe
echo Upload that file as a GitHub Release asset.
endlocal
