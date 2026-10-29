@echo off
setlocal

REM Se ejecuta desde la raíz del proyecto

set "CLIPPERX_ENV=.venv-whisperx"
if not exist "%CLIPPERX_ENV%\Scripts\python.exe" (
    echo No existe el entorno CUDA %CLIPPERX_ENV%.
    echo Ejecuta scripts\setup_env.ps1 primero.
    exit /b 1
)

call "%CLIPPERX_ENV%\Scripts\activate.bat"

python -m clipperx.cli doctor
if errorlevel 1 exit /b 1

echo Abriendo interfaz ClipperX...
python -m clipperx.cli ui

echo.
echo Presiona cualquier tecla para cerrar esta ventana...
pause >nul


endlocal

