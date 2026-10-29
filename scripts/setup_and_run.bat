@echo off
setlocal

pushd %~dp0..

if not exist .venv-whisperx\Scripts\python.exe (
    echo Falta .venv-whisperx. Ejecuta scripts\setup_env.ps1.
    exit /b 1
)

call .venv-whisperx\Scripts\activate.bat
python -m pip install -e ".[dev]"

if "%~1"=="" (
    python -m clipperx.cli ui
) else (
    python -m clipperx.cli %*
)

popd
endlocal

