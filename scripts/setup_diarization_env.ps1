param([string]$EnvPath = ".venv-diarization")

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path "$EnvPath\Scripts\python.exe")) {
    Write-Host "[ClipperX] Creando runtime aislado $EnvPath" -ForegroundColor Cyan
    python -m venv $EnvPath
}

$Python = Join-Path $EnvPath "Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[ClipperX] Instalando Torch 2.8 CUDA 12.8" -ForegroundColor Cyan
& $Python -m pip install `
    --index-url https://download.pytorch.org/whl/cu128 `
    torch==2.8.0 torchaudio==2.8.0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[ClipperX] Instalando Community-1 runtime" -ForegroundColor Cyan
& $Python -m pip install `
    --extra-index-url https://download.pytorch.org/whl/cu128 `
    pyannote.audio==4.0.7 torchcodec==0.7.0 "soundfile>=0.12"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m clipperx.diarization_worker doctor
if ($LASTEXITCODE -ne 0) {
    Write-Error "El runtime Community-1 no superó el diagnóstico."
    exit 1
}

Write-Host "[ClipperX] Runtime Community-1 listo." -ForegroundColor Green
