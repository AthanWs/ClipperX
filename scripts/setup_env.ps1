param([string]$EnvPath = ".venv-whisperx")

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path "$EnvPath\Scripts\python.exe")) {
    Write-Host "[ClipperX] Creando entorno $EnvPath" -ForegroundColor Cyan
    python -m venv $EnvPath
}

$Python = Join-Path $EnvPath "Scripts\python.exe"

Write-Host "[ClipperX] Actualizando pip/setuptools/wheel" -ForegroundColor Cyan
& $Python -m pip install --upgrade pip "setuptools>=68,<81" wheel

Write-Host "[ClipperX] Instalando dependencias principales" -ForegroundColor Cyan
& $Python -m pip install `
    --index-url https://download.pytorch.org/whl/cu124 `
    --extra-index-url https://pypi.org/simple `
    -e ".[dev]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Fuerza los wheels CUDA al final para impedir que PyPI los sustituya por CPU.
& $Python -m pip install --force-reinstall --no-deps `
    --index-url https://download.pytorch.org/whl/cu124 `
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m clipperx.cli doctor
if ($LASTEXITCODE -ne 0) {
    Write-Error "El entorno se instaló, pero el diagnóstico falló."
    exit 1
}

Write-Host "[ClipperX] Instalación completada." -ForegroundColor Green

