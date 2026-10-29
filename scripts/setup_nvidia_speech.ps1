param([switch]$DownloadModels)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskRuntime = Join-Path $taskRoot '.venv-nvidia-speech'
$taskPython = Join-Path $taskRuntime 'Scripts/python.exe'
$taskLock = Join-Path $taskRoot '.nvidia-speech-setup.lock'
$taskLockStream = [System.IO.File]::Open($taskLock, 'OpenOrCreate', 'ReadWrite', 'None')
try {
    if (-not (Test-Path -LiteralPath $taskPython)) {
        & (Join-Path $taskRoot '.venv-whisperx/Scripts/python.exe') -m venv $taskRuntime
        if ($LASTEXITCODE -ne 0) { throw 'Runtime creation failed' }
    }
    & $taskPython -m pip install --no-compile 'torch==2.6.0' 'transformers==5.20.2' 'accelerate==1.11.0' 'soundfile==0.13.1' 'numpy==2.2.6' 'librosa==0.11.0' --extra-index-url https://download.pytorch.org/whl/cu124 --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw 'Pinned NVIDIA speech dependencies failed' }
    & $taskPython -c "import torch, transformers; from transformers import ParakeetFeatureExtractor; assert torch.cuda.is_available(); ParakeetFeatureExtractor(); assert hasattr(transformers, 'AutoModelForTDT'); assert hasattr(transformers, 'AutoModelForAudioFrameClassification'); print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA speech runtime CUDA preflight failed' }
    if ($DownloadModels) {
        & $taskPython (Join-Path $PSScriptRoot 'download_nvidia_speech.py')
        if ($LASTEXITCODE -ne 0) { throw 'Pinned speech model download failed' }
    }
} finally {
    $taskLockStream.Dispose()
}
