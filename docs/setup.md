# Setup Windows para ClipperX

> **Nota:** esta guía es histórica (Conda, Python 3.10, CUDA 12.1) y ya no coincide con los scripts del repositorio. Para instalar ClipperX usa la sección [Quickstart](../README.md#quickstart) del README, que se basa en `scripts\setup_env.ps1` (Python 3.12, `.venv-whisperx`, Torch CUDA 12.4) y `scripts\setup_nvidia_speech.ps1`. Los problemas habituales están en [Troubleshooting](../README.md#troubleshooting).

## 1. Prerrequisitos de Hardware
- GPU: NVIDIA RTX 3060 (12 GB VRAM)
- CPU: Ryzen 5 3600 (6 núcleos, 12 hilos)
- RAM: 32 GB
- Almacenamiento: SSD NVMe recomendado (≥1 TB)

## 2. Software a Instalar
1. **Drivers NVIDIA**
   - Descargar últimos controladores Studio/Game Ready
   - Habilitar CUDA 12.x compatible con RTX 30xx

2. **CUDA Toolkit 12.x + cuDNN**
   - Instalar Toolkit (paths por defecto)
   - Registrar variables de entorno (`CUDA_PATH`, `Path`)

3. **FFmpeg**
   - Descargar build estático (Gyan.dev)
   - Descomprimir en `C:\Tools\ffmpeg`
   - Añadir `C:\Tools\ffmpeg\bin` al `Path`

4. **Python 3.10 + Miniconda**
   - Instalar Miniconda (x64)
   - Configurar `conda init powershell`

5. **Git + Git LFS**
   - `winget install Git.Git`
   - `git lfs install`

6. **Herramientas opcionales**
   - `aria2` (descarga rápida de modelos)
   - `jq` (manipulación JSON)
   - `Make` (a través de `choco install make`)

## 3. Crear Entorno Conda
```powershell
conda create -n clipperx python=3.10 -y
conda activate clipperx
pip install -U pip wheel setuptools
```

## 4. Instalar Dependencias Base
```powershell
# PyTorch con CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Librerías esenciales
pip install numpy scipy librosa soundfile
pip install faster-whisper==1.0.0
pip install pyannote.audio==3.1.1
pip install pyscenedetect
pip install opencv-python ultralytics
pip install streamlit gradio
pip install rich typer pydantic[dotenv]
```

## 5. Modelos y Pesos
- Crear carpeta `<repo>\models`
- Descargar modelos recomendados:
  - `faster-whisper-large-v3` (CTranslate2)
  - `Qwen2.5-7B-Instruct` (GGUF 4-bit)
  - `SigLIP-so400m-patch14-384`
  - `YOLOv8n.pt`
- Utilizar `aria2c -x 8 -d models <URL>` para acelerar descargas grandes

## 6. Verificación Inicial
```powershell
conda activate clipperx
python - <<'PY'
import torch
print('CUDA disponible:', torch.cuda.is_available())
print('Device name:', torch.cuda.get_device_name(0))
PY

ffmpeg -hide_banner -version

python - <<'PY'
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
print('Modelo cargado OK')
PY
```

## 7. Estructura Recomendada
```
ClipperX/
  data/
    raw/
    processed/
  docs/
  models/
  scripts/
  src/
    pipeline/
    utils/
```

## 8. Próximos Pasos
- Crear script `scripts/setup_env.ps1` para automatizar instalación
- Añadir `requirements.txt` y `environment.yml`
- Probar transcripción de video de ejemplo (≤15 min)
- Documentar troubleshooting (drivers, VRAM, permisos)


