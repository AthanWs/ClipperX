# Roadmap Inicial: Sistema Local de Clips Automáticos

## 0. Contexto Hardware
- GPU: RTX 3060 (12 GB VRAM)
- CPU: Ryzen 5 3600 (6C/12T)
- RAM: 32 GB DDR4
- Almacenamiento recomendado: SSD NVMe para `cache/` y `data/`

## 1. Estructura de Proyecto
- `docs/`: documentación (roadmaps, specs, guías del curso)
- `src/`: librerías Python y módulos del pipeline
- `scripts/`: CLI y utilidades PowerShell/Python
- `data/`: videos de entrada, audios normalizados, resultados intermedios
- `models/`: pesos descargados (ASR, LLM, VLM, YOLO)

## 2. Fases Técnicas
1. **Setup base (Semana 1)**
   - Instalar CUDA Toolkit 12.x, cuDNN, FFmpeg, Git, Python 3.10
   - Crear entorno Conda `clipperx` (PyTorch + torchaudio + CUDA 12)
   - Descargar utilidades (`ffmpeg`, `aria2c`, `jq`, `gsutil opcional`)
   - Probar GPU con `torch.cuda.is_available()`

2. **Pipeline ASR (Semanas 2-3)**
   - Implementar script `scripts/preprocess_audio.py`: normalizar audio a 16 kHz mono
   - Integrar `faster-whisper` (large-v3 int8) para transcripción por lotes (45 min)
   - Añadir `WhisperX` para alineación palabra-clipping + diarización `pyannote`
   - Guardar `transcripts/raw/*.jsonl` y `transcripts/diarized/*.jsonl`

3. **Segmentación & Scoring Textual (Semanas 4-5)**
   - Dividir transcript por ventanas dinámicas (60-90 s, tolerancia ±15 s)
   - Servir `Qwen2.5-7B-Instruct` vía Ollama/llama.cpp (4-bit) en localhost
   - Diseñar prompts para clasificar ventanas (`tema`, `relevancia`, `clip_hook`, `CTA`)
   - Persistir resultados en `analysis/text/*.json`

4. **Validación Visual + Audio (Semanas 6-7)**
   - Detectar escenas con `PySceneDetect` (`ContentDetector`)
   - Extraer frames clave (`ffmpeg select='gt(scene,0.4)'` + muestreo uniforme)
   - Calcular similitud texto↔imagen (SigLIP/OpenCLIP) y métricas de energía audio (`librosa`)
   - Fundir señales en score final (`score = 0.4T + 0.25V + 0.2A + 0.15D`)

5. **Edición Automática (Semanas 8-9)**
   - Tracking sujeto con `YOLOv8n/s` (Ultralytics) → generar `tracks/*.csv`
   - Filtros `ffmpeg` para crop dinámico + `scale=1080:1920`
   - Generar subtítulos SRT y overlays `drawtext`
   - Exportar clips finales en `exports/*.mp4`

6. **Orquestación + GUI (Semanas 10-12)**
   - Construir orquestador Python (`src/pipeline/orchestrator.py`)
   - Implementar GUI local (Streamlit) para configurar pesos, seleccionar resultados
   - Añadir reportes (`reports/*.html`) con thumbnails + metadatos
   - Preparar empaquetado (installer o instrucciones reproducibles)

## 3. Plan del Curso
1. **Módulo 1**: Introducción, objetivos, overview arquitectura
2. **Módulo 2**: Setup Windows, drivers, herramientas
3. **Módulo 3**: Limpieza audio y silencios (Auto-Editor)
4. **Módulo 4**: Transcripción larga con faster-whisper
5. **Módulo 5**: Alineación y diarización con WhisperX
6. **Módulo 6**: LLMs locales (Ollama + prompts)
7. **Módulo 7**: Segmentación temática y scoring textual
8. **Módulo 8**: Escenas, frames y embeddings CLIP/SigLIP
9. **Módulo 9**: Fusión multimodal y ranking
10. **Módulo 10**: Auto-reframe, subtítulos, overlays
11. **Módulo 11**: Orquestador, manejo de recursos, batch
12. **Módulo 12**: Proyecto final + checklist de QA

## 4. Tareas Inmediatas
- [ ] Documentar requisitos de software con links de descarga
- [ ] Crear script `scripts/setup_env.ps1` (conda env, paquetes base)
- [ ] Implementar `scripts/preprocess_audio.py`
- [ ] Definir formato estándar de JSON para transcripts y análisis



