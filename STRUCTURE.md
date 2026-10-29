# Estructura del Proyecto ClipperX

Mapa rápido de las carpetas del repositorio. Para instalar y usar ClipperX, consulta el [README](README.md).

## Directorios Principales

*   **`clipperx/`**: Código fuente principal de la aplicación (paquete Python).
    *   `config.py`: Configuración global; lee `.env` (ver [`.env.example`](.env.example)).
    *   `cli.py`: Punto de entrada de la línea de comandos (`python -m clipperx.cli --help`).
    *   `interface.py`: Interfaz de usuario de terminal (Textual).
    *   `tests/`: Suite de pytest (`python -m pytest clipperx/tests -q`).
*   **`scripts/`**: Instalación, descarga de modelos y pruebas de calificación.
    *   `setup_env.ps1`: Crea `.venv-whisperx` (Torch CUDA 12.4) e instala el paquete con extras `dev`.
    *   `setup_nvidia_speech.ps1`: Crea `.venv-nvidia-speech` para Parakeet/Nemotron; con `-DownloadModels` ejecuta `download_nvidia_speech.py`.
    *   `setup_diarization_env.ps1`: Crea `.venv-diarization` (opcional, backend pyannote Community-1).
    *   `download_nvidia_speech.py`: Descarga las versiones fijadas de Parakeet y Nemotron en `data/models/nvidia_speech/`.
    *   `setup_and_run.bat`: Reinstala el paquete en `.venv-whisperx` y abre la UI (o pasa sus argumentos al CLI).
    *   `benchmark_visual_models.py`, `qualify_*.py`, `render_visual_smoke.py`, `observe_manual_windows.py`, `validate_source_timeline.py`: Pruebas de calificación opcionales; no hacen nada salvo que se activen explícitamente.
*   **`tools/`**: Scripts de utilidad y experimentos de investigación cerrados (ver "Research notes" en el README).
    *   `check_deps.py`: Verificación de dependencias de CTranslate2/Transformers.
    *   `preprocess_audio.py`: Normaliza audio a WAV mono 16 kHz con FFmpeg.
*   **`assets/`**: Recursos versionados de render (imagen de marca, catálogo de SFX).
*   **`docs/`**: Documentación, investigación, especificaciones y planes de diseño.
*   **`data/`** *(local, ignorado por git)*: VODs, cachés, trabajos por VOD, modelos descargados y clips renderizados.
    *   `models/`: Modelos locales (`nvidia_speech/`, `face_identity/`, `style_embeddings/`).
*   **`logs/`** *(local, ignorado por git)*: Logs de ejecución.

## Ejecución

Para iniciar la aplicación, utiliza el archivo por lotes en la raíz:
`run_clipperx.bat`

Este script comprueba que exista `.venv-whisperx`, ejecuta `python -m clipperx.cli doctor` y, si el diagnóstico pasa, abre la interfaz con `python -m clipperx.cli ui`.
