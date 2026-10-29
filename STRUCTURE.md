# Estructura del Proyecto ClipperX

La estructura del proyecto ha sido reorganizada para mayor claridad y funcionalidad desde la raíz.

## Directorios Principales

*   **`clipperx/`**: Contiene el código fuente principal de la aplicación (paquete Python).
    *   `config.py`: Configuración global.
    *   `cli.py`: Punto de entrada de la línea de comandos.
    *   `interface.py`: Lógica de la interfaz de usuario.
*   **`tools/`**: Scripts de utilidad para tareas específicas.
    *   `convert_whisper.py`: Script para convertir modelos de Hugging Face a formato CTranslate2 (requerido para WhisperX).
    *   `check_deps.py`: Verificación de dependencias.
    *   `preprocess_audio.py`: Utilidad para preprocesamiento.
*   **`scripts/`**: Scripts de configuración e instalación.
    *   `setup_env.ps1`: Script de PowerShell para crear el entorno virtual e instalar dependencias.
    *   `setup_and_run.bat`: Helper para configurar y ejecutar.
*   **`data/`**: Almacenamiento de datos, modelos y salidas.
    *   `models/`: Ubicación recomendada para modelos convertidos.
*   **`docs/`**: Documentación del proyecto.
*   **`logs/`**: Logs de ejecución.

## Ejecución

Para iniciar la aplicación, utiliza el archivo por lotes en la raíz:
`run_clipperx.bat`

Este script activa el entorno virtual y lanza el módulo `clipperx`.
