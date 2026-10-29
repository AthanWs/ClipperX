# Curso: Automatización de Clips con IA Local

## Overview
- Modalidad: Video + práctica guiada
- Duración estimada: 12 módulos (≈ 30 h)
- Objetivo: Construir pipeline local en Windows para generar clips verticales desde streams largos

## Módulos
1. **Introducción y Objetivos**
   - Motivación, limitaciones hardware
   - Visión general del flujo (audio → texto → análisis → clips)

2. **Preparación del Entorno Windows**
   - Instalación drivers, CUDA, FFmpeg, Conda
   - Verificación con `torch.cuda`

3. **Gestión de Proyectos y Datos**
   - Estructura de carpetas, versiones, almacenamiento
   - Buenas prácticas con archivos grandes

4. **Preprocesamiento de Audio y Silencios**
   - Normalización con FFmpeg
   - Uso de Auto-Editor para limpiar pausas

5. **Transcripción Masiva (faster-whisper)**
   - Segmentación por lotes
   - Integración con GPU (INT8/FP16)

6. **Alineación y Diarización (WhisperX)**
   - Palabras con timestamps
   - Identificación de locutores (pyannote/Resemblyzer)

7. **LLMs Locales y Prompts Especializados**
   - Instalación de Ollama / llama.cpp
   - Diseño de prompts para clasificación y resumen

8. **Segmentación Temática y Scoring Textual**
   - Ventanas dinámicas
   - Generación de títulos, CTAs, keywords

9. **Escenas y Validación Visual**
   - PySceneDetect + CLIP/SigLIP
   - Detección de incoherencias texto-imagen

10. **Fusión Multimodal y Selección de Clips**
    - Diseño del score final (texto, visual, audio, diarización)
    - Selección top-N con reglas personalizadas

11. **Edición Automática y Reencuadre Vertical**
    - Tracking con YOLOv8
    - Subtítulos SRT, overlays con drawtext

12. **Orquestación Completa y Deploy Local**
    - Construcción de GUI (Streamlit)
    - Gestión de recursos, logs, empaquetado

## Proyectos / Prácticas
- **Mini-Proyecto 1**: Transcript + diarización de video 20 min
- **Mini-Proyecto 2**: Ranking textual y visual de 10 ventanas
- **Proyecto Final**: Pipeline end-to-end sobre stream de 2 h con 3 clips generados

## Evaluación
- Checklists por módulo
- Rubricas para proyecto final (calidad transcript, coherencia clips, UX)

## Recursos Complementarios
- Repositorios de referencia (FunClip, Auto-Editor, ShortGPT)
- Documentos de troubleshooting (drivers, VRAM)
- Plantillas de prompts y scripts base


