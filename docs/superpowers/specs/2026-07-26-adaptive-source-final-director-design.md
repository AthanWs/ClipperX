# Director visual temporal y aprendizaje de estilo sin pares

Fecha: 2026-07-26
Estado: aprobado por el usuario

## Contexto

ClipperX ya puede descubrir candidatos, auditar límites, conservar rangos no
contiguos, planificar clips globalmente y renderizar MP4 verticales. El problema
pendiente no es únicamente escoger un layout: debe reconstruir una edición
coherente a partir de VOD horizontales largos que cambian entre cámara completa,
gameplay, navegador, video reaccionado, ausencia de webcam y varias personas.

Existen dos colecciones distintas:

- 301 clips editados que describen el lenguaje visual deseado.
- 14 VOD representativos, 47,72 horas en total, que describen las entradas que el
  sistema deberá resolver.

No hay evidencia de correspondencia entre ambas colecciones. El diseño no debe
inventar pares fuente-resultado ni entrenar una transformación supervisada con
datos falsos.

## Objetivos

1. Detectar temporalmente qué fuentes visuales existen y cuál debe dominar.
2. Mantener la identidad del streamer aunque aparezcan otras caras más grandes.
3. Adaptar crop, posición, escala, fondo, subtítulos, título y marca por tramo.
4. Recuperar ejemplos del estilo histórico apropiados para la escena actual sin
   copiarlos como plantillas rígidas.
5. Traducir decisiones semánticas a un contrato JSON completo y validable.
6. Renderizar de forma reproducible con FFmpeg y revisar el MP4 resultante.
7. Funcionar incrementalmente sobre VOD de dos a seis horas sin enviar el video ni
   el transcript completo al modelo.

## No objetivos iniciales

- Fine-tuning con los clips históricos sin pares reales.
- Consultar un modelo VL a una cadencia fija durante todo el VOD.
- Añadir efectos aleatorios para aparentar dinamismo.
- Reemplazar la selección narrativa de clips por heurísticas visuales.
- Hacer tracking por frame cuando un crop estable por estado sea suficiente.

## Evidencia observada

Los VOD alternan cinco estados dentro de un mismo archivo: talking head, gameplay
con webcam, navegador o herramientas, contenido reaccionado y ausencia de cámara
útil. Los directos de *The Last of Us* mantienen una webcam pequeña y estable en
la esquina superior izquierda; `HOLA NO TENGO PC (1)` es casi todo cámara
completa; otros VOD cambian varias veces entre cámara, navegador y juegos.

Una prueba del detector actual mostró:

- webcam estable de gameplay: detección correcta;
- cámara completa: crop parcial incorrecto;
- tramo largo con varios layouts: región promediada inexistente;
- persona dentro del contenido sin webcam: falso positivo de webcam.

El corpus final, en cambio, privilegia cámara centrada y legible, títulos aislados,
subtítulos de alto contraste, marca reconocible y fondos negros o borrosos según la
información que se debe preservar. La webcam pequeña de la fuente normalmente se
amplía y recoloca en el resultado.

## Alternativas consideradas

### A. Plantillas fijas con detector de webcam

Ventaja: bajo costo y resultados predecibles.

Problema: falla al cambiar la cámara, confunde rostros del contenido, no sabe si
debe priorizar texto, juego o reacción y reproduce mal composiciones especiales.

### B. Modelo VL controlando directamente cada frame

Ventaja: flexibilidad máxima.

Problema: costo alto, inestabilidad temporal, crops que vibran y dificultad para
reproducir exactamente una decisión. No es necesario para una webcam estable.

### C. Director híbrido temporal con recuperación de estilo

Es el enfoque recomendado. El análisis local encuentra eventos y tracks; el VL
toma decisiones editoriales sobre ventanas concretas; un contrato estructurado
expresa el resultado; FFmpeg lo compila; una revisión posterior corrige errores.

## Arquitectura propuesta

### 1. Timeline multimodal de fuente

Un `SourceTimelineBuilder` produce estados reutilizables del VOD. Combina:

- cambios de escena e histograma;
- movimiento y diferencia estructural;
- detecciones y embeddings faciales;
- densidad de texto/OCR y regiones de interfaz;
- bordes, letterbox y rectángulos persistentes de webcam;
- métricas de voz, energía, emoción y silencios;
- transcript, speakers y límites narrativos existentes.

El muestreo base es barato y espaciado. Cuando cambia una señal, aparece una cara,
se mueve una región persistente o un candidato cruza el evento, se abre una
ventana densa alrededor del cambio. Así se obtiene precisión local sin analizar
cada segundo de seis horas con VL.

El timeline contiene estados explícitos como `full_camera`, `content_with_camera`,
`content_only`, `browser_text`, `reacted_video`, `multi_person` y `unknown`, con
confianza, evidencia y rango temporal.

### 2. Identidad y tracking del streamer

El sistema deja de usar la cara más grande como identidad. Se crea un perfil por
VOD a partir de tracks persistentes y segmentos de cámara completa. Los embeddings
faciales vinculan la misma persona entre layouts. La continuidad de voz y posición
solo aporta evidencia auxiliar: nunca asigna por sí sola una voz a una cara dentro
del video mezclado.

El recuadro de webcam se detecta por su rectángulo persistente, bordes y apariencia,
no únicamente expandiendo una cara con multiplicadores. La identidad y el recuadro
son objetos separados: puede existir streamer sin webcam superpuesta y webcam sin
una cara visible durante algunos frames.

Si la evidencia entra en conflicto, el director solicita frames adyacentes. No se
aplica silenciosamente el crop global ni se escoge otra cara por tamaño.

### 3. Biblioteca recuperable de estilo

Los clips históricos se convierten en `StyleCard`, no en pares de entrenamiento.
Cada tarjeta contiene:

- familia de escena;
- keyframes representativos;
- geometría aproximada de cámara y contenido;
- uso de negro, blur, cover o contain;
- posición de título, marca y subtítulos;
- ritmo de cambios, zooms y recursos especiales;
- fecha o época visual para priorizar la identidad reciente.
- calidad visual o aprobación del estilo, separada de la calidad narrativa del
  momento y de lo que se estaba diciendo.

Para cada tramo, se recuperan dos o tres tarjetas compatibles con su estado fuente
y función narrativa. El director recibe esos ejemplos como repertorio. Puede
apartarse de ellos cuando preservar el contenido lo exija y debe explicar por qué.

### 4. Director visual por clip

El director se ejecuta después de la composición narrativa y antes del render.
Recibe:

- rangos fuente aprobados, incluidos silencios visualmente relevantes;
- rol narrativo de cada rango;
- timeline local y tracks de identidad;
- frames de inicio, final, eventos y límites dudosos;
- métricas de audio y emoción;
- StyleCards recuperadas;
- zonas seguras de TikTok, Reels y Shorts.

Puede subdividir un `EditSegment` cuando cambia el estado visual sin alterar la
narrativa. También puede pedir otra ventana hacia atrás o adelante si el cambio
queda cerca del límite. El resultado no es un nombre de layout, sino una secuencia
de planos visuales explícitos.

### 5. Contrato `VisualCompositionPlan` v2

Cada `VisualShot` expresa como mínimo:

- rango fuente y rango de salida;
- fuente principal y razón editorial;
- capas con crop, posición, tamaño, anclaje y z-order;
- track de crop o keyframes cuando haya movimiento;
- región que debe permanecer visible;
- modo de relleno: `cover`, `contain_blur`, `black`, `solid` o `inherit`;
- zonas permitidas y prohibidas para subtítulos;
- título, marca y duración de cada overlay;
- zoom, paneo o punch-in con easing;
- transición de entrada y salida;
- ganancia, ducking, limitador y SFX opcional;
- evidencia y confianza de cada decisión.

`adaptive` deja de ser un alias que cae en el split por defecto. La adaptación
queda compilada en decisiones concretas. Un plan incompleto o geométricamente
inválido se rechaza; no se convierte silenciosamente en una plantilla arbitraria.

### 6. Compilador FFmpeg

El compilador es determinista y no decide la creatividad. Se encarga de:

- validar tiempos, capas, relaciones de aspecto y zonas seguras;
- interpolar tracks y suavizar keyframes;
- construir filtros de crop, scale, pad, overlay, blur y drawtext/ASS;
- preservar audio y sincronía al concatenar rangos no contiguos;
- aplicar normalización de voz, limitador de picos y ducking solicitado;
- exportar H.264/AAC 1080x1920, `yuv420p`, faststart y máximo 179 segundos;
- guardar el plan y métricas junto al MP4.

Las reglas deterministas protegen geometría, sincronía y límites de plataforma;
no sustituyen la decisión editorial del modelo.

### 7. Subtítulos, título y marca

Los subtítulos se planifican por plano, no con un `margin_v` global. El estilo base
reciente será amarillo pollo `#FFD51A`, negrita y borde negro, centrado ópticamente
y con pocas palabras por evento. La posición evita cara, HUD, OCR importante y
controles de plataforma.

El headline visual usa una banda segura superior, Impact o equivalente compacto,
y puede permanecer todo el clip o solo durante el hook según el StyleCard y la
carga visual. `platform_title` y `caption` siguen siendo metadata, no overlays.

La marca Kick usa el PNG aprobado, centrado en una zona reservada. No desplaza el
contenido a la izquierda para esquivar la interfaz de TikTok.

### 8. Revisión posterior al render

Una revisión local comprueba resolución, duración, audio, frames negros, cortes y
zonas seguras. Después, el VL inspecciona el inicio, final, cambios de layout,
keyframes y momentos de alta emoción del MP4 renderizado.

La revisión devuelve errores estructurados: cara cortada, identidad incorrecta,
texto tapado, contenido relevante perdido, hueco absurdo, cambio brusco o overlay
fuera de zona. El director emite un parche de plan y se permiten como máximo dos
rerenders por clip. La segunda corrección debe limitarse al plano fallido; si aún
falla, el resultado queda marcado para revisión en vez de ocultar el problema.

### 9. Interfaz

La interfaz debe exponer un preset `Athan adaptativo` y mostrar:

- storyboard con planos y rangos fuente;
- fuente dominante y StyleCards usadas;
- crop de cámara e identidad seleccionada;
- título, marca, subtítulos y zonas seguras;
- advertencias de baja confianza;
- acciones para aprobar, regenerar un plano o corregir identidad.

No se obliga al usuario a editar manualmente cada clip. Las correcciones se guardan
como pares fuente-resultado reales para futuras regresiones.

## Integración con el flujo narrativo existente

`Discovery` sigue encontrando momentos. `Boundary` sigue auditando contexto antes,
durante gaps y después. `Composition` sigue juntando fragmentos relacionados y
separando temas. El nuevo director visual consume el resultado de Composition y
puede subdividirlo solo por cambios visuales.

El transcript largo continúa procesándose por cursor y ventanas solapadas. El
modelo no mantiene una sesión infinita de Ollama ni recibe todo el VOD. Cada llamada
es autosuficiente, con contexto local, resumen acumulado y referencias por ID.

## Rendimiento y modelos

El análisis local y sus caches se calculan una vez por VOD. Las llamadas VL se
limitan a candidatos, eventos y revisiones, no a las 47,72 horas completas. El
contrato es independiente del proveedor: puede probarse con Qwen VL local o con un
modelo ligero de CLIProxyAPI sin cambiar el renderer.

Esta fase de diseño y auditoría no ejecuta ningún LLM. La selección de modelo se
validará con un conjunto pequeño antes de procesar un VOD completo.

## Validación

El corpus de entrada se divide en escenarios, no en archivos completos:

1. webcam estable sobre gameplay oscuro;
2. cámara completa;
3. transición cámara completa a navegador;
4. transición navegador a gameplay;
5. contenido con rostros ajenos;
6. segunda persona en el plano;
7. ausencia de webcam;
8. video vertical reaccionado;
9. navegador con texto pequeño;
10. movimiento o desaparición de la cámara.

Para cada escenario se guardan entrada, timeline, plan, MP4 y reporte de QA. Los
primeros renders aprobados por el usuario forman el corpus de pares reales.

## Criterios de aceptación

- El streamer no cambia de identidad por aparecer una cara ajena más grande.
- Ningún plano aprobado corta cabeza, boca o información marcada como esencial.
- Un cambio real de cámara o modo visual subdivide el plano cerca del evento.
- Gameplay, navegador y video reaccionado conservan su información crítica.
- Blur y negro se eligen por evidencia, no por layout fijo.
- Título, marca y subtítulos permanecen centrados y dentro de zonas seguras.
- Los rangos no contiguos conservan sincronía audiovisual y continuidad narrativa.
- El plan v2 valida y serializa sin pérdidas; no hay fallback visual silencioso.
- El resultado final es un MP4 1080x1920 reproducible y menor o igual a 179 s.
- El QA posterior puede señalar y corregir un plano sin regenerar todo el VOD.

## Decisión recomendada

Implementar primero el director híbrido con biblioteca de StyleCards, identidad
del streamer, timeline de estados y contrato visual v2. No hacer fine-tuning aún.
Los clips históricos guían la estética; los VOD actuales validan robustez; los
renders aprobados empiezan a crear los pares reales que hoy no existen.
