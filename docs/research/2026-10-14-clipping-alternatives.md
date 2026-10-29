# Alternativas externas frente a la visión de ClipperX

Revisión acotada de fuentes originales, 14 de octubre de 2026. No se instalaron
estos programas ni se ejecutaron sus pipelines. Las funciones documentadas no
son pruebas de calidad real sobre los directos de ATHANW. Exa no estaba disponible;
se consultaron repositorios oficiales directamente. Context7 no identificó
Chopify/OpenShorts ni el editor oficial de Diffusion Studio; sí ofreció SupoClip.

## Decisión

Mantener ClipperX como coordinador narrativo y visual, con FFmpeg como ejecutor.
No añadir un segundo pipeline de clipping ni cambiar el editor aprobado sólo
porque tenga MCP. Una conexión MCP expone controles; no demuestra comprensión
correcta de una reacción, identidad del streamer o coherencia de un montaje.

La ventaja pendiente de ClipperX es resolver y verificar conjuntamente: contexto,
rangos no contiguos, identidad, cambios visuales y estilo de los clips anteriores.
Eso sigue en desarrollo; no se afirma que hoy supere a las alternativas en un
benchmark ni que garantice viralidad. La prueba comparativa debe usar los mismos
fragmentos y medir contexto conservado, encuadre, identidad, subtítulos, tiempo
y consumo; las estrellas y los reclamos de marketing no sustituyen esa prueba.

| Opción | Hecho verificado y ventaja posible | Decisión para este proyecto |
| --- | --- | --- |
| Chopify | El código selecciona ventanas de transcripción con heurísticas o un LLM local, y exporta intervalos start/end. | Útil como baseline barato; no demuestra la dirección multimodal ni montaje de varios rangos que buscamos. |
| OpenShorts | El README ofrece MCP/API; permite un endpoint compatible para selección textual, pero conserva Gemini para layout y vídeo silencioso. | Posible baseline o referencia de operación. Integrarlo completo duplica ingestión, tareas y render; no es un reemplazo íntegro por Luna sin adaptación. |
| SupoClip | Su MCP es un cliente del backend REST, alojado por defecto y configurable a self-host. | Controlar otro clipper no añade por sí mismo la interpretación que falta. No conectarlo ahora. |
| Diffusion Studio | Editor programable con runtime, JSX, CLI y MCP; la interfaz usa Solid. | Interesante para entregar un proyecto editable más adelante. No hay ventaja medida que justifique sustituir el compilador determinista aprobado. |
| FilmCraft | CLI/MCP documentados; véase la investigación específica enlazada abajo. | Mismo papel opcional de revisión o entrega editable, no requisito para exportar MP4 automático. |

Chopify emplea listas de indicios léxicos mayoritariamente en inglés en el modo
heurístico. Por eso no asumiría calidad equivalente sobre español coloquial sin
medirla. El modo LLM es diferente y tendría que compararse por separado.
[Código de selección](https://github.com/mehbul/chopify/blob/main/score_clips.py),
[render y reencuadre](https://github.com/mehbul/chopify/blob/main/render_clips.py).

La descripción inicial de OpenShorts como análisis exclusivamente textual quedó
incompleta: el código actual también contiene rutas visuales para material sin
habla y decisiones por imagen. No equivale a que todas sus rutas funcionen con
cualquier servidor compatible. El núcleo declara MIT, con una excepción comercial
en cloud/. [README oficial](https://github.com/mutonby/openshorts),
[rutas y prompts visuales](https://github.com/mutonby/openshorts/blob/main/gemini_worker.py).

SupoClip declara AGPL-3.0-or-later en su documentación MCP; cualquier reutilización
de código requiere revisar la licencia aplicable, no sólo instalar un conector.
[MCP oficial](https://github.com/FujiwaraChoki/supoclip/blob/main/mcp/README.md).

Diffusion Studio declara MPL-2.0, excepto ciertos activos de marca. Su CLI actual
se llama diffusion y conserva dapi como alias. Las notas de versión documentan
ejecución de la aplicación en segundo plano; eso no es prueba de un despliegue
completo sin escritorio. [Repositorio](https://github.com/diffusionstudio/editor),
[versiones](https://github.com/diffusionstudio/editor/releases).

AutoClip no identifica por sí solo un repositorio: existen proyectos y forks
distintos con ese nombre. Uno que coincide con la investigación, artbyjazi/autoclip,
declara selección por transcripción y reencuadre por hablante, y reconoce que su
calidad de reencuadre todavía no pasó un conjunto fijo de referencia. No tomar
su estado E2E declarado como validación sobre nuestros streams.
[Repositorio identificado](https://github.com/artbyjazi/autoclip).

## Qué aprovechar sin ampliar el sistema

- Comparar contra un baseline de selección por transcripción, además de medir
  mejoras visuales; un JSON correcto no demuestra que se eligió un buen momento.
- Reutilizar artefactos por etapa y reintentos locales, ya previstos en ClipperX.
- Verificar inicio/contexto/desenlace y referencias de identidad explícitamente.
- Tratar funciones de publicación social, B-roll generado y doblaje como otra
  iniciativa: no forman parte del alcance autorizado de este avance.

Los editores tradicionales de la lista no fueron instalados ni comparados aquí.
Un editor manual puede facilitar retoques, pero no resuelve por sí mismo la
selección autónoma y contextual del directo. No hay evidencia para añadirlo como
dependencia obligatoria ahora.

Investigación complementaria: [ArtCraft y FilmCraft](2026-10-14-artcraft-filmcraft.md).
