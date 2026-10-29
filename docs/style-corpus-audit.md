# Auditoría del corpus de edición personal

Fecha: 2026-07-26

## Alcance

Corpus principal:

`R:\Streaming\(2025-Adelante)\Clips`

Corpus de fuentes representativas:

`R:\Descargas\streams`


La auditoría separa deliberadamente la calidad del momento elegido de la calidad
de su edición. El objetivo es reconstruir la gramática visual y sonora que debe
poder reproducir ClipperX sobre cualquier fragmento aprobado por el selector.

No se ejecutó ningún LLM externo. Se inspeccionaron técnicamente los 301 archivos,
se generó una lámina visual de cada uno, se revisaron 32 clips estratificados en
cinco puntos temporales y se abrieron diez fotogramas representativos a resolución
completa. Además, se muestrearon 12 puntos uniformes de cada uno de los 14 VOD
representativos, para un total de 168 fotogramas fuente.

Los clips editados y los VOD no se tratan como pares del mismo material. Los
primeros enseñan el lenguaje visual deseado; los segundos representan la clase de
entrada real que el sistema deberá resolver en el futuro.

## Inventario

- 301 videos, 21,03 GB y 2,60 horas.
- 296 videos son 1080x1920; tres son 864x1536, uno 360x640 y uno 1080x1080.
- 192 videos están a 60 fps, 105 a 30 fps y cuatro a 29,97 fps.
- Duración mediana: 22,88 segundos.
- 71 clips duran menos de 15 segundos; 137 duran entre 15 y 29 segundos.
- 298 usan H.264 y tres HEVC. Los 301 tienen audio AAC.
- En una muestra audiovisual estratificada de 32 clips, el volumen medio varía
  aproximadamente entre -14,6 y -26,4 dBFS. La mediana es -21,5 dBFS y varios
  archivos alcanzan 0 dBFS. La edición evita silencios largos, pero no mantiene
  una normalización sonora consistente.

Una medición facial conservadora sobre las 301 láminas detectó una cara útil en
170 fotogramas:

- 118 estaban centradas en la zona superior.
- 36 estaban centradas en la zona media.
- 134 eran grandes y 28 dominaban el encuadre.
- Las esquinas fueron excepcionales.
- Seis fotogramas contenían más de una cara detectable.

Esto confirma que el estilo histórico privilegia una cámara centrada y visible,
no una webcam pequeña escondida permanentemente en una esquina.

## Corpus de VOD fuente

- 14 directos, 65,28 GB y 47,72 horas.
- Trece están en 1280x720 y uno en 1920x1080.
- Las duraciones van de unos 27 minutos a más de seis horas y media.
- Los mosaicos temporales están en
  `data/style_audit/source_overview`; cada hoja cubre todo un VOD con doce
  muestras uniformes.

La muestra incluye cinco estados de entrada que pueden coexistir dentro del mismo
directo:

1. cámara completa o talking head;
2. gameplay con webcam pequeña superpuesta;
3. navegador, Discord, tiendas o herramientas con y sin webcam;
4. videos, trailers o contenido vertical reaccionado;
5. ausencia temporal de una cámara útil.

Los VOD de *The Last of Us* mantienen una webcam pequeña en la esquina superior
izquierda durante largos tramos oscuros de gameplay. Los directos variados cambian
entre cámara completa, navegador y varios juegos. `HOLA NO TENGO PC (1)` es casi
todo cámara completa. En otros directos una segunda persona entra físicamente al
plano, se reproduce contenido con rostros más grandes que el streamer o desaparece
la webcam.

Esto confirma cuatro requisitos:

- la identidad del streamer no puede deducirse de la cara más grande;
- un crop de cámara calculado una sola vez para todo el VOD es inválido;
- el tipo de escena debe detectarse temporalmente, aunque no haya silencio ni
  cambio de tema en el transcript;
- el contenido principal puede ser juego, navegador, video reaccionado, texto en
  pantalla o reacción facial, y esa prioridad puede cambiar dentro del clip.

Doce muestras uniformes sirven para caracterizar el corpus, pero no para localizar
con precisión todas las transiciones. En producción se necesita un detector local
de eventos que proponga cambios de estado y una inspección VL concentrada alrededor
de esos eventos, en vez de consultar al modelo a una cadencia fija.

Una prueba directa del detector actual confirmó el problema:

- acertó la webcam estable de *The Last of Us* en la esquina superior izquierda;
- trató una cámara completa como un crop parcial de 583x567 dentro de 1920x1080;
- promedió veinte minutos mixtos de `ÚLTIMO STREAM` en una región superior derecha inexistente;
- confundió una persona dentro del contenido con una webcam superior izquierda.

## Cómo relacionar fuente y resultado sin pares reales

No existe evidencia de que los 301 clips finales provengan de los 14 VOD
disponibles. Por tanto, no se debe intentar entrenar ni evaluar una transformación
supervisada fingiendo correspondencias.

La relación correcta es de dos corpus:

- **Corpus de estilo:** los clips finales aportan ejemplos recuperables de layout,
  jerarquía, fondos, tamaño de cámara, subtítulos, títulos, marca y ritmo.
- **Corpus de entrada:** los VOD aportan escenarios reales para pruebas de
  detección, segmentación temporal, identidad, reencuadre y robustez.

Durante la planificación, el sistema puede recuperar dos o tres clips de estilo
visualmente parecidos al tramo fuente —por ejemplo, gameplay con webcam, talking
head o reacción a navegador— y dárselos al director VL como ejemplos, no como
plantillas obligatorias. Los primeros pares fuente-resultado verdaderos se
obtendrán con los clips que genere ClipperX de estos VOD y que el usuario apruebe
o corrija. Esos pares sí podrán alimentar evaluaciones de regresión y, más
adelante, una adaptación del modelo si hubiera volumen suficiente.

## Gramática de edición observada

### 1. Contenido protagonista con cámara superpuesta

Se usa para gameplay, trailers, videos reaccionados o pantallas que necesitan
conservar detalle. El contenido ocupa casi todo el ancho y la cámara aparece como
un panel ancho centrado, normalmente abajo o junto al límite entre paneles.

Rasgos:

- Fondo negro cuando distraer con una copia borrosa perjudicaría el contenido.
- Título aislado en una banda superior.
- Cámara suficientemente grande para leer la reacción.
- Subtítulo cerca de la cámara, pero sin reemplazar el contenido reaccionado.
- La marca comparte la cabecera o una banda de separación.

### 2. Split clásico: cámara arriba y contenido abajo

Es la familia más frecuente en los clips históricos de gaming. La cámara ocupa
aproximadamente entre un tercio y la mitad superior; gameplay o contenido ocupa
el resto.

Rasgos:

- Ambos paneles están centrados.
- El subtítulo suele vivir cerca de la unión entre cámara y gameplay.
- El blur rellena únicamente el sobrante provocado por la relación de aspecto.
- No se usan bordes decorativos como requisito; el propio corte entre paneles
  construye la jerarquía.

Aunque esta familia representa bien el corpus, no debe ser el único default:
cuando el detalle del juego importa, poner la cámara arriba puede quitar demasiado
espacio útil al contenido.

### 3. Cámara dominante o talking head

Cuando el gesto, la historia o la reacción corporal son el contenido, la cámara
se amplía hasta ocupar la mayor parte del lienzo.

Rasgos:

- Encuadre centrado, generalmente de torso o primer plano.
- Copia borrosa del mismo plano arriba y abajo si el crop no llena 9:16.
- Subtítulos grandes de alto contraste sobre el tercio inferior.
- Marca separada del rostro.
- En algunos clips se cambia temporalmente desde split hacia un primer plano para
  enfatizar una reacción.

### 4. Reacción a contenido externo

La edición decide qué debe dominar en cada momento: video reaccionado, navegador,
chat, publicación o streamer. No existe una posición única correcta.

Rasgos:

- Picture-in-picture ancho y centrado, no necesariamente en una esquina.
- Alternancia entre contenido completo y reacción ampliada.
- Fondos negros para videos horizontales o interfaces con información crítica.
- Blur cuando el contenido central es estrecho y la periferia no contiene datos.
- Inserciones de capturas, videos verticales y pantallas compartidas.

### 5. Contenido sin cámara

En momentos donde la reacción facial no añade valor, algunos clips conservan solo
el juego, video o captura. El encuadre suele llenar el lienzo o usar una copia
borrosa para preservar el contenido completo.

### 6. Composición especial

Una minoría incorpora recursos que no caben en una plantilla básica:

- Tarjetas o títulos a pantalla completa.
- Memes y videos insertados.
- Recortes de la persona sobre fondos gráficos.
- Filtros de color, zooms abruptos y cambios a primer plano.
- Logos animados o intercambiados según la plataforma.
- Varias personas o fuentes verticales apiladas.

Estas piezas no deben convertirse en efectos aleatorios. Necesitan una intención
editorial explícita y evidencia visual de que refuerzan el momento.

## Texto y marca

Hay dos épocas claras:

- Estilo histórico: Impact blanco con borde negro, subtítulos centrados y marca
  Twitch/YouTube cercana al contenido.
- Estilo reciente: título grande separado del video, subtítulo amarillo intenso
  con borde negro y marca Kick `ATHANW`.

Los 17 clips de la carpeta `Subido` usan una identidad reciente mucho más
consistente. El título suele permanecer durante todo el clip, mientras el render
actual de ClipperX lo muestra solo durante los primeros cuatro segundos. Esa
diferencia debe ser una decisión de estilo explícita, no accidental.

## Qué merece conservarse

- La cámara se adapta al propósito del momento: dominante cuando la reacción
  importa y secundaria cuando el contenido visual importa.
- El centro óptico se mantiene; no se desplaza todo a la izquierda solo para
  esquivar botones de una plataforma.
- El blur es relleno, no decoración universal.
- El negro se usa para ordenar y separar contenido, no como hueco arbitrario.
- Los subtítulos tienen contraste fuerte y pocas palabras por evento.
- El título no tapa el gameplay ni la información principal.
- La marca es persistente y reconocible.
- Los cambios de escala tienen intención: reacción, punchline o cambio de fuente.

## Hábitos del corpus que no conviene copiar literalmente

- Algunos clips antiguos desperdician grandes zonas negras.
- El blur aparece en piezas donde un crop inteligente habría llenado mejor.
- El volumen varía demasiado y varios clips rozan 0 dBFS.
- Algunos subtítulos pisan cara, HUD o texto del contenido.
- Ciertas marcas y títulos ocupan más espacio del necesario.
- Las composiciones especiales no siguen una identidad única.

El corpus debe usarse como lenguaje visual, no como plantilla exacta ni como
estándar de calidad absoluto.

## Cobertura actual de ClipperX

| Necesidad | Estado actual | Veredicto |
|---|---|---|
| Exportar 1080x1920 H.264/AAC | Implementado | Cubierto |
| Conservar rangos no contiguos | Implementado | Cubierto |
| Elegir full content, full camera, split o PIP por segmento | Existe en contrato | Parcial |
| Adaptar geometría a resolución y aspecto de cámara | Implementado | Cubierto para un crop correcto |
| Detectar cámara por segmento | MediaPipe + estimación alrededor de cara | Parcial |
| Seguir una cámara que cambia dentro de un segmento | No existe tracking temporal real | Falta |
| Distinguir streamer de otra cara grande | El detector escoge la cara más grande | Falta |
| Detectar los límites reales del recuadro de webcam | Se infieren con multiplicadores desde la cara | Parcial |
| Elegir blur, negro, cover o contain según el contenido | Está fijado por cada layout | Falta |
| Reencuadrar gameplay según HUD, texto o zona importante | Crop centrado | Falta |
| Mover subtítulos para evitar cara, HUD o texto | Margen vertical global | Falta |
| Punch-in, zoom o movimiento con keyframes | El prompt lo menciona, el contrato no lo expresa | Falta |
| Transiciones distintas de corte | El contrato solo permite `cut` o `none` | Falta |
| Insertar memes, imágenes o B-roll desde el planner nuevo | No está en el contrato nuevo | Falta |
| Normalizar voz, limitar picos y mezclar SFX | Se conserva el audio fuente sin mezcla editorial | Falta |
| QA visual posterior al render | No hay ciclo de revisión y corrección | Falta |
| Usar el planner nuevo desde la interfaz | El README lo deja como flujo CLI | Falta |

## Fallas estructurales importantes

### El prompt promete efectos que el renderer no puede ejecutar

`EDIT_DIRECTION_RULES` pide zoom, punch-in y cambios motivados, pero
`EditSegment` no contiene escala, posición, keyframes, fondo, zona de subtítulos
ni efectos. El modelo puede razonar correctamente y aun así no existe un canal
estructurado para convertir su decisión en FFmpeg.

### La cámara se identifica por tamaño, no por identidad

`detect_webcam_region` elige la cara más grande de cada frame. En una reacción,
un invitado o una persona dentro del video reproducido puede ser más grande que
el streamer. El suavizado posterior puede mezclar posiciones distintas.

### El layout no es verdaderamente temporal

El renderer puede recibir crops diferentes por `EditSegment`, pero cada crop se
calcula sobre todo el segmento. Si la cámara se mueve a mitad del segmento, el
resultado puede promediar regiones incompatibles. Tampoco hay un detector local
de cambios de layout que subdivida automáticamente el segmento editorial.

### Los fondos son decisiones fijas

`_compose_contained` siempre genera un fondo borroso. Split usa cover para el
contenido y negro en la cámara. PIP usa un tamaño y una posición fijos. El corpus
demuestra que blur, negro, cover y contain deben seleccionarse por la información
visible, no únicamente por el nombre del layout.

### Los subtítulos no conocen el plano

El estilo ASS se selecciona globalmente y usa un `margin_v` fijo. No recibe las
regiones ocupadas por cara, HUD, chat, título o botones. Además, `dynamic_clean`
se mapea actualmente al template `tiktok` blanco, mientras el estilo visual
aprobado recientemente usa amarillo pollo.

## Enfoques posibles

### A. Plantillas fijas con detección de cámara

Barato y predecible. Se mantienen tres o cuatro composiciones y se escoge una por
clip. No cubre bien cambios internos, múltiples caras ni pantallas poco comunes.

### B. Composición completa decidida por VL en cada frame

Muy flexible, pero costosa, lenta e inestable. Puede producir encuadres que vibran
o cambian sin necesidad. No conviene como controlador directo de FFmpeg.

### C. Director visual híbrido y temporal — recomendado

1. Un analizador local construye un timeline de cámaras, caras, texto, bordes,
   cambios de escena, movimiento y regiones importantes.
2. Un modelo VL revisa contact sheets y eventos, identifica al streamer, decide
   qué fuente debe dominar y selecciona una familia de composición.
3. Un contrato JSON expresa capas, crops, fondos, subtítulos, marca, keyframes,
   transiciones y audio por tramo.
4. Un compilador determinista valida límites, suaviza movimientos y traduce la
   intención a filtros FFmpeg.
5. Una revisión VL inspecciona varios fotogramas del MP4 renderizado y solicita
   correcciones si hay caras cortadas, texto tapado, huecos absurdos o cambios
   bruscos.

La IA toma las decisiones editoriales. Las reglas deterministas se limitan a
seguridad geométrica, duración, suavizado y contratos válidos.


## Task 6: smoke real de ventanas visuales

Estado durable al 2026-07-26: **GATE LOCAL PHASE 1 APROBADO**.

Generación publicada:
`20260727T005135.883896Z-756b545435954af6b40af4914ff50440`.
Se reconstruyeron forzadamente ocho fuentes locales y se evaluaron las diez
ventanas humanas de `data/validation_visual/local_sources.json`. No se llamó a
LLM, VL ni proveedores y no se descargaron modelos durante la corrida.

- Escenarios reales ejecutados: 10/10.
- Pass/fail/ambiguous/unavailable: 10/0/0/0.
- `gate_passed: true`; `all_ten_real_windows_executed: true`.
- Muestras decodificadas e indexadas de forma comprobable: 347.
- Cache hits: 0; la corrida usó `--force` y analizador
  `source-visual-local-v3`.
- Runtime de construcción acumulado por fuente única: 645,567 s
  (10 min 45,567 s). La evaluación de assertions fue submilisegundo por
  escenario.
- YuNet SHA-256:
  `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- SFace SHA-256:
  `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`.

| Fuente | Muestras | Build s |
|---|---:|---:|
| `01_stable_gameplay_webcam.mp4` | 41 | 25,25 |
| `02_full_camera.mp4` | 28 | 34,17 |
| `03_camera_mode_transition.mp4` | 68 | 46,88 |
| `04_browser_to_gameplay.mp4` | 104 | 433,18 |
| `05_foreign_faces_multi_person.mp4` | 55 | 19,30 |
| `06_no_webcam.mp4` | 17 | 4,44 |
| `07_reacted_vertical_video.mp4` | 17 | 29,24 |
| `08_browser_small_text.mp4` | 17 | 53,09 |

Las transiciones semánticas usan `event_at` confirmado por una persona y exigen
una boundary visual cercana; el manifest ya no acepta conteos genéricos de
cambios. `expected_states` es exclusivo, los rectángulos verticales deben
persistir en al menos dos muestras y una ventana termina de forma semiabierta
(`start <= at < end`).

Este gate prueba la base visual local, no el editor autónomo completo. Las
fuentes del corpus son extractos cortos: todavía falta medir un VOD continuo de
2-4 horas. `04_browser_to_gameplay.mp4` consumió el 67 % del runtime total, por
lo que el análisis por muestra debe perfilarse antes de ese benchmark. También
siguen fuera de Phase 1 el director VL, active-speaker multiusuario, compilación
adaptativa FFmpeg y QA visual del MP4 final.

## Revalidación de la base visual - 2026-10-14

La base v7 pasó los diez escenarios reales con rebuild forzado, sin LLM ni
descargas. Run: `20261014T211048.520325Z-d4980a35c4fa4e7ba60dbf5c82f6cdcd`.
Ocho fuentes únicas, 500 muestras indexadas y 536,82 segundos de construcción
total; sin hits de caché. `04_browser_to_gameplay.mp4` concentró 385,59 segundos
para 171 muestras. La suite completa pasó 286 pruebas con dos advertencias de
dependencias; Ruff y compileall pasaron. El runtime usado fue `.venv-whisperx`.

Se corrigieron la lectura de perfiles entre discos, la recuperación transaccional
con distinto orden de archivos, la fusión de tracks compatibles sin solapamiento
y la procedencia de la identidad durante holds de cámara. Un track independiente
compatible con el perfil puede sostener geometría de cámara completa sin heredar
silenciosamente la identidad global. Esta incertidumbre queda explícita.

La ventana local `full_camera_to_browser` tenía anotaciones que incluían una
segunda transición real: se revisaron frames 30,5/30,75/31 y 43,5/44/45. El
evento queda en el primer escritorio estable observado (31 s) y la ventana
termina antes del retorno a cámara (43 s). Fue una corrección revisada por el
asistente, no una nueva aprobación humana. Las expectativas y tolerancias no
cambiaron; el retorno se verificó independientemente con pruebas y video real.
La procedencia está en `.superpowers/sdd/source-label-correction-20261014.md`.

El usuario reconfirmó shorts verticales, referencias históricas y marca ATHANW,
aplazó edición horizontal y chat futuro, y eligió GPT-6 Luna en CLIProxyAPI.
Un probe real de una imagen/JSON pasó (1.292 tokens reportados, 2,832 s). Es
compatibilidad básica, no un benchmark del director ni evidencia de viralidad.

El director, el compilador v2, el QA final y el benchmark continuo de 2-4 horas
siguen pendientes. No se extrapola el throughput de extractos al VOD completo.

## Contrato objetivo mínimo

Cada tramo visual debería poder expresar:

- rango fuente y orden de salida;
- fuente principal y fuentes secundarias;
- crop normalizado y región que debe mantenerse visible;
- posición, tamaño, anclaje y z-order de cada capa;
- modo de fondo: `cover`, `contain_blur`, `black`, `solid` o `inherit`;
- zona permitida para subtítulos y regiones prohibidas;
- tipo y duración de título y marca;
- keyframes de zoom, paneo o punch-in;
- transición de entrada y salida;
- ganancia, ducking, limitador y SFX opcional;
- razón editorial y evidencia usada.

## Recomendación

No conviene reemplazar el render actual por una única plantilla derivada del
corpus. Conviene conservar el diseño actual como una de las familias y construir
encima un director temporal híbrido. El primer objetivo debe ser resolver
identidad y tracking de cámara, selección adaptativa de fondo/crop y subtítulos
con zonas dinámicas. Después se añaden zooms, insertos y mezcla sonora.
# 2026-10-14: bounded style and visual-director recovery checkpoint

Owner selected GPT-6 Luna through the existing local CLIProxyAPI. Credentials
were read only in process memory from the named existing proxy; no copied .env
key, new provider account or global configuration was created. Operating context
was65,536tokens, output4,096, modelgpt-6-luna(low).

Only three historical edited references were analyzed/enriched, each with five
saved JPEGs, not a full video upload. After observed role/classification/geometry
failures and reviewed repairs, all three passed the bounded primary-role/band
checks: full_camera, reacted_video_with_camera and browser_with_camera; grouped
headline, creator mark in the observed band, camera present. Narrative remains
null and approval remains unreviewed; no source-VOD/output pair was invented.
Final rerun reported22,741tokens (7533/7562/7646), elapsed16.013/18.951/30.156s
including local hashing/publication. Library fingerprint:
09ca924a8dd02ee9d3baa7e9422429667035c0e9a5ea97b07eb486624363cf8c.
This is not pixel-perfect box/palette or human/render acceptance. Source hashes
and provenance remain preserved; no full301 corpus or fullVOD inference began.

Local MiniLM retrieval dependency was announced and installed only under this
project, pinned revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Offline CPU
encoding and compatible-family/cache behavior were verified, not semantic ranking
quality over a large Spanish corpus. Missing-family coverage stays sparse.

The stateless visual director passed59initial focused tests and, after independent
review fixes,71focused/285expanded regressions plus targeted lint/typing/import
isolation. Re-review closed streamer-crop association and requested-image
starvation gaps. These use fake clients/synthetic media. Real three-case director
benchmark, CLI wrapper integration, deterministic v2 compiler and rendered QA
remain separate pending gates. Conservative MSER/face/motion geometry can require
review rather than asserting unsupported crops; MSER candidates are not OCR.

## Task8 bounded benchmark harness - prepared, not executed

The portable three-case gate is prepared for the owner-selected
`gpt-6-luna(low)` through local CLIProxyAPI, provider `cliproxy`, operating context
65,536 and explicit output budget4,096. Inventory and actual director inference
remain not_run in this implementation lane. No passing model, token count,
latency or provider cost is claimed before the controller executes the gate.
Proxy billing is unknown/unpriced; estimated provider cost is null.

Cases reuse read-only Phase1 v7 caches and short original validation references:
stable gameplay with top-left webcam0..10s (source duration30s), full camera
0.5..6.5s (duration20s), and full camera/browser transition24..38s (duration60s).
The transition must split exactly at cached30.75s, preserving24..30.75 and
30.75..38. Source fingerprints remain eb7b85fc8da2036d,908a3fddb208bfde and
a229b36524b38277, respectively. Context is source-clamped editorial+/-15s.
Portable cases store cache/subset references; actual paths and generated local
subsets/evidence stay in ignored machine configuration/run artifacts.

The public director performs host authority, crop, identity, evidence and timing
validation. Benchmark limits are initial call plus one followup per case,
<=5source+3style images per call. Cases can run separately with a shared explicit
run directory, then aggregate without inference. Aggregation requires all3cases
with matching model, context, budgets, source/timeline/style and code signatures.
Completed cases cannot accidentally rerun. Until aggregation, selection remains
incomplete. Only the selected owner target is tested; no globally cheapest claim
is possible with unknown proxy rates. The final aggregate appends actual dated
metrics here, including explicit none_passed if appropriate.

Offline fake-client tests distinguish schema success from host acceptance; a
schema-valid incorrect streamer identity fails. Cached subset checks preserve
ordered face evidence/time association. These tests do not measure real provider
quality. All3StyleCards remain unreviewed, narrative null, no source pairing;
missing gameplay style coverage remains sparse. No fullVOD/all301 run or render.


<!-- visual-benchmark:0c35fa3834ccae034511a749c6786df97eba1e057ed01c062f5a2d9007371015 -->
## Bounded director benchmark - 2026-10-15T01:21:50.246288+00:00

Provider cliproxy; model gpt-6-luna(low); context 65536; output 4096.
Only the owner-selected target was tested; proxy rates are unknown/unpriced.
Outcome: none_passed; selected: None; cheapest: unknown.
Estimated provider cost: null. Source files are short Phase1 v7 references.

- stable_gameplay_webcam: fail; source [[0, 10]]; schema 1.0; invariants 0.3333333333333333; calls 1; evidence iterations 0; tokens prompt/completion/total 27245/1268/28513; elapsed 41.104s; call latencies [28.716]s.
- full_camera: fail; source [[0.5, 6.5]]; schema 1.0; invariants 0.3333333333333333; calls 1; evidence iterations 0; tokens prompt/completion/total 24200/973/25173; elapsed 24.629s; call latencies [11.634]s.
- full_camera_to_browser: fail; source [[24, 30.75], [30.75, 38]]; schema 0.0; invariants 0.3333333333333333; calls 1; evidence iterations 0; tokens prompt/completion/total None/None/None; elapsed 38.424s; call latencies [26.66]s.

Measured artifacts: `O:\Escritorio\ClipperX\data\validation_visual\runs\model_benchmark\owner-luna-20261014-r26-01`.
This measures host/schema geometry and source preservation, not semantic editing,
rendered quality or human approval. All three StyleCards remain unreviewed,
narrative null and without invented source pairing. Missing style families stay sparse.
No full VOD, all301 enrichment or MP4 rendering was performed.

<!-- visual-benchmark:795c0bdde7b7a7fc5451c68728a5a4acc841112572f44393f6017a4ce46b46d6 -->
## Bounded director benchmark - 2026-10-15T01:44:00.612830+00:00

Provider cliproxy; model gpt-6-luna(low); context 65536; output 4096.
Only the owner-selected target was tested; proxy rates are unknown/unpriced.
Outcome: none_passed; selected: None; cheapest: unknown.
Estimated provider cost: null. Source files are short Phase1 v7 references.

- stable_gameplay_webcam: fail; source [[0, 10]]; schema 0.0; invariants 0.3333333333333333; calls 1; evidence iterations 0; tokens prompt/completion/total None/None/None; elapsed 37.469s; call latencies [24.213]s.
- full_camera: fail; source [[0.5, 6.5]]; schema 1.0; invariants 0.3333333333333333; calls 1; evidence iterations 0; tokens prompt/completion/total 24395/1061/25456; elapsed 23.379s; call latencies [11.874]s.
- full_camera_to_browser: fail; source [[24, 30.75], [30.75, 38]]; schema 0.0; invariants 0.3333333333333333; calls 1; evidence iterations 0; tokens prompt/completion/total None/None/None; elapsed 41.818s; call latencies [30.274]s.

Measured artifacts: `O:\Escritorio\ClipperX\data\validation_visual\runs\model_benchmark\owner-luna-20261014-r27-01`.
This measures host/schema geometry and source preservation, not semantic editing,
rendered quality or human approval. All three StyleCards remain unreviewed,
narrative null and without invented source pairing. Missing style families stay sparse.
No full VOD, all301 enrichment or MP4 rendering was performed.

## Current native renderer qualification — 2026-10-16

This is a separate renderer gate, not a reinterpretation of the historical
model benchmark above. Current prerequisite run `native5772-a-20261016-15`
and complete synthetic run `native5772-b-20261016-03` passed unchanged strict
numerical criteria and Root inspected all eight prerequisite images plus ten
full synthetic images and five six-cell overlay boundary sheets. Approval is
recorded separately in `.superpowers/sdd/native5772-a15-result.md` and
`.superpowers/sdd/native5772-b3-result.md`; actual summaries retain their
`MEASURED_VISUAL_ACCEPTANCE_PENDING` status and were not rewritten as human
approval.

The synthetic source is a generated twelve-second 1920×1080 sequence, not a
user VOD. The output is eight seconds, 480 natural 1080×1920/60 H.264 NVENC
frames, `yuv420p`, AAC 48 kHz stereo with exactly 384000 presentation samples
per channel and faststart. It joins source ranges 0.5–3, 5–7, and 8–11.5 seconds
with 150/120/210 output frames. BLACK, SOLID, CONTAIN_BLUR, moving crop,
enlarged camera, source-audio normalization/silence, impact cue, subtitles and
finite alpha PNG layers were exercised. All thirty independent boundary crops
pass their original activation, PTS/checksum and RGB tolerances; sampled source
markers, moving geometry, audio pulses and visual flashes also pass. Full
independent audio/video decode passes.

Measured command/controller time: prerequisite 33.5694792/32.687 seconds
(77 native children); complete synthetic 102.3841637/101.515 seconds
(63 children: two fixture, 39 public, 22 independent validation). These are
small-campaign measurements, not long-VOD throughput or an OS hard quota.
Both completed synthetic fixture/controller jobs report `own0=true` and
`files_clear=true`; public return requires its existing clearance checks but
does not persist a final public `own0` snapshot. Historical retained uncertainty
was not deleted or relabelled. Program API calls in this qualification are zero;
coding-model usage/cost is separate and unpriced.

Exact synthetic output:
`data/processed/phase3-native-qualification/native5772-b-20261016-03/public/render_95a915328be26eaef01d0ccc9a532f3b/clip.mp4`,
3658594 bytes, SHA-256
`26f53be2f966b5becf9e262ab67766dc0d0458160f49170edc1bd580ad0f907e`,
with exactly five clip sidecars. Summary SHA-256
`1621283fc3c18ebbe83badc97f9c8ab5a78e4344fefbcbe4ca8c1b08d70a1836`.

The first-shot synthetic camera deliberately covers part of its test title to
exercise layer order; it is not a consumer layout recommendation. The synthetic
quadrant PNG is not the channel brand. Original fonts remain preserved; their
selected runtime file identity is still unverified under the user's explicit
waiver, while rendered finish was inspected. No human style approval or
source-result learning pair has been created.

Three evidence-backed real-window plans/renders, the full Phase 3 regression
gate (including existing typing errors), all Phase 4 QA/interface
work, human output-quality approval and the final full-input automatic workflow
remain pending. This section does not declare the product complete.

## Evidence-backed manual real render qualification — 2026-10-16

Fresh current-owned input observation `observation5773-20261016-01` selected
twelve lossless frames across seven effective windows of at most two seconds.
Its29 actual children include three source probes, fourteen video/incidental
audio children and twelve PNG encoders. Root inspected all images and exact
PTS/successor coverage/byte associations. Summary SHA-256:
`210fd4e483cf4d3b9a324cf995cd9b2e82cc9676805044b35090a5d9cf8eac5c`.
All seven controls retired; final controller `own0`/`files_clear` true. Twelve
selected RAW scratch files were retired; intended PNG evidence remains. Stat
fingerprints were rechecked; original VOD bodies were not hashed or rewritten.

Portable technical templates and README live under
`data/validation_visual/manual_plans/phase3-root-*-v1.template.json` and
`phase3-root-v1-README.md`. Machine-local adopted plans/evidence/source mappings
are in ignored `data/processed/phase3-native-qualification/` data. They are manual
technical plans with no model, identity-track assertion, new text or brand;
they do not constitute reviewed style cards or source-result learning pairs.

All three literal-public real outputs below passed current strict media/decode/
PTS/AAC/faststart/sidecar checks, and Root personally inspected all twelve
standard rendered diagnostic PNGs, plus three extra actual dissolve frames.
Every delivery is1080×1920/60 H.264 NVENC `yuv420p` with AAC48k stereo and exactly
one MP4 plus five clip sidecars. The shared186-pin manifest is
`root-5773-c-manifest.json`, SHA-256
`9c6a5c2f3ab2c34c22b0f30f19fec1cc9383329e5045e2162644972c34322ecd`.

| Case / current local source | Source windows → output | Frames / AAC presentation samples | Command / controller seconds | Native children |
| --- | --- | --- | --- | --- |
| gameplay /01_stable_gameplay_webcam.mp4 / statFPeb7b85fc8da2036d | .5–1.5 →0–1 |60 /48000 |27.3703904 /26.172 |28 |
| camera /02_full_camera.mp4 /908a3fddb208bfde | .5–6.5 →0–6 |360 /288000 |242.6525387 /240.422 |32 |
| transition /03_camera_mode_transition.mp4 /a229b36524b38277 |28–31,31–32 →0–3,3–4 |240 /192000 |109.9408423 /107.609 |36 |

Exact deliveries, relative to `data/processed/phase3-native-qualification/`:

- `native5772-g-20261016-01/public/render_5516c05a0b835b5d67f32adf52350b2b/clip.mp4`,1134290B,SHA-256 `f146e5a15ee7dde97a92ea8ab4cbb5d8b0f950646fa33bd07f3d1ce66b82341c`.
- `native5772-c-20261016-01/public/render_824aabfeab307c6124b56884dff3d68e/clip.mp4`,6509814B,SHA-256 `152857964bc9eae60561bf80a998f366ff3f577d60316581f54bc9f650d0e750`.
- `native5772-t-20261016-01/public/render_f20065aa89f509ae8451be4519db8c2d/clip.mp4`,3001225B,SHA-256 `d324409bfaa2fd0d9b8e8af2d76481d594ebcbed53bf014642e5f9e54aae303e`.

AAC skip1024 in all three; discard padding128/768/512 respectively. Controller
summaries SHA-256 `51ee756dad865cbcb4b5864684c1125547d16ab93cd8d0fe32065455240e5630`,
`86f5b975720998da35a6f78ad2d434a6e207ac96b497dbfeca40d0540d7e064b`,
`f029e557e84a09622813bea9c3646b2f449b996631e788b6b0d43b50cfac6b01`.
Both completed fixture jobs in each case report `own0`/`files_clear` true;
public return follows its existing clearance contract, without an archived
public final snapshot. Original uncertain historical files were not cleaned.

The observed webcam is344×270, larger than the old approximate box. Game
captions/context and the full head remain visible in sampled output. Camera
foreground keeps the sampled head and moving phone; original soft focus is not
recovered by this renderer. Transition preserves the actual in-source dissolve
at source30.7/.75/.8; the layout change at31 is a manual later-stable-state
decision, not an asserted exact dissolve endpoint. A separate three-child
owned output diagnostic verified actual rendered PTS/checksums and Root viewed
those three fade images; summary SHA-256
`5076616aae95ed43a33964be22bc0711cc3efeec0d21cb9c7dad9bc6780af463`.

These conservative plans intentionally retain black padding/embedded-camera
duplication and omit new Kick branding/captions. They qualify source continuity
and geometry, not final consumer style, human approval or virality. The
six-second camera campaign was slow; no realtime or full-VOD throughput promise
is supported. Actual global Phase3 regression/static gates and all Phase4 work
are still required. No model inference or full-VOD run occurred here.

### Required Phase3 global regression and motion-cost follow-up (2026-10-16)

The first complete global regression is **not passing**: pytest recorded3169
passed,13failed,5guarded-native skips and2dependency deprecations; cold mypy
recorded339diagnostics in33files. Ruff and compilation passed. Twelve pytest
failures were traced to the numerical-reference helper using a removed FFmpeg9
file-graph option. The reviewed helper correction passed all12affected cases
(18finite synthetic children), without changing production graphs or oracles.
The other failure is the unchanged121-frame moving-minification control
exceeding90seconds; it remains a required check, not waived coverage.

Two reviewed typing-preservation units closed their original136and119owned
diagnostics in scoped checks, with144and154passing preservation cases
respectively; the latter has an existing different-volume skip. These scoped
results do not replace the still-required full-project gate. Remaining repair
units and the next full regression are pending. Exact reports and original
failures are retained under `.superpowers/sdd/`.

One separately reviewed two-child motion experiment measured the candidate
libplacebo option `dynamic_constants=yes`, without changing production policy.
It completed121RGBA48×48frames in48.86seconds and matched the original-policy
frozen output at frames0/42/60/120 byte-for-byte (192row comparisons). Both
children identified NVIDIA GeForce RTX3060 and retired all owned resources and
temporary RAW files. Controller summary SHA-256
`47c1f9fe069995b7fd67915e749bff3bc12763b26153192eb27b341b2f1105ae`.

This is not proof of causal acceleration, cold-cache behavior, stable speed,
all-frame equivalence or production readiness: the original control previously
completed in43.047seconds under conditions not reproduced here. No option was
promoted merely to make a test pass. The next required complete gate runs
pytest and then mypy without overlapping project workloads, retaining the
original90second limit and exact oracle. Phase4 QA/interface, full-input
production validation and specific human quality approval remain pending.

### Updated global gate and controlled motion result (2026-10-16)

All reviewed typing repairs are now complete. The second full gate recorded
3,202 passed tests, one motion timeout and five guarded-native skips, with no
warnings. Cold mypy passed all156 source files; full Ruff and compilation also
passed. All163 source/configuration inventory entries remained unchanged
through that gate. The durable C:pytest base preserves the diagnostics and
exercised the different-volume profile regression.

A subsequent reviewed A1/B/A2 diagnostic did not establish a repair. The
original policy completed121 frames in102.89seconds under an exploratory180s
budget, exceeding the unchanged90s acceptance requirement. The dynamic-constant
candidate then timed out at90s; the controller stopped before A2 as specified.
Both owned jobs retired their resources and RAW files; failed-output transfer
uncertainty is preserved as such. No full-array equality was established and
the candidate was not promoted. Exact evidence and limitations are recorded
in `.superpowers/sdd/diag5775-motion-result.md` and the consumed diagnostic
directory. The source-backed performance investigation continues; Phase3 exit,
Phase4, finished-clip human approval and full-VOD acceptance remain open.
