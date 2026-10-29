# Qué falta para afirmar «paso el directo y sale todo»

Estado observado al retomar, 14 de octubre de 2026. Este registro no sustituye
los planes adaptativos aprobados ni autoriza saltarse sus gates.

1. Descubrimiento visual independiente. El planner actual recorre los lotes de
   transcripción, pero su petición de descubrimiento contiene texto y no imágenes
   ni propuestas de eventos visuales (`planner_engine._discovery_user_prompt`).
   La evidencia audiovisual se solicita después de tener un candidato. Por tanto,
   una acción inesperada sin habla puede no llegar a ser candidata. Dirigir bien
   el encuadre de un clip ya elegido no resuelve por sí solo esta cobertura.
2. Cantidad versus exhaustividad. PlannerRunLimits usa por defecto target_clips=3
   y max_candidates=6. El bucle sí recorre todos los lotes; no hay evidencia de
   que simplemente abandone el resto del transcript. Esos límites son útiles para
   validación acotada, pero no prueban que se aproveche cada momento publicable.
   No se eliminaron los presupuestos ni se inició inferencia masiva para simular
   exhaustividad. Hará falta medir cobertura y permitir un modo de producción
   con presupuesto explícito, sin rellenar con clips malos.
3. Escala real. La campaña v7 valida diez ventanas y ocho fuentes, no un VOD
   continuo de tres horas. Una ventana especialmente densa tardó unos386segundos
   para171muestras. Antes de procesar el archivo completo hay que perfilar y medir
   coste/tiempo total, reanudación y consumo de disco; no extrapolar una velocidad
   garantizada a partir de los casos pequeños.
4. Dirección, render y QA. La biblioteca y el director de Phase2 aún deben
   conectarse al compilador v2 de Phase3 y a la revisión acotada/UI de Phase4.
   Las pruebas de esquema y la detección de cámara no equivalen a un MP4 terminado
   que conserve contexto, identidad, audio y legibilidad.
5. Validación de gusto y viralidad. Las referencias históricas permiten describir
   estilo; no prueban una asociación con VODs ni que una nueva pieza sea buena.
   Las primeras pruebas reales de estilo revelaron errores marca/título pese a
   pasar esquema. Tras reparar los prompts y aportar geometría medida sin roles
   semánticos, la repetición acotada pasó las tres comprobaciones principales de
   familia, rol y banda; no prueba precisión píxel a píxel, calidad narrativa ni
   aceptación del director. Un score de similitud no es una probabilidad de
   viralidad. La aceptación humana inicial sigue siendo distinta del QA automático;
   no se fabricarán aprobaciones ni pares de entrenamiento.
6. Texto detectado frente a texto entendido. detect_text_regions usa MSER para
   estimar cajas parecidas a texto; no es OCR semántico confirmado. Un span real
   de30segundos contiene5690candidatos. El director debe acotar su representación
   sin fingir que cada candidato está leído o que todos son importantes para la
   historia. La protección conservadora puede restringir recortes/acercamientos;
   su utilidad visual debe comprobarse sobre los MP4, no sólo con un esquema.

Se conserva la visión: verticales con contexto y estilo ATHANW, rangos relacionados
aunque no sean contiguos, cambios visuales motivados y procesamiento reanudable.
Edición horizontal larga, chat de futuros directos, generación de B-roll y
publicación en cuentas sociales no se añadieron a este avance.

Orden de trabajo vigente: recuperar bases → biblioteca/dirección → compilador →
QA/interfaz → validar una salida con el usuario. La cobertura de descubrimiento
silencioso y el rendimiento de un VOD completo deben quedar resueltos o claramente
acotados antes de afirmar autonomía integral; no basta con cambiar de editor.
