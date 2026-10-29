"""Prompt templates for LLM calls (OpenAI-compatible API)."""

from __future__ import annotations


HEADLINE_PROMPT = """
Actúa como un experto en marketing viral para TikTok y Shorts.
Analiza el siguiente texto de un video y genera un título (headline) altamente atractivo (clickbait moderado, emocional o curioso).

Texto:
"{text}"

Reglas:
1. Máximo 6 palabras.
2. Usa mayúsculas para énfasis solo si es necesario.
3. El título debe evocar curiosidad, humor o sorpresa.
4. Genera 3 hashtags relevantes (#).

Responde EXCLUSIVAMENTE en formato JSON:
{{
  "headline": "Título aquí",
  "keywords": ["tag1", "tag2", "tag3"]
}}
"""


VISION_PROMPT = """
Actúa como un Director de Edición Senior. Tu tarea es analizar este fotograma para determinar CÓMO debe editarse este clip para TikTok (formato vertical 9:16).

Analiza VISUALMENTE la imagen y su composición.
Contexto de audio (si ayuda): "{context}"

Responde UNICAMENTE en JSON con los siguientes campos:

1. "description": Describe detalladamente qué se ve en pantalla. Ej: "Es un gameplay de Call of Duty con una webcam superpuesta en la esquina derecha", "Es un video de YouTube siendo reaccionado por el streamer en pantalla completa", "Es solo la cara del streamer hablando a la cámara".
2. "category": Clasifica la escena en una de estas opciones:
   - "GAMEPLAY": Se ve un videojuego activo ocupando la mayor parte de la pantalla.
   - "REACTION": El streamer está viendo/reaccionando a un video/trailer/contenido de terceros.
   - "TALKING_HEAD": El streamer ocupa toda la pantalla (Just Chatting) o la atención principal es su cara sin contenido de fondo relevante.
   - "IRRELEVANT": Pantallas de carga, menús aburridos, negro, borroso.
3. "layout_suggestion": ¿Cómo deberíamos editar esto?
   - "SPLIT": (Dividir pantalla) Si hay contenido relevante (juego/video) + una webcam clara. Pondremos el contenido arriba y la cara abajo.
   - "FULL_CAM": (Recorte centrado) Si es un "TALKING_HEAD" o si la webcam es lo único importante.
   - "FULL_CONTENT": Si es un trailer o cinemática donde la cara no importa o no está.
   - "DISCARD": Si la categoría es IRRELEVANT.

Ejemplo de JSON esperado:
{{
  "description": "Streamer reaccionando a un trailer de película en ventana pequeña.",
  "category": "REACTION",
  "layout_suggestion": "SPLIT",
  "relevante": true,
  "motivo": "Reacción activa interesante"
}}
"""


EDITORIAL_BRIEF_PROMPT = """Eres un director editorial de contenido para redes sociales.
Se te da una muestra representativa de la transcripción de un stream (fragmentos del inicio, medio y final).

MUESTRA:
{sample}

Analiza el stream completo y genera un brief editorial que servirá como guía para seleccionar los mejores clips.

Responde ÚNICAMENTE con JSON válido:
{{
  "content_type": "<tipo: gameplay, just chatting, reaction, podcast, tutorial, etc>",
  "tone": "<tono dominante: humor, competitivo, educativo, casual, emocional, etc>",
  "streamer_style": "<descripción breve del estilo del streamer>",
  "valuable_moments": "<qué tipos de momentos tienen más valor viral en ESTE stream>",
  "language": "<idioma principal detectado>",
  "summary": "<resumen de 2-3 frases de qué pasó en el stream>"
}}"""


CHUNK_CLIP_PROMPT = """Eres un editor experto en clips virales para TikTok, Reels y Shorts.

Se te da un fragmento de transcripción de un stream con timestamps en segundos.
Los primeros segmentos pueden solaparse con el fragmento anterior — los timestamps te indican exactamente cuándo ocurre cada cosa, úsalos para orientarte.

CONTEXTO EDITORIAL DEL STREAM:
{editorial_brief}

TRANSCRIPCIÓN:
{chunk}

RANGO DE ESTE FRAGMENTO: {chunk_start}s → {chunk_end}s

Identifica los mejores momentos para clips virales dentro de este rango.
Un buen clip:
- EMPIEZA en (o 1-2s antes de) el PICO — el remate, la reacción, el reveal, la jugada — NO en el wind-up ni en charla de relleno. Cold-open directo en lo fuerte.
- Lleva una emoción de ALTA ACTIVACIÓN: risa, asombro, ira/crítica, suspense, o un gancho de curiosidad (open-loop).
- Cierra natural, sin colas muertas.
Usa el contexto editorial para saber qué momentos son más valiosos en ESTE stream.

REGLAS:
- Solo clips cuyos timestamps estén dentro de {chunk_start}s → {chunk_end}s
- Duración ideal: 15-45s. Máximo absoluto: 179s. Corto y dinámico > largo y plano.
- Prioriza emoción fuerte, sorpresa, humor, polémica o información impactante.
- El "start" debe caer justo en lo interesante (recorta el wind-up); no cortes a mitad de un remate.
- Si no hay ningún momento que valga la pena, devuelve lista vacía

Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{{"clips": [{{"start": <float>, "end": <float>, "headline": "<max 6 palabras en español>"}}]}}"""


GAP_AUDIT_PROMPT = """Eres un editor experto en clips virales. Estás evaluando un silencio o pausa dentro de un clip.

CONTEXTO DEL CLIP:
- Tema del clip: {clip_headline}
- Duración del gap: {gap_duration:.1f} segundos (de {gap_start:.1f}s a {gap_end:.1f}s)

CONTEXTO TEXTUAL:
Antes del gap: {before_text}
Después del gap: {after_text}

AUDIO DEL GAP:
{audio_features}

Se te dan hasta 3 frames del gap (inicio, mitad y fin) para que puedas ver qué ocurre visualmente durante la pausa.

En short-form el AIRE MUERTO es la causa #1 de swipe: córtalo salvo que aporte tensión real.

Decide:
- "keep": SOLO si el gap aporta tensión real (suspense, reacción facial fuerte, anticipación de un reveal). Por defecto NO.
- "cut": silencio/relleno sin nada que aporte → cortar (lo más común).
- "split": cambio de tema sin relación → dividir.

REGLAS:
- Sé agresivo cortando: un gap de >1s sin reacción visible ni tensión = cut.
- Mantén un gap solo si quitarlo arruinaría el remate o el suspense.
- Si el contenido antes y después no tiene relación → split.

Responde ÚNICAMENTE con JSON válido:
{{"action": "keep|cut|split", "cut_start": <float|null>, "cut_end": <float|null>, "split_at": <float|null>, "reason": "<explicación breve>", "confidence": <float 0-1>}}"""


VIRALITY_SCORE_PROMPT = """Eres un analista de RETENCIÓN y viralidad para TikTok/Reels/Shorts. Estos algoritmos premian la retención: que la gente vea hasta el final, re-vea y comparta. Puntúa con criterios basados en evidencia (no en "suena bien").

TÍTULO DEL CLIP: {headline}

TEXTO DEL CLIP:
{text}

RÚBRICA (cada dimensión 0-25):

HOOK (0-25): el espectador decide en los PRIMEROS 1-3 SEGUNDOS (ahí se va ~la mitad).
- 0-5: arranca en wind-up/relleno/silencio, o promete algo que no cumple.
- 6-12: tarda en arrancar; el momento fuerte llega tarde.
- 13-18: engancha pronto, hay gancho claro.
- 19-25: el primer 1-2s detiene el scroll (acción, reacción fuerte, pregunta o remate inmediato).

ENGAGEMENT (0-25): ¿mantiene la atención sin caídas?
- 0-5: aire muerto, ritmo lento, fácil swipe.
- 6-12: interesante pero con tramos planos.
- 13-18: buen ritmo, sin tramos muertos.
- 19-25: open-loop/curiosidad que obliga a llegar al final; cero relleno.

VALUE (0-25): ¿entretiene, informa o emociona de forma memorable?
- 0-5: sin valor. 13-18: valor claro (se ríe, aprende, se emociona). 19-25: memorable.

SHAREABILITY (0-25): se comparte por EMOCIÓN DE ALTA ACTIVACIÓN (asombro, risa, ira/indignación, suspense), NO por tristeza/calma. Suma: relatabilidad/identidad ("esto soy yo") y controversia (da que comentar/criticar).
- 0-5: emocionalmente plano. 19-25: dispara emoción fuerte + ganas de mandárselo a alguien o de comentar/criticar.

HOOK_TYPE: "question" | "statement" | "statistic" | "story" | "contrast" | "action" | "none"
EMOTION: la emoción de ALTA ACTIVACIÓN dominante: "humor" | "outrage" | "awe" | "suspense" | "curiosity" | "none"

Responde ÚNICAMENTE con JSON válido:
{{"hook": <int 0-25>, "engagement": <int 0-25>, "value": <int 0-25>, "shareability": <int 0-25>, "total": <int 0-100>, "hook_type": "<tipo>", "emotion": "<emoción>", "reasoning": "<1-2 frases>"}}"""


BROLL_SUGGEST_PROMPT = """Eres un editor de video experto. Analiza el texto de este clip y sugiere momentos donde insertar B-roll (material de archivo) para mejorar el impacto visual.

TEXTO DEL CLIP:
{text}

DURACIÓN TOTAL: {duration:.1f} segundos

Busca momentos donde:
1. El hablante menciona algo visual que se podría mostrar (lugares, objetos, acciones)
2. Hay una pausa natural o transición temática
3. El contenido se beneficiaría de una imagen ilustrativa

REGLAS:
- Máximo 3 inserciones de B-roll por clip
- Cada B-roll debe durar entre 1.5 y 4 segundos
- El search_term debe ser en inglés (para Pexels API)
- El timestamp es relativo al inicio del clip
- No insertes B-roll en momentos de alta emoción facial

Responde ÚNICAMENTE con JSON válido:
{{"suggestions": [{{"timestamp": <float>, "duration": <float>, "search_term": "<keyword en inglés>"}}]}}"""


BOUNDARY_AUDIT_PROMPT = """Eres un editor experto en clips virales. Estás auditando el {boundary_side} de un clip.

CLIP:
- Lado auditado: {boundary_side} del clip
- Duración actual del clip: {clip_duration:.1f} segundos
- Tema del clip: {clip_headline}

CONTEXTO TEXTUAL:
Antes del boundary: {before_text}
Después del boundary: {after_text}

AUDIO EN EL BOUNDARY:
{audio_features}

En short-form la retención se decide en los PRIMEROS 1-3 SEGUNDOS. Decide si ajustar el {boundary_side}.

REGLAS:
- Si es el INICIO: debe enganchar de inmediato. Recorta cualquier wind-up, silencio o charla de relleno para EMPEZAR justo en (o 1-2s antes de) el momento fuerte/pico. Si ya cae en lo interesante, no toques.
- Si es el FIN: cierre natural, sin cortar a mitad de reacción ni dejar cola muerta/silencio final.
- Máximo ajuste: 12 segundos en cualquier dirección.
- Si el boundary ya está bien, devuelve todos los valores en 0

Responde ÚNICAMENTE con JSON válido:
{{"extend_before_sec": <float 0-12>, "extend_after_sec": <float 0-12>, "trim_before_sec": <float 0-12>, "trim_after_sec": <float 0-12>, "reason": "<explicación breve>", "confidence": <float 0-1>}}"""
