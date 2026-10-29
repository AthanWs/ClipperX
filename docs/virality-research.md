# Virialidad y retención en short-form — investigación accionable para ClipperX

> Informe de investigación (multi-fuente, citado). Secciones 1-3 y 5 = fuentes de
> creator-economy/plataforma (regla de oro, no constantes de laboratorio). Sección 4 =
> neurociencia peer-reviewed. La conclusión clave para nosotros está al final.

## 1. Hook y retención (segundo a segundo)
- **La decisión es en 0-3 s.** ~50-60% del drop-off ocurre en los primeros 3s *(vendor)*. Si retienes <~60% pasados 3s, el algoritmo te ahoga.
- **Curva sana:** caída inicial pequeña que se aplana, con repuntes (rewatch) y pico de loop al final. **Curva mala:** acantilado en 0-3s (hook débil o "falsa promesa") o sangrado lineal sin re-hooks.
- **Por qué hacen swipe:** sin payoff/curiosidad en ~3s; **hook que no cuadra con el contenido**; aire muerto/silencio; pacing lento (bloque >5s sin cambio); sin captions; baja energía.
- **Fórmulas de hook:** afirmación contundente/contraria · pregunta/curiosity-gap · pattern-interrupt (movimiento/sonido inesperado) · resultado-primero.
- **Reglas:** el elemento "stop" en los **primeros ~2s**. Para clips de directo: **empezar en el payoff/pico, NO en el wind-up** (cold-open en el momento fuerte). **Re-hook cada ~5s** (ningún bloque de 5s sin cambio) + pattern-interrupt deliberado hacia el **25-35s**.

## 2. Emociones → qué métrica disparan
Berger & Milkman (JMR 2012): **es el AROUSAL, no la valencia.** Alta activación (asombro, ira, ansiedad, diversión) → viral; baja (tristeza) → no.

| Emoción / táctica | Shares | Comentarios | Watch-time |
|---|---|---|---|
| Asombro (awe) | ★★★ | ★ | ★★ |
| **Humor / diversión** | ★★★ | ★★ | ★★★ (loops) |
| **Ira / indignación moral** | ★★★ | ★★★ | ★★ |
| Ansiedad / suspense | ★★ | ★ | ★★★ (open loop) |
| Tristeza (baja activación) | ✗ | ★ | ★ |
| Curiosity gap / open loop | ★ | ★★ | ★★★ |
| Relatabilidad / identidad | ★★★ | ★★ | ★★ |
| Controversia / hot take | ★★ | ★★★ | ★★ |

Shares → awe/humor/outrage + identidad. Comentarios → controversia/hot-take (no gimmicks). Watch-time → open loop + suspense + pacing apretado + loop.

La **crítica/rabia SÍ da viralidad** (la gente comparte y comenta para corregir), pero predice *shares*, no compromiso real, y el rage-bait crudo arriesga penalización.

## 3. ¿El engagement-bait funciona?
- **Meta lo demota explícitamente** (pedir likes/comentarios/tags "sin call-to-action real"). TikTok igual (engagement inauténtico → flag/baneo).
- **Sostenible (lo que hacen los top):** "cebar" con SUSTANCIA — un hot-take debatible, un open loop, un detalle "¿lo pillaste?" que invita a re-ver, un error deliberado que la gente corrige en comentarios.
- **Para la herramienta:** preferir destacar clips intrínsecamente debatibles/curiosos antes que estampar "comenta 👇".

## 4. Neurociencia del compartir (lo que de verdad dice la evidencia)
- **Falk (PNAS 2017):** sistema de **valor (estriado ventral + vmPFC)** + **auto-relevancia (mPFC/PCC)** + **mentalización (TPJ, dmPFC, precúneo)** → la actividad en estas regiones **predice el compartir a nivel POBLACIONAL**.
- **Knutson (neuroforecasting):** es el **núcleo accumbens (NAcc)** el que "le gana al comportamiento" prediciendo éxito agregado.
- **STEPPS (Berger):** Social currency · Triggers · Emotion · Public · Practical value · Stories.

> **⚠️ LO CLAVE PARA NOSOTROS:** (1) El mejor predictor neuronal de viralidad es el **sistema de recompensa (NAcc / estriado ventral) — SUBCORTICAL**, que el TRIBE **cortical NO tiene.** Eso **explica nuestro null.** (2) La propia literatura dice: *"no puedes meter a los espectadores en un escáner; lo útil es la TEORÍA (qué optimizar) + proxies de comportamiento (retención, rewatch, share-rate) como sustituto de la señal neuronal."* → El camino práctico **no es predecir cerebro por clip**, es **operacionalizar la teoría con señales detectables + validar con retención.**

## 5. Principios de edición para retención
Cortar aire muerto/silencios · jump cuts · **zoom/punch-in a la cara en el momento clave** · SFX/risers (riser antes del reveal, golpe en el corte) · captions quemados (palabra a palabra, +15-25% retención *vendor*) · B-roll en tramos de talking-head · speed ramps · **open loop/curiosity gap** (la palanca #1 de completado) · payoff a tiempo · **loop sin costura** (el final empalma con el inicio).

## 6. Operacionalización (spec de build) — qué puede DETECTAR ClipperX
| Principio | AUDIO | VIDEO | TRANSCRIPT/LLM | Acción |
|---|---|---|---|---|
| Encontrar el momento | pico RMS, risa, aplauso | pico de movimiento, cambio de escena | LLM: punchline, hot-take, clímax | scorear ventanas, pico = centro |
| Hook = empezar en payoff | localizar pico de energía/risa | frame de máxima expresión | LLM: frase del reveal | IN ~1-2s antes del pico, cold-open |
| Cortar aire muerto | silencio >300-500ms, muletillas | frame estático | huecos/muletillas en transcript | auto-trim; cambio cada 2-4s |
| Pacing / jump cuts | límites de frase/respiración | scene-cut detection | límites de cláusula | cortar en cláusulas; interrupt si >5s estático |
| Zoom a la cara | subida de volumen en la línea | **face detect + speaker activo** | LLM marca la línea clave | punch-in en la cara en esa línea |
| SFX/stingers | onset para sync | puntos de corte | LLM marca reveal | riser antes, golpe en corte |
| Captions | ASR + timestamps | safe-area/cara | timing por palabra, énfasis | quemar captions, resaltar keyword emocional |
| Open loop | — | — | LLM: detectar teaser + su respuesta | reordenar: teaser al inicio, payoff al final |
| Emoción objetivo | risa, voz tensa, gasp | clase de expresión facial | **LLM: sentimiento + arousal + palabras de outrage** | etiquetar emoción → ruta share vs comment |
| Re-hook cada ~5s | timeline de cambios audio | timeline de cambios visual | detección de cambio de tema | flag bloques >5s sin cambio → meter interrupt |
| Evitar bait penalizado | — | — | clasificador LLM de frases prohibidas | suprimir overlays baity |

**Prioridades de mayor palanca:** (1) detección de pico audio (energía+risa+aplauso) para *encontrar* el momento; (2) auto-trim de silencios/muletillas; (3) captions palabra-a-palabra con énfasis; (4) punch-in por face-detection en la línea clave; (5) pasada LLM que etiqueta tipo de hook, emoción/arousal dominante, posiciones de open-loop/payoff, y screening de engagement-bait.

## Conclusión operativa
La verdad de campo es **conductual**: instrumentar **retención temprana, rewatch y share-rate** como proxy de la "señal de valor", y dejar que esas métricas afinen las heurísticas de la Sección 6. La neurociencia justifica *qué optimizar*, no garantiza que un clip concreto sea viral.

## Fuentes
- Berger & Milkman, *What Makes Online Content Viral?* (JMR 2012): https://jonahberger.com/wp-content/uploads/2013/02/ViralityB.pdf
- Scholz, Baek, O'Donnell, Falk et al., *A neural model of valuation and information virality* (PNAS 2017): https://www.pnas.org/doi/10.1073/pnas.1615259114
- Genevsky, Tong & Knutson, *When Brain Beats Behavior* (J. Neurosci. 2017): https://pmc.ncbi.nlm.nih.gov/articles/PMC5588458/
- Brady & Crockett, MAD model: https://journals.sagepub.com/doi/full/10.1177/17456916231187395 · Brady et al. (Science Advances): https://www.science.org/doi/10.1126/sciadv.abe5641
- Berger, STEPPS: https://jonahberger.com/contagious-framework-stepps/
- Meta Transparency, Engagement bait: https://transparency.meta.com/en-gb/features/approach-to-ranking/content-distribution-guidelines/engagement-bait
- TikTok policies: https://www.tiktok.com/safety/en/policies-and-engagement
- OpusClip hook formulas: https://www.opus.pro/blog/tiktok-hook-formulas
- Highlight detection (audio+visual): https://arxiv.org/pdf/2407.13933
