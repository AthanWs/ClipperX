# Exact decimal overlay boundaries

The second real Luna preview response places subtitles at y=0.48, height=0.34 and the bottom forbidden area at y=0.82. Their decimal edges touch, but binary float addition produces 0.8200000000000001 and the current overlap guard rejects the shot. This is a locally reproduced program defect, not evidence that the first discarded response had the identical defect.

Treat the serialized decimal values of the existing normalized rectangles as exact for the four strict intersection comparisons. Use standard-library Fraction(str(value)); touching edges have no area and are not intersections. Real positive overlap, even smaller than 1e-9, remains forbidden. Do not add epsilon, relocate overlays, remove forbidden regions, change containment policy, weaken source protection, or imply that fixing this one validator makes the captured model plan host-ready.

Scope: only the overlap helper and focused contract tests. Keep the model responses, original inputs, failed receipts, native build namespaces, all render timeouts and all original assets unchanged. Local captured-output revalidation costs zero inference. The user delegated logical corrections and continued execution; no new provider or architecture is selected by this fix.
