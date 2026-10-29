# Plantilla vertical multiplataforma

ClipperX renderiza en 1080x1920 y reserva esta zona conservadora para elementos
críticos: 90 px a la izquierda, 190 px a la derecha, 110 px arriba y 420 px abajo.
No es un límite oficial común: TikTok varía la zona con el largo del caption y
los add-ons, mientras Meta y YouTube usan comprobadores o guías visuales. Estos
márgenes son la intersección operativa elegida para evitar botones, descripción y CTA.

- Margen superior vacío: 10% del lienzo, equivalente a 192 px en 1080x1920.
- Headline: Impact de 46 px, centrado en una banda negra de 120 px situada después
  del margen superior durante los primeros 4 segundos; nunca tapa gameplay.
- Marca: PNG transparente de Kick + ATHANW, centrado en la misma banda y mostrado
  después de que desaparece el headline.
- Subtítulos: amarillo `#FFD51A`, borde negro opaco y márgenes simétricos de
  170 px para mantener el centro óptico exacto.
- Cámara: se detecta por segmento, se expande al borde de su esquina y se contiene
  completa; el desenfoque solo rellena los laterales del panel.
- Composición split: gameplay en modo cover sin bordes desenfocados, una banda
  negra de 110 px para subtítulos, cámara de 480 px y 370 px negros abajo.
- Cámara: imagen interna ampliada al 65% del ancho, recortando únicamente la parte
  inferior sobrante y usando laterales negros para conservar cabeza y rostro.
- Adaptación: la altura del panel y el ancho interno se recalculan con el aspecto
  real de cada webcam. Recuadros verticales, cuadrados, 4:3 y 16:9 conservan el
  mismo eje central y el mismo borde inferior seguro sin compartir un crop fijo.
- Resolución fuente: todas las coordenadas de cámara y contenido se calculan sobre
  las dimensiones detectadas; 720p y 1080p con el mismo aspecto producen la misma
  composición final 1080x1920.

## Perfil TikTok medido

Sobre una captura real de 828x1792, el viewport 9:16 ocupa aproximadamente 828x1472.
Convertido al lienzo 1080x1920 de ClipperX:

- Los controles derechos empiezan cerca de x=942 y bajan aproximadamente de y=918
  a y=1660. Los overlays críticos terminan en x=890.
- La descripción y el audio empiezan cerca de y=1530. La cámara termina en y=1504
  y el resto queda sobre negro.
- El logo ocupa la banda y=0..120; el headline comienza en y=172 y los subtítulos
  se centran alrededor de y=1030, fuera de ambas zonas de interfaz.

La comprobación final debe hacerse también en la vista previa de cada plataforma,
porque la interfaz cambia según dispositivo, caption, CTA y tipo de publicación.
