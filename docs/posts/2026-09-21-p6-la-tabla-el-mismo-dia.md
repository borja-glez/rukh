# La tabla, el mismo día

*Borrador del post 4 de la serie de Rukh, y guion del vídeo de cierre de la fase 1. Escrito el
2026-09-21 al cerrar P6. Todos los números salen de `docs/benchmarks.md`, de `artifacts/eval/` y
de `docs/decisiones-de-ejecucion.md`; ninguno está redondeado a favor.*

---

Rukh es un modelo de lenguaje de ajedrez que he construido desde cero, módulo a módulo, para
aprender cómo funcionan por dentro. Seis módulos después tengo trece modelos publicados, un curso
que los construye paso a paso y una demo en la que se juega contra ellos en el navegador. Este
post cierra la primera fase, y va de la parte que menos apetece hacer y más enseña: **volver a
medirlo todo el mismo día**.

## Las tres cifras

Si solo te llevas tres números de la fase 1, que sean estos.

**1535.** El Elo estimado de `medium-v4`, el decoder de 115 M de parámetros del que salen todos los
afinados y todos los alineados, contra una escalera de Stockfish a fuerza limitada, con un
intervalo de 1476 a 1599. No es un Elo de Lichess y no hay conversión honesta; es el número que
ordena las etapas del proyecto entre sí, medidas con el mismo instrumento.

**+57 (36-79).** Lo que el alineamiento con DPO dentro de política añadió a ese modelo, medido
como se mide una diferencia: 1 600 partidas del alineado contra su base, no restando dos filas de
la tabla. La tabla de hoy da 1632 para el alineado y 1535 para la base, es decir, +97; la de hace
dos días daba +56. Ninguna de las dos es la medida.

**0 y 17.** Lo que se mueve la escalera cuando no cambia nada: el mismo modelo, la misma semilla,
dos veces con el rival por reloj (1538 y 1538) y dos veces con un presupuesto de nodos (1541 y
1524), con la máquina parada. Hace dos hitos, dos tiradas idénticas dieron 1498 y 1558 y lo
achaqué al reloj. Estaba mal: el rival aleatoriza a propósito con los dos límites, y lo que vi
entonces fue una máquina cargada o mala suerte a 1,18 σ.

## Por qué volver a medir

Cada módulo había cerrado con su medición, hecha el día que cerró, con el harness de ese día.
Entre medias el harness cambió: se corrigieron cuatro peldaños de la escalera, se partió la clave
de la caché, se añadió un baseline. Una tabla con filas de días distintos no es una tabla, es un
álbum.

`rukh eval nightly` recorre el catálogo de modelos publicados, descarga lo que falte, fusiona cada
adaptador LoRA sobre su base, mide y reescribe la tabla y el informe. Once etapas, dos horas y
cuarenta y ocho minutos, sin tocar nada a mano.

Y la comparación entre la tabla nueva y la vieja es la mejor lección del curso sobre
reproducibilidad. Legalidad, top-1, top-3, puzles y entropía **no se movieron nada**: a
temperatura 0,05 el muestreo es casi determinista y esas medidas no juegan contra nadie. La única
columna que se movió es la única que juega contra Stockfish, y se movió lo que su intervalo dice
que se puede mover: siete de nueve diferencias caben en un semiancho de intervalo.

| Etapa | Al cerrar su hito | El 2026-09-21 |
|---|---|---|
| `tiny-greedy` | 921 (713-1040) | 778 (479-904) |
| `small-v3-greedy` | 1365 (1293-1423) | 1425 (1367-1484) |
| `medium-v4-greedy` | 1504 (1446-1558) | 1535 (1476-1599) |
| `medium-v4-dpo-onpolicy-greedy` | 1560 (1500-1617) | 1632 (1567-1706) |
| `medium-v4-grpo-greedy` | 1572 (1512-1640) | 1577 (1526-1632) |

## Lo que no se puede decir de un modelo así

Este es el trozo del curso que más me costó escribir, y va en el vídeo tal cual.

- **«Juega a 1500 Elo».** Gana lo que ganaría un jugador de unos 1500 contra Stockfish limitado a
  0,1 segundos por jugada, con una máscara que rescata sus jugadas ilegales. Contra personas no se
  ha medido.
- **«El alineado es 97 puntos mejor».** Es restar dos ruidos de ±55. La diferencia medida es +57
  (36-79).
- **«Entiende ajedrez».** Predice el siguiente token con un 54 % de acierto sobre lo que jugaron
  humanos de 1800 en adelante. No tengo un instrumento que mida «entender».
- **«Reproduce el número».** Reproduce el intervalo. Dos corridas idénticas del reward model
  dieron 72,91 % y 74,21 %; dos escaleras idénticas, 1538 y 1538 un día y 1498 y 1558 otro.

## Un baseline que no entrené yo

Una tabla solo con mis modelos es una tabla contra mí mismo. Metí dos baselines, y los dos se
midieron con **mi** harness, no con el suyo: los mismos peldaños, la misma máscara, el mismo
criterio de puzles, el mismo día.

El Qwen3 de 4 000 millones de parámetros afinado con QLoRA sobre PGN (M4) es la fila más baja de
la tabla: 62,5 % de legalidad, 12,5 % de top-1, un Elo que solo se puede acotar por arriba. Un
vocabulario de jugadas y cinco millones de partidas compran más que cinco veces más parámetros
con unos miles de partidas encima.

El nanoGPT de ajedrez de Adam Karvonen (8 capas, 25,7 M, PGN carácter a carácter) lo reimplementé
sin depender de su repositorio y lo puse en la misma escalera, con su formato de prompt exacto y
una regla más dura que la suya ante una jugada ilegal (él reintenta cinco veces; yo pregunto una
y la cuento). Salió en 1328 (1250-1391), con un 99,6 % de
legalidad que supera a mi `small-v3`, y por debajo de `small-v3` en puzles difíciles y en Elo
(1425, 1367-1484). Es la fila que sitúa el proyecto: un modelo de 39 M con vocabulario de jugadas
gana a uno de 26 M carácter a carácter con tres veces más partidas, y el de 115 M ya no tiene con
quién compararse en la tabla.

## Lo que cuesta el int8

En el segundo módulo medí que el int8 de `medium-v4` elige la misma jugada que el modelo original
en el 95,1 % de las posiciones, y dejé la pregunta importante sin responder: ¿cuánto peor juega
el 4,9 % restante? La paridad no lo dice. Hay que jugar. Puse los tres grafos ONNX (fp32, fp16, int8) en la misma
escalera, con el rival por nodos porque ONNX Runtime juega en CPU: **1472, 1535 y 1524**, con
intervalos de unos ±55 que se superponen. El int8 no cuesta nada que 160 partidas puedan ver. Y el
fp32, que es la misma red que el checkpoint, salió 60 puntos por debajo del fp16: otra vez el suelo
del instrumento, en la misma tarde. Para ver el coste del int8 hará falta un enfrentamiento directo
fp16 contra int8 de varios cientos de partidas; con la escalera, no.

## Lo que queda

Veintiuna model cards regeneradas desde la tabla, cada una con la lección que la construyó, su
fila, su demo y el comando `rukh pull` que la trae. Una colección en Hugging Face igual al
catálogo. Y una demo con tres modos: jugar, una arena en la que dos etapas juegan entre sí en tu
navegador con la misma aritmética que mi harness, y los 150 puzles de la suite en vivo.

La fase 2 empieza donde esta termina: el modelo deja de ser lo único que hay y pasa a ser una
herramienta que algo más usa.

---

## Guion del vídeo (8-10 minutos)

1. **Tablero** (30 s): la demo jugando contra `medium-fp16`. «Esto es Rukh. Seis módulos, trece
   modelos, y hoy toca lo aburrido: medirlo todo otra vez.»
2. **La tabla** (2 min): `/proyecto/`, ordenar por Elo, por puzles, por legalidad. Las columnas se
   mueven por separado; señalar `dpo-onpolicy` (más Elo, menos top-1).
3. **Las tres cifras** (2 min): 1535, +57 (36-79), 0 y 17. Enseñar `LadderFloor`.
4. **Lo que se movió** (1,5 min): la tabla vieja frente a la nueva. «Nada se movió salvo lo que
   juega contra Stockfish, y eso se movió lo que su intervalo permite.»
5. **Lo que no se puede decir** (1,5 min): las cuatro frases, leídas.
6. **Los baselines** (1 min): Qwen y Karvonen, en mi harness.
7. **La arena** (1 min): `medium-fp16` contra `medium-dpo-fp16`, veinte partidas, y el número de
   partidas que haría falta para creerse algo.
8. **Cierre** (30 s): la colección, el curso, y la fase 2.
