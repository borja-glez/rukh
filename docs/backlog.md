# Backlog

Lo que se ha visto y no se hace ahora. Cada entrada: qué, por qué se aplaza, cuándo tocaría.
Se limpia al cerrar cada hito; lo que entra en un plan sale de aquí.

## Aplazado desde P0

- **Fetch real de datos** (`rukh data fetch` sin `--dry-run`): la ejecución existe (DuckDB
  `COPY` por mes + manifiesto) pero no se ha probado contra `hf://` con red. Se verifica y se
  ajusta en P1 con `limit` pequeño antes del volcado completo.
- **Filtro de plies** (`min_plies`): se aplica en P1 tras convertir `movetext` a UCI
  (`docs/spec/01`, pipeline paso 2), no en SQL.
- **Exclusión de variantes**: en P0 es `Event NOT ILIKE '%variant%'`; P1 la afina con la lista
  real de `Event` de Lichess.
- **Asset de Stockfish para Linux** en `scripts/get_stockfish.py`: definido
  (`stockfish-ubuntu-x86-64-avx2.tar`) pero no ejecutado en CI; la CI no necesita el motor.
- **OG por lección** en el curso (D-005), ver `docs/decisiones-de-ejecucion.md`.

## Aplazado desde P4

- **La clave del caché de evaluación es global y debería ser por suite.** `config_sha` mezcla
  ajustes que solo afectan a los puzles (`puzzles_use_header`) con otros que solo afectan a las
  partidas (`elo_games`, `rungs`, `elo_move_time`), así que cambiar uno invalida lo otro. Dos
  consecuencias medidas en este hito: el barrido por condición no puede reutilizar sus partidas de
  `@1800` para la evaluación canónica (doce minutos de Stockfish repetidos), y subir `elo_games` de
  20 a 40 tira las veinte partidas ya jugadas aunque la partida `rung:index` sea exactamente la
  misma —su semilla es `seed + index`, independiente del total—. **Cuándo:** P6, que es donde
  `rukh eval nightly` va a reconstruir la tabla entera y donde el ahorro se nota. **Coste de
  hacerlo:** invalida los 17 MB de `cache-greedy.sqlite` que ya tienen las etapas publicadas, así
  que conviene hacerlo junto a una tirada completa y no a mitad de un hito.

- **Las tablas de Markdown del curso no tienen contenedor con scroll.** Las tablas que escriben
  los componentes (`.table-wrap`) sí lo tienen; las que se escriben en MDX heredan `.prose table`
  y desbordan la página en móvil en cuanto pasan de cuatro columnas —medido en M4: 415 px de
  `scrollWidth` contra 390 px de ventana, con una tabla de cinco columnas y otra de seis—. Se ha
  resuelto estrechando esas dos tablas, que es lo correcto para el texto pero no para el problema.
  **Por qué se aplaza:** la solución general es un plugin `rehype` que envuelva cada `<table>`, y
  bajo Astro 7.3 los `markdown.rehypePlugins` corren sobre el procesador `unified` de
  `@astrojs/markdown-remark`, que ya no se instala por defecto desde que Sätteri es el procesador
  de Markdown; habilitarlo cambia el pipeline de Markdown de las cuatro lecciones publicadas.
  **Cuándo:** cuando toque revisar el pipeline del curso (P6 o el primer módulo que necesite una
  tabla ancha de verdad), con una prueba `e2e` de desbordamiento en las cuatro lecciones, no solo
  en la última.

- **Dos filas del encoder llevan el nombre de su directorio de corrida en la tabla pública.**
  `eval-encoder-heads-rank30-20260920-111151` y `eval-encoder-heads-rank80-20260920-111329` salen
  tal cual en `/proyecto/`, porque `--stage` no se pasó y el defecto es el nombre de la carpeta.
  Las mediciones son buenas; el nombre es ruido en una página pública, justo lo que D-099 acababa
  de limpiar por otra razón. **Por qué se aplaza:** renombrar una fila no existe —`rukh eval drop`
  solo retira— y volver a correrlas con `--stage` cuesta su evaluación entera; además son
  mediciones de P3 y tirarlas es decisión de quien las hizo. **Cuándo:** con la reconstrucción de
  la tabla de P6, o antes si se vuelven a evaluar las cabezas por cualquier otro motivo.

- **La escalera de Elo no es reproducible porque el rival va por tiempo.** Dos tiradas idénticas de
  `medium-elo` a `<w1800>` dieron 1498 y 1558 (D-107): `chess.engine.Limit(time=0.1)` más
  `UCI_LimitStrength` hacen que la semilla fije nuestro muestreo y no el suyo. El suelo de
  reproducibilidad queda en unos 40 Elo de una sigma, que es más de lo que separa a las condiciones
  contiguas de cualquier barrido. **La alternativa:** limitar por **nodos** en vez de por tiempo,
  que además deja de depender de lo ocupada que esté la máquina. **Por qué se aplaza:** cambia el
  rival, así que invalida todos los Elo publicados y obliga a recalibrar los ocho peldaños.
  **Cuándo:** P6, junto con la reconstrucción de la tabla, y midiendo antes cuánto se estrecha de
  verdad la reproducibilidad — que es el único motivo para pagar la recalibración.

## Aplazado desde P5

- **La galería usa `cp_best` del grupo también para la métrica de validación en las corridas
  cortas.** `best_available` arregla la métrica publicada (D-123), pero las tres corridas del
  barrido de temperatura se midieron antes y su columna de recompensa es la del grupo. No se
  rehacen: lo que esas corridas miden es la proporción de grupos planos, que no depende de la
  referencia. Si alguna vez se quiere la serie completa con la métrica buena, son doce minutos.
- **GRPO no aprende legalidad en la configuración por defecto.** Con `restrict_to_legal: true` la
  puerta nunca se dispara. `grpo-legality.yaml` existe para probar si GRPO puede devolver el peaje
  que DPO se gastó (D-121), y esa corrida se queda para cuando haya presupuesto de motor: el
  gradiente de legalidad es minúsculo porque solo el 0,25 % de las candidatas salen ilegales.
- **El criterio de +50 Elo se cumple en la estimación puntual y no en el extremo del intervalo.**
  Con 800 partidas agrupadas, `off-policy` da +67 (45 a 89). Llevarlo a 1 600 partidas bajaría el
  intervalo a unos ±16 y cruzaría el 50 por el extremo. Son siete minutos de máquina y se hace
  cuando haya una tanda libre.
- **`rukh eval match` no reutiliza partidas entre corridas.** Cada dirección juega sus 400 desde
  cero aunque el libro de aperturas sea el mismo. Un caché por `(modelo A, modelo B, apertura,
  color)` ahorraría la mitad al repetir una dirección, y es lo que hace falta para subir a 1 600
  partidas sin pagarlas enteras.
