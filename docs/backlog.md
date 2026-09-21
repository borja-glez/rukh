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
- ~~**El criterio de +50 Elo se cumple en la estimación puntual y no en el extremo del intervalo.**
  Llevarlo a 1 600 partidas cruzaría el 50 por el extremo.~~ **Hecho, y no cruzó**: 1 600 partidas
  dan +65,25 con el intervalo de **49,74** a 80,76. Se queda a 0,26 Elo y **no se juegan más**
  (D-127): añadir partidas hasta que el número cruce el umbral es ajustar el experimento al
  criterio.
- **`rukh eval match` no reutiliza partidas entre corridas.** Cada dirección juega sus 400 desde
  cero aunque el libro de aperturas sea el mismo. Un caché por `(modelo A, modelo B, apertura,
  color)` ahorraría la mitad al repetir una dirección, y es lo que hace falta para subir a 1 600
  partidas sin pagarlas enteras.

## Aplazado desde P6

- **Maia-2 como baseline condicionado por Elo.** `maia2` 0.11 fija `torch>=2.8,<2.9` frente al
  2.11 del proyecto; el diseño que cabe es un proceso hijo en su propio entorno de `uv`
  (`--isolated --with maia2 --with "torch==2.8.*"`) que recibe FEN y dos Elo por JSON lines y
  devuelve la jugada, implementando `Player.choose(board)` sin historial (D-138). Sus pesos vienen
  de Google Drive con `gdown`: fijar el id del fichero y cachearlo bajo `checkpoints/`. **Cuándo:**
  cuando la pregunta del condicionado por Elo (M4) vuelva a abrirse, o si la fase 2 necesita un
  rival humano-como a distintas fuerzas.
