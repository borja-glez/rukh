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
