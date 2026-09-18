# Decisiones y desviaciones de ejecución

Registro vivo, en orden cronológico. Cada entrada: qué se decidió, por qué, qué cuesta si está mal.
Las decisiones de diseño previas viven en `docs/spec/08-riesgos-y-decisiones.md`; aquí solo lo que
se decide o se desvía durante la ejecución.

## P0 · Scaffold (2026-09-18)

### D-001 · Commits en inglés con Conventional Commits
- **Qué:** los mensajes de commit de los tres repos van en inglés siguiendo Conventional Commits
  (`feat:`, `fix:`, `docs:`, `ci:`, `chore:`, `test:`, `refactor:`), sin coautoría ni referencias a
  herramientas o sesiones.
- **Por qué:** orden explícita de Borja al lanzar `/goal` el 2026-09-18; prevalece sobre GOAL.md,
  CLAUDE.md y `docs/spec/07` (que decían "en español, imperativo"). Los docs llevan nota fechada.
- **Si está mal:** coste nulo de revertir (solo afecta a mensajes futuros).

### D-002 · TypeScript fijado a `^6`
- **Qué:** `typescript@^6` en las dos webs aunque npm ya publica 7.0.2.
- **Por qué:** `docs/spec/04` lo exige ("TypeScript 6, nunca 7"); el portfolio de referencia usa 6.
- **Si está mal:** subir la versión es un cambio de una línea y `astro check`.

### D-003 · Stockfish 19 `windows-x86-64-universal` descargado por script
- **Qué:** `scripts/get_stockfish.py` descarga el asset `stockfish-windows-x86-64-universal.zip`
  de la release `sf_19` a `tools/stockfish/` (gitignored). No se añade al PATH del sistema;
  `rukh.engine.find_stockfish()` lo localiza (o `RUKH_STOCKFISH`).
- **Por qué:** no había Stockfish en la máquina; sf_19 no publica un asset `avx2` separado (el
  binario universal elige la mejor variante en tiempo de ejecución).
- **Si está mal:** cambiar la URL del script.

### D-004 · Capturas E2E como artefactos de CI, no comparación de píxeles
- **Qué:** los E2E de la demo en los tres viewports comprueban invariantes (sin scroll horizontal,
  tablero cuadrado ≤ 640 px, panel a la derecha/debajo, objetivos ≥ 44 px) y guardan capturas que la
  CI sube como artefacto. No se hace `toHaveScreenshot`.
- **Por qué:** las baselines de píxeles difieren entre Windows (local) y el runner Linux; mantener
  dos juegos de baselines en P0 es coste sin valor. `docs/spec/05` decía "capturas comparadas".
- **Si está mal:** añadir `toHaveScreenshot` con baselines generadas en el contenedor de Playwright.

### D-005 · OG por lección aplazado
- **Qué:** en P0 el curso usa una imagen OG estática; las OG por lección con satori + resvg se hacen
  cuando haya más de una lección (P1 o P2).
- **Por qué:** reducir el alcance de P0 a lo que pide GOAL.md.
- **Si está mal:** es una ruta `/og/[...].png` aislada.

### D-006 · Repos remotos, DNS y Dokploy los crea Borja
- **Qué:** todo P0 se hace en local; al final se pide a Borja crear `borja-glez/{rukh,rukh-lab,rukh-web}`,
  los registros DNS y las dos apps en Dokploy. `gh` está autenticado como `borja-glez`, así que se
  ofrece crear los repos desde aquí con su OK.
- **Por qué:** punto de intervención marcado en GOAL.md.

### D-007 · Python 3.12 fijado con `uv` (el del sistema es 3.13)
- **Qué:** `.python-version` = 3.12 y `requires-python = ">=3.12,<3.13"`.
- **Por qué:** `docs/spec/02` fija 3.12 y las ruedas de torch cu128 verificadas son cp312.
- **Si está mal:** cambiar el rango y regenerar `uv.lock`.

### D-008 · El plan de P0 quedó en el commit inicial de `main`
- **Qué:** `docs/plans/2026-09-18-p0-scaffold.md` entró en `chore: initial commit` de `rukh` en vez
  de en la rama. Se deja así; no se reescribe historia.
- **Si está mal:** ninguno; es documentación.

### D-009 · Patrones de `.gitignore` anclados a la raíz
- **Qué:** `data/`, `mlruns/`, `tools/` y `checkpoints/` pasan a `/data/`, `/mlruns/`, `/tools/` y
  `/checkpoints/`.
- **Por qué:** el patrón sin anclar ocultaba `src/rukh/data/` y `configs/data/` (git no los veía y
  `ruff` tampoco los revisaba). Detectado en la tarea 2 de P0.
- **Si está mal:** ninguno; solo cambia qué carpetas de primer nivel se ignoran.

### D-010 · `rukh.data.fetch.run` ejecuta la descarga en vez de dejar un marcador
- **Qué:** con `dry_run=False`, `run` hace `COPY` con DuckDB por mes a
  `<out_dir>/year=YYYY/month=MM/games.parquet`, cuenta filas, calcula sha256 y escribe
  `manifest.json`. `limit` se aplica por mes. Probado con un parquet local; no se ha ejecutado
  contra `hf://` en P0 (queda en `docs/backlog.md` para P1).
- **Por qué:** el plan pedía la interfaz `run(cfg, dry_run)` y la regla de no dejar marcadores;
  la implementación es pequeña y coincide con el paso 1 del pipeline de `docs/spec/01`.
- **Si está mal:** P1 la sustituye o la ajusta con datos reales.

### D-011 · `TRY_CAST` en el filtro de control de tiempo
- **Qué:** el filtro de `min_base_seconds` usa `TRY_CAST(split_part(TimeControl, '+', 1) AS INTEGER)`
  en vez del `CAST` que dicen literalmente `docs/spec/01` y el plan de P0. Las filas cuyo base no es
  un entero (partidas por correspondencia, que Lichess exporta con `TimeControl = '-'`) se descartan
  en silencio en lugar de abortar la consulta entera.
- **Por qué:** detectado en la ola de correcciones de P0: con `CAST`, un solo mes con partidas por
  correspondencia hacía fallar todo el `COPY`. Hay test unitario con un parquet local que incluye
  una fila con `'-'`.
- **Si está mal:** si en P1 hace falta conservar esas partidas, se cambia el predicado; el manifest
  sigue registrando `min_base_seconds`.

### D-012 · `hive_partitioning = false`, rutas relativas en el manifest y `min_plies_deferred`
- **Qué:** (1) `read_parquet(..., hive_partitioning = false)`: DuckDB detectaba `year=YYYY/month=MM`
  en la ruta e inyectaba columnas `year` y `month`, y el alias explícito `'YYYY-MM' AS month`
  acababa como `month_1`; ahora solo existe la columna `month` del alias. (2) `FetchPlan.out_paths`,
  `FetchPlan.manifest_path` y `Manifest.files[].path` son rutas relativas a `out_dir` en forma
  POSIX (`year=2025/month=01/games.parquet`); `FetchPlan.out_dir` lleva la ruta absoluta una sola
  vez y el `--dry-run` la imprime en su propia línea. (3) El manifest registra `min_plies_deferred`
  en lugar de `min_plies`, porque este paso no filtra por plies: se hace en P1 tras convertir
  `movetext` a UCI (ver `docs/backlog.md`).
- **Por qué:** ola de correcciones de P0. El manifest es el registro de procedencia y no debe
  contener rutas de máquina ni afirmar filtros que no se aplicaron. El test unitario de `run`
  escribe el parquet de origen bajo `year=2025/month=01/` para ejercitar el layout real.
- **Si está mal:** cambiar el alias o el formato de rutas es un cambio local en
  `src/rukh/data/fetch.py`; los manifests escritos en P0 no se han publicado.
