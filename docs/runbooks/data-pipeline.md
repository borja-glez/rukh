# Ejecutar el pipeline de datos (P1)

## Cuándo

Al regenerar los datasets `chorcat/rukh-*` y el tokenizador desde cero, o al añadir meses.
Requisitos: `uv sync --extra cu128 --group dev`, unos 40 GB libres en `E:` (ver `docs/spec/01`),
red para `fetch`, `evals`, `puzzles` y `elite`, y `HF_TOKEN` en el entorno solo para `publish`.
Todo se ejecuta desde `rukh/` con `uv run rukh data <paso>`. Cada paso lee su sección de
`configs/data/pipeline.yaml` (`--config` para otro fichero) y deja un `manifest.json` con
filtros, conteos y sha256 en su carpeta de salida. Ningún paso toca la red salvo los marcados.

Tiempos medidos o estimados en la máquina de referencia (RTX 5090, 64 GB, 15 procesos).

## Comandos, en orden

| # | Paso | Comando | Red | Tiempo | Salida |
|---|---|---|---|---|---|
| 1 | Recorte de Lichess | `uv run rukh data fetch --config configs/data/lichess-2025-01-02.yaml` | sí | 15-30 min | `data/raw/year=YYYY/month=MM/games.parquet`, `data/raw/manifest.json` |
| 2 | SAN → UCI | `uv run rukh data uci` | no | 10-20 min (6 M partidas, 15 procesos) | `data/uci/year=YYYY/month=MM/games.parquet` (`game_id, uci, n_plies, white_elo, black_elo, result, time_control, utc_date, eco, month`), `data/uci/manifest.json` |
| 3 | Tokenizador y fixture | `uv run rukh data tokenize --scheme uci --export-fixture` | no | segundos | `artifacts/tokenizer/vocab.json`, `artifacts/tokenizer/fixtures/games.json` |
| 4 | BPE real y estadísticas | `uv run rukh data tokenize --scheme bpe --stats --export-fixture` | no | 2-4 min (200 000 partidas para el BPE, 20 000 para las estadísticas) | `artifacts/tokenizer/bpe.json` (sobrescribe el de la fixture), `artifacts/web/tokenizer-stats.json`, fixture con `bpe_ids` nuevos |
| 5 | Empaquetado | `uv run rukh data tokenize --scheme uci --pack` (y `--scheme bpe --pack`; `--scheme san --pack` es lento) | no | uci/bpe: 2-5 min por mes; san: ~1 ms por partida (≈ 1,5 h para 6 M) | `data/tokens/<esquema>/{train,val}/{tokens.npy,starts.npy,meta.json}` |
| 6 | Posiciones | `uv run rukh data positions` | no | 5-10 min (300 000 partidas) | `data/positions/positions.parquet`, `data/positions/parts/` |
| 7 | Evaluaciones | `uv run rukh data evals` (en segundo plano) | sí | 1-3 h: 20 ficheros remotos de ~2 GB cada uno | `data/evals/part-NN.parquet`, `data/evals/positions-eval.parquet` |
| 8 | Puzles | `uv run rukh data puzzles` | sí | 3-6 min (877 MB) | `data/puzzles/puzzles.parquet` |
| 9 | Pares DPO | `uv run rukh data pairs` | no | 3-8 min | `data/pairs/pairs.parquet` |
| 10 | Elite | `uv run rukh data elite` | sí | 5-10 min (2 zips de ~80 MB + conversión) | `data/elite/games.parquet`, zips y PGN extraídos |
| 11 | Tramos de Elo | `uv run rukh data elo-bins` | no | 1-2 min | `data/elo-bins/games.parquet` |
| 12 | Publicación | `uv run rukh data publish --name <nombre> --dry-run` y después sin `--dry-run` | solo sin `--dry-run` | según subida (2-3 GB en total) | `data/publish/<nombre>/README.md` (card) y el repo en el Hub |

Nombres válidos para `publish` (`--list` los imprime): `rukh-games-1800`, `rukh-games-elite`,
`rukh-elo-bins`, `rukh-positions-eval`, `rukh-puzzles-split`, `rukh-pairs-dpo` y `rukh-tokenizer`.

Notas:

- El paso 3 se puede ejecutar antes que el 2: el vocabulario fijo no depende de datos y la fixture
  sale de `tests/fixtures/games.pgn`. El `bpe.json` que hay en git está entrenado solo con esas 20
  partidas (600 tokens) para que la paridad Python/TypeScript funcione sin datos; el paso 4 lo
  sustituye por el de 4 096 tokens y regenera `bpe_ids` en la fixture. Tras el paso 4 hay que
  volver a ejecutar `pnpm sync:tokenizer` en `rukh-web` y `rukh-lab`.
- `--games <parquet>` en `tokenize` cambia el mes con el que se entrena el BPE y se calculan las
  estadísticas (por defecto `data/uci/year=2025/month=01/games.parquet`).
- `workers: 0` en `uci`, `positions` y `elite` significa `cpu_count() - 1`. En Windows los procesos
  usan `spawn`: los comandos funcionan desde el CLI o desde un `.py`, nunca desde un script leído por
  stdin.
- Los pasos 9 y 12 (`rukh-positions-eval`, `rukh-pairs-dpo`) necesitan `positions-eval.parquet`; si
  `evals` no ha terminado, el consolidado cubre solo las partes descargadas (ver abajo).

## Reanudar `evals`

`rukh data evals` escribe una parte por fichero remoto (`data/evals/part-00.parquet` …
`part-19.parquet`) y salta las que ya existen, así que se puede interrumpir y relanzar:

```powershell
uv run rukh data evals                # todo lo que falte (0-19)
uv run rukh data evals --files 0-4    # solo un rango
uv run rukh data evals --files 7,9    # ficheros sueltos
```

Cada ejecución termina consolidando `positions-eval.parquet` con las partes presentes y anota en
el manifiesto `files_present`, `files_fetched_now` y `coverage` (posiciones con evaluación entre
las posiciones muestreadas). Una parte a medias se escribe primero como `.parquet.tmp` y solo se
renombra al terminar: si hay un `.tmp` huérfano, se borra y se relanza. Para forzar la descarga
de una parte, borrar su `part-NN.parquet`.

## Verificación

- `data/<paso>/manifest.json` existe y sus `counts` son plausibles: `uci` conserva del orden del
  95 % de las filas (`illegal` y `short` pequeños); `positions.distinct` < `positions.positions`;
  `evals.coverage` ≥ 0,6 con los 20 ficheros; `pairs` con las tres fases al mismo tamaño; `puzzles`
  con 2 000 `test` por tramo; `elo-bins` sin tramo por encima de `n_per_bin`.
- `artifacts/web/tokenizer-stats.json`: `uci.pct_le_max_len` cerca del 95 % y `bpe` con fusiones
  de varias jugadas en `bpe_longest_tokens`.
- `uv run pytest -m unit -q` verde (no usa `data/`).
- `pnpm test` en `rukh-web` y `rukh-lab` tras `pnpm sync:tokenizer` (paridad con la fixture).
- Tras `publish`, abrir `https://huggingface.co/datasets/chorcat/<nombre>` y comprobar la card
  (licencia `cc0-1.0`, tabla de columnas, bloque de filtros).

## Si falla

- `uv sync` o `uv run` fallan con "rukh.exe ... utilizado por otro proceso": hay un `rukh data`
  corriendo en segundo plano; usar `uv run --no-sync` hasta que termine.
- `fetch`/`evals` con errores de red de DuckDB (`httpfs`): relanzar; `evals` reanuda por partes,
  `fetch` repite el mes entero.
- `evals` con `coverage` bajo: comprobar que `remote_pattern` en `pipeline.yaml` apunta a los
  ficheros reales del dataset (`data/data_0000.parquet` … `data_0019.parquet`) y que
  `positions.parquet` usa la misma normalización de FEN (cuatro campos, casilla al paso solo si la
  captura es legal).
- `publish` sin `HF_TOKEN`: `huggingface_hub` devuelve 401; exportar el token en la sesión o usar
  `--dry-run` para revisar la card antes.
- `tokenize --scheme san --pack` demasiado lento: es conversión UCI→SAN con python-chess en un solo
  proceso; ejecutarlo en segundo plano o solo sobre el mes de validación.
