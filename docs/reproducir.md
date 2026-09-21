# Reproducir el curso módulo a módulo

Cada módulo del curso construye algo sobre lo que dejó el anterior: M1 deja datos, M2 deja un
decoder, M3 un encoder, M4 una familia afinada sobre `medium-v4`, M5 tres modelos alineados sobre
el mismo `medium-v4`. Este documento dice, para cada módulo, **qué tiene que haber en disco para
empezarlo, cómo conseguirlo si te saltaste lo anterior, y qué comandos lo producen todo**, con lo
que tardaron en la máquina de referencia (RTX 5090, 64 GB, Windows 11).

Tres reglas que valen para todos los módulos:

1. **El código es siempre `main`.** No hay una rama por módulo: una rama congelada al cerrar M2
   llevaría la escalera de Elo con cuatro peldaños mal etiquetados (D-070), que se corrigió después
   y cambió el número publicado de todos los modelos. Los hitos están marcados con etiquetas
   (`git tag -l 'p*'`) por si quieres ver cómo estaba el repo el día que se cerró cada uno; para
   ejecutar, quédate en `main`.
2. **Los checkpoints se nombran por su corrida, sin fecha.** Las configs y las lecciones dicen
   `checkpoints/medium-v4/best.pt`. El bucle de entrenamiento escribe
   `checkpoints/medium-v4-20260919-174623/` (con `unique_run_name`), y `rukh pull medium-v4` escribe
   `checkpoints/medium-v4/best.pt`. Los dos valen: cuando la carpeta sin fecha no existe, cualquier
   comando o config busca la corrida con fecha más reciente de ese nombre. Lo mismo para carpetas
   de adaptadores (`checkpoints/lora-e4`).
3. **`rukh pull --module mN` trae del Hub lo que el módulo N necesita para empezar.** Modelos a
   `checkpoints/<corrida>/…` y datasets a su `data/…`. Lo que ya está en disco no se toca
   (`--force` lo sustituye). `rukh pull --list` imprime el catálogo entero.

Sobre la exactitud: los datos, las configs y las semillas son los mismos, y los comandos son los
que se ejecutaron. Lo que **no** se repite exactamente es la GPU: dos corridas idénticas del
reward model dieron 72,91 % y 74,21 % (D-113), y dos escaleras idénticas 1498 y 1558 (D-107). Cada
lección publica sus números con ese margen delante; si el tuyo cae dentro, has reproducido el
módulo.

## M0 · Taller

Nada que descargar del Hub. `uv sync --extra cu128 --group dev`, `rukh info`,
`scripts/get_stockfish.py`, `rukh engine check`, `rukh mlflow ui`. El recorte de datos se lanza al
final del módulo:

| Comando | Tiempo | Deja |
|---|---|---|
| `uv run rukh data fetch --config configs/data/lichess-2025-01-02.yaml` | 15-30 min, red | `data/raw/year=2025/month={01,02}/games.parquet` (6 M partidas 1800+) |

## M1 · Datos y tokenización

**Punto de partida:** `data/raw/` de M0, o nada: `rukh pull rukh-games-1800` deja directamente el
resultado del paso 2 en `data/uci/` (966 MB) y el lab 2 queda hecho.

| # | Comando | Tiempo | Deja |
|---|---|---|---|
| 1 | `uv run python labs/m1/explore.py` | s | consultas DuckDB sobre `data/raw` |
| 2 | `uv run rukh data uci` | 10-20 min | `data/uci/` (5 896 388 partidas) |
| 3 | `uv run rukh data tokenize --scheme bpe --pack` | 3-8 min | `artifacts/tokenizer/bpe.json`, `data/tokens/bpe/` |
| 4 | `uv run rukh data tokenize --scheme san --pack` | ~1,5 h (opcional) | `data/tokens/san/` |
| 5 | `uv run rukh data tokenize --scheme uci --stats --export-fixture --pack` | 5-10 min | `data/tokens/uci/{train,val}`, `artifacts/web/tokenizer-stats.json`, fixture |
| 6 | `uv run rukh data positions` | 5-10 min | `data/positions/positions.parquet` |
| 7 | `uv run rukh data evals` | 1-3 h, red (reanudable) | `data/evals/positions-eval.parquet` |
| 8 | `uv run rukh data puzzles` | 3-6 min, red | `data/puzzles/puzzles.parquet` |
| 9 | `uv run rukh data pairs` | 10 s | `data/pairs/pairs.parquet`, `data/pairs/dpo-prompts.parquet` |
| 10 | `uv run rukh data elite` | 5-10 min, red | `data/elite/games.parquet` (dos meses; M2 lo amplía a 44) |
| 11 | `uv run rukh data elo-bins` | 1-2 min | `data/elo-bins/games.parquet` |
| 12 | `uv run python labs/m1/loader_check.py`, `labs/m1/bpe_merges.py` | s | las salidas de los labs 4 y 5 |

Atajos del Hub: `rukh-positions-eval` (paso 7, el caro), `rukh-puzzles-split` (8), `rukh-pairs-dpo`
(9, con `dpo-prompts.parquet`), `rukh-tokenizer` (3 y 5, el vocabulario y el BPE; los tokens
empaquetados no se publican porque salen de `tokenize --pack` en minutos).

## M2 · El decoder

**Punto de partida:** `data/tokens/uci/` (M1 paso 5), `data/puzzles/` y `data/evals/` para la suite
completa. `rukh pull --module m2` trae `rukh-games-1800`, `rukh-tokenizer`, `rukh-puzzles-split`,
`rukh-positions-eval`, `rukh-games-elite` y los dos decoders publicados; los tokens se empaquetan
con el paso 5 de M1.

| # | Comando | Tiempo | Deja |
|---|---|---|---|
| 1 | `uv run python labs/m2/params.py`, `labs/m2/causal_mask.py` | s | los números del lab 1 y 2 |
| 2 | `uv run rukh train --config configs/train/tiny.yaml` | 3 min | `checkpoints/tiny-*/best.pt` |
| 3 | `uv run rukh eval --model checkpoints/tiny/best.pt --suite quick` | ~10 min | `artifacts/eval/tiny/` |
| 4 | `uv run rukh train --config configs/train/small.yaml` | 42 min | `checkpoints/small-*/best.pt` |
| 5 | `uv run rukh eval --model checkpoints/small/best.pt --config configs/eval/greedy.yaml --stage small-greedy` | ~30 min (160 partidas) | `artifacts/eval/small-greedy/`, fila en `artifacts/web/results.json` |
| 6 | `uv run rukh play --ckpt checkpoints/small/best.pt --games 1` | s | una partida en consola |
| 7 | `uv run rukh export --ckpt checkpoints/small/best.pt --out artifacts/onnx/small --fp16 --int8 --check-parity` | ~12 min | `model{,-fp16,-int8}.onnx`, `parity.json` |
| 8 | `uv run python labs/m2/attention_export.py`, `labs/m2/replay_export.py --with-legality` | min | `artifacts/web/attention.json`, `training-replay.json` |
| 9 | `uv run rukh publish model --ckpt checkpoints/small/best.pt --repo chorcat/rukh-small --stage small-greedy --onnx artifacts/onnx/small --dry-run` | min | `artifacts/publish/chorcat/rukh-small/` |

**La cuarta parte de M2 (más datos, no más red)** es la que produce el modelo del que parten M4 y
M5, y no está en el spec original: salió de medir por qué `small` no llegaba a donde se creía.

| # | Comando | Tiempo | Deja |
|---|---|---|---|
| 10 | `uv run rukh data elite --config configs/data/pipeline-elite44.yaml` | ~1,5 h, red (44 zips) | `data/elite/games.parquet` (13 146 352 partidas) — o `rukh pull rukh-games-elite` |
| 11 | `uv run rukh data tokenize --config configs/data/pipeline-v4.yaml --scheme uci --pack` | 5 min | `data/tokens-v4/uci/` (1 681 M tokens, 3,3 GB) |
| 12 | `uv run rukh train --config configs/train/medium-v4.yaml` | 4 h 3 min | `checkpoints/medium-v4-*/best.pt` — o `rukh pull medium-v4` |
| 13 | `uv run rukh eval --model checkpoints/medium-v4/best.pt --config configs/eval/greedy.yaml --stage medium-v4-greedy` | ~30 min | 1504 Elo (IC 1446-1558), 99,8 % legal |

Las corridas intermedias de esa parte (`small-v2.yaml` sobre `pipeline-v2.yaml`, `small-v3.yaml`
sobre `pipeline-v3.yaml`) están en el repo y se ejecutan igual; `rukh-small` en el Hub es
`small-v3`.

## M3 · El encoder

**Punto de partida:** `data/tokens/uci/` (M1), `data/evals/positions-eval.parquet` y `data/uci/`
(las etiquetas y los prefijos). `rukh pull --module m3` trae además `encoder-mmm-v4` y
`encoder-v4`, para saltarse el preentrenamiento grande o llegar directo al lab 4.

| # | Comando | Tiempo | Deja |
|---|---|---|---|
| 1 | `uv run python labs/m3/bidirectional.py` | s | las tres comprobaciones del lab 1 |
| 2 | `uv run rukh train encoder --config configs/train/encoder-mmm.yaml` | 15 min | `checkpoints/encoder-mmm-*/best.pt` (15 M, 75,2 % de acierto en jugadas tapadas) |
| 3 | `uv run rukh train heads --config configs/train/encoder-heads-moves.yaml --mode probe` (y `last-n`, `full`) | 1-3 min cada uno | `checkpoints/encoder-heads-moves-*/best.pt` |
| 4 | `uv run rukh train heads --config configs/train/encoder-heads-moves.yaml --mode last-n --curve` | ~6 min | la curva 10/25/50/100 % dentro del checkpoint |
| 5 | `uv run rukh eval encoder --model checkpoints/encoder-heads-moves/best.pt --stage encoder` | ~5 min | F1 frente a la heurística, correlaciones |
| 6 | `uv run rukh train encoder --config configs/train/encoder-mmm-v4.yaml` | 40 min | `checkpoints/encoder-mmm-v4-*/best.pt` (39 M, 81,4 %) — o `rukh pull encoder-mmm-v4` |
| 7 | `uv run rukh train heads --config configs/train/encoder-heads-v4.yaml --mode last-n` | 3 min | `checkpoints/encoder-heads-v4-*/best.pt` — o `rukh pull encoder-v4` |
| 8 | `uv run rukh eval encoder --model checkpoints/encoder-heads-v4/step-4000.pt --stage encoder-v4` | ~5 min | 18,6 % de F1 (+9,7 sobre la línea base), Spearman 0,666 |
| 9 | `uv run rukh export --ckpt checkpoints/encoder-heads-v4/step-4000.pt --out artifacts/onnx/encoder --kind encoder --fp16 --int8 --check-parity` | ~5 min | el encoder que sirve la demo |
| 10 | `uv run rukh encoder embed --positions data/evals/positions-eval.parquet --out artifacts/embeddings/positions.npy --ckpt checkpoints/encoder-heads-v4/step-4000.pt` | min | embeddings para la fase 2 |
| 11 | `uv run python labs/m3/value_bar_export.py` | min | `artifacts/web/value-bar.json` |

## M4 · Fine-tuning e instrucción

**Punto de partida:** `checkpoints/medium-v4/best.pt` (M2, parte 4, o `rukh pull medium-v4`),
`data/uci/`, `data/elite/games.parquet` (44 meses) y el tokenizador. `rukh pull --module m4` lo
trae todo, incluidos los cinco resultados del módulo por si solo quieres medirlos.

Los comandos, en orden y con tiempos, están en
[`runbooks/afinado-y-adaptadores.md`](runbooks/afinado-y-adaptadores.md): el tramo bajo de Elo,
el corpus plano, los dos afinados, los dos adaptadores, el barrido por condición con su control,
Qwen3 con QLoRA, las exportaciones y la publicación.

## M5 · Alineamiento

**Punto de partida:** `checkpoints/medium-v4/best.pt`, `data/pairs/dpo-prompts.parquet` (M1
paso 9, o `rukh pull rukh-pairs-dpo`), `data/puzzles/` y `data/uci/` para la suite, Stockfish.
`rukh pull --module m5` trae eso y los tres modelos alineados.

Los comandos, en orden, están en [`runbooks/alineamiento.md`](runbooks/alineamiento.md): el
instrumento y su control, el reward model, los pares on-policy, los dos DPO, GRPO, los
enfrentamientos en las dos direcciones, la galería y la publicación.

## M6 · Evaluar, exportar, publicar

**Punto de partida:** todo lo que los cinco módulos anteriores publicaron. `rukh pull --module m6`
trae los doce modelos a las rutas que leen las configs, más las partidas, los puzles y las
posiciones de la suite. Hace falta Stockfish. Ningún paso entrena nada; todos miden, exportan o
publican.

| Paso | Comando | Tarda | Deja |
|---|---|---|---|
| 1 | `uv run python labs/m6/ladder_check.py --games 40 --nodes 200000` | ~25 min | los ocho peldaños medidos contra el ancla, por nodos |
| 2 | `uv run rukh eval --model checkpoints/medium-v4/best.pt --config configs/eval/ladder-time.yaml --stage ladder-time-a --no-cache` (y `-b`; luego `ladder-nodes.yaml`, `-a` y `-b`) | ~12 min cada una | las cuatro tiradas del suelo del instrumento (D-137) |
| 3 | `uv run python labs/m6/ladder_floor_export.py --decision nodes` | s | `artifacts/web/ladder-floor.json` |
| 4 | `uv run rukh eval nightly` | horas | la tabla entera: `artifacts/eval/<etapa>/`, `artifacts/web/results.json`, `docs/benchmarks.md`, `artifacts/eval/nightly.json` |
| 5 | `uv run rukh eval karvonen` | ~20 min | el baseline público en la tabla (`karvonen-8l`) |
| 6 | `uv run python labs/m6/parity_cost.py --onnx artifacts/onnx/medium-v4` | ~1 h (ORT en CPU) | fp32, fp16 e int8 del mismo modelo en la misma escalera; `artifacts/web/parity-cost.json` |
| 7 | `uv run python labs/m6/puzzles_export.py` | s | `artifacts/web/puzzles.json`, los 150 puzles de la demo |
| 8 | `uv run rukh publish cards --dry-run` y después sin `--dry-run` | ~3 min | cada `README.md` del Hub regenerado desde la tabla y el curso |
| 9 | `uv run rukh publish collection` | s | la colección `Rukh` del Hub, igual al catálogo de `rukh pull` |

`rukh eval nightly --dry-run` imprime el plan (qué etapa, qué fichero, qué config) sin medir nada;
`--only tiny,small` acota; `--no-pull` no descarga lo que falte. El nightly fusiona los adaptadores
LoRA sobre su base antes de medirlos y retira las filas que ninguna etapa del catálogo reclama.

## Qué hay en el Hub y a qué corrida corresponde

| `rukh pull` | Repositorio | Escribe | Es |
|---|---|---|---|
| `tiny` | `chorcat/rukh-tiny` | `checkpoints/tiny/best.pt` | M2 lab 3 |
| `small` | `chorcat/rukh-small` | `checkpoints/small-v3/best.pt` | `small-v3`, M2 parte 4 |
| `medium-v4` | `chorcat/rukh-medium` | `checkpoints/medium-v4/best.pt` | M2 parte 4; base de M4 y M5 |
| `encoder-mmm-v4` | `chorcat/rukh-encoder-mmm` | `checkpoints/encoder-mmm-v4/best.pt` | M3 paso 6 |
| `encoder-v4` | `chorcat/rukh-encoder` | `checkpoints/encoder-heads-v4/step-4000.pt` | M3 paso 7 |
| `medium-elo`, `medium-masters` | `chorcat/rukh-medium-{elo,masters}` | `checkpoints/<nombre>/step-3800.pt` | M4 pasos 4 y 5 |
| `lora-e4`, `lora-d4` | `chorcat/rukh-lora-{e4,d4}` | `checkpoints/lora-{e4,d4}/adapter.safetensors` | M4 paso 6 |
| `qwen3-pgn-qlora` | `chorcat/rukh-qwen3-pgn-qlora` | `checkpoints/qwen3-pgn-qlora/` | M4 lab 8 |
| `rm` | `chorcat/rukh-rm` | `checkpoints/rm/reward.pt` | M5 paso 3 |
| `medium-v4-dpo-onpolicy` | `chorcat/rukh-medium-dpo` | `checkpoints/medium-v4-dpo-onpolicy/dpo.pt` | M5 paso 6 |
| `medium-v4-grpo` | `chorcat/rukh-medium-grpo` | `checkpoints/medium-v4-grpo/grpo.pt` | M5 paso 9 (la tasa lenta, D-126) |
| `rukh-games-1800` … `rukh-pairs-onpolicy` | `chorcat/<nombre>` (datasets) | su `data/…` | M1, M2 parte 4 y M5 |

Lo que **no** está en el Hub y se regenera en minutos: los tokens empaquetados (`data/tokens*`),
el tramo bajo de Elo y su corpus plano (M4, pasos 1-3), las rebanadas de estilo y el PGN de Qwen
(M4, 3d-3e), el DPO fuera de política (M5, paso 5) y los pares emparejados
`data/pairs-offpolicy-matched/` que el lab 6 de M5 construye a partir de `dpo-prompts.parquet`.
