# Afinar el decoder y servir adaptadores de estilo (P4)

## Cuándo

Al rehacer la familia de M4 —el modelo condicionado por Elo, el de maestros y los dos adaptadores
de estilo—, o al entrenar un adaptador nuevo. También al exportar el grafo que permite cambiar de
estilo en el navegador sin descargar otro modelo.

Requisitos: `uv sync --extra cu128 --extra hf --group dev`, el corpus de P1 en disco, Stockfish en
`tools/stockfish/` (o `RUKH_STOCKFISH`), y `HF_TOKEN` solo para los pasos de `publish`. Todo se
ejecuta desde `rukh/`.

**Regla que no se salta:** ningún trabajo pesado de CPU mientras corre una escalera de Elo. El
rival tiene 0,1 s por jugada, así que robarle CPU lo debilita y el Elo del modelo sale inflado. Las
exportaciones esperan a que no haya ninguna evaluación en marcha, y entre trabajos largos se dejan
15-20 minutos de descanso.

## Comandos, en orden

| # | Paso | Comando | Tiempo | Salida |
|---|---|---|---|---|
| 1 | Tramo bajo de Elo | `uv run rukh data fetch --config configs/data/lichess-low-elo.yaml` | ~2 min | `data/raw-low/...` |
| 2 | Corpus plano por bandas | `uv run rukh data elo-bins --config configs/data/pipeline-elo.yaml` | ~10 min | `data/elo-bins-v2/`, `manifest.json` |
| 3 | Tokenizar | `uv run rukh data pack --tokens data/tokens-elo` | ~5 min | `data/tokens-elo/uci/{train,val}` |
| 4 | Afinado condicionado | `uv run rukh train --config configs/train/medium-elo.yaml` | ~30 min | `checkpoints/medium-elo-*/step-3800.pt` |
| 5 | Afinado de maestros | `uv run rukh train --config configs/train/medium-masters.yaml` | ~30 min | `checkpoints/medium-masters-*/step-3800.pt` |
| 6 | Adaptadores de estilo | `uv run rukh train --config configs/train/lora-e4.yaml` (y `lora-d4.yaml`) | ~15 min cada uno | `checkpoints/lora-*/adapter.safetensors` |
| 7 | Barrido por condición | `uv run rukh eval sweep --model <ckpt> --elos 1200,1500,1800,2000,2100,2400 --config configs/eval/greedy-sweep.yaml --stage medium-elo` | ~26 min por condición | `artifacts/eval/medium-elo-elo-sweep/` |
| 8 | Control sobre el modelo base | el mismo comando con `--model <base>` y `--elos 1200,2100` | ~52 min | `artifacts/eval/medium-v4-elo-sweep/` |
| 9 | Evaluación canónica de cada etapa | `uv run rukh eval --model <ckpt> --config configs/eval/greedy.yaml --stage <nombre>` | ~26 min | `artifacts/eval/<nombre>/`, fila en `artifacts/web/results.json` |
| 10 | Qwen por el mismo harness | `uv run rukh eval qwen --adapter checkpoints/qwen3-pgn-qlora --config configs/eval/greedy.yaml` | ~40 min | `artifacts/eval/qwen3-pgn-qlora/` |
| 11 | Tabla de benchmarks | `uv run rukh eval benchmarks` | segundos | `docs/benchmarks.md` |
| 12 | Exportar los modelos | `uv run rukh export --ckpt <ckpt> --out artifacts/onnx/<nombre> --fp16 --int8 --check-parity` | ~12 min cada uno | `model{,-fp16,-int8}.onnx`, `parity.json` |
| 13 | Exportar el grafo adaptable | `uv run rukh export --ckpt <base> --out artifacts/onnx/medium-lora --fp16 --int8 --check-parity --adapter-inputs --adapter checkpoints/lora-e4-*` | ~15 min | lo mismo, con `lora_a` y `lora_b` como entradas |
| 14 | Publicar modelos | `uv run rukh publish model --ckpt <ckpt> --repo rukh-<nombre> --onnx artifacts/onnx/<nombre>` | minutos | repo en el Hub |
| 15 | Publicar adaptadores | `uv run rukh publish adapter --run checkpoints/lora-e4-* --repo rukh-lora-e4 --base rukh-medium --effect artifacts/publish/effects/lora-e4.json` | segundos | repo de 1,6 MB con `web/adapter.bin` |
| 16 | Publicar el Qwen | `uv run rukh publish qwen --run checkpoints/qwen3-pgn-qlora --repo rukh-qwen3-pgn-qlora` | minutos | repo `peft` tal cual |

Los pasos 14-16 admiten `--dry-run`: preparan la carpeta entera en `artifacts/publish/` y no tocan
la red. **Conviene revisar la card antes de subir**, porque se genera a partir de la corrida y
puede delatar que una medición falta.

## Lo que hay que mirar y no está en el comando

- **`best.pt` no es el resultado de un afinado.** El bucle lo guarda cuando baja la pérdida de
  validación, y un afinado que cambia de corpus a propósito la sube desde el principio: `best.pt`
  se queda en el paso 200. Se publica el **último** checkpoint. El bucle avisa en el log.
- **Un adaptador no se reanuda con `--resume`.** Sus checkpoints llevan los pesos fundidos, que ya
  no dicen dónde acababa el adaptador. El bucle lo rechaza diciéndolo.
- **La cabecera hay que forzarla en un barrido.** Las posiciones de validación y casi todos los
  puzles llevan el Elo real de la partida de la que salieron, así que sin `force_header: true` la
  legalidad, el top-1 y los puzles salen idénticos en todas las filas — correcto por su propia
  definición e indistinguible de «la condición no hace nada».
- **Antes de pedir más partidas, haz la cuenta.** `uv run python labs/m4/games_needed.py --only
  1500,2000,2400` dice cuántas harían falta para separar cada par. Un par que necesita doscientas
  es una hora; uno que necesita veinticuatro mil es una diferencia que no está.
- **Si una métrica sube con el tratamiento, mídela también sin él.** El paso 8 existe por eso, y en
  P4 contestó que el 89 % del recorrido de Elo ya estaba en el modelo sin afinar.
- **La escalera no se repite a sí misma.** Mismo modelo, misma semilla, dos tiradas: 1498 y 1558.
  El rival juega por tiempo y con `UCI_LimitStrength`, que aleatoriza a propósito. Comparar dentro
  de una tirada es válido; entre tiradas, no.

## Tamaños y paridad medidos (2026-09-20)

| Fichero | Tamaño | Paridad de la jugada elegida |
|---|---:|---:|
| `medium-elo/model.onnx` | 461,6 MB | 100,00 % |
| `medium-elo/model-fp16.onnx` | 231,4 MB | 99,90 % |
| `medium-elo/model-int8.onnx` | 121,8 MB | 96,40 % |

La paridad se mide sobre mil posiciones de validación y se escribe en `parity.json` al lado de los
ficheros, para que la card pueda citar el número del fichero exacto que recomienda descargar.
