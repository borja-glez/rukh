# 02 · Modelos y entrenamiento

Hardware: RTX 5090 (32 GB, sm_120, CUDA 12.8), 64 GB RAM, Windows 11. Python 3.12 con `uv`.
Versiones verificadas el 2026-09-18: `torch 2.11.0+cu128` (última rueda cp312 del índice cu128),
`transformers 5.17.0`, `trl 1.13.0`, `peft 0.21.0`, `datasets 5.0.1`, `accelerate 1.15.0`,
`onnx 1.22.0`, `onnxruntime 1.30.0`, `mlflow 3.16.1`. Sin `nvcc`: no se compila flash-attn; se usa SDPA.

Expectativas calibradas con trabajos previos: un nanoGPT de 50M parámetros entrenado sobre 16M partidas
de Lichess a nivel de carácter alcanzó ~1300 Elo con 99,8 % de jugadas legales en un día de GPU
(Karvonen); DeepMind, con 270M parámetros y 10M partidas anotadas por Stockfish, llegó a 2895 Elo
blitz. Objetivo honesto de este curso: **1200-1500 Elo** con el decoder propio de 30-50M, y mejoras
medibles (50-150 Elo) con Elo-conditioning, DPO y GRPO.

## Componente 1 · `MoveDecoder` (desde cero)

- Arquitectura GPT: embedding de tokens + posiciones aprendidas (y una variante con RoPE para
  comparar), N bloques pre-norm con atención multi-cabeza causal (SDPA), MLP GELU, tied embeddings.
  Tamaños: `tiny` (6 capas, d=256, 4 cabezas, ~5M) para iterar, `small` (12 capas, d=512, 8 cabezas,
  ~40M) como modelo del curso, `medium` (16 capas, d=768, ~110M) opcional de noche.
- Entrada: `<bos> <w1800> <b1900> e2e4 e7e5 … <1-0> <eos>` (UCI, vocabulario fijo). Contexto 200.
- Objetivo: siguiente token con entropía cruzada; los tokens de Elo y resultado también se predicen
  (ayuda a que el modelo "sepa" quién juega).
- Receta: AdamW (β=0,9/0,95, wd 0,1), lr 6e-4 con warmup 1 000 pasos y coseno, batch 256 × 200 tokens,
  bf16, clipping 1,0, `torch.compile`. En la 5090, `small` procesa ~1M tokens/s → 3-6M partidas
  (~500M tokens) en 2-4 epochs ≈ 4-8 h. Se guarda checkpoint cada 1 000 pasos y se registra en MLflow.
- Inferencia: jugada = `argmax` o muestreo con temperatura/top-k **con enmascarado de jugadas ilegales**
  (se calcula la lista legal con `python-chess` o `chess.js` y se filtran los logits). Se mide tanto
  con máscara como sin ella: la tasa de legalidad sin máscara es la métrica de "comprensión".
- Tests unitarios: formas, causalidad (cambiar un token futuro no altera logits pasados), paridad
  ONNX/PyTorch, muestreo enmascarado nunca produce jugadas ilegales.

## Componente 2 · `PositionEncoder` (desde cero)

- Encoder bidireccional (6-8 capas, d=384) sobre la misma secuencia de jugadas UCI (prefijo de partida)
  o, alternativa didáctica, sobre 64 tokens de casilla (pieza-o-vacío) + turno + enroques desde el FEN.
  Se implementan las dos entradas y se compara: es la lección de "qué es una buena representación".
- Preentrenamiento: **masked move modeling** (15 % de tokens enmascarados, 80/10/10) sobre partidas.
- Cabezas (fine-tuning): (a) `value`: regresión del `cp` de Stockfish acotado (tanh de cp/400) o
  clasificación en 5 tramos; (b) `blunder`: la última jugada pierde ≥ 100 cp (etiquetas del cruce con
  evaluaciones); (c) `result`: quién gana. Curvas por número de etiquetas (10/25/50/100 %) y fine-tuning
  del probe → últimas capas → completo.
- Salida secundaria: embeddings de posición (pooling CLS/mean) para recuperación de posiciones
  parecidas (fase 2).

## Componente 3 · Fine-tuning e instrucción (sobre `MoveDecoder`)

- **Completo con maestros**: seguir preentrenando con `rukh-games-elite` (lr 1e-4). Medir: sube el
  Elo, baja la diversidad.
- **Elo-conditioning** (instruction tuning): reentrenar/afinar con `rukh-elo-bins` y evaluar que
  `<w1500>` juega peor que `<w2400>` de forma medible (Elo estimado por condición). Comparar con Maia.

> **Nota (2026-09-20, P4):** medido y **no se cumple** por la vía del Elo. Con 160 partidas por
> condición ni las estimaciones son monótonas ni los intervalos se separan, y el cálculo de potencia
> dice que entre `<w1500>` y `<w2000>` harían falta 24 420 partidas por condición para separar una
> diferencia de −0,013 en tasa de puntos: no es tamaño de muestra, es que no hay diferencia. El
> control lo remata: el modelo **sin** afinar recorre 161 Elo por el mismo eje usando cabeceras que
> nunca entrenó, así que la columna de Elo no distingue «condicionamiento» de «un prefijo
> desconocido estorba». Lo que sí cambia de forma medible y ordenada es el **estilo**: entropía
> analítica de la primera jugada monótona en las seis condiciones (1,596 → 1,992 bits), top-1 con
> máximo en `<w1800>` y legalidad que baja al ensanchar el repertorio. Detalle en D-100, D-101 y
> D-106 de `docs/decisiones-de-ejecucion.md`.
- **LoRA por estilo**: adaptadores r=8-16 sobre las proyecciones de atención, uno por repertorio o
  jugador (p. ej. partidas de un jugador concreto de la Elite DB), intercambiables en la demo.
  Implementado a mano en el decoder propio (para entender LoRA) y con `peft` sobre el modelo HF.
- **Puente al ecosistema HF**: envolver `MoveDecoder` como `PreTrainedModel` con `config.json` propio
  (o convertir al formato GPT-2 de HF, que es equivalente), de modo que `SFTTrainer`, `DPOTrainer`,
  `GRPOTrainer` y `peft` funcionen sin código especial. Además, QLoRA de `Qwen3-0.6B/1.7B` sobre PGN
  como texto (`transformers` + `peft` + `trl`) para comparar un LLM general con el modelo propio.

## Componente 4 · Alineamiento con recompensas verificables

- **Reward model**: `PositionEncoder` + cabeza escalar entrenado con `RewardTrainer` (Bradley-Terry)
  sobre pares (posición, jugada mejor ≻ jugada peor) de `rukh-pairs-dpo`. Métrica: exactitud en
  pares held-out y correlación con `cp`.
- **DPO** con `DPOTrainer` sobre el decoder (β 0,1; lr 5e-6; pares on-policy: se muestrean 4 jugadas
  del modelo, se puntúan con Stockfish o con las evaluaciones, se forma el par). Métrica: Elo antes y
  después, y `rewards/margins`.
- **GRPO** con `GRPOTrainer`: para cada posición (prompt = prefijo de partida) se generan G=8 jugadas;
  recompensas: legalidad (puerta), Δcp respecto a la mejor jugada (normalizado, con tope), bonus por
  mate, penalización por repetición. Sin vLLM (generación en el propio trainer, modelos pequeños).
  300-600 pasos ≈ 1-2 h. Se documenta reward hacking observado (p. ej. preferir tablas seguras) y su
  corrección (peso de "progreso").
- Alternativa barata para comparar: rejection sampling fine-tuning (mejor de N por Stockfish → SFT).

## Componente 5 · Harness de evaluación (`ml/rukh/eval/`)

Una tabla para todas las etapas (base, tiny/small/medium, maestros, Elo-cond, LoRA, RM, DPO, GRPO,
Qwen-QLoRA, Karvonen-50M, Maia):

| Métrica | Cómo |
|---|---|
| Legalidad sin máscara | % de jugadas legales sobre 10 000 posiciones de validación |
| Exactitud de siguiente jugada | Top-1 y top-3 contra la jugada humana en validación (por tramo de Elo) |
| Puzles | % resueltos por tramo de dificultad (1000-1500 / 1500-2000 / 2000+), 2 000 puzles por tramo |
| Elo estimado | 100 partidas contra Stockfish `UCI_Elo` en 1320/1500/1800/2000 (+ `Skill Level` para más bajo), ambos colores, con máscara de legalidad; Elo por regresión logística de resultados; intervalo de confianza |
| Δcp medio | Pérdida media de centipeones por jugada respecto a Stockfish profundidad 12 en 500 posiciones |
| Diversidad | Entropía de aperturas jugadas en 200 partidas propias |

Todo reproducible con `rukh eval --model <id> --suite full`, caché SQLite, informe Markdown y JSON,
y `rukh eval nightly` para la tabla completa.

## Componente 6 · Exportación y despliegue

- ONNX con `torch.onnx.export` (dynamo, opset 18), solo el eje de batch dinámico y longitud fija 200
  (o dinámica con prueba de paridad); fp16 para WebGPU e int8 dinámico para WASM. `small` ≈ 80 MB en
  fp16, ≈ 40 MB int8. Paridad PyTorch/ORT ≥ 99,9 % de jugadas iguales en 1 000 posiciones.
- Inferencia en navegador con `onnxruntime-web` 1.30 directo (modelo propio, no transformers.js):
  un worker por modelo, cache en Cache API, consentimiento con tamaño. Lecciones heredadas del
  proyecto anterior que aplican a cualquier demo con ORT en navegador: importar
  `onnxruntime-web/webgpu` para WebGPU; el build asyncify no admite llamadas asíncronas concurrentes
  (crear sesiones en serie y serializar `run`); mantener el literal `new Worker(new URL(...))` en el
  punto de llamada con Vite; los modelos de más de ~300 MB no caben en algunos navegadores embebidos.
- Publicación en Hugging Face de **todo** lo entrenado, en la colección `chorcat/rukh`:
  - modelos: `rukh-tiny`, `rukh-small` (PyTorch `.safetensors` + carpeta `onnx/` con fp16 e int8
    en el mismo repo), `rukh-encoder`, `rukh-small-masters` (SFT), `rukh-small-elo`,
    `rukh-rm`, `rukh-small-dpo`, `rukh-small-grpo`;
  - adaptadores (peft): `rukh-lora-<estilo>`, `rukh-qwen3-pgn-qlora`;
  - artefactos: `rukh-tokenizer` (vocabulario JSON, tokens especiales, fixtures de paridad);
  - datasets: los seis `rukh-*` de `docs/01`.
  Model cards en inglés generadas desde MLflow (métricas de la tabla, datos, receta, limitaciones,
  enlace a la demo con `?stage=`) y licencia Apache-2.0. `rukh publish <run>` sube pesos, ONNX y
  card en un solo paso; la demo lee los pesos del Hub (nunca de git).
- Bot en Lichess (opcional, M6): cuenta BOT nueva + token OAuth con `bot:play`, `lichess-bot`
  (AGPL, solo se ejecuta) con el modelo como motor UCI mínimo en Python. Da un Elo "real" de bots.

## MLOps ligero

MLflow local (SQLite) con `report_to="mlflow"` en los trainers de TRL y registro manual en los bucles
propios; cada run guarda seed, SHA de git, hash del manifiesto de datos y config. Configs YAML +
pydantic (`extra="forbid"`); CLI `rukh` con typer: `data | train | eval | export | serve | publish`.
CI en CPU: `uv sync --extra cpu`, ruff, pytest `-m unit` (formas, máscaras, tokenizador, paridad ONNX
del `tiny`, recompensas, parser de PGN) en < 3 min; `gpu` y `slow` solo en local.
