# 03 · Currículo

Dos fases, un artefacto. Cada módulo: teoría justa (solo lo que el lab necesita para decidir), lab
guiado con TODOs y solución de referencia, artefacto público, visualización en la web del curso y
"demo verde" al terminar. Ritmo objetivo: 8-10 h/semana.

## Fase 1 · Un modelo de lenguaje de ajedrez desde cero (5 semanas, ≈ 44 h)

| # | Módulo | h | Al terminar funciona… | Artefacto público |
|---|---|---|---|---|
| M0 | Taller: repo, entorno, datos y tablero | 3 | `uv sync`, datos recortados descargados, tablero en la demo con jugadas legales | Repo, manifiesto de datos |
| M1 | Del PGN al tensor: datos y tokenización | 6 | Dataloader que sirve lotes de las tres tokenizaciones; comparativa | `chorcat/rukh-games-1800`, tokenizador |
| M2 | El decoder: un GPT que juega | 8 | `rukh-small` juega ≥ 1200 Elo con ≥ 99 % legales; juegas contra él en la web | `chorcat/rukh-small` (+ONNX), demo v1, post 1 |
| M3 | El encoder: entender la posición | 6 | Valor y detección de errores en vivo en la demo | `chorcat/rukh-encoder` |
| M4 | Fine-tuning e instrucción | 7 | Selector "juega como 1500/2000/2400" y adaptadores de estilo | `-elo`, `-lora-*`, comparativa con Qwen-QLoRA, post 2 |
| M5 | Alineamiento con recompensas verificables | 8 | Modelos DPO y GRPO medibles; galería de reward hacking | `-rm`, `-dpo`, `-grpo`, post 3 |
| M6 | Evaluar, exportar, publicar | 6 | Tabla única, model cards, demo final con selector de etapa, bot de Lichess | Informe, bot, post 4 (vídeo) |

### M0 · Taller (3 h)
Tres repos (ver 07), `uv` con `cu128`, descarga del recorte de datos con `rukh data fetch`
(lanzada al inicio; tarda 1-2 h en segundo plano), Stockfish instalado y verificado, MLflow, tests en
CPU, web del curso con la lección 0 y la demo con tablero y jugadas legales (sin modelo). Casi todo lo
hace Claude; Borja verifica `rukh info`, `pnpm dev` y el despliegue.

### M1 · Datos y tokenización (6 h)
Teoría justa: qué es un token y por qué la elección importa; vocabulario fijo frente a BPE; tokens
especiales como interfaz de control; padding, máscaras y streaming; fugas entre splits.
Labs: (1) explorar el parquet con DuckDB (distribuciones de Elo, duración, aperturas); (2) SAN→UCI y
limpieza con `python-chess`; (3) las tres tokenizaciones y sus estadísticas (tokens por partida,
cobertura); (4) `Dataset` y `DataLoader` con empaquetado por bloques y máscara causal; (5) entrenar
un BPE con `tokenizers` y mirar las fusiones. Visualización: tokenizador de partidas en vivo (pegas un
PGN y ves los tokens de cada representación). Recorte: n-gramas, word2vec, RNN (lectura de 10 min).

### M2 · El decoder (8 h)
Teoría justa: embeddings y posiciones (aprendidas/RoPE), atención con máscara causal, multi-cabeza,
bloque pre-norm, tied embeddings; receta de entrenamiento; muestreo y enmascarado de jugadas
ilegales; por qué el modelo "aprende el tablero" sin verlo (world models: lectura de Karvonen).
Labs: (1) `MoveDecoder` a mano con tests de formas y causalidad; (2) entrenar `tiny` en 15 min y ver
la curva; (3) lanzar `small` de noche; (4) evaluación: legalidad, exactitud, primeras partidas contra
Stockfish limitado; (5) exportar a ONNX y jugar en la demo (WebGPU). Visualización: mapa de atención
de una partida; "ver emerger el juego" con checkpoints por paso (legalidad y Elo por iteración).

### M3 · El encoder (6 h)
Teoría justa: bidireccional vs causal, MLM trasladado a jugadas, pooling, cabezas de
clasificación/regresión, fine-tuning parcial vs completo, curvas por número de etiquetas, cambio de
representación (jugadas vs casillas).
Labs: (1) masked move modeling 20-40 min; (2) cabezas de valor y error con etiquetas del cruce de
evaluaciones; (3) comparación de las dos entradas; (4) probe lineal vs fine-tuning; (5) indicador de
valor y alerta de error en la demo. Visualización: barra de evaluación del encoder frente a Stockfish
en una partida.

### M4 · Fine-tuning e instrucción (7 h)
Teoría justa: por qué afinar; catastrophic forgetting; instrucciones como tokens de control;
LoRA (rango, α, módulos), QLoRA y NF4; el ecosistema HF (config, PreTrainedModel, trainers).
Labs: (1) SFT con maestros y su efecto en diversidad; (2) Elo-conditioning y evaluación por condición
(comparar con Maia); (3) LoRA a mano sobre el decoder; (4) `peft` sobre el modelo envuelto en HF;
(5) QLoRA de Qwen3 sobre PGN como texto y comparación honesta. Visualización: selector de Elo en la
demo; explorador del rango LoRA (SVD de ΔW).

### M5 · Alineamiento (8 h)
Teoría justa: el LLM como política; reward model y Bradley-Terry; DPO (β, referencia implícita,
por qué no hace falta RM); RLVR/GRPO (ventaja relativa por grupo, KL, diseño de recompensas
verificables, reward hacking); PPO en 10 min de contexto.
Labs: (1) RM con `RewardTrainer` y pares de evaluaciones; (2) pares on-policy y DPO; (3) recompensas
con tests unitarios; (4) GRPO 300 pasos en vivo (`tiny`) y `small` de noche; (5) tabla antes/después.
Visualización: curvas de recompensa por función; grupo de G jugadas con ventajas coloreadas sobre el
tablero; galería de hacking.

### M6 · Evaluar, exportar, publicar (6 h)
Teoría justa: métricas de dominio vs proxies; intervalos de confianza del Elo; reproducibilidad;
cuantización y paridad; model cards y licencias; qué NO decir de un modelo así.
Labs: (1) `rukh eval nightly` y la tabla; (2) exportaciones fp16/int8 y paridad; (3) model cards
desde MLflow; (4) bot de Lichess (opcional); (5) demo final con selector de etapa y comparación lado a
lado; (6) README maestro y post con vídeo. Visualización: leaderboard de etapas; comparador de dos
modelos jugando entre sí.

## Fase 2 · IA agéntica: el entrenador (4 semanas, ≈ 30 h)

| # | Módulo | h | Al terminar funciona… |
|---|---|---|---|
| A1 | Embeddings y RAG: posiciones y libros | 6 | "En posiciones parecidas, los maestros jugaron…" y comentarios con citas de Capablanca; RAGAS |
| A2 | Agente con herramientas (LangGraph) | 7 | Agente entrenador que analiza una partida con motor, modelo, libro y buscador; trazas en LangSmith |
| A3 | Evaluación y observabilidad | 4 | Suite de evaluación en LangSmith; regresión por commit |
| A4 | Multiagente | 5 | Jugador vs entrenador vs redactor del informe |
| A5 | MCP | 3 | Servidor MCP con las herramientas de ajedrez, usable desde cualquier cliente |
| A6 | Capstone | 5 | Web del entrenador: juegas, te comenta en vivo, informe final |

Detalle en `06-agentes.md`.

## Mapa de cobertura frente a los programas de referencia (sin nombrarlos en el curso)

- **Se mantiene con propósito**: tokenización y vocabularios (M1), datasets y dataloaders (M1),
  atención/MHA/posiciones/decoder (M2), encoder y MLM y clasificación de secuencias (M3), fine-tuning
  parcial y completo, instruction tuning, LoRA/QLoRA, ecosistema HF (M4), RM, DPO, RLHF en contexto,
  GRPO (M5), evaluación y cuantización (M6), embeddings, vector DB, RAG, RAGAS (A1), LangChain
  básico, agentes y herramientas, LangGraph, LangSmith (A2-A3), multiagente (A4), MCP (A5), capstone (A6).
- **Recortado explícitamente**: GAN/VAE/difusión, RNN/LSTM/seq2seq, one-hot/BoW/word2vec/GloVe,
  n-gramas (solo perplejidad), traducción, soft prompts, PPO completo, InstructLab, BLEU/ROUGE.
- **Cheatsheets de entrevista** al final de cada módulo: 8-10 preguntas con respuesta corta.

## Calendario

S1 M0+M1 (descarga de datos lanzada el día 1) · S2 M2 (entrenamiento `small` de noche entre sesiones)
· S3 M3 y mitad de M4 · S4 resto de M4 + M5 (GRPO `small` de noche) · S5 M6 y post con vídeo ·
S6-S9 A1-A6. GPU total ≈ 15-20 h, casi toda desatendida.
