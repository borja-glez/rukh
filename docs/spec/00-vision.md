# 00 · Visión

## Qué es

Un curso práctico propio, en español, que enseña IA generativa (y después IA agéntica) construyendo un
único proyecto de principio a fin: **un modelo de lenguaje que juega al ajedrez**. El proyecto se elige
porque permite cubrir el ciclo de vida completo de un modelo moderno en semanas, no meses:

- El dominio es cerrado y verificable: la legalidad y la calidad de cada jugada las decide un motor,
  no una persona. Eso hace posibles el reward model, DPO y GRPO con recompensas reales y baratas.
- Los datos son abiertos, enormes y con licencia CC0 (Lichess): no hay que negociar nada.
- Los modelos son pequeños (decenas de millones de parámetros): se entrenan en horas en la RTX 5090
  y caben en el navegador.
- La demo se entiende en cinco segundos: un tablero, y juegas contra el modelo que has entrenado.
  Sencilla, cuidada y usable igual en ordenador, tablet y móvil.

## Para quién y para qué

- **Para Borja**, como aprendizaje real: cada concepto de los dos programas de referencia (IA
  generativa con LLMs; RAG y agentes) se aprende haciéndolo sobre el mismo artefacto y midiendo.
- **Para el portfolio, el CV y LinkedIn**: la historia "entrené un modelo desde cero, lo alineé con RL de
  recompensas verificables, lo desplegué en WebGPU y lo convertí en un agente con herramientas" es
  exactamente el relato de un AI Architect, con cifras honestas y artefactos públicos (modelos,
  datasets, demo, posts).
- **Tres repos en GitHub** (`rukh` para el modelo, `rukh-lab` para el curso, `rukh-web`
  para jugar) y **todo lo entrenado en Hugging Face** (`chorcat/rukh-*`: modelos, adaptadores y
  datasets con card).
- **Publicable** bajo `lab.rukh.borjaglez.com` (curso) y `rukh.borjaglez.com` (demo), con el sistema de
  diseño del portfolio.

## Criterios de diseño del curso

1. **Un artefacto, todo el ciclo.** Cada módulo mejora el mismo modelo o añade una pieza que lo usa.
2. **Teoría bajo demanda.** Un concepto entra cuando un lab lo necesita para tomar una decisión
   (qué tokenización, qué pooling, qué β en DPO). Nada de capítulos de teoría sueltos.
3. **Baseline obligatorio y tabla única.** Toda etapa se mide igual (Elo estimado frente a Stockfish
   limitado, tasa de jugadas legales, precisión en puzles por dificultad) y se compara con lo anterior
   y con modelos públicos (Karvonen 50M, Maia). Sin número no hay model card.
4. **Demo siempre verde.** Cada semana termina con algo jugable y publicado.
5. **Honestidad sobre el "desde cero".** El decoder y el encoder son propios y se entrenan desde
   cero; los LLMs generales (Qwen) solo se usan para comparar y, en la fase 2, para comentar partidas.
6. **Sin referencias a los cursos de origen.** Los conceptos se explican de cero con el proyecto como
   hilo; los notebooks anteriores no se reutilizan.
7. **Bilingüe.** Curso y docs en español; código, README, model cards y páginas de proyecto en inglés.

## Qué se aprende (mapa de conceptos → módulo)

| Concepto | Dónde se aprende con el proyecto |
|---|---|
| Tokenización (vocabulario fijo, BPE, tokens especiales) | M1: tres tokenizaciones de partidas comparadas |
| Datasets, dataloaders, padding, máscaras, streaming | M1: de parquet a lotes |
| Embeddings y posiciones (aprendidas, RoPE) | M2 |
| Atención, multi-cabeza, máscara causal, decoder GPT | M2: decoder a mano |
| Receta de entrenamiento (AdamW, warmup, bf16, clipping) | M2 |
| Muestreo (temperatura, top-k, enmascarado) | M2 |
| Encoder bidireccional, MLM, pooling, clasificación | M3: masked move modeling + cabezas |
| Fine-tuning completo vs parcial, curvas por etiquetas | M3-M4 |
| Instruction tuning (condicionar el comportamiento) | M4: token de Elo objetivo |
| LoRA, QLoRA, adaptadores intercambiables | M4 |
| Ecosistema HF (transformers, peft, trl, datasets) | M4: QLoRA de Qwen sobre PGN |
| Reward model, Bradley-Terry | M5 |
| DPO (β, referencia implícita) | M5 |
| GRPO / RL con recompensas verificables, reward hacking | M5 |
| Evaluación, harness reproducible, model cards | M6 |
| Cuantización, ONNX, inferencia en navegador | M2 y M6 |
| Embeddings para recuperación, vector DB, RAG, RAGAS | A1 |
| Agentes con herramientas, LangGraph, LangSmith | A2-A3 |
| Multiagente, MCP, capstone | A4-A6 |

## Qué queda fuera a propósito

- Audio, voz, visión y vídeo.
- RLHF con etiquetas humanas a escala (el motor las sustituye; hay un DPO pequeño con preferencias de
  Borja sobre comentarios en la fase 2).
- Traducción y seq2seq; RNN, n-gramas, word2vec (se mencionan en una lectura de 10 minutos).
- Entrenar LLMs generales desde cero.

## Éxito

Fase 1 cumplida cuando: el decoder propio supera 1200 Elo estimado con ≥ 99 % de jugadas legales; el
encoder supera a las heurísticas en detección de errores; DPO o GRPO mejoran medibles sobre el
fine-tuning; todo está publicado (modelos, datasets recortados, demo, curso con seis módulos) y hay
un post con vídeo. Fase 2 cumplida cuando el entrenador agéntico comenta una partida completa con
citas recuperadas, trazas en LangSmith y evaluación RAGAS documentada.
