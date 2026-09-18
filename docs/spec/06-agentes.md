# 06 · Fase 2: IA agéntica sobre el mismo proyecto

Objetivo: convertir el modelo y sus datos en un **entrenador de ajedrez agéntico** que analiza y
comenta partidas, con recuperación de conocimiento, herramientas, trazas, evaluación y una interfaz
MCP. Cada módulo del programa de agentes añade una capa sin rehacer nada de la fase 1.

Versiones verificadas (2026-09-18): `langchain 1.4.2`, `langchain-core 1.6.3`, `langgraph 1.2.11`,
`langsmith 0.13.0`, `langchain-mcp-adapters 0.3.2`, `mcp 2.2.0`, `ragas 0.4.3`, `chromadb 1.5.9`,
`faiss-cpu 1.15.1`, `sentence-transformers 6.1.0`, `ollama 0.6.2`, `openai 3.16.1` (cliente para
endpoints OpenAI-compatibles locales: LM Studio, Ollama, llama-server). Sin APIs de pago: el LLM
comentarista es un modelo local (Qwen3.5-9B/27B en LM Studio u Ollama) y, para la demo pública, uno
pequeño servido en CPU o el puente a la 5090.

## A1 · Embeddings y RAG (6 h)

- **Recuperación de posiciones**: embeddings del `PositionEncoder` (M3) para 2-5 M de posiciones de
  partidas de maestros → índice FAISS/Chroma → "en posiciones parecidas, los maestros jugaron X con
  resultado Y". Comparar con recuperación exacta por FEN y con hashing de posiciones.
- **Libro de aperturas** como recuperación estructurada (`chess-openings`): nombre ECO de cualquier
  prefijo de partida.
- **RAG textual**: libros en dominio público de Gutenberg (Capablanca #33870, E. Lasker #5614 y los
  que se verifiquen bajo el tema "Chess"), limpieza, chunking por sección, embeddings con
  `multilingual-e5-small` o `bge-m3` (local), Chroma; el comentarista cita el pasaje. Evaluación con
  RAGAS (faithfulness, answer relevance) sobre 50 preguntas doradas escritas por Borja.
- Teoría justa: embeddings y similitud, chunking, top-k y reranking, evaluación de recuperación.

## A2 · Agente con herramientas (7 h)

- LangGraph: grafo `analizar_partida` con nodos de planificación, llamada a herramientas y redacción;
  estado tipado (pydantic); memoria de la partida.
- Herramientas tipadas (Python, sobre `python-chess`, Stockfish y los modelos propios):
  `legal_moves(fen)`, `engine_eval(fen, depth)`, `model_move(fen, stage, elo)`, `model_value(fen)`,
  `similar_positions(fen, k)`, `opening_name(moves)`, `book_passages(query, k)`, `solve_puzzle(fen)`.
- LLM comentarista con `langchain-openai` contra el endpoint local; prompts en español; salida
  estructurada (pydantic) para el comentario por jugada: jugada, valoración, alternativa, cita opcional.
- Trazas en LangSmith (proyecto `rukh`), con etiquetas por versión de prompt y de modelo.
- Teoría justa: bucle de agente, tool calling, estado y control de flujo en LangGraph, límites y
  guardarraíles (máximo de pasos, validación de argumentos).

## A3 · Evaluación y observabilidad (4 h)

- Dataset de evaluación en LangSmith: 40 posiciones con comentario dorado (Borja + motor) y 20
  adversariales (posiciones triviales, mates en uno, inyección en el PGN).
- Evaluadores: exactitud de la jugada recomendada frente a Stockfish, consistencia entre valoración y
  `cp`, presencia de cita cuando se afirma algo de teoría, longitud, idioma. Juez LLM local de otra
  familia como segundo evaluador.
- Regresión por commit: `rukh agent eval` sube el run a LangSmith; umbrales en CI opcional.

## A4 · Multiagente (5 h)

- Tres agentes: **jugador** (usa `model_move` con la etapa y el Elo elegidos), **entrenador**
  (interviene cuando `engine_eval` detecta un error del humano o del jugador, con explicación y
  alternativa) y **redactor** (al terminar, informe en Markdown: aperturas, errores por fase, puzles
  recomendados del dataset por tema). Orquestación en LangGraph con supervisor; handoffs explícitos.
- Teoría justa: cuándo separar agentes, coste de coordinación, evaluación de un sistema multiagente.

## A5 · MCP (3 h)

- Servidor MCP (`mcp` 2.2) que expone las herramientas de A2 (`rukh mcp serve`), con esquemas y
  ejemplos; cliente de prueba con `langchain-mcp-adapters`; uso desde Claude Code o cualquier cliente
  MCP para analizar una partida. Seguridad: solo `127.0.0.1`, sin estado persistente, tiempo máximo por
  llamada.

## A6 · Capstone (5 h)

- Backend FastAPI (`rukh/src/rukh/serve/`) con SSE que encapsula el grafo; la demo (`Coach.tsx`) muestra
  el comentario en vivo y el visor de trazas.
- Informe final descargable (Markdown/JSON).
- Página del proyecto en inglés actualizada con la arquitectura agéntica, y post 5 con vídeo.

## Datos y artefactos de la fase 2

| Artefacto | Dónde |
|---|---|
| Índice de posiciones (embeddings + metadatos) | `chorcat/rukh-positions-index` (dataset) |
| Corpus de libros limpio y troceado | `chorcat/rukh-chess-books` (dominio público, con fuente por chunk) |
| Preguntas doradas y adversariales | `ml/gold/` en el repo |
| Prompts versionados | `rukh/src/rukh/prompts/*.jinja` |
