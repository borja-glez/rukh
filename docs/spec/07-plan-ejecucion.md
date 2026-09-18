# 07 · Plan de ejecución

## Tres repos públicos (decisión de Borja: una pieza, un repo)

Carpeta de trabajo `E:\work\ai\chess-lm\` (no es un repo; guarda estos docs y los tres clones):

```
rukh/                     borja-glez/rukh · modelo, datos, entrenamiento, evaluación, agentes
├─ pyproject.toml            un paquete `rukh`; extras cpu/cu128; `uv`
├─ src/rukh/              {data,tokenize,models,train,rewards,eval,export,serve,agent,mcp,prompts}
├─ configs/ · tests/{unit,gpu,slow} · labs/ (notebooks finos) · gold/ (dorados, git)
├─ artifacts/web/            JSON precomputado para el curso (< 1 MB por fichero, versionado)
├─ data/ · mlruns/           (gitignored)
├─ docs/                     spec/ (estos docs migrados) · plans/ · adr/ · runbooks/ · model-cards/
│                            benchmarks.md · backlog.md · decisiones-de-ejecucion.md
├─ Dockerfile                backend FastAPI de la fase 2 (opcional)
└─ .github/workflows/ml.yml  CPU: uv sync --extra cpu, ruff, pytest -m unit, paridad ONNX

rukh-lab/              borja-glez/rukh-lab · curso → lab.rukh.borjaglez.com
├─ src/{pages,content,layouts,components,islands,styles,data} · public/
├─ scripts/sync-data.mjs · scripts/sync-tokenizer.mjs · docs/design-system.md (fuente de verdad)
├─ e2e/ · Dockerfile · nginx/ · .github/workflows/ci.yml · lighthouserc.json
└─ package.json · tsconfig.json · eslint.config.js · .prettierrc · README.md · LICENSE

rukh-web/                borja-glez/rukh-web · jugar contra el modelo → rukh.borjaglez.com
├─ src/{pages,islands,workers,lib/chess-lm,styles} · public/ort/
├─ e2e/ (partida, tres viewports, axe) · Dockerfile · nginx/ · .github/workflows/ci.yml
└─ package.json · tsconfig.json · eslint.config.js · .prettierrc · README.md · LICENSE
```

Licencias: código MIT en los tres; contenido del curso CC BY-NC-SA 4.0; pesos Apache-2.0. Lo que se
comparte entre repos se copia con script y se vigila con un test de hash (tokens CSS, tokenizador TS,
vocabulario y fixtures); no hay paquetes npm ni workspaces cruzados. Los planes de cada hito viven en
`rukh/docs/plans/` aunque el hito toque los tres repos; cada repo recibe sus propios commits y su
rama `pN-<slug>`.

Convenciones: commits en español, imperativo, sin coautoría ni referencias a herramientas; código y
comentarios en inglés; TypeScript 6; pnpm 10; Node ≥ 24; Python 3.12 con `uv`; `ruff`; pydantic
`extra="forbid"`; tests `unit` en CPU < 3 min.

> **Nota (2026-09-18):** los commits van en **inglés con Conventional Commits** (`feat:`, `fix:`,
> `docs:`, `ci:`, `chore:`…), sin coautoría ni referencias a herramientas o sesiones. Decisión D-001 en
> `docs/decisiones-de-ejecucion.md`; sustituye a "commits en español, imperativo" de este párrafo.

## Hitos (P = plataforma y producto; cada uno cierra un módulo del curso con su lección publicada)

| Hito | Alcance | Módulo | Verificación |
|---|---|---|---|
| **P0 Scaffold** | Tres repos con CI propia; tokens del portfolio copiados en ambas webs; `rukh-lab` con landing y lección 0; `rukh-web` con la pantalla única responsive, tablero y jugadas legales (sin modelo); `rukh` con CLI `rukh info`, MLflow y estos docs en `docs/spec/`; Dockerfiles en la raíz de cada web; dos apps en Dokploy con dominio y HTTPS | M0 | `pnpm check/test/build` verdes en las dos webs; `uv run pytest -m unit`; `curl -I` de ambos dominios; Lighthouse ≥ 0,95 escritorio y móvil; capturas en tres viewports |
| **P1 Datos** | `rukh data fetch/uci/tokenize/positions/puzzles/pairs/publish`; manifiesto; datasets `chorcat/rukh-*` y `chorcat/rukh-tokenizer` publicados; tokenizador TS en la demo y sincronizado al curso; lección M1 con `TokenizerPlayground`; dataloaders probados | M1 | Conteos del manifiesto; tests de tokenización (paridad Python↔TS por fixture); dataset cards en el Hub |
| **P2 Decoder** | `MoveDecoder` tiny/small, entrenamiento con MLflow, harness v1 (legalidad, exactitud, Elo vs Stockfish), export ONNX fp16/int8 con paridad, `chorcat/rukh-small`, demo v1 jugable con consentimiento y WebGPU, lección M2 con `AttentionMap` y `TrainingReplay`, post 1 | M2 | `small` ≥ 1200 Elo y ≥ 99 % legales; partida completa en Chrome con WebGPU; paridad ONNX ≥ 99,9 % |
| **P3 Encoder** | `PositionEncoder`, MMM, cabezas de valor y error, embeddings, `chorcat/rukh-encoder`, `EvalBar` en la demo, lección M3 | M3 | Supera heurísticas en error (≥ 5 puntos F1); barra de valor en vivo |
| **P4 Fine-tuning** | SFT maestros, Elo-conditioning, LoRA a mano y con peft, envoltura HF del decoder, QLoRA Qwen sobre PGN, selector de Elo y adaptadores en la demo, lección M4, post 2 | M4 | Elo por condición monótono; tabla comparativa; adaptadores intercambiables en la demo |
| **P5 Alineamiento** | RM, pares on-policy, DPO, recompensas con tests, GRPO, galería de hacking, `chorcat/rukh-{rm,dpo,grpo}`, `GrpoGroup` y `RewardCurves`, lección M5, post 3 | M5 | DPO o GRPO ≥ +50 Elo sobre su base con IC; recompensas testeadas |
| **P6 Cierre fase 1** | `rukh eval nightly`, tabla única, model cards, demo final con selector de etapa, arena modelo vs modelo, puzles en vivo, bot de Lichess (opcional), README maestro, página EN de proyecto, enlace desde borjaglez.com, post 4 con vídeo, lección M6 | M6 | Tabla completa reproducible; demo verificada en Chrome; CI verde |
| **A1-A6** | Fase 2 según `06-agentes.md`: índice de posiciones y RAG (A1), agente LangGraph + LangSmith (A2), evaluación (A3), multiagente (A4), MCP (A5), backend + `Coach.tsx` + informe + post 5 (A6) | A1-A6 | RAGAS documentado; suite LangSmith; partida comentada de principio a fin |

## Cómo se ejecuta cada hito

1. **Plan escrito** por hito (`rukh/docs/plans/AAAA-MM-DD-pN-<slug>.md`, aunque toque los tres repos) con tareas de 2-4 h, ficheros,
   interfaces, tests y verificación; deriva de estos docs (el spec).
2. **Rama** `pN-<slug>` desde `main`; una tarea = un commit revisado; revisión final de la rama;
   una única ola de correcciones; Borja decide merge y push (Dokploy despliega en `main`).
3. **Verificación real** siempre por el controlador: entrenamientos en la 5090, partidas en Chrome
   con WebGPU, Docker con cabeceras. Los subagentes no tienen GPU ni navegador.
4. **Lección del módulo escrita en el mismo hito** (no al final), con su visualización alimentada por
   el JSON que exporta el lab.
5. **Ledger** de decisiones y desviaciones en `rukh/docs/decisiones-de-ejecucion.md` (versionado).
6. **Publicar antes de cerrar**: el hito no se cierra hasta que sus modelos, adaptadores y datasets
   están en el Hub con card y la demo los carga desde allí.

## Puntos donde Borja interviene (el resto es autónomo)

- Crear los repos `borja-glez/rukh`, `rukh-lab` y `rukh-web` en GitHub y las dos apps en
  Dokploy con sus dominios (P0).
- Token de Hugging Face con permiso de escritura en `chorcat` (P1) y, si se hace el bot, cuenta BOT de
  Lichess y token OAuth `bot:play` (P6).
- Escuchar/mirar: partidas de prueba en la demo, en ordenador y en móvil (P2, P6), 50 preguntas doradas y 40 comentarios
  dorados (A1, A3), grabar los vídeos de los posts.
- Decidir merges (fin de cada hito).

## Calendario orientativo (8-10 h/semana)

S1 P0+P1 · S2 P2 · S3 P3 + mitad de P4 · S4 resto de P4 + P5 · S5 P6 · S6-S9 A1-A6.
Los entrenamientos largos (`small` en P2, GRPO en P5, embeddings de 5 M posiciones en A1) se lanzan
al final de una sesión y se recogen en la siguiente.
