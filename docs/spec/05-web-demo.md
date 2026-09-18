# 05 · La web de demostración (`rukh.borjaglez.com`)

Aplicación y repo separados del curso (`borja-glez/rukh-web`): aquí se juega contra el modelo y se
ve lo que sabe. En la fase 1 es 100 % en el navegador (sin backend). En la fase 2 gana un backend
opcional para el entrenador agéntico.

## Principios (fijados por Borja)

- **Sencilla de entrada.** La v1 es una sola pantalla: tablero, panel del modelo (etapa, Elo, estado de
  carga) y lista de jugadas con "nueva partida", "deshacer" y "exportar PGN". Nada más hasta que el
  modelo juegue bien. Las funciones posteriores (arena, puzles, cascada de latencia, "sin máscara",
  adaptadores) entran en un cajón "más" o en pestañas, sin tocar la pantalla principal.
- **Bonita.** Mismo sistema de diseño que el portfolio y el curso (tokens `light-dark()`, Bricolage
  Grotesque + IBM Plex Mono, `Rail`/`SectionHead`); piezas SVG de `cm-chessboard` con colores del
  tema; animación de jugada ≤ 150 ms, respetando `prefers-reduced-motion`; estados de carga y error
  cuidados (skeleton del tablero, progreso de descarga con MB, mensaje claro si no hay WebGPU).
- **Usable en ordenador, tablet y móvil.**
  - Layout: tablero a `min(100vw - 2·gutter, 100dvh - cabecera - controles, 640px)`; panel a la
    derecha a partir de 900 px, debajo del tablero por debajo (móvil y tablet vertical); tablet
    horizontal como escritorio. Sin scroll horizontal nunca; `env(safe-area-inset-*)` en móvil.
  - Entrada: tocar-tocar y arrastrar en táctil y ratón; casillas destino resaltadas; promoción con
    selector grande; objetivos táctiles ≥ 44 px; sin nada que dependa solo de `hover`.
  - Rendimiento móvil: `small-int8` por defecto en WASM cuando no hay WebGPU; aviso de tamaño y
    `navigator.connection.saveData` antes de descargar; el modelo no se carga hasta que el usuario
    pulsa "jugar"; presupuesto JS inicial ≤ 120 KB gz sin contar ORT.
  - Verificación: Lighthouse móvil y escritorio ≥ 0,95; E2E Playwright en tres viewports (390×844,
    820×1180 y 1280×800) con capturas comparadas en CI; axe limpio; Borja prueba en su móvil real en
    P2 y P6.

## Fase 1 · Jugar contra el modelo (todo en el navegador)

- **Tablero**: `cm-chessboard` 8.14 (MIT, SVG, responsive) + `chess.js` 1.4 (BSD-2) para reglas y
  jugadas legales. Se descarta `chessground` (GPL-3) para mantener la demo bajo MIT. Sin Stockfish en
  el navegador en la fase 1 (`stockfish` npm es GPL-3 y pesa 205 MB desempaquetado): la evaluación en
  vivo la da el encoder propio.
- **Modelo**: `onnxruntime-web` 1.30 (MIT) en un worker por modelo; WebGPU con fallback WASM
  (`import 'onnxruntime-web/webgpu'`; sesiones creadas en serie; `run` serializado). Registro de
  etapas con tamaños reales: `small-fp16` (~80 MB), `small-int8` (~40 MB), `elo`, `dpo`, `grpo`,
  `encoder` (~30 MB). Consentimiento antes de descargar con tamaño y licencia; Cache API; gestor para
  borrar modelos.
- **Jugada del modelo**: el worker recibe la secuencia de jugadas UCI (más tokens de Elo objetivo y
  color), devuelve logits del siguiente token; el hilo principal enmascara con `chess.js`, muestrea
  (temperatura/top-k configurables) y mueve. Opción "sin máscara" para enseñar la tasa de legalidad
  real (muestra la jugada ilegal propuesta y la marca).
- **Selectores**: etapa del modelo (base → maestros → Elo → DPO → GRPO), Elo objetivo (1200-2400 en
  pasos de 100, solo en modelos condicionados), adaptador LoRA de estilo, color, temperatura.
- **Paneles**: probabilidades de las 5 mejores jugadas del modelo sobre el tablero (flechas con
  opacidad), barra de valor del encoder y alerta de error (M3), historial en SAN, exportar PGN,
  cascada de latencia (tokenización → inferencia → muestreo), "qué ha salido por la red".
- **Modos**: jugar; "modelo contra modelo" (dos etapas enfrentadas, con Elo estimado en vivo por
  resultados acumulados en el navegador); puzles (cargar 50 puzles por dificultad desde JSON y ver la
  precisión del modelo en vivo).
- **Compartir**: enlace con `?stage=&elo=&fen=` que reproduce una posición; OG dinámico con el tablero.

## Fase 2 · Entrenador agéntico (backend opcional)

- Backend FastAPI en `rukh/src/rukh/serve/` (o desplegado en el servidor de Dokploy si cabe en CPU;
  el LLM comentarista corre en la 5090 de Borja vía puente local o se usa un modelo pequeño en CPU):
  `POST /coach/analyze` (SSE con las herramientas que va llamando el agente y el comentario),
  `POST /coach/report`, `GET /health`. Stockfish en el servidor (GPL, ejecutado como proceso).
- La demo muestra las trazas del agente (herramienta, argumentos, resultado) junto al tablero: es el
  "visor de trazas" de la lección A2.
- Sin cuentas ni datos personales; límite de peticiones; el backend puede estar apagado y la demo
  sigue funcionando en modo fase 1.

## Estructura

```
rukh-web/
  src/pages/index.astro            (una página; todo son islas Preact)
  src/islands/Board.tsx            (tablero + reglas + selección de jugada)
  src/islands/ModelPanel.tsx       (etapa, Elo, temperatura, consentimiento, progreso)
  src/islands/MoveList.tsx         (historial SAN, deshacer, nueva partida, PGN)
  src/islands/EvalBar.tsx          (encoder, P3)
  src/islands/More.tsx             (cajón: sin máscara, latencia, red, adaptadores)
  src/islands/Puzzles.tsx, Arena.tsx, Coach.tsx (P6 y fase 2)
  src/workers/decoder.worker.ts, encoder.worker.ts
  src/lib/chess-lm/tokenizer.ts    (vocabulario UCI fijo; misma lógica que en Python)
  src/lib/chess-lm/vocab.json      (sincronizado desde chorcat/rukh-tokenizer)
  src/lib/chess-lm/fixtures/       (paridad Python↔TS)
  src/lib/registry.ts              (etapas, tamaños, URLs del Hub)
  src/styles/tokens.css, base.css  (copia del portfolio; test de hash contra el curso)
  public/ort/<version>/            (runtime ORT autoalojado, copiado en prebuild)
  e2e/                             (Playwright: partida, viewports, axe)
  Dockerfile · nginx/ · .github/workflows/ci.yml
```

Mismo stack que el curso (Astro + Preact, tokens del portfolio, TypeScript 6), pero repo independiente:
no hay paquetes compartidos. Los tokens CSS se copian del portfolio (el curso guarda la fuente de verdad
en `docs/design-system.md`); el tokenizador TS vive aquí y el curso lo copia con `pnpm sync:tokenizer`;
el vocabulario y las fixtures vienen del repo ML vía el Hub. Un test de hash en CI avisa si alguna copia
diverge.

## Calidad y despliegue

- E2E Playwright en modo `?mock=1` (modelo simulado que juega la primera jugada legal): partida de 10
  jugadas con ratón y con toques (emulación táctil), cambio de etapa, puzle resuelto, axe, en los tres
  viewports con capturas; verificación con modelo real en Chrome por el controlador y en el móvil por
  Borja.
- nginx: COOP/COEP en `/` (WASM multihilo) e `include mime.types` con `wasm`/`onnx`; CSP con hashes;
  `connect-src` limitado a `huggingface.co`, `*.hf.co` y el backend propio.
- Dominio `rukh.borjaglez.com` en Dokploy: app propia desde `borja-glez/rukh-web`, Dockerfile en
  la raíz, auto-deploy en `main`.
- Licencias mostradas en la página: modelos Apache-2.0, datos CC0 (atribución a Lichess), librerías
  MIT/BSD.
