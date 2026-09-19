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

### D-013 · `style-src-attr 'unsafe-inline'` en las dos webs; `script-src` y `style-src` solo por hash
- **Qué:** la CSP de `rukh-lab` y de `rukh-web` permite estilos en línea únicamente en atributos
  (`style-src-attr 'unsafe-inline'`). `script-src` y `style-src` siguen siendo `'self'` más hashes
  generados por Astro, sin `'unsafe-inline'`, y los tests E2E de CSP lo comprueban.
- **Por qué:** en el curso, expressive-code colorea la sintaxis con atributos `style="--0:…"` por
  token, que un hash no puede cubrir; en la demo, cm-chessboard posiciona la pieza arrastrada con
  `setAttribute('style', …)`. Ninguno de los dos admite una alternativa por clase sin bifurcar la
  librería. Un atributo `style` no ejecuta código y no puede cargar recursos externos con
  `img-src 'self' data:` y `font-src 'self'`, así que el riesgo se limita a la presentación.
- **Si está mal:** son dos líneas de `styleDirective` en cada `astro.config.mjs`; si expressive-code
  o cm-chessboard pasan a clases o a hojas con hash, se elimina `'unsafe-inline'` y el test de CSP
  se endurece para prohibirlo también en `style-src-attr`.

### D-014 · Preparación de la CSP para P2 (ORT en el navegador)
- **Qué:** cuando la demo cargue `onnxruntime-web` (P2), `script-src` necesitará
  `'wasm-unsafe-eval'` (la compilación de WASM está bloqueada por un `script-src` con hashes sin él)
  y hay que tener en cuenta que la CSP en `<meta>` no gobierna los *dedicated workers*: si se quiere
  cubrir el worker del modelo, la CSP debe ir como cabecera en nginx (`Content-Security-Policy`)
  además de, o en lugar de, la meta que genera Astro.
- **Por qué:** hallazgo de la revisión de P0; se anota aquí para que P2 no lo herede como sorpresa.
- **Si está mal:** nada que deshacer; es una nota para el plan de P2.

### D-015 · Lighthouse en Windows: `lhci` falla al limpiar el perfil de Chrome
- **Qué:** `lhci autorun` (chrome-launcher 1.2.1) termina con `EPERM` al borrar su perfil temporal
  en Windows, después de generar los informes. En local, Lighthouse se ejecuta contra un Chrome
  headless lanzado aparte (`chrome.exe --headless=new --remote-debugging-port=9333 --user-data-dir=…`)
  con el CLI de Lighthouse y `--port=9333`. La CI (ubuntu) usa `lhci` tal cual.
- **Por qué:** fallo conocido de chrome-launcher en Windows; no afecta al runner.
- **Si está mal:** los scripts `lighthouse*` de los `package.json` no cambian; es solo el modo de
  ejecutarlos en la máquina de referencia.

## Verificación de P0 (2026-09-18, máquina de referencia)

Evidencia obtenida por el controlador, no por subagentes:

- `rukh`: `uv run rukh info` → Python 3.12.11, torch 2.11.0+cu128, `cuda True`, GPU `NVIDIA GeForce
  RTX 5090`; `uv run rukh engine check --elo 1400 --plies 40` → Stockfish 19, `UCI_Elo` 1320-3190,
  partida real (1-0 en 35 plies); `uv run pytest -m unit -q` → 57 passed; ruff limpio.
- `rukh-web`: `docker build` + `docker run` (nginx) → cabeceras `Content-Security-Policy:
  frame-ancestors 'none'`, HSTS, `X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`,
  COOP `same-origin`, COEP `require-corp`; `pieces/standard.svg` como `image/svg+xml`. Suite E2E
  ejecutada contra el contenedor en **Chrome real** (`channel: 'chrome'`, headed, DPR 1,5) en los tres
  viewports: 24 pruebas verdes (partida por clic, por toque y por arrastre, deshacer con blancas y
  con negras, nueva partida, exportar PGN, axe, CSP, layout 390/820/1280, 960×900 y 844×390).
  Lighthouse contra el contenedor (Chrome 153): móvil 0,98 / 1 / 1 / 1; escritorio 1 / 1 / 1 / 1.
- `rukh-lab`: contenedor con las mismas cabeceras, `gzip`, `/_astro/` inmutable, 404 y
  `/curso` → `/curso/` relativo; tema persistido sin flash tras recarga (sin errores de CSP en
  consola); lección 0 con las salidas reales del CLI; 0 botones por debajo de 44 px a 390 px.
  Lighthouse: `/`, `/curso/` y la lección → móvil 0,98-0,99 / 1 / 1 / 1; escritorio 1 / 1 / 1 / 1.
- Pendiente de Borja (punto de intervención de P0): repos en GitHub, DNS, apps en Dokploy, merge de
  `p0-scaffold` y prueba en su móvil real. La CI de los tres repos se verá en verde tras el push.

## P1 · Datos (2026-09-18/19)

### D-016 · La rama `p1-datos` nace de `p0-scaffold`
- **Qué:** en los tres repos, `p1-datos` sale de `p0-scaffold` en lugar de `main`.
- **Por qué:** P0 sigue sin fusionar (falta el punto de intervención de Borja: repos en GitHub,
  DNS, Dokploy y prueba en su móvil). Partir de `main` habría dejado P1 sin el scaffold, el CLI,
  la configuración pydantic ni la CI que P1 usa en cada paso.
- **Si está mal:** al fusionar P0 primero y P1 después no hay conflicto; si se decidiera fusionar
  P1 antes, habría que rebasarla sobre `main` (historia lineal, sin cambios de contenido).

### D-017 · Tope de 3 000 000 de partidas por mes en el recorte de Lichess
- **Qué:** `configs/data/lichess-2025-01-02.yaml` lleva `limit: 3000000`, así que cada mes aporta
  como mucho 3 M partidas (los primeros ~12 ficheros parquet del mes, es decir los primeros días)
  y los dos meses suman 6 M. El manifiesto lo registra como recorte temporal.
- **Por qué:** sondeo del 2026-09-18 sobre el primer fichero de 2025-01: de 1 394 617 partidas,
  **251 618 (18 %)** pasan los filtros de `docs/spec/01` (ambos Elo ≥ 1800, base ≥ 180 s,
  terminaciones normales). Extrapolado a los 72 ficheros del mes son ≈ 18 M partidas/mes, muy por
  encima de los 3-6 M que pide el spec para los dos meses; sin tope, `fetch` tardaría horas y
  llenaría el disco sin mejorar el entrenamiento de M2.
- **Si está mal:** subir o quitar `limit` y relanzar `fetch`; el resto del pipeline no cambia. El
  sesgo que introduce es temporal (solo los primeros días del mes), no de fuerza ni de ritmo.

### D-018 · Los pares DPO salen de las líneas múltiples del dataset de evaluaciones
- **Qué:** `rukh data pairs` construye `chosen`/`rejected` a partir de las varias líneas por FEN de
  `Lichess/chess-position-evaluations` (≈ 7 de media), sin ejecutar Stockfish. `chosen` es la
  primera jugada de la mejor línea y `rejected` la de una línea al menos 100 centipeones peor para
  el que mueve.
- **Por qué:** el dataset ya trae multi-PV con profundidad alta; generar los pares con el motor
  costaría horas de GPU/CPU por unos datos que ya existen y que además son reproducibles por
  cualquiera. `docs/spec/01` no exigía Stockfish para este paso.
- **Si está mal:** los pares se pueden regenerar con el motor (`rukh.engine`) reutilizando las
  mismas posiciones; el esquema de columnas no cambiaría.

### D-019 · Empaquetado en flujo con ventanas alineadas a `<bos>`
- **Qué:** `pack.py` escribe un único flujo `tokens.npy` (uint16) con las partidas codificadas
  enteras y concatenadas más `starts.npy` con el desplazamiento de cada `<bos>`; el `PackedDataset`
  corta ventanas de `block` tokens que empiezan en un `<bos>`. Con `block = 200`, los tokens de una
  partida más allá de la posición 200 **no se ven** durante el entrenamiento.
- **Por qué:** un flujo continuo evita rellenar cada partida hasta el bloque (menos padding, menos
  memoria) y alinear a `<bos>` garantiza que el modelo siempre ve el prefijo de control
  (`<bos>`, Elo de ambos, jugadas) en el mismo sitio. `docs/spec/01` admite truncar a 200 tokens.
- **Si está mal:** `start_at_game=False` ya recorre el flujo en bloques consecutivos (ve todas las
  partidas completas, a cambio de ventanas que empiezan a mitad de partida); no hay que reempaquetar.

### D-020 · El lab de M1 aplaza la máscara causal a M2
- **Qué:** la lección M1 del curso se queda en datos y tokenización (labs 1-5 de `docs/spec/03`) y
  no incluye el lab 4 tal como está escrito allí: la máscara causal se explica y se implementa en
  M2, junto con la atención.
- **Por qué:** la máscara solo se entiende con el bloque de atención delante; en M1 sería un
  fragmento de código sin contexto y alarga una lección que ya cubre tres esquemas de tokenización,
  el empaquetado y los dataloaders.
- **Si está mal:** es mover una sección de `m2` a `m1` en el MDX; no afecta al código.

### D-021 · El BPE se entrena sobre la partida UCI sin espacios
- **Qué:** el esquema `bpe` codifica `e2e4e7e5g1f3…` (jugadas concatenadas), no `e2e4 e7e5 g1f3…`.
  La fixture de paridad (`artifacts/tokenizer/fixtures/games.json`) usa ese mismo texto y los dos
  tokenizadores TS lo replican.
- **Por qué:** el pre-tokenizador obligatorio es `WhitespaceSplit`, así que con espacios ninguna
  fusión puede cruzar de una jugada a la siguiente y el lab 5 ("mira qué aperturas enteras aprende
  el BPE", `docs/spec/03`) se queda sin objeto. Sin espacios, cada partida es una palabra y las
  fusiones largas son secuencias de apertura reales.
- **Si está mal:** cambiar `bpe_text()` en `src/rukh/tokenize/bpe.py`, regenerar `bpe.json` y la
  fixture, y sincronizar las dos webs; los tests de paridad detectan cualquier olvido.

### D-022 · Los scripts de los labs de M1 viven en `labs/m1/`
- **Qué:** `labs/m1/{explore,loader_check,bpe_merges}.py` son copias exactas de los bloques de
  código de la lección M1, con un README que dice de qué paso del pipeline depende cada uno.
- **Por qué:** la lección pide guardarlos en el repo `rukh` para ejecutarlos; tenerlos versionados
  evita que el lector los copie mal y permite que `ruff` los revise.
- **Si está mal:** si la lección y el script divergen, manda la lección; el README lo advierte.

## P2 · Decoder (2026-09-19)

### D-023 · El código de P2 se adelanta mientras se descarga el recorte de P1
- **Qué:** la rama `p2-decoder` nace de `p1-datos` y se desarrolla el modelo, el entrenamiento, la
  evaluación y la exportación mientras `rukh data fetch` sigue descargando los dos meses (tarda
  varias horas: son ~12 GB de parquet remoto por mes para quedarse con 3M partidas). El código de P1
  ya está completo y revisado; lo único pendiente de P1 es la ejecución real y la publicación.
- **Por qué:** el protocolo de hitos (plan → rama → tareas → revisión → una ola → parada) se respeta
  en los dos hitos; lo que se solapa es la espera de una descarga, no el trabajo. La alternativa era
  dejar la máquina parada varias horas.
- **Si está mal:** `p2-decoder` se rebasa sobre `p1-datos` cuando P1 cierre; los artefactos que P1
  regenera (`artifacts/tokenizer/*`, `artifacts/web/tokenizer-stats.json`) se traen con un merge.

### D-024 · `torch.compile` no está disponible en este Windows: el bucle entrena en eager
- **Qué:** `maybe_compile` intenta `torch.compile` con una pasada de calentamiento (un lote de la
  forma real) antes del bucle. En esta máquina Inductor falla con
  `torch._inductor.exc.TritonMissing` (no hay Triton ni MSVC, y no hay `nvcc`), así que
  `dynamo.config.suppress_errors = True` se traga el error en 3 s y el modelo se ejecuta en eager
  aunque `compile: true` siga en el YAML. El rendimiento medido en la 5090 sin compilar es de
  **440 000 tokens/s** con lotes de 256×200, o sea que `small` ve 1 500 M de tokens en algo más de
  una hora.
- **Por qué:** el plan de P2 ya preveía este riesgo ("`torch.compile` en Windows sin `nvcc`: si
  falla, `compile: false`"). Dejar la opción a `true` y fallar hacia eager es mejor que apagarla en
  la configuración: el mismo YAML compila en una máquina con Triton (Linux, o Windows con Triton
  instalado) y aquí no rompe nada. La pasada de calentamiento existe para que el fallo ocurra en un
  sitio controlado y no a mitad del entrenamiento.
- **Si está mal:** instalar Triton para Windows y volver a medir; si compilar diera menos del 10 %
  de mejora, poner `compile: false` en los dos YAML y ahorrarse la pasada de calentamiento. La
  cifra de tokens/s está en MLflow (`tokens_per_s` y `real_tokens_per_s`) de cada run.

### D-025 · El harness de Elo adjudica las partidas cortadas por el contexto
- **Qué:** una partida que llega al límite de contexto (`block - 4` = 196 medias jugadas) terminaba
  con `*` y `score_of` la contaba como tablas. Ahora `play_game` devuelve el FEN final y
  `record_of` la adjudica: con el motor del propio peldaño (`engine.analyse` a profundidad 8) gana
  quien tenga 200 centipeones o más; sin motor, el recuento de material con el mismo margen (2
  peones). El informe y `results.json` dicen cuántas partidas se cortaron y cuántas se
  adjudicaron, por peldaño. Además los cuatro peldaños por debajo de 1320 se declaran en el
  informe, en `results.json` y en la card como anclas nominales de `Skill Level` (no son fuerzas
  medidas), el tiempo por jugada sube de 0,05 s a 0,1 s, y cuando todas las partidas se ganan (o
  se pierden) se marca `separated` y se publica una cota unilateral en vez de un intervalo del
  95 % de anchura cero.
- **Por qué:** medio punto por partida cortada empuja el ajuste hacia el centro de los peldaños,
  justo en el umbral de 1200 que GOAL pone como criterio de aceptación: una tabla regalada por
  quedarse sin contexto no es una tabla. Y un intervalo simétrico calculado sobre resultados
  separados es un número inventado; el bootstrap devuelve siempre el mismo tope. Las anclas y el
  tiempo por jugada se dicen en voz alta porque el intervalo solo cubre el ruido de muestreo: la
  incertidumbre real de la calibración de Stockfish no está dentro.
- **Si está mal:** el margen (`ADJUDICATION_CP`) y la profundidad (`ADJUDICATION_DEPTH`) están en
  `src/rukh/infer/game.py` y la caché de evaluación ya incluye la configuración en su clave, así
  que cambiarlos vuelve a jugar las partidas en vez de reutilizar las viejas. Si se prefiere no
  adjudicar, basta con no llamar a `adjudicate` en `record_of`: el campo `cut` seguiría estando.

### D-026 · La legalidad se publica dos veces: argmax y muestreada
- **Qué:** `rukh eval` mide la legalidad sin máscara con las dos definiciones y las publica las
  dos: `legality_argmax` (el token más probable, sin temperatura ni top-k) es el titular y
  `legality_sampled` (temperatura 0,6 y top-k 20, como juega la demo) va al lado, con las dos
  definiciones escritas en `report.md`, en `results.json` y en la model card.
- **Por qué:** el listón del ≥ 99 % de `GOAL.md` habla de lo que saben los pesos, y eso es el
  argmax; medirlo con un muestreo a temperatura 0,6 mezcla una propiedad del modelo con un ajuste
  del muestreador y siempre da peor. Publicar solo el argmax sería quedarse con el número bonito,
  así que van los dos con su definición al lado.
- **Si está mal:** `legality(..., mode=...)` acepta las dos y el informe imprime lo que haya; si
  algún día el listón se redefine sobre el muestreo, solo cambia qué columna es el titular en
  `WebRow.legality`.

### D-027 · El exportador moderno de ONNX perdía contra la consola de Windows
- **Qué:** `torch.onnx.export(dynamo=True)` imprime marcas de verificación Unicode mientras
  trabaja. Con la consola en cp1252 (el valor por defecto en Windows) eso lanzaba
  `UnicodeEncodeError` **dentro** del exportador, `export_onnx` lo tomaba por un fallo del
  exportador y caía al tracer antiguo de TorchScript. `src/rukh/export/onnx.py` reconfigura ahora
  los flujos con `errors="replace"` mientras dura la exportación, y un test comprueba que el
  exportador que corre es `dynamo` y que no hubo aviso.
- **Por qué importa:** el grafo del tracer antiguo produce un `model-fp16.onnx` que onnxruntime
  **rechaza** (`Type (tensor(float16)) … does not match expected type (tensor(float))`), y ese es
  justo el fichero que carga la demo. Con el exportador moderno, fp16 e int8 cargan y eligen la
  misma jugada que PyTorch. El eje dinámico de secuencia lo comprueba `verify_dynamic_seq`
  ejecutando el fichero a **dos** longitudes (`sequence_lengths` devuelve la longitud trazada y
  una vecina legal), no a tres: la comprobación manual a 8/16/64 con paridad de 1e-7 la hizo el
  controlador aparte y no está en el código.
- **Si está mal:** se quita el contexto `_utf8_console()` y se acepta el tracer antiguo, pero
  entonces hay que exportar el fp16 de otra manera.

### D-028 · `onnx`, `onnxscript`, `onnxruntime`, `onnxconverter-common` y `safetensors` son dependencias
- **Qué:** estaban en el spec pero no en `pyproject.toml`, así que toda la ruta de exportación
  nunca se había ejecutado (seis tests saltaban). Ahora son dependencias del proyecto y la suite
  corre entera: 355 tests sin saltos.
- **Si está mal:** `onnxscript` solo hace falta para el exportador moderno; sin él se vuelve al
  caso de D-027.

### D-029 · nginx sirve `.mjs` como `text/javascript` por una regla propia
- **Qué:** una `location ~* \.mjs$` con `default_type text/javascript` en `nginx/default.conf`, y
  `e2e/fixtures/coi-server.mjs` (el servidor del proyecto E2E `isolated`) lee esas declaraciones
  para que la prueba se caiga si la regla desaparece.
- **Por qué:** el `mime.types` que trae nginx (revisado en 1.29) tiene entrada para `js` y ninguna
  para `mjs`, así que el cargador de ONNX Runtime que vive junto al `.wasm`
  (`ort-wasm-simd-threaded.asyncify.mjs`) salía como `application/octet-stream`. Con
  `X-Content-Type-Options: nosniff` —que está en `security-headers.conf` y no se va a quitar— el
  navegador se niega a evaluarlo, el `import()` dinámico que hace el worker falla y la sesión no
  llega a crearse nunca. Es un fallo que solo aparece en producción: en `astro dev` y en `vite
  preview` el tipo lo pone la herramienta, no nginx.
- **Si está mal:** la alternativa es añadir `mjs` al `mime.types` de la imagen, pero eso es editar
  un fichero de nginx dentro del Dockerfile; la `location` está a la vista y se prueba.

### D-030 · ONNX Runtime se publica de un solo hilo y con dos artefactos
- **Qué:** el worker fija `ort.env.wasm.numThreads = 1` y `ort.env.wasm.proxy = false`, y
  `scripts/copy-assets.mjs` copia a `public/ort/<versión>/` **solo** el par asyncify
  (`ort-wasm-simd-threaded.asyncify.mjs` y `.wasm`, 27 MB) más un `.wasm.gz` precomprimido que
  nginx sirve con `gzip_static` (6,6 MB por la red). `postbuild` (`copy-assets.mjs --verify`) lee
  del bundle construido todos los `ort-wasm*.{mjs,wasm}` que el runtime puede pedir y rompe la
  build si alguno no está publicado.
- **Por qué:** `onnxruntime-web/webgpu` usa esa misma build asyncify para WebGPU y para el respaldo
  en WASM, así que las otras tres variantes (`threaded`, `jsep`, `jspi`) serían 60 MB de más en la
  imagen. Pedir más hilos bajo aislamiento de origen cruzado (COOP/COEP están puestos para el WASM
  multihilo) hace que ORT vaya a buscar justo esos artefactos que no publicamos, y el worker de
  proxy sería un cuarto fichero para mover el trabajo fuera del hilo principal, que es donde este
  código ya está. Un hilo es además suficiente: WebGPU es el camino rápido y el respaldo en WASM
  se acepta lento a propósito. El `.wasm` no entra en `gzip_types` (comprimir 27 MB en cada
  petición sería peor que servirlos), de ahí el `.gz` escrito en tiempo de build.
- **Si está mal:** subir `numThreads` obliga a copiar la variante `threaded` y a mantener el
  aislamiento de origen cruzado en todas las rutas; `ORT_ASSETS` en `scripts/copy-assets.mjs` es la
  única lista que hay que tocar, y `--verify` avisa en la build siguiente.

### D-031 · Tamaños provisionales de las etapas y un `.onnx` de juguete versionado
- **Qué:** `STAGE_SIZE_MB` en `src/lib/registry.ts` declara 6 / 80 / 40 MB para `tiny-int8`,
  `small-fp16` y `small-int8`: son las estimaciones del plan, no medidas, y el controlador las
  actualiza en ese único sitio cuando `rukh export --fp16 --int8` diga los tamaños reales. Solo
  alimentan el texto del consentimiento y el respaldo de la barra de progreso cuando la respuesta
  no trae `content-length`, nunca la descarga. Aparte, `public/test/toy-decoder.onnx` (un decoder de
  `n_layer=1, n_head=2, d_model=8` con el vocabulario real de 2 030 y `block` 200) está versionado
  en el repo web para que el E2E recorra el camino real del worker sin bajarse 40 MB del Hub; se
  llega a él solo con `?stage=test` y no aparece en el selector.
- **Por qué:** el consentimiento tiene que decir un tamaño antes de que exista el modelo, y mentir
  con un `0 MB` sería peor que una estimación etiquetada como tal. El juguete se versiona porque un
  E2E que dependa de la red (o de un entrenamiento previo) no es un E2E: con él, la prueba del
  contrato del modelo, el fallback a WASM y la cascada de latencia se ejecutan en CI en segundos.
  Se regenera con `rukh.export.export_onnx`, la misma función que produce los modelos reales, para
  que lleve los mismos metadatos `rukh_*` y salga del mismo exportador (dynamo).
- **Si está mal:** los tamaños viven en una constante con un test que comprueba que cada etapa la
  usa; cambiarlos es una línea. Si el juguete se queda desfasado respecto al contrato real, se
  vuelve a exportar con `export_onnx` sobre un `MoveDecoder` de juguete y se recomprueba que pese
  menos de 200 KB. Una capa y no dos: el exportador dynamo cuelga de cada nodo su traza de pila
  (`pkg.torch.onnx.stack_trace`), que en un modelo tan pequeño son unos 1,3 KB por nodo y pesan más
  que los pesos; con dos capas el fichero se iba a 224 KB. Quedan 154 KB, menos que los 173 KB del
  fichero anterior, que además lo había escrito el tracer antiguo (D-027).
