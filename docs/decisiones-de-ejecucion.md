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

### D-032 · La exportación borra el fichero de datos externos que deja el exportador
- **Qué:** el exportador dynamo escribe los pesos en `<nombre>.onnx.data` y apunta los
  inicializadores allí; al escribir los metadatos, `onnx.save` los vuelve a meter dentro del
  `.onnx`, así que el sidecar queda muerto. `set_metadata` lo borra ahora, y solo cuando ha
  comprobado que ningún inicializador sigue siendo externo. Un test verifica que la carpeta de
  salida contiene exactamente un fichero.
- **Por qué:** ese sidecar pesa lo mismo que el modelo y se habría subido al Hub y servido al
  navegador sin que nadie lo leyera nunca.
- **Si está mal:** se quita el borrado y hay que publicar los dos ficheros juntos.

### D-033 · DuckDB escribe sin preservar el orden de inserción
- **Qué:** `rukh.data.db.connect` fija `preserve_insertion_order = false` (configurable en
  `pipeline.yaml`).
- **Por qué:** con el valor por defecto, un `COPY` de un escaneo remoto grande retiene el
  resultado entero en memoria para que las filas de salida conserven el orden de entrada. La
  primera descarga real de un mes llegó a **20 GB de memoria residente** para un fichero de 2,5 GB
  y el sistema empezó a matar procesos en segundo plano. Ningún paso posterior depende del orden
  de las partidas (los splits son por mes y los muestreos llevan semilla), así que ese buffer no
  compra nada.
- **Si está mal:** si alguna vez hace falta un orden determinista en el parquet, se pone a `true`
  en la configuración de ese paso y se paga la memoria.

### D-034 · La descarga de P1 se dejó terminar con la configuración antigua
- **Qué:** la descarga de los dos meses se lanzó antes de que existiera el helper de DuckDB, así
  que corre sin límite de memoria y sin D-033. Se decidió **no** reiniciarla: enero (2,5 GB) ya
  estaba escrito y febrero iba por la mitad.
- **Por qué:** reiniciar costaba otras dos horas de descarga y el proceso seguía avanzando. Se
  liberó memoria parando los contenedores de verificación y se dejó de lanzar trabajos en segundo
  plano mientras durase.
- **Si está mal:** basta con volver a lanzar `rukh data fetch` para el mes que falte; el manifiesto
  se reescribe con los dos meses.

## P3 · Encoder (2026-09-19)

### D-035 · El esquema `squares` son 47 tokens y 69 posiciones, con el enroque en un solo token
- **Qué:** `SQUARE_VOCAB` tiene **47** tokens (`<pad>`, `<mask>`, `<cls>`, `<empty>`, las 12
  piezas, los 2 turnos, las 16 combinaciones de enroque, `ep:none` más los 8 ficheros, y 4 tramos
  de reloj) y toda secuencia mide exactamente **69** posiciones: `<cls>`, las 64 casillas en orden
  por columnas (a1…a8, b1…, h8), turno, enroque, al paso y reloj.
- **Por qué:** la enumeración del plan (64 + turno + 4 enroques + al paso + reloj) suma 71 y el
  número duro del plan era 69. Los cuatro derechos de enroque son cuatro bits de un mismo hecho,
  así que viajan en **un** token de 16 valores; el hueco que eso libera se usa para un `<cls>`
  propio, que le da a `pool("cls")` un ancla real en este esquema en vez de tomar prestada la
  casilla a1. El orden de las casillas es el de `uci_vocab.squares()`, el mismo que usan los
  extremos de las jugadas: una sola enumeración en todo el proyecto.
- **Nota:** el token 68 (el reloj) es **constante** en todo el conjunto de datos. Un `fen4` no
  lleva contadores y la tabla de P1 es toda `fen4`, así que siempre sale `clock:0`. Se mantiene en
  el diseño porque una posición en vivo de la demo sí trae los contadores, y una representación
  que cambia de forma entre entrenamiento y servicio sería peor que un hueco constante.
- **Si está mal:** `vocab_hash()` fija la identidad del esquema; cambiar la enumeración invalida
  todo encoder entrenado con ella, exactamente como el vocabulario UCI de P1.

### D-036 · `ignore_index = -100` porque `0` es `<pad>` en los dos esquemas
- **Qué:** la pérdida de MMM usa `MMM_IGNORE_INDEX = -100` para "aquí no se predice nada", no el
  `0` del decoder.
- **Por qué:** `0` es `<pad>` en los **dos** esquemas, así que no puede hacer además de "no
  predicho" sin que la pérdida pierda la capacidad de distinguir los dos casos. La razón que se
  escribió primero —"`0` es etiqueta válida en casillas y no en jugadas"— era falsa: `0` es
  `<pad>` en los dos, y las secuencias de `squares` ni siquiera llevan relleno, porque miden 69
  siempre. El decoder se lo puede permitir porque su objetivo es el token siguiente de un flujo
  empaquetado y allí `<pad>` nunca es objetivo; aquí la distinción tiene que ser explícita, y
  `-100` está fuera de todo vocabulario.
- **Si está mal:** es una constante en `rukh.models.encoder` y el `ignore_index` de una sola
  llamada a `cross_entropy`.

### D-037 · El preentrenamiento MMM solo corre sobre `moves`
- **Qué:** `MmmConfig.input` es `Literal["moves"]`: no hay preentrenamiento enmascarado del
  esquema `squares`.
- **Por qué:** MMM necesita un flujo de tokens empaquetado y P1 solo produce uno, el de jugadas
  UCI (`data/tokens/uci`), que además es el mismo que entrena el decoder, de modo que los dos
  modelos ven las mismas partidas. Las posiciones de P1 viven en un parquet de FENs, no en un
  flujo; enmascarar casillas sueltas de un tablero sería otro objetivo (más cerca de un
  autoencoder de posiciones) y otro hito.
- **Consecuencia:** un encoder de `squares` empieza siempre desde pesos aleatorios, y esa es la
  línea base honesta contra la que se compara el preentrenado.
- **Si está mal:** hace falta empaquetar un flujo de tokens de casillas en P1 y quitar el
  `Literal`.

### D-038 · Las cabezas afinan los dos esquemas, y el prefijo de `moves` es *una* línea
- **Qué:** `rukh.train.heads` acepta `input: moves` además de `squares`.
  `rukh.data.labels.game_moves` vuelve a unir las partidas de P1
  (`data/uci/year=*/month=*/games.parquet`, semi-join por `game_id`, **una fila por partida**) y
  cada ítem es `[<bos>, elo(blancas), elo(negras)] + uci.split()[:ply]`, recortado por la
  izquierda con `infer.sampler.prompt_ids` (que conserva las tres cabeceras) y rellenado por lote
  con su `attention_mask`.
- **Por qué:** sin esto el encoder preentrenado con MMM (`moves`) **no tenía dónde ir**: las
  cabezas solo sabían leer `squares`, así que el preentrenamiento quedaba desconectado del resto
  de P3 y la comparación entre las dos representaciones que pide el spec no existía. La
  alternativa —preentrenar MMM sobre casillas— es D-037.
- **Salvedad, documentada en el código, en la card y aquí:** la tabla supervisada está
  deduplicada por `fen4`, así que el prefijo es **una** línea que llega a esa posición, no
  necesariamente la de la partida etiquetada. La posición, el valor y el veredicto de error son
  los mismos; la historia puede no serlo.
- **`game_moves` queda fuera de `build_labels`** a propósito: el camino `squares` no debe pagar
  una unión con dos meses de partidas que no usa.
- **Si está mal:** `load_encoder_for` compara el esquema del checkpoint con el de la configuración
  y se niega en vez de reinterpretar el vocabulario; volver atrás es restringir otra vez
  `HeadsConfig.input`.

### D-039 · La curva por número de etiquetas viaja dentro del checkpoint
- **Qué:** `label_curve` escribe sus puntos (`label_curve=[...]`) en **todos** los checkpoints de
  la última tirada (la de más etiquetas) con `train.checkpoint.attach_payload`, y
  `rukh eval encoder` los lee de ahí.
- **Por qué:** antes nadie escribía esa clave, así que `label_curve_points()` devolvía siempre
  `[]`, la tabla "labels needed" no se dibujaba nunca y la curva —la lección del módulo y un
  entregable de `GOAL.md`— era inalcanzable. Un número que solo existió en una línea de log es un
  número que nadie puede poner en una tabla.
- **Si está mal:** la clave es una sola (`CURVE_KEY`, con un test que ata al lector y al escritor)
  y la alternativa era un `curve.json` al lado de la tirada, que se pierde al mover los pesos.

### D-040 · Se comprueban los dos criterios de `GOAL.md`, y el valor va contra la puntuación acotada
- **Qué:** `EncoderResult` lleva `meets_goal` (margen de F1 ≥ 5 puntos),
  `value_correlation_meets_goal` (≥ 0,80) y `meets_all_goals`. La correlación se mide contra
  `tanh(cp / value_scale)` —la puntuación acotada con la que se entrena la cabeza— y el titular es
  **Spearman**, con Pearson al lado.
- **Por qué:** `GOAL.md` tiene dos criterios numéricos y solo se comprobaba el F1. Y correlacionar
  una salida acotada en `(-1, 1)` contra el `cp` crudo compara peras con manzanas: un mate vale
  ±9 99x y un puñado de filas decidiría el Pearson de todo el conjunto. Spearman es además la
  lectura honesta de "el modelo sabe qué posición es mejor", que es de lo que habla el listón.
- **Si está mal:** son dos constantes (`GOAL_MARGIN`, `GOAL_VALUE_CORRELATION`) y el informe
  imprime contra qué se correlacionó cada fila.

### D-041 · La caché de la heurística se indexa por la posición, no por `game_id:ply`
- **Qué:** la clave de un veredicto de la línea base es `fen|fen_before|last_move`, y
  `heuristic_fields()` incluye ahora `labels`.
- **Por qué:** con `f"{game_id}:{ply}"`, apuntar `positions_eval` a otro parquet reutilizaba en
  silencio veredictos de posiciones distintas que caían en el mismo `(partida, ply)`. Con la clave
  por posición ocurre lo contrario, que es lo deseable: dos tablas que comparten una posición
  comparten la respuesta. `fen_before` entra en la clave porque `fen` más `last_move` no permite
  recuperar la pieza capturada, y la diferencia de material entre las dos posiciones es
  literalmente todo el juicio de la heurística.
- **Efecto visible:** el fixture de dos copias de la misma partida pasó de 16 filas de caché a 8,
  y hay un test que lo fija.
- **Si está mal:** `HEURISTIC_KEY` sigue existiendo para invalidar todo de golpe.

### D-042 · La exportación del encoder: `clear_value_info`, `_align_cast_outputs` y lote 2
- **Qué:** tres detalles de `rukh.export` que solo aparecen con el encoder.
  - `clear_value_info` borra las anotaciones de forma de los valores intermedios antes de
    cuantizar. El exportador moderno anota la forma del ejemplo trazado, no la del grafo dinámico;
    onnxruntime las ignora, pero `quantize_dynamic` vuelve a inferir formas en modo estricto y
    rechaza el fichero, así que el int8 del encoder moría en una anotación opcional.
  - `_align_cast_outputs` arregla los nodos `Cast` que `convert_float_to_float16` deja diciendo
    `to=float32` mientras su salida ya está declarada float16. El encoder lo provoca porque el
    *pooling* castea la máscara de relleno al tipo del estado oculto; sin el arreglo, onnxruntime
    rechaza el fp16 con `Type (tensor(float16)) … does not match expected type`.
  - el grafo se traza con un **lote de 2**, no de 1: `torch.export` especializa una dimensión cuyo
    ejemplo vale 1 (la especialización 0/1), con lo que el eje del lote quedaba fijo y la segunda
    posición de la demo fallaba dentro de un `reshape`. Con 2 el eje sigue siendo dinámico y el
    fichero acepta igualmente un lote de 1. El ejemplo son ids reales, no ceros: `<pad>` en todas
    las posiciones es una fila enteramente enmascarada, que el encoder rechaza a propósito.
- **Si está mal:** los tres tienen su test; el de `squares` comprueba además que no hay eje de
  secuencia que hacer dinámico (siempre 69 tokens).

### D-043 · La guarda de la máscara se salta bajo `export` **y** bajo `compile`
- **Qué:** `PositionEncoder._key_mask` rechaza una fila enteramente enmascarada (softmax daría
  NaN), pero la comprobación se salta cuando `torch.compiler.is_exporting()` **o**
  `torch.compiler.is_compiling()`.
- **Por qué:** leer un tensor para decidir si lanzar es justo lo que una captura de grafo no puede
  representar. Bajo `torch.export` mandaba el exportador al tracer antiguo (D-027); bajo
  `torch.compile` era peor por silencioso: Dynamo no puede probar la condición y rompía el grafo
  en **cada** paso del bucle de MMM, o sea una sincronización y dos medios grafos por paso a
  cambio de una comprobación que ya había pasado. En modo *eager* no cambia nada y el test del
  `ValueError` sigue igual.
- **Si está mal:** es una función de una línea (`_tracing()`).

### D-044 · El sondeo de compilación pasa la máscara que pasa el bucle
- **Qué:** el `probe` de `train.mmm` llama a `masked_step(..., model.padding_mask(warm))`, y
  `train.heads` compila con un lote real del propio dataset, máscara incluida.
- **Por qué:** sondear con `attention_mask=None` compila un grafo que el bucle no ejecuta nunca y
  se paga una recompilación en el paso 1, que es exactamente el coste que el sondeo existe para
  adelantar.
- **Además:** `train_heads` llama ahora a `maybe_compile`, así que el `compile: false` de
  `configs/train/encoder-heads.yaml` describe un mando que existe; `tokens_dir`, la única clave
  heredada que este bucle no usa, se excluye de los parámetros que van a MLflow y al checkpoint.
- **Y:** bajo `last-n`, `set_training_mode` deja en `eval` el prefijo congelado y solo las últimas
  `last_n` capas y la norma final conservan su dropout. Antes el prefijo congelado seguía en
  `train()` y metía ruido que nadie podía aprender ni absorber.

### D-045 · Los fixtures compartidos de los tests viven en `tests/unit/helpers_labels.py`
- **Qué:** `GAME`, `game_rows` y `source_frame` se importan como `from helpers_labels import ...`.
- **Por qué:** `tests/` no es un paquete (no hay `__init__.py`), así que
  `from tests.unit.test_eval_encoder import ...` fallaba con `ModuleNotFoundError` y dejaba la
  suite en rojo. pytest añade `tests/unit` al `sys.path`, de modo que el import desnudo funciona
  sin convertir los tests en paquete, que habría cambiado la resolución de imports de los demás
  ficheros. `ruff` necesita saberlo: `known-first-party` incluye `helpers_labels`.
- **Si está mal:** la alternativa es añadir `__init__.py` a `tests/` y `tests/unit/` y volver al
  import con puntos.

### D-046 · Los nombres de métrica se saneaan antes de llegar a MLflow
- **Qué:** `_metric_name()` en `src/rukh/eval/suite.py` sustituye por `_` cualquier carácter que
  MLflow no acepta. Los tramos de Elo se llaman `<1800` y `2600+`, y MLflow solo admite
  alfanuméricos, `_`, `-`, `.`, espacio y `/`.
- **Por qué:** la primera evaluación real de `tiny` murió con
  `MlflowException: Invalid value "top1/2600+"` **después** de haber jugado todas las partidas
  contra Stockfish. La excepción llega al final del todo, así que se pierde el trabajo entero por
  una etiqueta. Hay test de regresión.
- **Si está mal:** los nombres de los tramos en el informe y en `results.json` no cambian; solo
  cambia cómo se llaman en MLflow.

## P2 · Resultados reales del decoder (2026-09-19)

### D-047 · La temperatura de muestreo vale más de 200 puntos de Elo
- **Qué:** el mismo checkpoint de `small` mide **785 Elo (IC 680-896)** con la configuración por
  defecto de la suite (temperatura 0,6, top-k 20) y **1007 Elo (IC 920-1101)** con muestreo casi
  determinista (temperatura 0,05, top-k 1). La legalidad sin máscara no cambia (99,40 % por argmax)
  pero la muestreada sube de 99,00 % a 99,40 %.
- **Por qué importa:** la cifra de Elo no es una propiedad del modelo sino del par modelo+muestreo.
  Cualquier tabla que compare etapas tiene que fijar el muestreo, y la demo (que juega a 0,6 para
  ser variada) es entre 200 y 250 Elo más débil que el mismo modelo jugando a lo seguro.
- **Decisión:** la tabla publica las dos filas (`small` y `small-greedy`) y la model card explica
  la diferencia. El valor de referencia para el criterio de aceptación es el determinista.
- **Si está mal:** rehacer la tabla con un único muestreo fijado y documentado.

### D-048 · `small` no alcanza los 1200 Elo: se documenta y se escala a `medium`
- **Qué:** `small` (38 971 392 parámetros, 20 000 pasos, 1 024 M tokens, 42 minutos en la 5090)
  mide 1007 Elo determinista, 99,40 % de legalidad sin máscara, 51,1 % de top-1 y 79,4 % de top-3.
  Cumple el listón de legalidad (≥ 99 %) y no el de Elo (≥ 1200).
- **Por qué:** el recorte son 5,9 M partidas frente a las 16 M con las que Karvonen llegó a ~1300
  Elo con 50M parámetros. La pérdida de validación seguía bajando (1,52) al terminar, así que no
  está saturado.
- **Decisión:** `docs/spec/08` autoriza "más pasos, más meses o `medium`". Se lanza `medium`
  (115 120 128 parámetros, misma receta y mismo presupuesto de tokens, ~2 h) para saber si el
  cuello es capacidad o datos. Las dos cifras se publican tal cual.
- **Si está mal:** la alternativa es descargar más meses (unas 4 h por mes con la configuración
  actual) y reentrenar `small`.

### D-049 · La cuantización a int8 cambia la jugada elegida en el 4,6 % de las posiciones
- **Qué:** paridad del export sobre 1 000 posiciones de validación: fp32 **100 %**, fp16
  **99,80 %**, int8 **95,40 %**. El criterio del spec es ≥ 99,9 %.
- **Por qué importa:** la demo sirve `small-int8` (43,5 MB) por defecto en móvil y con
  `saveData`, así que un usuario de móvil juega contra un modelo medibemente distinto del que
  aparece en la tabla. fp16 (78,8 MB) se queda a una décima del listón.
- **Decisión pendiente de Borja:** o se acepta int8 como "modelo de móvil" con su propia fila en
  la tabla y un aviso en la demo, o se sirve fp16 a todo el mundo y se asume la descarga de 79 MB.
  Hasta decidirlo, la demo no se publica con int8 por defecto.
- **Si está mal:** cuantización por canal o dejar en fp32 las capas sensibles subiría la paridad a
  costa de tamaño.

### D-050 · Los laboratorios fijan la salida a UTF-8 antes de imprimir
- **Qué:** `labs/m1/explore.py`, `labs/m1/bpe_merges.py`, `labs/m2/{params,causal_mask}.py` y
  `labs/m3/bidirectional.py` llaman a `sys.stdout.reconfigure(encoding="utf-8")` tras los imports.
- **Por qué:** la consola de Windows es cp1252 y DuckDB dibuja sus tablas con caracteres de marco,
  así que el primer `print` de un resultado moría con `UnicodeEncodeError`. Le pasa a cualquiera
  que siga la lección en Windows, no solo a nosotros. Es el mismo fallo que D-027, en otro sitio.
- **Si está mal:** en una terminal que ya habla UTF-8 la llamada no hace nada.

### D-051 · Los tests entrenan su propio BPE en vez de leer el del repo
- **Qué:** `tests/unit/test_pack.py` entrena un BPE de 600 tokens en `tmp_path` en lugar de cargar
  `artifacts/tokenizer/bpe.json`.
- **Por qué:** `rukh data tokenize --scheme bpe` reescribe ese artefacto con el vocabulario real de
  4 096, y la batería se puso en rojo la primera vez que el pipeline corrió de verdad. Un test no
  puede depender de un fichero que el pipeline regenera legítimamente.
- **Si está mal:** el coste es un segundo de entrenamiento por test.

### D-052 · Las cachés de evaluación salen de git
- **Qué:** `/artifacts/eval/**/*.sqlite` pasa a `.gitignore` y las dos que se habían colado se
  quitan del índice. Los informes y los `results.json` sí se quedan: son el registro.
- **Por qué:** una caché es una aceleración local, no un resultado, y crece con cada tirada.

### D-053 · Los puzles se puntúan desde la partida real, no desde su propia solución
- **Qué:** `rukh data puzzles` pasa a leer `Lichess/chess-puzzles-with-games` (`with_games: true`,
  el valor por defecto en `configs/data/pipeline.yaml`; `with_games: false` vuelve a
  `Lichess/chess-puzzles`). El `movetext` de la partida se convierte a UCI con
  `rukh.data.uci.san_tokens` y se reproduce hasta que el FEN de cuatro campos del tablero
  (`rukh.data.positions.fen4`) coincide con el del puzle: esas jugadas se guardan en `prefix_uci`,
  junto a `prefix_plies`, `white_elo` y `black_elo`. `rukh.eval.puzzles` arma el prompt como
  `<bos> <wXXXX> <bXXXX>` más esas jugadas, igual que `rukh.infer.sampler.prompt_ids`.
- **Por qué:** `rukh eval --suite quick` daba `puzzles: 0.0107 solved` con un modelo que juega
  legal el 99,4 % de las veces y acierta la jugada humana el 51,1 %. El prompt era `<bos>` más la
  propia línea de solución del puzle: una secuencia que no es una partida y que ni siquiera
  empieza en la posición inicial. Un decoder entrenado con prefijos de partidas no puede
  responder a eso, así que el 1,07 % medía el prompt, no la táctica. GOAL.md pide "puzles por
  tramo" como criterio de aceptación de P2/P6, y un número pegado al suelo por construcción no
  sirve de criterio.
- **Detalles:** los puzles cuyo prefijo no se puede reconstruir (la posición no aparece,
  `movetext` ilegal o ausente) se descartan y se cuentan en el manifiesto
  (`counts.dropped_no_prefix`, `filters.prefix.dropped`). En un sondeo de 3 000 puzles de dos de
  los 14 ficheros del dataset, con los filtros de siempre, se reconstruyeron 3 000: la pérdida
  esperada es < 0,1 %. Un parquet del camino antiguo (sin `prefix_uci`) se sigue evaluando con el
  prompt de antes, el informe lo marca como `line-only` y añade una nota diciendo que ese número
  no es comparable. La caché de evaluación guarda el estilo de prompt en cada intento y no reusa
  uno del otro estilo, para que el 1,07 % no reaparezca disfrazado.
- **Si está mal:** la alternativa es no medir puzles con el decoder (solo con el encoder, que sí
  recibe la posición) y quitar el criterio del GOAL; volver a la fuente anterior es un `false` en
  la configuración.

### D-054 · El cuello de `small` son los datos, no la capacidad
- **Qué:** `medium` (115 120 128 parámetros, misma receta y mismo presupuesto de tokens, 1 h 40 en
  la 5090) mide **1091 Elo (IC 990-1194)**, 52,9 % de top-1, 23,9 % de puzles y 99,40 % de
  legalidad, frente a los **1007 Elo (IC 920-1101)**, 51,1 % y 22,1 % de `small`. Triplicar los
  parámetros compra 84 Elo, con intervalos que se solapan.
- **Por qué importa:** responde la pregunta que abrió D-048. Con 5,9 M de partidas, el modelo de
  39M ya está cerca de lo que ese corpus permite; Karvonen llegó a ~1300 Elo con 16 M de partidas
  y 50M de parámetros. Para acercarse a 1200 hay que traer más meses, no más capas.
- **Decisión:** las dos filas se publican. El modelo del curso sigue siendo `small` (entrena en 42
  minutos y cabe en el navegador); `medium` queda documentado como la comprobación.
- **Si está mal:** repetir con más meses de datos y el mismo `small` separaría las dos hipótesis
  del todo.

### D-055 · El F1 de error se mide en un punto de operación elegido aparte, no en 0,5
- **Qué:** `rukh eval encoder` sobre el encoder afinado real
  (`checkpoints/encoder-heads-moves-20260919-095307/best.pt`, 7 453 filas etiquetadas del
  conjunto retenido) daba **F1 0,0000** (precisión 0, exhaustividad 0) frente a **0,0824** de la
  heurística de material (P 0,0436, R 0,7653): "−8,2 puntos de F1, por debajo del listón". La
  tasa base de error es **3,72 %** y las probabilidades de la cabeza van de **0,0039 a 0,2568**,
  con media 0,0436: en el umbral fijo 0,5 la cabeza **no se dispara nunca**, así que ese F1 era
  necesariamente cero. El mismo modelo, en las mismas filas, tiene **ROC AUC 0,7193** y en su
  mejor umbral (0,118) llega a **F1 0,1826** (P 0,148, R 0,2383): más del doble de la heurística,
  **+10 puntos**, muy por encima de los +5 que pide GOAL.md.
- **Por qué importa:** el modelo no estaba roto; lo estaba el punto de operación. Con una clase
  positiva del 3,7 %, un F1 en un umbral arbitrario mide la calibración de un sigmoide que nadie
  calibró, no la calidad de la representación, y la exactitud no mide nada en absoluto (un
  modelo que siempre dice "no hay error" saca 96,3 %). Publicar −8,2 puntos habría sido tan
  falso como publicar +10 sin decir de dónde sale el umbral.
- **Decisión:** las filas etiquetadas se parten en dos **por `game_id`** (nunca por posición: la
  misma política que `rukh.data.labels`, y por la misma razón). La mitad `tune` elige el umbral
  que maximiza F1 allí; la mitad `score` es donde se miden el F1, la precisión y la exhaustividad
  que se publican y con los que se calcula la comparación de GOAL.md, contra la heurística medida
  en esas mismas filas. Nunca se elige el umbral en las filas que luego se puntúan. El informe
  lleva ahora los dos puntos de operación (el afinado, con su umbral, y el fijo 0,5), el tamaño
  en filas y partidas de las dos mitades, precisión y exhaustividad en ambos, y **ROC AUC** y
  **precisión media** (el área bajo la curva P-R, el resumen correcto para una clase rara),
  implementadas a mano sobre rangos, sin `scipy` ni `scikit-learn`. La heurística es una regla de
  sí/no sin umbral que afinar: el informe lo dice explícitamente, para que la comparación no sea
  injusta en silencio en el otro sentido. El aviso de la tasa base (la exactitud no significa
  nada, y el F1 en un umbral arbitrario casi tampoco) está en `report.md` y en la model card,
  porque es la lección más transferible del módulo. Cuando la partición degenera (pocas partidas,
  o todos los errores en una mitad) el umbral se elige y se mide en las mismas filas, el informe
  lo marca como cota superior y no como número retenido.
- **Detalles:** `value_correlation_meets_goal` no cambia y Spearman sigue siendo el titular del
  criterio de valor. La fila de la web gana `blunder_threshold`, `blunder_f1_fixed`,
  `blunder_roc_auc`, `blunder_average_precision` y `blunder_base_rate`; MLflow registra lo mismo.
- **Si está mal:** la alternativa es calibrar la cabeza (Platt o isotónica sobre la mitad `tune`)
  y volver a un umbral fijo de 0,5 con sentido, o cambiar el criterio de GOAL.md a la precisión
  media, que no depende de ningún umbral. Las dos son más honestas que un 0,5 sin calibrar;
  ninguna de las dos se decide con una sola tirada.

### D-056 · Resultados reales del encoder: un criterio cumplido y otro no
- **Qué:** con 438 093 posiciones etiquetadas (10 % de validación, split por partida) y el encoder
  preentrenado con MMM (75,2 % de acierto en las jugadas ocultas, 16 minutos en la 5090):

  | Esquema | F1 de error (ajustado) | Margen sobre la heurística | ROC AUC | Spearman valor↔cp | Pearson |
  |---|---|---|---|---|---|
  | `moves` (preentrenado con MMM) | 0,179 | **+9,0** | 0,738 | 0,407 | 0,442 |
  | `squares` (desde cero) | 0,146 | **+5,7** | 0,696 | 0,422 | **0,688** |

  Los dos **cumplen** el listón de error (≥ heurística + 5 puntos). **Ninguno** llega al 0,80 de
  correlación de valor que pide GOAL: el mejor Spearman es 0,42.
- **Qué dice la comparación de representaciones** (la pregunta de `docs/spec/02`, Componente 2):
  la línea de jugadas detecta mejor los errores (el preentrenamiento ayuda a saber qué acaba de
  pasar), y el tablero correlaciona mucho mejor el valor en Pearson (0,69 frente a 0,44): ver las
  piezas es mejor para "cuánto vale esto", ver la línea es mejor para "qué se acaba de tirar".
- **Por qué falla el valor:** 4 000 pasos de afinado, 488 159 posiciones con etiqueta de las 5 M
  muestreadas (9,8 % de cobertura del cruce), y un `tanh(cp/400)` que comprime justo donde hay más
  densidad. Es el mismo diagnóstico que D-054: faltan datos etiquetados, no capacidad.
- **Decisión:** se publican las dos filas tal cual. El criterio de valor queda **no cumplido** y
  documentado, como manda GOAL.
- **Si está mal:** más pasos de afinado, más ficheros del dataset de evaluaciones (hay 20 y el
  cruce solo recuperó el 9,8 %) o una escala de valor menos comprimida.

### D-057 · Afinar menos red gana, y las etiquetas se saturan a las 110 000
- **Qué:** dos resultados del afinado del encoder, ambos a 4 000 pasos sobre las mismas etiquetas:
  - **Modo de afinado** (MAE del valor): `probe` (tronco congelado) 0,1429 · `full` (todo) 0,1354 ·
    **`last-n` (los dos últimos bloques y la norma final) 0,1180**. Afinar *menos* red gana.
  - **Curva por número de etiquetas** (`last-n`): 10 % (43 809 filas) 0,1217 y 95,59 % de acierto de
    error; 25 % (109 523) 0,1192 y 96,78 %; 50 % (219 046) 0,1210 y 96,78 %; 100 % (438 093) 0,1215
    y 96,78 %. **Plana a partir del 25 %.**
- **Por qué importa:** responde la pregunta que el módulo M3 plantea. Con 110 000 etiquetas de
  Stockfish el encoder ya está donde va a estar; las 330 000 restantes no compran nada medible. Y
  el orden `last-n > full > probe` es el argumento de que la adaptación específica de la tarea vive
  en los últimos bloques, mientras que mover los primeros aleja la representación que construyó el
  preentrenamiento.
- **Honestidad:** una tirada por modo y por punto. La dispersión entre los cuatro puntos de la curva
  (0,1192-0,1217) es del mismo tamaño que la distancia a la tirada suelta de `last-n` (0,1180), así
  que el orden es sugerente, no establecido; haría falta repetir con varias semillas. La única
  diferencia que supera claramente ese ruido es el acierto de error al 10 % (95,59 % frente a 96,78 %).
- **Si está mal:** repetir con tres semillas por punto separaría señal de ruido.

### D-058 · Las opciones del CLI se comprueban en el parser, no en el `--help`
- **Qué:** `tests/conftest.py` expone `cli_options("train", "heads")`, que devuelve las banderas que
  declara el comando de Click. Cinco tests que hacían `assert "--config" in result.output` ahora
  preguntan al parser.
- **Por qué:** Typer dibuja la ayuda con Rich y, cuando hay color (el runner de GitHub lo activa;
  una consola de Windows no), los códigos de escape caen **dentro** del nombre de la opción, así
  que la subcadena no aparece aunque la ayuda muestre la bandera. Los cinco tests pasaban en local
  y tumbaron la primera CI de `main`. Reproducido con `FORCE_COLOR=1`.
- **Si está mal:** para comprobar el texto renderizado hay `plain()` en el mismo `conftest`, que
  quita los escapes.

### D-059 · Las tablas con desplazamiento llevan `tabindex` y nombre
- **Qué:** los diez envoltorios `.table-wrap` de las lecciones y los dos de los componentes
  (`ResultsTable`, `TokenizerStats`) pasan a `tabindex="0"` con `role="region"` y `aria-label`, más
  un anillo de foco visible.
- **Por qué:** axe lo marcó como `scrollable-region-focusable` (impacto **serio**) en la CI de
  `main`: una caja que se desplaza solo con el ratón deja fuera a quien navega con teclado. En
  local pasaba porque la tabla no desbordaba a ese ancho; el runner sí la desbordó. Es un fallo de
  accesibilidad real, no una prueba quisquillosa.
- **Si está mal:** quitar el `tabindex` devuelve la violación.

### D-060 · Las model cards publican los listones y la paridad de la cuantización
- **Qué:** las dos plantillas de card (`model.md.jinja`, `encoder.md.jinja`) ganan tres bloques:
  - **Listones de aceptación**: una tabla con los criterios de `GOAL.md` (decoder: legalidad sin
    máscara por argmax ≥ 99 % y Elo ≥ 1200; encoder: F1 de error ≥ heurística + 5 puntos y
    correlación valor↔cp ≥ 0,80) frente a lo medido, con veredicto "met"/"not met". El veredicto se
    calcula con las constantes del propio harness (`GOAL_LEGALITY` y `GOAL_ELO` nuevas en
    `rukh.eval.suite`; `GOAL_MARGIN` y `GOAL_VALUE_CORRELATION` ya existían en `rukh.eval.encoder`),
    nunca se escribe a mano, y un listón cuya métrica falta **se omite** en vez de darse por
    fallado. Cuando uno no se cumple, la card lo dice además **en prosa** con el motivo del
    registro: D-054 para el Elo (5,9 M partidas frente a 16 M, y `medium` con el triple de
    parámetros solo compró 84 Elo: el cuello son los datos) y D-056 para el valor (4 000 pasos de
    afinado y 9,8 % de cobertura de etiquetas).
  - **Paridad de la cuantización**: `rukh export` escribe ahora `parity.json` junto a los ficheros
    ONNX (acuerdo y deriva máxima por precisión, número de posiciones, de qué población salieron y
    qué exportador las produjo) y `rukh publish model` lo copia al repositorio y lo cita. La card
    del decoder dice que int8 cambia la jugada elegida en el 4,6 % de las posiciones (D-049) y que
    por tanto un móvil en WASM juega contra un modelo medibemente distinto del de la tabla; la del
    encoder dice que sus tres precisiones coinciden en el 100 % de las decisiones de error, que es
    el contraste que hace legible el número del decoder.
  - **El muestreo forma parte de la cifra** (D-047): el decoder publica los dos puntos de operación
    del mismo checkpoint (1007 Elo casi determinista, 785 a temperatura 0,6). El segundo se lee de
    `artifacts/eval/<stage>-greedy/results.json` y solo se publica si el `model_sha` de las dos
    tiradas coincide: si no, serían dos modelos y la diferencia no sería del muestreo.
- **Por qué:** `GOAL.md` manda documentar la cifra tal cual, y el registro y las lecciones ya lo
  dicen; la card era el único sitio donde un lector no podía saber qué listón se cumplía. Publicar
  "1007 Elo" sin decir que el objetivo eran 1200, o servir `model-int8.onnx` sin decir cuánto se
  aleja del checkpoint, es contar la mitad de la medición.
- **Detalles:** ningún número se escribe en la plantilla; todos salen del contexto de la card, de
  `parity.json` o de las entradas del registro citadas arriba. Sin `parity.json` la card no dice
  nada de paridad (ausencia significa "no comprobado", que no es "comprobado y perfecto").
- **Si está mal:** quitar `parity.json` del `upload_folder` deja las cifras sin respaldo
  comprobable; volver a un solo punto de operación exige fijar el muestreo en toda la tabla.
