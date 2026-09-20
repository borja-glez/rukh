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

### D-061 · Se publica el encoder afinado con `last-n`, no el `full`
- **Qué:** `chorcat/rukh-encoder` lleva los pesos del afinado `last-n` (últimos dos bloques y la
  norma final), no los del completo. Medido sobre el mismo conjunto: F1 de error 0,1804 frente a
  0,1791, margen +9,2 frente a +9,0, ROC AUC 0,740 frente a 0,738 y, sobre todo, correlación de
  valor **Spearman 0,520 frente a 0,407** (Pearson 0,648 frente a 0,442).
- **Por qué:** al preparar la publicación se iba a subir el checkpoint de `last-n` con las métricas
  del `full`, que son de otro modelo. Se evaluó el que se publica y resultó ser además el mejor,
  coherente con D-057.
- **Si está mal:** el criterio de correlación sigue sin cumplirse (0,52 frente a 0,80), y la card
  lo dice.

### D-062 · La validación se queda en 100 000 partidas y el resto del mes entrena
- **Qué:** `TokenizeConfig` gana `train_months` (lista) y `val_games`. Con `val_games: 100000`, la
  validación son las primeras 100 000 partidas de `val_month` y las otras ~2,85 M se unen al
  entrenamiento. `pack_month` pasa a ser un caso de `pack_games(sources)`, que concatena trozos
  `GameSlice(path, skip, take)` en un solo flujo.
- **Por qué:** P1 dedicó el mes 02 entero a validación: 238 657 571 tokens parados para un bucle
  que lee 50 lotes (640 000 tokens) por evaluación. El decoder entrenó con 240 M tokens únicos,
  6,2 por parámetro frente a los ~20 razonables, y repitió el corpus 4,3 veces (1,024 B tokens
  vistos). La huella está en las curvas: el hueco train/val pasa de 0,021 en el paso 10 000 a
  0,060 en el 20 000 mientras el top-1 solo sube 0,75 puntos en los últimos 5 000 pasos. El cuello
  era repetición de datos, no capacidad — lo que ya sugería D-054.
- **Medido:** el corpus recuperado da 5 796 388 partidas y **470 650 377 tokens** de entrenamiento
  (12,1 por parámetro) contra 100 000 partidas y 8 076 148 tokens de validación; las dos cifras
  suman exactamente las 5 896 388 del corpus, así que ninguna partida se pierde ni se duplica.
- **Comparabilidad:** los mismos pesos (`small` paso 20 000) miden **1,5197 de pérdida y 0,5120 de
  top-1 en las dos validaciones**, la vieja y la nueva, hasta el cuarto decimal. El objetivo de
  val/loss ≤ 1,42 se traslada sin recalibrar.
- **Si está mal:** entrenamiento y validación salen ahora del mismo mes, con partidas distintas.
  Se pierde la propiedad "un mes que el modelo no ha visto" y con ella la prueba de deriva
  temporal, que vuelve en cuanto haya un mes nuevo descargado.

### D-063 · La precisión que sirve la demo la decide el backend, no el tamaño de pantalla
- **Qué:** `defaultStageId` deja de mirar `mobile`. Sirve `small-fp16` allá donde haya WebGPU,
  teléfono incluido, y reserva `small-int8` para el respaldo WASM y para `saveData`. El encoder
  mantiene int8 en móvil.
- **Por qué:** int8 cambia la jugada en el 4,6 % de las posiciones (D-049), así que la regla
  anterior servía a los móviles un modelo que no es aquel cuyo Elo publican las cards, y lo hacía
  en hardware que sí puede con fp16 — los teléfonos llevan tiempo con WebGPU. fp16 diverge en el
  0,2 %. El tamaño de pantalla no dice nada sobre la aritmética disponible.
- **La asimetría del encoder es medida, no preferencia:** su int8 coincide con el checkpoint en el
  100 % de las posiciones de paridad, en las tres precisiones, así que ahí los 15 MB salen gratis.
- **Si está mal:** un móvil sin WebGPU y sin `saveData` baja 78,8 MB en vez de 43,5 MB, una sola
  vez y cacheado. `navigator.gpu` se consulta de forma síncrona porque `parseQuery` lo es; si
  existe pero no da adaptador, el worker cae a WASM por su cuenta y reporta el backend real.

### D-064 · 24 meses de la Lichess Elite Database entran en el corpus de entrenamiento
- **Qué:** `data/elite` pasa de 2 meses (541 085 partidas) a **24** (2023-09 a 2025-08,
  **6 676 794 partidas**) y se suma al entrenamiento por `extra_train_parquets`. Nunca a la
  validación, que sigue siendo las 100 000 partidas congeladas de 2025-02.
- **Por qué:** la sonda de condicionamiento mostró que el eje de Elo está comprimido, no muerto:
  dentro del rango entrenado las distribuciones distan 0,004-0,047 nats, frente a 0,39 contra una
  cabecera por debajo del suelo del corpus. La causa es la composición: solo el **3,4 %** de
  nuestras partidas tiene a las blancas en 2400+. En la base de élite es el **93,6 %**.
- **Corrección a la primera lectura:** se dijo que forzar una cabecera alta *empeora* la predicción
  (50,93 % con `<1800>` contra 50,07 % con `<2800>`). Eso se midió contra continuaciones de
  jugadores de 1800-2100, así que medía lo contrario de lo que parecía. Sobre `data/uci-strong`
  (partidas de 2200+) la curva **se invierte y hace pico en `<2200>`**, la banda real de esas
  partidas: 41,70 → 41,89 → **42,20** → 41,95 → 41,64 % para 1800/2000/2200/2400/2800. El
  condicionamiento funciona en la dirección correcta; lo que falla es la magnitud.
- **Además resuelve la palanca de datos:** Hugging Face está devolviendo HTTP 429 a las descargas
  de meses nuevos y `database.nikonoel.fr` es otro servidor, así que esta vía no depende de aquella.
- **Calidad verificada, no supuesta:** 88,4 jugadas de media (frente a ~77 del corpus general) y
  **solo el 0,5 %** de las partidas con menos de 180 s de base, así que la base de élite viene
  prácticamente sin bullet y es coherente con nuestro filtro `min_base_seconds: 180`.
- **Cuidado al leer la pérdida:** el entrenamiento deja de parecerse a la validación, que es 53 %
  sub-2000. Se añade `data/uci-strong` (10 230 partidas de 2200+ **dentro** de las 100 000
  congeladas, nunca entrenadas por ninguna corrida) para tener una pérdida comparable sobre juego
  fuerte, y el Elo se mide con partidas en vez de inferirse.
- **Si está mal:** el modelo dedicaría capacidad a imitar un juego que el listón no premia; se
  vería como Elo estancado pese a mejor pérdida sobre `uci-strong`, y la alternativa sería usar
  élite solo como afinado final y no en el preentrenamiento.

### D-065 · Recuperar el mes de validación vale más que triplicar los parámetros
- **Medido** (misma receta, mismo muestreo, misma validación congelada):

  | Corrida | Parámetros | Tokens únicos | Pasos | val/loss | top-1 | hueco train/val |
  |---|---|---|---|---|---|---|
  | `small` v1 | 38 971 392 | 240 068 954 | 20 000 | 1,5197 | 51,20 % | **+0,0603** |
  | `medium` v1 | 115 120 128 | 240 068 954 | 20 000 | 1,4782 | 52,31 % | — |
  | **`small` v2** | 38 971 392 | 470 650 377 | 28 000 | **1,4703** | **52,49 %** | **+0,0242** |

- **`small` con 39 M de parámetros bate a `medium` con 115 M** en las dos validaciones (general
  1,4703 frente a 1,4782; fuerte 1,4652 frente a 1,4880). Confirma D-054 con una medida directa:
  el cuello eran los datos, y la capacidad extra de `medium` se estaba gastando en memorizar un
  corpus repetido 4,3 veces.
- **El hueco train/val cae a la mitad** (0,0603 → 0,0242), que es justo lo que predice el
  diagnóstico de repetición: menos épocas sobre el mismo material, menos memorización.
- **Traducción a Elo:** −0,0494 nats × 2132 Elo/nat ≈ **+105 Elo** (~1112 estimado). Sigue corto
  del listón de 1200, así que la palanca de datos no se agota aquí.
- **Nota:** v2 es el primer modelo que predice mejor el juego de 2200+ (1,4652) que el promedio
  del corpus (1,4703); v1 y `medium` iban al revés.
- **Si está mal:** el Elo se mide con partidas, no con la recta; la estimación de +105 solo sirve
  para decidir si merece la pena seguir, y se sustituye por la medición en cuanto haya CPU libre.

### D-066 · La élite ensancha el eje de Elo; la cantidad de datos por sí sola, no
- **Medido** al terminar las tres corridas, todas con 28 000 pasos salvo v1 (20 000), mismo
  planificador, misma semilla y la misma validación congelada:

  | Modelo | Par. | general | top-1 | fuerte 2200+ | top-1 |
  |---|---|---|---|---|---|
  | `small` v1 | 39 M | 1,5197 | 51,20 % | 1,5391 | 50,84 % |
  | `medium` v1 | 115 M | 1,4782 | 52,31 % | 1,4880 | 52,06 % |
  | `small` v2 | 39 M | **1,4703** | **52,49 %** | 1,4652 | 52,69 % |
  | `small` v3 | 39 M | 1,4770 | 52,19 % | **1,4537** | **53,02 %** |

- **El intercambio es el diseñado:** v3 cede 0,0067 nats en juego promedio y gana 0,0115 sobre
  juego fuerte respecto a v2. Frente a v1 son **−0,0854 nats en juego fuerte**, ~+182 Elo por la
  pendiente medida.
- **El eje de condicionamiento se ensancha 2,4 veces.** KL de `<2800>` contra `<1800>`: 0,0472 en
  v1, 0,0487 en v2, **0,1147 en v3**. La ganancia por condicionar sobre juego fuerte pasa de
  +0,42 puntos de top-1 (v2, pico en `<2200>`) a **+0,88** (v3: 52,15 % con `<1800>` frente a
  53,03 % con `<2200>`).
- **La conclusión que importa:** v2 duplicó los datos y dejó el eje exactamente igual que v1
  (KL 0,0487 frente a 0,0472). Lo que lo abre es la **composición**, no el volumen. Sin el corpus
  de élite, el hito de "juega como 1500/2000/2400" de P4 no tenía de dónde salir.
- **Si está mal:** la sonda mide distribuciones, no fuerza. Que el eje se ensanche no garantiza
  Elo; eso se comprueba con partidas y es lo que decide qué cabecera sirve la demo.

### D-067 · `PackedDataset` mandaba su índice entero a cada worker y eso rompió a los 19 M
- **Síntoma:** `medium-v4` murió al arrancar con `UnpicklingError: pickle data was truncated`,
  antes del primer paso, sobre el corpus de 18 942 740 partidas. Las mismas configuraciones
  funcionaban con 12 473 182.
- **Causa:** `starts` es un `int64` por partida y se cargaba en memoria. Los workers del
  `DataLoader` se lanzan con *spawn* en Windows, así que el dataset se serializa una vez por
  worker: **95 MB con 12,5 M partidas, 145 MB con 18,9 M**, y a 145 MB la tubería se corta.
- **Arreglo:** `__getstate__` excluye `tokens` y `starts`, y `__setstate__` los remapea desde
  disco en el worker; `starts` pasa además a `mmap_mode="r"`. El pickle del dataset real baja de
  **145 MB a 0,5 KB** y las ventanas son idénticas tras el viaje (comprobado sobre el índice
  1 234 567 del corpus v4).
- **Por qué importa más allá del susto:** era un techo de escalado silencioso. Cualquiera que
  ampliara el corpus se lo habría encontrado, y el mensaje de error no señala a los datos.
- **Si está mal:** el coste es reabrir dos memmaps por worker al arrancar, una vez por época.

### D-068 · La carga de CPU infla el Elo en ~13 puntos, dentro del ruido
- **Duda:** Stockfish juega a 0,1 s por jugada, así que bajo carga busca menos profundo y regala
  Elo. Si el sesgo fuese grande, ninguna medición tomada mientras la GPU entrena sería comparable
  con las publicadas, y habría que serializar toda la noche.
- **Medido:** el mismo checkpoint de `small` v1, misma configuración, misma semilla, con un caché
  aparte para no pisar las partidas originales:

  | Condición | Elo | IC 95 % | Puntuación |
  |---|---|---|---|
  | máquina libre (lo publicado) | 1006,8 | 920-1101 | 0,2656 |
  | bajo carga (entrenando + descargando) | 1019,7 | 931-1116 | 0,2750 |

- **Conclusión:** **+12,9 Elo**, una séptima parte del intervalo de confianza. La diferencia de
  puntuación es 0,0094 sobre 160 partidas, con error típico 0,035: indistinguible del ruido. Para
  efectos de 100-200 Elo no es un confusor, así que las mediciones pueden ir en paralelo con el
  entrenamiento anotando el sesgo.
- **Si está mal:** el sesgo no es cero y se suma en la dirección favorable, así que cualquier
  resultado que quede a menos de ~15 Elo del listón hay que repetirlo con la máquina parada antes
  de declararlo cumplido.

### D-069 · El condicionamiento alto no da Elo, aunque el eje funcione
- **Medido** con la suite completa (160 partidas, muestreo determinista, `configs/eval/greedy*.yaml`):

  | Etapa | Elo | IC 95 % | legal argmax | top-1 | puzles |
  |---|---|---|---|---|---|
  | `small` v1 @1800 | 1007 | 920-1101 | 99,40 % | 51,10 % | 22,07 % |
  | `small` v2 @1800 | 1070 | 975-1167 | 99,30 % | 51,80 % | **26,93 %** |
  | `small` v3 @1800 | **1095** | 1006-1188 | 99,10 % | **52,40 %** | 26,73 % |
  | `small` v3 @2600 | 1058 | 974-1175 | 99,10 % | 52,40 % | 26,73 % |

- **Pedirle a v3 que juegue a 2600 no mejora nada**: 1058 frente a 1095, con intervalos muy
  solapados. La prueba de 32 partidas había dado 0,703 contra 0,609 a favor de `<2600>`; era
  ruido, y con 160 partidas se cae.
- **El eje sí funciona** (D-066: KL 2,4 veces mayor, +0,88 puntos de top-1 sobre juego fuerte).
  Lo que no ocurre es la traducción a fuerza: imitar las elecciones de un 2400 no gana partidas
  sin búsqueda táctica. Sirve para el hito de P4 «juega como 1500/2000/2400», no para el listón
  de Elo.
- **La pendiente pérdida→Elo estaba sobreestimada.** Se predijo +105 Elo para v2 con 2132
  Elo/nat; lo medido son +63 sobre 0,0494 nats, o sea ~1275 Elo/nat. Llegar a 1200 desde 1007
  exige ~0,151 nats, no 0,09.
- **Vigilar:** la legalidad sin máscara baja monótonamente (99,40 → 99,30 → 99,10 %). Sigue sobre
  el listón del 99 % pero el margen se adelgaza corrida a corrida.
- **Los puzles suben mucho más que el Elo**: 22,07 → 26,93 %, y por bandas +8,7 puntos en
  1000-1500 frente a +1,1 en 2000+.

### D-070 · Cuatro de los ocho rivales del harness tenían el Elo inventado y estaba mal por ~500
- **Qué estaba mal:** los escalones `skill-0..3` se metieron en la escalera para llegar *por
  debajo* del suelo de 1320 de `UCI_Elo` y se etiquetaron 800/950/1100/1250 sobre esa suposición.
  Es falsa. Ninguno de los cuatro está por debajo de 1320.
- **Cómo se detectó:** por una contradicción interna, no buscando aprobar el criterio. `small` v2
  puntuaba 0,725 contra `uci-1320` y perdía 20-0 contra `skill-3`, etiquetado 1250. Ninguna
  medición del modelo puede resolver eso, porque el modelo es lo que se está midiendo; motor
  contra motor sí.
- **Medido** (40 partidas por pareja, 0,1 s por jugada, colores alternados, anclado en `uci-1320`):

  | Escalón | Etiqueta vieja | Medido | Error |
  |---|---|---|---|
  | `skill-0` | 800 | **1381** | +581 |
  | `skill-1` | 950 | **1467** | +517 |
  | `skill-2` | 1100 | **1589** | +489 |
  | `skill-3` | 1250 | **1678** | +428 |

- **El método se valida con sus propios controles:** `uci-1500` midió **+179** Elo sobre
  `uci-1320` frente a los +180 nominales. Si el procedimiento estuviera sesgado, ese control
  habría fallado. `UCI_Elo` sí se comprime más arriba (`uci-1800` midió +215 sobre `uci-1500`, no
  +300), lo que es una salvedad para los escalones altos y no para el rango donde jugamos.
- **Consecuencia sobre todo lo publicado:**

  | Etapa | Publicado | Corregido | IC 95 % |
  |---|---|---|---|
  | `small` v1 | 1007 | **1359** | 1293-1429 |
  | `small` v2 | 1070 | **1407** | 1344-1462 |
  | `small` v3 @1800 | 1095 | **1425** | 1367-1485 |
  | `small` v3 @2600 | 1058 | 1397 | 1340-1450 |

- **El listón de 1200 nunca se falló**: incluso `small` v1, ya publicado en Hugging Face con
  «Elo bar not met», estaba en 1359. Las model cards y `docs/plans/*` dicen lo contrario y hay que
  corregirlas.
- **Lo relativo no cambia:** la corrección sube a los cuatro modelos por igual, así que todas las
  comparaciones de D-065, D-066 y D-069 siguen en pie; el trabajo de datos de hoy vale +66 Elo
  (1359 → 1425) en la escala corregida igual que valía +88 en la torcida.
- **Lo que sigue sin resolverse:** el número absoluto depende de fiarse del `UCI_Elo` de Stockfish
  a 0,1 s por jugada, un régimen para el que no está calibrado. Los controles lo respaldan entre
  1320 y 1500 y lo desmienten por encima.
- **Si está mal:** la escalera corregida ya no tiene ningún rival por debajo de 1320, así que un
  modelo débil queda mal acotado por abajo. Para los actuales (~1400) la escalera los rodea.

### D-071 · DPO compra Elo y rompe el criterio de legalidad
- **Medido** con la suite completa y la escalera corregida de D-070:

  | Etapa | Elo | IC 95 % | legal argmax | top-1 | puzles |
  |---|---|---|---|---|---|
  | `small` v1 | 1359 | 1293-1429 | 99,40 % | 51,10 % | 22,07 % |
  | `small` v2 | 1407 | 1344-1462 | 99,30 % | 51,80 % | 26,93 % |
  | `small` v3 | 1425 | 1367-1485 | 99,10 % | 52,40 % | 26,73 % |
  | **`small` v3 + DPO** | **1460** | 1401-1517 | **98,90 %** | **53,20 %** | **27,87 %** |

- **+35 Elo sobre v3**, y a la vez sube top-1 y puzles. Es la única palanca del día que no está
  limitada por lo fuertes que fueran los jugadores del corpus: un par de preferencia dice «esta
  jugada es 100 cp mejor que aquella», que no aparece en ninguna partida humana.
- **Rompe la legalidad:** 98,90 % frente al listón de 99 %. La serie venía bajando (99,40 → 99,30
  → 99,10) y DPO la cruza. El mejor modelo que cumple **los dos** criterios sigue siendo `small`
  v3 con 1425 Elo y 99,10 %.
- **Hay volante:** `nll_weight` está en 0,5. Subirlo ancla más la política a la referencia, lo que
  debería recuperar legalidad a cambio de parte del Elo. Falta medir esa curva.
- **Si está mal:** el par proviene de posiciones que están en el corpus de entrenamiento; lo nuevo
  es la preferencia, no la posición. Si el efecto fuese memorización, no se vería en puzles de un
  conjunto distinto, y ahí también sube (26,73 → 27,87 %).

### D-072 · `medium` con datos suficientes: 1504 Elo y la mejor legalidad del proyecto
- **Medido** (escalera corregida de D-070, muestreo determinista):

  | Modelo | Par. | Tokens únicos | Elo | IC 95 % | legal argmax | top-1 | puzles |
  |---|---|---|---|---|---|---|---|
  | `small` v1 | 39 M | 240 M | 1359 | 1293-1429 | 99,40 % | 51,10 % | 22,07 % |
  | `medium` v1 | 115 M | 240 M | — | — | — | 52,31 % | — |
  | `small` v2 | 39 M | 471 M | 1407 | 1344-1462 | 99,30 % | 51,80 % | 26,93 % |
  | `small` v3 | 39 M | 1 095 M | 1425 | 1367-1485 | 99,10 % | 52,40 % | 26,73 % |
  | `small` v3 + DPO | 39 M | 1 095 M | 1460 | 1401-1517 | 98,90 % | 53,20 % | 27,87 % |
  | **`medium-v4`** | 115 M | 1 681 M | **1504** | **1446-1558** | **99,80 %** | **54,40 %** | **37,50 %** |

- **Los dos criterios cumplidos con holgura:** Elo 1504 con el intervalo entero por encima de
  1200, y legalidad 99,80 %, la más alta de cualquier etapa del proyecto.
- **El diagnóstico de la legalidad estaba equivocado.** Se leyó la serie 99,40 → 99,30 → 99,10 %
  como una tendencia a vigilar. No lo era: era un modelo de 39 M estirándose sobre un corpus
  creciente. Con capacidad suficiente, fuerza y legalidad suben juntas.
- **Los puzles pasan de 22,07 % a 37,50 %**, un 70 % relativo, que es donde más se ve la mejora
  real de juego: `medium-v4` en juego fuerte mide 1,3242 de pérdida frente a los 1,4537 de v3.
- **Contra el `medium` original:** misma arquitectura, mismos 115 M de parámetros, 1,4782 de
  pérdida con 240 M de tokens repetidos 4,3 veces frente a **1,3733** con 1 681 M y 1,46 épocas.
  No era la arquitectura.
- **Si está mal:** el hueco train/val de `medium-v4` es +0,0667, parecido al +0,0603 que en v1
  señalaba memorización, así que 1,68 B tokens tampoco sobran para 115 M de parámetros.

### D-073 · Modelo final: `medium-v4` + DPO, 1529 Elo, los dos criterios cumplidos
- **Medido** con la escalera corregida (D-070) y muestreo determinista:

  | Modelo | Elo | IC 95 % | legal argmax | top-1 | puzles |
  |---|---|---|---|---|---|
  | `small` v1 (publicado) | 1359 | 1293-1429 | 99,40 % | 51,10 % | 22,07 % |
  | `small` v2 | 1407 | 1344-1462 | 99,30 % | 51,80 % | 26,93 % |
  | `small` v3 | 1425 | 1367-1485 | 99,10 % | 52,40 % | 26,73 % |
  | `small` v3 + DPO | 1460 | 1401-1517 | 98,90 % | 53,20 % | 27,87 % |
  | `medium-v4` | 1504 | 1446-1558 | 99,80 % | 54,40 % | 37,50 % |
  | **`medium-v4` + DPO** | **1529** | **1470-1583** | **99,80 %** | 53,40 % | **38,77 %** |

- **Listón de Elo ≥ 1200: cumplido con el intervalo entero por encima.** Legalidad ≥ 99 %:
  cumplida con 99,80 %, la más alta del proyecto.
- **DPO no cuesta legalidad cuando hay margen.** El 98,90 % de `small` v3 + DPO (D-071) no era un
  defecto de DPO sino falta de holgura en 39 M parámetros: sobre `medium-v4` la legalidad se queda
  clavada en 99,80 % antes y después, y los +25 Elo salen gratis. `nll_weight` 0,5 y `lr` 2e-6 en
  los dos casos.
- **El coste real de DPO es el top-1** (54,40 → 53,40 %), que es lo esperado: deja de imitar la
  continuación humana para preferir la mejor jugada. Los puzles, que sí miden calidad, suben
  (37,50 → 38,77 %).
- **Reparto de los +170 Elo del día** (1359 → 1529): recuperar el mes de validación +48, corpus de
  19 M partidas con élite +18, capacidad ya justificada por los datos +79, DPO +25, y
  condicionamiento por Elo **0**.
- **Si está mal:** los intervalos de `medium-v4` y `medium-v4` + DPO se solapan (1446-1558 frente a
  1470-1583), así que los +25 de DPO no están separados del ruido por sí solos; lo que los sostiene
  es que top-1 baja y puzles suben a la vez, que es la firma esperada y no la del azar.

### D-076 · La cabeza de valor entrenaba magnitud y el criterio mide orden
- **Qué estaba mal:** `MultiHead.loss` usaba `F.mse_loss` sobre `tanh(cp/400)`, que mide cuánto se
  acerca cada predicción a su etiqueta. `GOAL.md` puntúa la cabeza con **Spearman**, que solo mira
  el orden. Son objetivos distintos y la diferencia era visible: el esquema `squares` tenía mejor
  Pearson (0,688 frente a 0,648) y peor Spearman (0,422 frente a 0,520) que `moves` — aprendía la
  escala y se dejaba el ranking.
- **Arreglo:** `pairwise_rank_loss`, un término por pares ponderado por la distancia entre
  etiquetas, detrás de `HeadWeights.value_rank` (por defecto 0, así que las corridas anteriores
  siguen siendo reproducibles). La ponderación importa: sin ella el gradiente se iría a separar
  posiciones casi iguales, que es justo donde un evaluador sin búsqueda no puede ganar.
- **Medido** (mismo encoder de 15 M, mismo afinado `last-n`, solo cambia la pérdida):

  | `value_rank` | Pearson | Spearman | F1 error | margen |
  |---|---|---|---|---|
  | 0 (MSE sola) | 0,6476 | 0,5205 | 0,1804 | +9,17 |
  | **1,0** | **0,6621** | 0,6395 | 0,1774 | +8,87 |
  | 3,0 | 0,6114 | **0,6601** | 0,1188 | +3,01 |
  | 8,0 | 0,5497 | 0,6596 | 0,1663 | +7,76 |

- **No era un intercambio:** se esperaba ceder Pearson para ganar Spearman y sube todo, así que la
  pérdida anterior estaba peor alineada con la tarea en los dos ejes. El Spearman se satura en
  ~0,66 a partir de peso 1; se elige **1,0** por el mejor Pearson y el margen holgado.
- **Ojo con el peso 3,0:** su F1 de error cae a +3,01 y rompería el criterio, pero el peso 8,0
  vuelve a +7,76. No es monótono, así que es ruido de tirada única — el mismo aviso de D-057.

### D-077 · El encoder sube a 39 M y el criterio de valor sigue sin cumplirse, con una razón medida
- **Qué:** `PositionEncoder` pasa de 15 052 800 a **38 971 392 parámetros** (12 capas, d=512, el
  tamaño del decoder `small`) preentrenado con MMM sobre el corpus de 1 681 M tokens en vez del de
  240 M. Top-1 de jugada enmascarada: **81,40 %** frente al 75,2 % anterior.
- **Medido** con la pérdida de ordenación en peso 1,0:

  | Variante | Pearson | Spearman | F1 error | margen |
  |---|---|---|---|---|
  | publicado (15 M, MSE) | 0,6476 | 0,5205 | 0,1804 | +9,17 |
  | 15 M + orden | 0,6621 | 0,6395 | 0,1774 | +8,87 |
  | **39 M + orden** | **0,7261** | **0,6665** | **0,1855** | **+9,68** |

- **El reparto dice dónde está el valor:** la pérdida dio **+0,119** de Spearman y triplicar la
  capacidad **+0,027**. Subir a 87 M daría un par de centésimas y no acercaría el listón.
- **Por qué hay techo, medido por bandas de |cp|** (10 000 posiciones):

  | Rango de `cp` | Posiciones | Spearman |
  |---|---|---|
  | 0-50 | 6 099 (61 %) | 0,4625 |
  | 50-150 | 2 775 (28 %) | 0,5844 |
  | 150-400 | 496 (5 %) | 0,5196 |
  | **400+** | 626 (6 %) | **0,8797** |

  **Donde la ventaja está decidida el encoder ya pasa de 0,80.** Falla en posiciones casi
  igualadas, que son el 61 % del conjunto: ordenarlas exige ver táctica y este modelo no tiene
  búsqueda. El criterio global está dominado por el caso en que el enfoque tiene un techo
  estructural.
- **Decisión:** no se mueve el listón; se publica el número real (0,6665, **no cumplido**) con el
  desglose al lado, y la decisión sobre el criterio es de Borja.
- **Si está mal:** una tirada por variante. El desglose por bandas es la medida más informativa y
  la más barata de repetir.

### D-078 · DPO sin ancla destruye el modelo: la medida que faltaba en el registro
- **Qué faltaba:** D-071 y D-073 documentan el DPO ya anclado (`nll_weight` 0,5) y dan por sabido
  que el puro no vale. La tirada que lo demuestra no estaba escrita, así que el curso habría
  enseñado de oídas justo donde predica lo contrario.
- **Medido** sobre `small` v3, β 0,1, lr 5e-6, una época, `nll_weight` **0**:

  | | general | top-1 | fuerte 2200+ |
  |---|---|---|---|
  | antes de DPO | 1,4770 | 52,19 % | 1,4537 |
  | DPO puro | **1,8892** | 47,29 % | 1,8312 |

  Margen final 4,40 y aciertos en pares reservados **bajando** de 0,9068 a 0,8960: el margen crecía
  sin mejorar el orden.
- **Mecanismo:** con una respuesta de un solo token, empujar hacia abajo el logit de la jugada
  rechazada sube todas las demás por el softmax compartido, incluidas las malas. Un objetivo que
  solo dice «esto no» no dice adónde va la probabilidad que libera.
- **Con ancla** (`nll_weight` 0,5, lr 2e-6): margen 0,84, aciertos 0,9249, y la imitación cede solo
  0,021 nats.
- **Corrección que debe viajar con el episodio:** D-071 concluyó «DPO rompe la legalidad» (98,90 %)
  y D-073 lo desmintió — sobre `medium-v4` la legalidad se queda en 99,80 % antes y después. No era
  el método, era la falta de holgura en 39 M parámetros. Enseñarlo sin esa corrección sería
  generalizar desde un solo punto de operación.

## P4 · Fine-tuning e instrucción (2026-09-20)

### D-079 · El corpus entero empieza en 1800, así que media escala de Elo nunca se entrenó
- **Qué se encontró:** `configs/data/lichess-2025-01-02.yaml` fija `min_elo: 1800` y el filtro se
  aplica a **los dos** jugadores. Todo lo que ha visto cualquier modelo del proyecto —los 19 M de
  partidas de `tokens-v4`, la Elite DB incluida— está entre 1800 y ~3100.
  `data/elo-bins/manifest.json` lo confirma sin ambigüedad: su bin más bajo es **1800**.
- **Consecuencia:** los tokens `<w1000>`…`<w1700>` y sus gemelos de negras **nunca han recibido un
  gradiente**. Su embedding sigue en la inicialización. Pedirle al modelo publicado que «juegue
  como 1500» no le pide nada: le pone delante un vector aleatorio.
- **Qué reinterpreta:** D-069 midió que condicionar a 2600 no daba Elo y concluyó que imitar a un
  fuerte no compra táctica. Eso sigue en pie para el extremo alto. Pero el extremo **bajo**, que es
  el que pide el criterio de P4 y el que un jugador humano querría, no se había probado nunca
  porque no había datos con los que probarlo.
- **Qué se hace:** un segundo corpus con el mismo filtro salvo el rango (`min_elo: 1000`,
  `max_elo: 1799`, `configs/data/lichess-low.yaml`) en su propio `out_dir`, para que `data/uci`
  siga significando exactamente aquello con lo que se entrenó lo publicado.
- **Si está mal:** que el eje no se aprenda ni con datos, y entonces el problema no era el corpus
  sino la capacidad del condicionamiento por prefijo. Lo decidirá `rukh eval sweep`, no una
  opinión.

### D-080 · Leer un mes por `hf://` dejó de funcionar; se descarga shard a shard
- **Qué pasó:** el primer intento de descargar la banda baja murió con `HTTP 429 Too Many Requests`
  después de minutos de trabajo y sin fichero parcial. El segundo, ya autenticado y con ocho
  reintentos, se quedó **cincuenta minutos a 0,02 MB/s** con 8,6 GB de buffer en memoria y cero
  bytes escritos.
- **Por qué:** un mes son 72 ficheros de ~1 GB y leerlos en remoto son miles de peticiones de
  rango contra un solo host. El anfitrión responde 429 y DuckDB entra en una espera que se
  cuadruplica a cada intento. Medido el mismo día: descargar **un shard entero** va a 14,8 MB/s.
- **Qué se hace:** cada shard se descarga una vez con `huggingface_hub` (que cachea, reanuda y
  espera bien), se filtra en local a un fichero de parte y se borra. El pico de disco es un shard,
  el mes reanuda donde se quedó, y `limit` ahora **para la descarga** en vez de solo recortar el
  resultado: con 505 547 partidas de la banda 1000-1799 en el primer shard, 1,2 M salen de tres.
- **Lo que no cambia:** el filtro. `where_clause` se construye una vez y se usa en los dos sitios
  —la consulta que imprime el ensayo y cada shard descargado— para que lo documentado sea lo que
  corre.
- **Coste:** una conexión DuckDB ahora crea un secreto de Hugging Face cuando hay token local, y el
  presupuesto de reintentos sube de 3 a 8. Firmar las peticiones también sube el límite del
  anfitrión, así que es lo correcto aunque no hubiera 429.

### D-081 · El corpus balanceado es plano a propósito, y se estrecha donde se estrecha el mundo
- **Construido:** `data/elo-bins-v2`, bandas de 200 Elo de 1000 a 2600 sobre las dos mitades del eje
  (`data/uci-low` y `data/uci`), 150 000 partidas por banda.
- **Medido:** siete bandas llenas (1000-2200), 146 232 en la de 2400 y **23 191** en la de 2600.
  Total **1 219 423 partidas, 95 722 318 tokens**. La cabecera queda plana entre 59 335 y 87 662
  partidas por bin de 100 desde `<w1000>` hasta `<w2400>`, frente a la pirámide 20:1 del corpus real.
- **Por qué plano:** un corpus con la forma real enseña que `<w1500>` es *raro*, que no es lo mismo
  que enseñar qué *significa*. La frecuencia de una condición y su contenido son cosas distintas y
  aquí solo interesa la segunda.
- **Lo que no se puede arreglar:** no hay 150 000 partidas de 2600+ en dos meses de Lichess. El
  manifiesto publica el reparto real. El criterio de `GOAL.md` (1500 < 2000 < 2400) vive entero
  dentro de la parte plana, así que la cola fina no lo compromete.
- **Ancho de banda 200 y no 100:** con 100 las cuatro bandas de club tendrían la misma resolución
  que las de maestro y se quedarían sin partidas; con 400 se difuminaría la diferencia entre 1500 y
  1800, que es justo la que hay que enseñar.

### D-082 · `val_remainder_trains`: una regla razonable que habría deshecho el experimento
- **Qué pasaba:** el empaquetador parte el mes de validación en dos y suma el resto al
  entrenamiento (es lo que recuperó 238 M de tokens en D-062). Sobre el corpus balanceado eso añade
  **2,85 millones** de partidas de 1800+ encima del reparto plano y lo deshace en silencio.
- **Qué se hace:** `val_remainder_trains: false` en la config del afinado. El conjunto de
  validación sigue siendo exactamente el mismo con el que se miden las demás corridas —para que las
  pérdidas sean comparables— y el de entrenamiento es el corpus balanceado y nada más.
- **Por qué no se quita la regla:** para un preentrenamiento sigue siendo correcta y valiosa. Lo que
  hacía falta era un interruptor, no una marcha atrás.

### D-083 · El afinado por Elo cuesta 0,025 nats de imitación fuerte
- **Medido** sobre el mismo conjunto de validación (100 000 partidas de 2025-02, 1800+) y con el
  mismo medidor:

  | Modelo | pérdida val | top-1 val |
  |---|---|---|
  | `medium-v4` (paso 48 000) | 1,3733 | 54,73 % |
  | `medium-elo` (paso 3 800) | **1,3987** | **54,66 %** |

- **Es el precio esperado y es pequeño.** La validación mide imitación de juego 1800+, que es
  precisamente lo que este afinado deja de optimizar: dos tercios de su corpus son partidas de club.
  Que suba no es un defecto, es la definición de lo que se está haciendo.
- **Receta:** 3 800 pasos × 51 200 tokens = 194,56 M (unas dos pasadas), `lr` 1e-4 (dieciséis veces
  por debajo del preentrenamiento), warmup 200, coseno a 0,1. ~20 min en la 5090.
- **Lo que decide el hito no es esta tabla** sino el barrido por condición: lo que importa es qué
  compró ese cuarto de nat.

### D-084 · `best.pt` no es el resultado de un afinado, y ahora el bucle lo dice
- **Qué pasó:** `best.pt` se quedó congelado en el **paso 200** de `medium-elo`. Es correcto según
  su definición (menor pérdida de validación) y es la trampa: en un afinado que cambia de corpus a
  propósito la pérdida de validación sube desde el principio, así que «el mejor» es el modelo casi
  sin tocar.
- **Por qué es peligroso:** quien evaluara ese fichero después estaría midiendo los pesos
  equivocados, y los números saldrían perfectamente plausibles. No hay excepción que lo delate.
- **Qué se hace:** cuando hay `init_from` y `best.pt` no es el último paso, el bucle escribe un
  aviso que nombra el checkpoint que **sí** es el resultado y explica por qué. Con test.

### D-085 · La LoRA escrita a mano es LoRA: comprobado contra `peft`, paso a paso
- **Por qué hacía falta:** un `alpha` en el sitio equivocado, `A` y `B` intercambiadas o una
  inicialización distinta dan un modelo que entrena, converge y produce números creíbles. Ninguna da
  LoRA.
- **Cómo se comprueba:** dos modelos sobre los mismos pesos base, uno con `apply_lora` y otro con
  `peft.get_peft_model`; se copia nuestra `A` en la suya (`B` es cero en los dos por construcción) y
  los dos dan pasos de SGD sobre el mismo lote. Las pérdidas se comparan **paso a paso** con
  tolerancia 1e-5.
- **Medido:** iguales en los seis pasos, y el recuento de parámetros entrenables coincide.
- **Detalle que obliga a elegir el objetivo con cuidado:** la comparación se hace sobre `attn.proj`
  y no sobre `q`/`v`, porque `peft` no puede expresar lo mismo que nosotros sobre una matriz `qkv`
  fusionada: con `target_modules=["qkv"]` adapta las tres proyecciones con **una** pareja `A`/`B` de
  2304 filas, mientras que nuestra implementación da una por rango. Son parametrizaciones distintas
  y compararlas mediría esa diferencia, no la corrección del código.

### D-086 · Tres cosas que exige `PreTrainedModel` y que no salen en los tutoriales
- **`_tied_weights_keys` es un diccionario** `{copia: origen}` en `transformers` 5, no una lista. Con
  una lista, `save_pretrained` revienta con `'list' object has no attribute 'keys'`.
- **Hay que llamar a `post_init()`** al final del constructor. Sin él no existe
  `all_tied_weights_keys` y cualquier guardado o carga falla con un `AttributeError` sobre un
  atributo que uno nunca escribió. En esta versión `post_init` no toca los pesos, así que llamarlo
  después de construir el decoder es seguro.
- **La config no puede tener un miembro llamado `decoder`.** `transformers` lo lee como la mitad
  decodificadora de un par encoder-decoder e intenta llamarle `to_dict()`; un método enlazado se
  convierte en una excepción la primera vez que algo pide una `GenerationConfig`. Se llama
  `decoder_config()`.
- **`labels` no se vuelve a desplazar.** El flujo empaquetado ya guarda `y` un paso por delante de
  `x`; desplazarlas otra vez dentro de `forward`, como hace casi todo `transformers`, entrenaría al
  modelo a predecir la jugada de después de la siguiente, con una pérdida de aspecto normal.

### D-087 · El harness se abre a cualquier jugador, sin mover una coma del camino del decoder
- **Por qué:** M4 tiene que pasar un modelo de lenguaje general por la misma escalera que el
  decoder. Una comparación cuyas dos mitades corren por código distinto mide también el código.
- **Cómo:** `play_game` se parte en `play_game_with(player, opponent, ...)` más un `DecoderPlayer`
  que es el camino original palabra por palabra. `play_rung`/`play_rungs` aceptan un `player`
  opcional. Los 639 tests siguieron verdes sin tocar ninguno, que es la comprobación de que la
  refactorización no cambió comportamiento.
- **Lo que el protocolo obliga a decir:** `choose` devuelve la jugada **y** si la propuesta sin
  máscara era legal. Son dos preguntas distintas: la partida tiene que seguir, así que una propuesta
  ilegal se rescata, pero el rescate no puede esconder que ocurrió.

### D-088 · Un modelo de texto puede fallar de tres maneras que el decoder no tiene
- **El decoder** solo puede equivocarse de una forma: jugada legal en la posición equivocada. Su
  vocabulario *es* el conjunto de jugadas.
- **Un modelo que escribe SAN** puede además escribir algo que no es una jugada (`Nf9`), una jugada
  bien formada que esta posición no permite, o SAN **ambigua** que dos piezas podrían satisfacer y
  que no desambiguó (`Nd2` en vez de `Nbd2`).
- **Se cuentan por separado**, nunca sumadas en «ilegal». Sumarlas escondería justo lo que cuesta la
  representación, que es la mitad de lo que enseña la comparación.

### D-089 · Los puzles llevan el Elo real de sus jugadores, y eso habría falseado el barrido
- **Qué habría pasado:** casi todos los puzles del conjunto traen `white_elo`/`black_elo` de la
  partida de la que salieron, y `start_history` los usa cuando están. En un barrido por condición
  eso deja la columna de puzles **idéntica en las cinco filas** — correcto por su propia definición,
  e indistinguible de la evidencia de que la condición no hace nada.
- **Qué se hace:** `puzzles_use_header`, activo solo en el barrido, fuerza la cabecera de la
  condición en todos los puzles; y la caché del barrido es otra, porque estas tentativas responden a
  otra pregunta.
- **La caché vieja no se invalida:** el campo entra en la clave **solo cuando está activo**. Una
  clave de caché es la promesa de que dos corridas midieron lo mismo, y un interruptor apagado *es*
  el comportamiento con el que se jugaron las partidas cacheadas de P2 y P3.

### D-090 · Se mide y se publica la entropía de aperturas, que llevaba desde el spec en `n/a`
- **Por qué ahora:** el spec predice del afinado con maestros que «sube el Elo y baja la
  diversidad». Una tabla que solo mide la primera mitad no puede comprobar esa frase.
- **Dos números, porque fallan distinto.** La **entropía de líneas** (N auto-partidas, entropía de
  Shannon de las líneas distintas) es la que pide el spec y depende del muestreo: al ajuste casi
  determinista con el que se comparan las etapas, cualquier decoder juega una sola partida y saca 0.
  La **entropía de la primera jugada** es analítica, no tiene varianza entre tiradas y es la
  comparable entre etapas tal cual; su techo es log₂(20) = 4,32 bits.
- **La primera se publica siempre con la temperatura a la que se leyó** (D-047 otra vez).
- **Detalle de implementación que este modelo obliga:** las auto-partidas van en lote y el lote es
  rectangular por construcción. Las posiciones son *aprendidas*, así que rellenar por la izquierda
  desplazaría cada token real a una posición en la que nunca se entrenó, y rellenar por la derecha
  dejaría un `<pad>` en la columna que predice. El código falla antes que rellenar.

### D-091 · El eje pasó de ruido a significado, y se ve sin jugar una sola partida
- **Medido** con `rukh eval openings`: la distribución del propio modelo sobre las veinte primeras
  jugadas legales, sin temperatura, sin top-k y sin semilla. Es analítica, así que no tiene varianza
  entre tiradas y no hay que promediar cientos de auto-partidas para leerla.

  | Cabecera | `medium-v4` entropía | su jugada preferida | `medium-elo` entropía | su jugada preferida |
  |---|---|---|---|---|
  | `<w1200>` | 2,8556 | e2e4 43,7 % | **1,5955** | e2e4 66,9 % |
  | `<w1500>` | **2,9369** | e2e4 41,8 % | **1,6470** | e2e4 64,8 % |
  | `<w1800>` | 1,7695 | e2e4 59,6 % | 1,7408 | e2e4 60,5 % |
  | `<w2100>` | 1,9033 | e2e4 52,6 % | 1,8809 | e2e4 53,5 % |
  | `<w2400>` | 2,0104 | e2e4 46,1 % | 1,9916 | e2e4 47,2 % |

- **La huella del suelo del corpus está en los pesos.** En `medium-v4` hay un escalón entre
  `<w1500>` (2,9369 bits) y `<w1800>` (1,7695): **1,17 bits** de discontinuidad exactamente donde
  `min_elo: 1800` cortaba los datos. Por debajo del suelo el modelo está *menos* decidido que en
  cualquier cabecera entrenada, que es la firma del ruido y no la de la debilidad: un prefijo
  aleatorio no le pide nada, le confunde.
- **Después del afinado el eje es monótono y tiene el signo correcto:** 1,5955 → 1,6470 → 1,7408 →
  1,8809 → 1,9916 bits. Un jugador de club abre con `1. e4` o `1. d4` y poco más; el repertorio se
  ensancha con la fuerza. El modelo condicionado reproduce esa forma, y lo hace **en orden**.
- **Por encima de 1800 los dos modelos coinciden** (1,74 frente a 1,77; 1,99 frente a 2,01): el
  afinado no deshizo lo que ya estaba, solo llenó lo que faltaba.
- **Por qué esta medida vale aparte del Elo:** no cuesta una sola partida, no tiene ruido de
  muestreo y no depende de Stockfish. Si la escalera de Elo saliera ambigua, esto seguiría siendo
  evidencia de que la cabecera dejó de ser ruido.

### D-092 · Un `<style>` de componente `.astro` no llega a una lección MDX, y nada avisa
- **Qué pasó:** las tres figuras nuevas de M4 llevaban su CSS —incluidas las animaciones— en el
  bloque `<style>` de su propio componente, que es lo que documenta Astro. Se construyeron sin
  error, sin aviso y sin animación: en Chrome, `document.styleSheets` no contenía **ninguna** regla
  de esos componentes, mientras que las de `global.css` sí estaban.
- **Por qué no se había visto:** ninguna figura anterior del curso tenía `<style>`. Todas se
  pintan con atributos SVG y tokens, así que el proyecto llevaba tres módulos sin tocar ese camino.
- **Qué se hace:** el CSS de las figuras vive en `src/styles/global.css`, que es la misma regla que
  ya seguían las islas (`.vb*`, `.tr*`). Y un e2e comprueba lo que un build verde no comprueba:
  que cada figura tiene al menos un elemento con una animación corriendo.
- **La clase de fallo:** silencioso y estético. No hay excepción que lo delate, y revisarlo en una
  captura tampoco sirve si uno no sabe que debería moverse. La única defensa es una aserción sobre
  `getComputedStyle(...).animationName`.

### D-093 · Dos trampas de MDX que el navegador enseña y el build no
- **Un `<Term>` al principio de línea parte el párrafo.** MDX trata una línea que empieza por `<`
  como un bloque, así que cuando Prettier movió un `<Term>` al inicio de una línea, el párrafo que
  lo contenía se convirtió en dos. Renderiza sin error y se lee como dos frases rotas. Hay un e2e
  que falla si algún `<p>` de la prosa empieza en minúscula y no por código en línea.
- **Las casillas de tarea son campos de formulario sin etiqueta.** Los `- [ ]` que se copiaron de
  los documentos de plan renderizan como `<input type="checkbox">` sin `<label>`: **24 violaciones
  críticas** de axe en una sola lección. Las lecciones de M2 y M3 no tienen ni una; usan viñetas.
- **Los dos se detectaron en Chrome real**, no en `astro check` ni en el build, que pasaron los dos
  en verde con la lección rota.

### D-094 · QLoRA funciona en sm_120 y a 0,6 B no ahorra lo que dice su nombre
- **Medido** con la misma corrida de dos pasos, cambiando solo `four_bit`:

  | Precisión | pesos en la tarjeta | pico durante el entrenamiento |
  |---|---|---|
  | bf16 | 1 192 MB | 1 963 MB |
  | 4 bits NF4 | **851 MB** | **1 956 MB** |

- **El ahorro real es 341 MB sobre los pesos (29 %) y 7 MB sobre el pico (0,4 %).** El pico lo
  dominan las activaciones y el estado del optimizador, que la cuantización no toca. La razón de
  existir de QLoRA —caber en una tarjeta que sin ella no llegaría— no aplica a esta escala, y el
  módulo lo dice en vez de presentar la técnica como si se hubiera demostrado algo.
- **Por qué solo 341 MB y no ~900:** `bitsandbytes` no cuantiza la tabla de embeddings ni la cabeza
  atada, y en Qwen3-0.6B esa tabla son 151 936 × 1024 = **155,6 M parámetros, el 25,9 %** del
  modelo. Un cuarto de los pesos se queda en bf16 por construcción.
- **La contabilidad también engaña si no se corrige:** `numel()` sobre un `Params4bit` devuelve la
  mitad de los parámetros que representa, porque hay dos valores por byte. Sin corregirlo, la línea
  de «solo el 0,764 % es entrenable» saldría el doble de favorable.
- **`bitsandbytes` 0.50.2 instala y carga** en Windows con CUDA 12.8 y sm_120 sin `nvcc`. Entra en
  el extra `hf`.

### D-095 · El ensayo de dos pasos encontró un fallo que habría aparecido a los veinticinco minutos
- **Qué se hizo:** antes de la corrida real de Qwen (media hora de GPU) se ejecutó la misma receta
  con `max_steps: 2` y cien partidas, y la evaluación con una suite de doce posiciones, nueve
  puzles y una partida.
- **Qué encontró:** `estimate(records, bootstrap=...)` — el parámetro se llama `samples`. Un
  `TypeError` que solo se dispara después de jugar todas las partidas, es decir, al final de la
  fase más cara de la evaluación.
- **La regla:** una receta nueva se ensaya con el presupuesto más pequeño en el que todavía pasa por
  todas sus fases. Todo lo que puede fallar en un afinado falla en los primeros treinta segundos —
  la descarga, el backend de cuantización, la API del entrenador, el relleno del tokenizador, el
  guardado— y descubrirlo al final es la forma cara.

### D-096 · Los factores de LoRA se creaban en CPU con el modelo ya en CUDA
- **Qué pasó:** `apply_lora` construía `A` y `B` con `torch.empty(...)` sin dispositivo, y el bucle
  aplica los adaptadores **después** de mover el modelo y cargar el checkpoint (cargar necesita los
  nombres de módulo de un decoder limpio). Primer `forward` en la GPU: `Expected all tensors to be
  on the same device`.
- **Por qué los tests no lo veían:** los catorce tests de `test_lora.py` corren en CPU, donde el
  fallo es invisible. Lo destapó un ensayo de veinte pasos sobre el modelo real —el mismo
  procedimiento que en D-095— y ahora hay un test que comprueba la *propiedad* (los factores viven
  donde el peso que corrigen) en vez del dispositivo en el que toque ejecutarse.

### D-097 · Fundir un adaptador es exacto a 3,6e-6 relativo sobre 115 M de parámetros
- **Medido** sobre `medium-v4` con un adaptador real de `r = 8`: diferencia máxima entre el modelo
  con envoltorios y el mismo modelo fundido, **6,8e-5 absoluta sobre logits de escala 18,9**, o sea
  3,6e-6 relativa, y **la jugada elegida es la misma** en todas las posiciones probadas.
- **Por qué importa el número y no solo el test:** el test unitario comprueba la igualdad sobre un
  decoder de juguete con tolerancias de `float32`; la afirmación que hace la lección —«publicar un
  adaptador de 1,6 MB y aun así exportarlo como un modelo cualquiera»— es sobre el modelo de 115 M,
  donde el error se acumula por dieciséis capas. Es exacta en lo que importa (el argmax) y no bit a
  bit, y conviene decirlo así.
- **El adaptador pesa 1,6 MB**, que es exactamente lo que predice `2 · r · d_model · capas ·
  objetivos` = 393 216 números en `float32`.

### D-098 · M1 prometía lo que M4 tuvo que desmentir, y se corrige donde estaba escrito
- **Qué decía la lección de M1**, publicada y en `vigente`: «en M4, cuando quieras que juegue como
  un 1500, no habrá que entrenar nada nuevo: bastará con poner `<w1500>` al principio y muestrear».
  La ficha de M1 repetía la misma frase.
- **Por qué era falso:** tres secciones más abajo, en esa misma página, está el recorte con
  `min_elo: 1800`. El mecanismo que M1 explica es correcto —el Elo va delante, condiciona todas las
  jugadas— y la promesa daba por hecho que los datos lo alimentaban.
- **Qué se hace:** se borra la promesa y se pone en su lugar un aviso que cuenta el error, cómo se
  encontró (contando: cero de doce cabeceras en 1 681 069 636 tokens) y lo que costó (un corpus
  nuevo y un afinado). No se reescribe la historia: se deja dicho qué decía antes.
- **Lo que no había que tocar:** los Elo de M1 y M2 ya son los corregidos de D-070 (1425 y 1397
  para `small` v3 a `<w1800>` y `<w2600>`; 1359 para `small` v1), y M2 ya cuenta su propia
  corrección. El problema estaba solo en la promesa.

### D-099 · La tabla única servía Elo retractados en `/proyecto/`
- **Qué pasaba:** `configs/eval/greedy.yaml` tenía `web_results: null`, así que ninguna de las
  corridas deterministas —las que producen todos los números publicados— escribía en
  `artifacts/web/results.json`. La tabla se quedó con las filas de la suite `full`, medidas antes
  de que D-070 corrigiera la escalera: **1091** para `medium`, **1007** y **785** para `small`,
  **64** para `tiny`. La página en inglés del proyecto las servía.
- **Qué se hace:** `greedy.yaml` escribe en la tabla; las etapas publicadas se vuelven a evaluar
  (las partidas y los puzles están en `cache-greedy.sqlite`, así que son minutos y no otra hora de
  Stockfish); y las filas que ninguna corrida corregida reemplaza se **retractan** con
  `rukh eval drop`, que es una operación que faltaba.
- **Por qué existe el comando en vez de editar el JSON:** una medición puede resultar equivocada, y
  cuando lo es la tabla tiene que poder dejar de llevarla. Es explícito y por etapa a propósito:
  tirar una fila es tirar una medición, y debería costar decir su nombre.

### D-100 · El criterio 1 no se cumple, y la aritmética dice que no es cuestión de partidas
- **Qué se midió:** el barrido con las seis condiciones, 160 partidas cada una contra los ocho
  peldaños de Stockfish, con la cabecera forzada también en las posiciones de validación y en los
  puzles (D-088). Elo: 1425, 1549, 1498, 1538, 1606, 1644 para `<w1200>`…`<w2400>`. **Ni monótono
  ni separado.** Span de 219 Elo entre extremos.
- **Lo que pide `GOAL.md`:** `Elo(<w1500>) < Elo(<w2000>) < Elo(<w2400>)` con intervalos. El primer
  par va del revés: 1549 contra 1538, once puntos de Elo, y la tasa de puntos contra la escalera
  —que es el número que se mide, el Elo es una transformación de él— va de 0,466 a 0,453.
- **La pregunta que había que contestar antes de pedir más máquina:** ¿faltan partidas o no hay
  diferencia? Es aritmética. La tasa de puntos es una proporción, su error típico es
  `sqrt(p(1-p)/n)` y dos intervalos del 95 % dejan de tocarse cuando la distancia entre las tasas
  supera `1,96 (se1 + se2)`, que para proporciones cerca de la mitad es
  `n > 3,84 / (delta p)^2` partidas por condición. Con los números medidos
  (`labs/m4/games_needed.py`):

  | par | tasa | delta | partidas necesarias |
  |---|---|---:|---:|
  | `<w1500>` → `<w2000>` | 0,466 → 0,453 | **−0,013** | **24 420** |
  | `<w2000>` → `<w2400>` | 0,453 → 0,569 | +0,116 | 284 |
  | `<w1500>` → `<w2400>` | 0,466 → 0,569 | +0,103 | 357 |
  | `<w1200>` → `<w2400>` | 0,331 → 0,569 | +0,238 | 64 ✅ ya separado |

- **Qué se decide:** el criterio 1 **no se cumple** y se dice así, con la tabla delante. No se
  vuelve a correr `<w2000>`: veinticuatro mil partidas por condición son sesenta horas de
  Stockfish para estrechar el intervalo alrededor de una diferencia que no está. Sí se vuelven a
  correr `<w1500>` y `<w2400>` con 400 partidas (`configs/eval/greedy-sweep-50.yaml`), que es lo
  que el mismo cálculo pide para el par exterior del criterio, y que convierte «no separado con
  160» en una afirmación medida en un sentido o en el otro.
- **Por qué esto es un resultado y no una excusa:** los extremos del eje **sí** están separados
  (1200 contra 2400, con 64 partidas habría bastado y se jugaron 160), la entropía analítica de la
  primera jugada es monótona en las **seis** condiciones sin una sola partida de por medio
  (1,596 → 1,647 → 1,741 → 1,841 → 1,881 → 1,992 bits) y los puzles son planos
  (37,0 – 38,2 %). Las tres cosas dicen lo mismo: **la cabecera mueve el estilo, no la fuerza
  táctica**. Que es exactamente lo que P3 sospechaba y no podía afirmar, porque entonces las
  cabeceras por debajo de 1800 ni siquiera se habían entrenado.

### D-101 · La cabecera sí mueve dos columnas que en la primera pasada salían constantes
- **Qué cambió:** con `force_header` extendido a las posiciones de validación (D-088), la legalidad
  sin máscara y el top-1 dejan de ser idénticos en todas las filas y se ordenan solos:

  | cabecera | legal argmax | top-1 |
  |---|---:|---:|
  | `<w1200>` | 100,00 % | 51,40 % |
  | `<w1500>` | 100,00 % | 53,90 % |
  | `<w1800>` | 99,80 % | **54,10 %** |
  | `<w2000>` | 99,70 % | 54,00 % |
  | `<w2100>` | 99,70 % | 53,40 % |
  | `<w2400>` | 99,70 % | 53,30 % |

- **Cómo se lee:** el top-1 tiene un máximo en `<w1800>` y baja hacia los dos lados. No es ruido:
  las posiciones de validación son partidas reales con su reparto real de Elo, cuya media está
  cerca de 1800, así que la cabecera que mejor predice la jugada siguiente es la que describe a
  los jugadores que la hicieron. Pedirle al modelo que juegue como 2400 lo hace **peor** prediciendo
  jugadas de un jugador medio, y eso es lo correcto.
- **Y la legalidad baja al subir la cabecera**, de 100,00 % a 99,70 %. Tres décimas son tres
  jugadas de mil, pero el signo es el mismo que el de la entropía: cuanto más ancho el repertorio,
  más lejos de las aperturas trilladas y más ocasiones de escribir algo ilegal.

### D-102 · Los adaptadores viajan como entradas del grafo, no fundidos en los pesos
- **El problema:** fundir un adaptador y exportar el ONNX cuesta 221 MB por estilo en `medium`.
  Dos estilos son 442 MB de descarga para mover 1,6 MB de corrección, que es tirar por tierra la
  única cosa que LoRA compra.
- **Qué se hace:** el grafo se exporta con `A` y `B` como **entradas**, apiladas por capas en dos
  tensores `(capas, rangos, r, d)` y `(capas, rangos, d, r)` (`rukh.export.adapter`). El navegador
  cambia de estilo subiendo 1,6 MB, no descargando otro modelo.
- **Las tres propiedades que lo hacen honesto, y las tres están en `tests/unit/test_export_adapter.py`:**
  con ceros el grafo **es** el modelo base (por eso el fichero adaptable *sustituye* al ordinario en
  vez de sumarse a él, y una demo sin estilo elegido no está corriendo otro modelo); con un
  adaptador distinto la respuesta cambia sin tocar el fichero; y la paridad contra PyTorch con el
  mismo adaptador cargado se mide sobre posiciones reales.
- **Lo que no admite:** un adaptador cuyos rangos no midan todos lo mismo. Los factores viajan como
  un tensor cada uno, así que todas las capas tienen que adaptar la misma matriz con los mismos
  anchos —cierto para consulta, clave y valor sobre el `qkv` fundido, falso en cuanto entra el MLP,
  cuyo `fc` es cuatro veces más ancho—. Ese caso sigue teniendo `merge_lora` y una exportación
  normal; lo único que no tiene es el intercambio en caliente. `adapter_layout` lo dice con esas
  palabras en vez de apilar algo que no cuadra.
- **Detalle de implementación que sí importa:** la corrección se arma con `cat` sobre todo el ancho
  de salida en vez de escribirse en una rodaja de un tensor de ceros. La asignación por índice
  exporta como `ScatterND`, que onnxruntime web ejecuta en CPU aunque el resto vaya por WebGPU.

### D-103 · El fichero dice cuánto mide un adaptador válido, así que el navegador no hay que decírselo
- **El problema de siempre en esta demo:** ORT Web no expone `metadata_props`, así que todo lo que
  el exportador escribe dentro del `.onnx` (`rukh_block`, `rukh_vocab_size`) es inalcanzable desde
  el navegador y tiene que viajar por el registro (D-031). Un adaptador tenía pinta de necesitar lo
  mismo: formas, rango, orden de los tensores, un JSON al lado.
- **Lo que se hace en vez de eso:** las entradas `lora_a` y `lora_b` se declaran con **forma fija**,
  y las formas de las entradas sí las expone ORT (`inputMetadata`). Así que el propio fichero dice
  cuántos `float32` tiene un adaptador suyo, y `readAdapterShape` lo lee. Un fichero descargado de
  cualquier otro tamaño es un desajuste que la página puede **nombrar** en vez de leer como basura
  sobre los pesos correctos, que es la peor forma de estar mal: un modelo que juega legal y fatal.
- **Y «sin estilo» no es un caso especial.** El selector en «Sin estilo» alimenta un adaptador de
  ceros, que por construcción es el modelo base exacto. Ni se recrea la sesión ni se vuelve a
  descargar nada: el camino de código es el mismo con estilo y sin él, y el E2E comprueba que
  quitar el adaptador devuelve **el mismo número**, no uno parecido.
- **Cómo se verifica en un navegador de verdad:** `scripts/make_toy_web_models.py` escribe, con las
  mismas funciones que escriben los ficheros publicados, un decoder de juguete exportado con
  factores como entradas (186 KB) y un adaptador para él (256 bytes). `e2e/adapter.spec.ts` carga
  el modelo una vez, lee la distribución que publica para una posición, cambia de estilo, la vuelve
  a leer y comprueba tres cosas: que cambia, que al quitarlo vuelve exactamente, y que solo se ha
  pedido **un** `.onnx` en toda la prueba. Esa última aserción es el hito entero en una línea.
- **Lo que no se toca:** `toy-decoder.onnx`, el de juguete que ya estaba versionado, se deja como
  está. Regenerarlo con otra semilla cambiaría las jugadas que espera media docena de pruebas del
  navegador, y no hay ninguna razón para hacerlo; el script lo reescribe solo con `--plain`.

### D-104 · Un valor por defecto de la especificación rompió el fp16 de todo modelo intercambiable
- **El síntoma:** `onnxruntime` se niega a cargar el fichero fp16 con
  `Type Error: Type (tensor(float16)) of output arg (val_20) of node (node_ConstantOfShape_18)
  does not match expected type (tensor(float))`. El fp32 carga perfectamente, y el fp32 no lo sirve
  nadie.
- **La causa:** la corrección de LoRA se arma con `cat` sobre todo el ancho de la proyección, y los
  rangos que nadie adapta son un `ConstantOfShape` de ceros. El exportador lo escribe **sin**
  atributo `value`, porque la especificación dice que el valor por defecto es un cero de
  `float32`. `convert_float_to_float16` retipa el grafo alrededor y no toca el nodo, así que el
  grafo pasa a declarar `float16` donde el nodo sigue produciendo `float32`.
- **La familia entera del fallo:** es el mismo error que ya arreglaba `_align_cast_outputs` —un
  `Cast` cuyo atributo `to` se quedó diciendo el tipo viejo— escrito de otra forma. Aquí no hay un
  atributo equivocado: hay un atributo **ausente**, y lo que cambia bajo los pies es el valor por
  defecto que lo sustituía. Un parámetro implícito es una dependencia como cualquier otra, y las
  herramientas que reescriben un fichero no la ven.
- **Qué se hace:** `_align_constant_outputs` retipa el `value` de todo `Constant` y
  `ConstantOfShape` que no cuadre con lo que el grafo declara, y le **pone** el atributo al que no
  lo tenía. Y el test que lo cubre no comprueba que el fichero cargue: comprueba que, ya cargado,
  el adaptador sigue cambiando la respuesta en fp16 y en int8. Cargar es el mínimo; lo que había
  que saber es si la cuantización deja intactas las dos multiplicaciones del adaptador, cuyos dos
  operandos vienen de fuera del grafo y no son inicializadores.
- **Cuándo se encontró:** en el modelo de juguete, antes de la exportación real de 221 MB. La
  prueba que lo destapó cuesta catorce segundos; la que no se escribió habría costado cuarenta y
  cinco minutos de exportación y una demo rota en el navegador.

### D-105 · La familia de P4 se construye sobre `medium-v4` y no sobre `small`
- **Qué dice `GOAL.md`:** publicar `rukh-small-masters` y `rukh-small-elo`. Lo que se publica es
  `rukh-medium-masters` y `rukh-medium-elo`.
- **Por qué:** `medium-v4` es el modelo que la demo sirve y el que tiene margen para que el eje se
  note. `small` saca 1007 de Elo y 51,1 % de top-1; pedirle además que el estilo cambie con la
  cabecera es pedirle que mueva una aguja que apenas se ve. Los dos afinados y los dos adaptadores
  parten del mismo `medium-v4-20260919-174623/best.pt`, que es también la base sobre la que la card
  de cada adaptador dice que se monta.
- **Coste de la desviación:** 221 MB por etapa en la demo en vez de 75, y la descarga se pide con
  el consentimiento delante. A cambio, las mediciones del hito se hacen sobre el modelo cuyos
  números ya están publicados, así que cada comparación es contra una fila que existe.
- **Lo que cambia con el nombre:** nada más. El recorte, la receta, la suite y los criterios son
  los del plan; lo único que se sustituye es el tamaño del modelo base, y queda escrito aquí para
  que nadie busque `rukh-small-elo` en el Hub.

### D-106 · El control que impide cantar victoria: el modelo **sin** afinar mueve el mismo Elo
- **Por qué se hizo:** `medium-elo` puntúa 1425 pidiéndole 1200 y 1606 pidiéndole 2100, un salto de
  181 Elo con los intervalos separados. Leído solo, eso es «el condicionamiento da fuerza». Hay una
  segunda explicación: que **cualquier** modelo puntúe menos con una cabecera baja, y entonces el
  salto no diría nada del afinado. Se distinguen con un control, y el control es correr el mismo
  barrido sobre el modelo **base**, que por debajo de 1800 no tiene cabecera ninguna: `<w1200>` es
  para él un vector de la inicialización.
- **Lo que salió:**

  | cabecera | modelo | Elo | IC 95 % | legal | top-1 | puzles | entropía 1.ª |
  |---|---|---:|---|---:|---:|---:|---:|
  | `<w1200>` | `medium-v4` (base) | 1419 | 1358-1479 | 99,60 % | 49,80 % | 36,42 % | 2,856 |
  | `<w1200>` | `medium-elo` | **1425** | 1361-1479 | **100,00 %** | **51,40 %** | **37,02 %** | **1,596** |
  | `<w2100>` | `medium-v4` (base) | 1580 | 1512-1649 | 99,80 % | 53,80 % | 38,27 % | 1,903 |
  | `<w2100>` | `medium-elo` | 1606 | 1543-1670 | 99,70 % | 53,40 % | 38,08 % | 1,881 |

- **La conclusión, que es incómoda y es la correcta:** el modelo **base** también recorre el eje,
  161 Elo, con los intervalos separados. Y no puede ser condicionamiento, porque su `<w1200>` nunca
  recibió un gradiente. Lo que le pasa al base es otra cosa: un prefijo desconocido le **estorba**,
  y estorbarle cuesta unos ciento sesenta puntos. El afinado mueve 181, que está dentro del ruido
  de los 161. **La columna de Elo no distingue las dos causas**, así que por sí sola no es evidencia
  de que la cabecera signifique algo.
- **Dónde sí se distinguen, en la misma tirada:** a `<w1200>`, con el mismo Elo, el modelo afinado
  escribe **cero** jugadas ilegales contra cuatro de mil del base, acierta 1,6 puntos más de top-1,
  resuelve 0,6 puntos más de puzles y —lo más claro— tiene **1,26 bits menos** de entropía de
  primera jugada: 1,596 contra 2,856. Es decir, el afinado convirtió «ruido en la entrada» en «un
  jugador de club decidido», y eso se ve en todo menos en el resultado contra Stockfish.
- **Y a `<w2100>` los dos modelos coinciden** en las cinco columnas, como tenía que ser: esa
  cabecera siempre estuvo entrenada y el afinado no debía tocarla. Que no la tocara es la prueba de
  que no hubo olvido catastrófico.
- **La regla que deja:** cuando una métrica sube con el tratamiento, córrela también sobre el
  modelo sin tratar. Si sube igual, la métrica no mide el tratamiento. Cuesta una hora de máquina y
  es la diferencia entre publicar un hallazgo y publicar un artefacto.

### D-107 · La misma escalera, dos veces, con la misma semilla: 1498 y 1558
- **Qué pasó:** `medium-elo` a `<w1800>` se midió dos veces por caminos distintos —la fila `@1800`
  del barrido y la evaluación canónica— con **la misma** configuración en todo lo que afecta a las
  partidas: mismos ocho peldaños, `elo_games: 20`, `elo_move_time: 0.1`, `seed: 42`, misma
  temperatura, mismo `top_k`. Salió **0,4094 → 1498** en una y **0,4750 → 1558** en la otra.
- **Por qué no es un fallo:** la semilla fija *nuestro* muestreo, no el de Stockfish. El rival juega
  con un límite de **tiempo** (`chess.engine.Limit(time=0,1)`) y con `UCI_LimitStrength`, que además
  aleatoriza a propósito para acertar el Elo pedido. Así que «las mismas 160 partidas» no son las
  mismas partidas: son 160 partidas nuevas contra un rival que no se repite.
- **Y el tamaño cuadra con la aritmética:** la diferencia de tasa es 0,0656 y el error típico de la
  diferencia entre dos tiradas independientes de 160 partidas con `p ≈ 0,44` es
  `sqrt(2 p (1-p) / 160) = 0,0555`. Son **1,18 σ**. Ruido de manual.
- **Lo que esto significa para el hito:** el suelo de **reproducibilidad** del instrumento es de
  unos 40 Elo de una sigma, es decir, unos ±80 al 95 %. Los pares contiguos del barrido están a 11,
  40, 51 y 38 puntos. Están **por debajo de lo que el instrumento repite**, y eso no lo arregla
  ninguna cantidad de partidas mientras el rival vaya por tiempo: lo arreglaría hacerlo
  determinista (límite por **nodos** o por profundidad) y volver a calibrar la escalera. Es la misma
  conclusión de D-100 vista desde el otro lado, y esta se puede enseñar con dos números.
- **Qué se hace ahora:** nada en las mediciones —las dos son correctas y las dos se publican, cada
  una diciendo de qué corrida sale—. La tabla única lleva la canónica (1558), que es la que
  comparte suite con el resto de etapas; el barrido lleva la suya, que es la que comparte suite con
  las otras cinco condiciones. Comparar **dentro** de una tirada es válido; comparar **entre**
  tiradas es lo que este apunte existe para desaconsejar.
- **La regla que deja:** antes de explicar una diferencia pequeña, mide cuánto se mueve tu montaje
  cuando no cambias nada. Aquí no hubo que montar nada: bastó con que dos caminos distintos
  midieran lo mismo sin querer.

### D-108 · El módulo se reformula sobre lo medido, y el criterio se queda incumplido
- **Decisión de Borja (2026-09-20):** «el valor del curso está en que cada módulo enseñe algo
  cierto, no en aprobar el criterio que se escribió antes de medir». No se persigue el criterio 1
  con más máquina; M4 se reformula sobre lo que el hito **sí** estableció.
- **El título de la lección cambia** de «enseñarle a jugar peor» —que promete lo que no se
  cumplió— a **«cambiar el estilo sin cambiar la fuerza»**, y la pregunta del módulo pasa a ser una
  sola: *¿qué cambia un afinado y qué no?*
- **Lo que el módulo enseña, con las mediciones que lo sostienen:**
  - el afinado cambia el **comportamiento** de forma total y barata: 1,6 MB (0,34 % del modelo)
    llevan `1. e4` del 59,64 % al 99,85 % sin coste medible en legalidad, top-1 ni puzles;
  - ordena el **repertorio** a lo largo del eje sin jugar una partida: entropía monótona 6 de 6;
  - y **no mueve la competencia**: ni el condicionado ni el de maestros separan su intervalo de Elo
    del modelo del que salieron;
  - más la mitad metodológica, que es la que se reutiliza: partidas necesarias (D-100), corrida de
    control (D-106) y suelo de reproducibilidad (D-107).
- **Por qué esto no es rebajar el listón:** el criterio se publica **incumplido**, con sus números y
  su explicación, en `GOAL.md`, en el ledger y en la lección. Lo que se reformula es el **objetivo
  docente**, no la medición. Y el resultado explica por qué M5 existe: si imitar mejor no da
  táctica, hace falta otra herramienta —recompensas, DPO, GRPO—, que es exactamente el hito
  siguiente.
- **Lo que sí se mide antes de cerrar, porque es el mecanismo y no el criterio:** toda la suite
  juega con `temperature: 0.05, top_k: 1`, es decir, **la moda** de la distribución. La diferencia
  entre un 1200 y un 2400 no está en la moda —los dos hacen la recaptura obvia— sino en la cola.
  La entropía de la primera jugada, que lee la distribución entera, es monótona; los puzles y el
  Elo, que leen la moda, son planos. Así que se corre el barrido una vez a **temperatura 1,0 con
  top-k 20** en `<w1200>` y `<w2400>`: si la brecha se abre, el eje controla la fuerza y lo que
  faltaba era dejar de muestrear el pico; si sale plano, es un segundo nulo y la lección lo dice.
  Una hora de máquina, y contesta una pregunta en vez de precisar un no.

## Publicación en Hugging Face (2026-09-19)

Once repos en `chorcat`, todos con card en inglés:

| Modelos | Tamaño | Datasets | Filas |
|---|---|---|---|
| `rukh-tokenizer` | 0,4 MB | `rukh-games-1800` | 5 896 388 |
| `rukh-tiny` | 61,5 MB | `rukh-games-elite` | 541 085 |
| `rukh-small` | 434,9 MB | `rukh-elo-bins` | 423 829 |
| `rukh-medium` | 1 275,4 MB | `rukh-positions-eval` | 488 159 |
| `rukh-encoder` | 169,9 MB | `rukh-puzzles-split` | 312 000 |
| | | `rukh-pairs-dpo` | 13 887 |

Cada modelo lleva pesos en `safetensors`, los tres ONNX y `onnx/parity.json` con la medición de
fidelidad. Paridad de la jugada elegida: `small` fp16 99,80 % e int8 95,40 %; `tiny` 99,80 % y
96,10 %; `medium` **100 %** y 96,50 %; el encoder, 100 % en las tres precisiones sobre la decisión
de error.
