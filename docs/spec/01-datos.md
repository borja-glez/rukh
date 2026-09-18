# 01 · Datos

Todas las fuentes verificadas el 2026-09-18. Salvo indicación, licencia **CC0 1.0** (Lichess publica
sus volcados bajo CC0; las copias en Hugging Face heredan la licencia).

## Fuentes principales

| Dataset | Dónde | Tamaño | Formato | Para qué |
|---|---|---|---|---|
| Partidas estándar valoradas | `Lichess/standard-chess-games` (HF) | 6 771 826 271 partidas; particionado `year=/month=`; un mes reciente ≈ 72 parquet ≈ 73 GB | Parquet; columnas abajo | Preentrenamiento del decoder, encoder, Elo-conditioning |
| Puzles | `Lichess/chess-puzzles` (HF) | 6 100 960 puzles, 877 MB | Parquet: `PuzzleId, GameId, FEN, Moves (UCI), Rating, RatingDeviation, Popularity, NbPlays, Themes[], OpeningTags[]` | Evaluación por dificultad; SFT táctico; recompensas |
| Evaluaciones de Stockfish | `Lichess/chess-position-evaluations` (HF) | 394 669 566 posiciones, 20 parquet ≈ 42 GB | `fen, line (PV en UCI), depth, knodes, cp, mate` | Etiquetas de valor y de "jugada mejor" sin ejecutar Stockfish: cabezas del encoder, reward model, pares DPO |
| Aperturas ECO | `Lichess/chess-openings` (HF) | 3 704 aperturas | `eco, name, pgn, uci, epd` | Libro de aperturas para RAG y para la demo |
| Base de datos Elite | `database.nikonoel.fr` (zip mensuales, 60-100 MB) | Partidas 2500+ contra 2300+, sin bullet | PGN | SFT "como los maestros"; origen Lichess (CC0 de facto, sin licencia explícita en la web: citar) |
| Volcados PGN originales | `database.lichess.org` (`.pgn.zst` mensuales) | Alternativa a los parquet | PGN con `%eval` en ~6 % y `%clk` | Solo si hace falta el PGN literal |

Columnas de `standard-chess-games`: `Event, Site, White, Black, Result, WhiteTitle, BlackTitle,
WhiteElo (int16), BlackElo (int16), WhiteRatingDiff, BlackRatingDiff, UTCDate, UTCTime, ECO, Opening,
Termination, TimeControl, movetext`. `movetext` es SAN numerado con comentarios `%clk` (desde 2017) y
`%eval` en ~6 % de partidas. Los Elo son Glicko-2. Nota de la ficha: está previsto añadir una columna
`UCI`; hasta entonces la conversión SAN→UCI la hace `python-chess`.

## Fuentes secundarias (referencia y baselines)

| Recurso | Uso | Licencia |
|---|---|---|
| `adamkarvonen/chess_llms` y `adamkarvonen/chess_games` (HF) | Baseline público: nanoGPT de 8-16 capas sobre PGN a nivel de carácter; el 50M llegó a ~1300 Elo con 99,8 % legales tras 16M partidas y un día de GPU. Sus datasets (6-9 GB de PGN, 16M partidas) sirven para comparar tokenizaciones | MIT |
| `google-deepmind/searchless_chess` (ChessBench) | Referencia de techo: 10M partidas anotadas con Stockfish 16 (15 000 M puntos); modelos 9M/136M/270M; el 270M alcanzó 2895 Elo blitz en Lichess. Los ficheros de action-value pesan 1,1 TB: NO se descargan; se cita | Apache-2.0 (código) |
| Maia-2 / Maia-3 (`CSSLab/maia2`, `CSSLab/maia3`) | Baseline de predicción de jugada humana por nivel de Elo (Elo-conditioning). Comparación en M4 | GPL-3 (código); pesos propios |
| Libros en dominio público (Project Gutenberg) | RAG de la fase 2: Capablanca, *Chess Fundamentals* (#33870); Edward Lasker, *Chess Strategy* (#5614); más títulos bajo el tema "Chess" de Gutenberg, verificando cada uno | Dominio público |

## Recorte de trabajo (decisión por defecto)

No se descargan 73 GB por mes. Con DuckDB o Polars sobre `hf://datasets/Lichess/standard-chess-games/data/year=2025/month=0[1-2]/*.parquet` se filtra en remoto con *predicate pushdown* y solo se materializa lo que pasa el filtro:

- `WhiteElo >= 1800 AND BlackElo >= 1800`
- `TimeControl` con base ≥ 180 s (excluye bullet; `TimeControl` es `"base+incremento"`)
- `Termination IN ('Normal', 'Time forfeit')`
- ≥ 20 medias jugadas y sin `Event` de variantes
- Se eliminan los comentarios `%clk`/`%eval` del `movetext` y se convierte a UCI con `python-chess`

Estimación: 3-6 millones de partidas por dos meses recientes, ≈ 2-3 GB de texto UCI. Es el volumen con el que Karvonen entrenó modelos de 25-50M; en la 5090 basta para 30-50M parámetros en 6-12 h. Si sobra tiempo, se añaden meses.

Conjuntos derivados (todos publicados en `chorcat/` como datasets con model card y con los filtros exactos en un manifiesto JSON):

| Derivado | Contenido | Módulo |
|---|---|---|
| `rukh-games-1800` | Partidas filtradas, UCI, con `WhiteElo, BlackElo, Result, TimeControl, UTCDate` | M1-M2 |
| `rukh-games-elite` | Elite DB (2500+/2300+) en el mismo formato | M4 (SFT maestros) |
| `rukh-elo-bins` | Muestreo equilibrado: hasta N partidas por tramo de 100 Elo | M4 (Elo-conditioning) |
| `rukh-positions-eval` | Posiciones de las partidas cruzadas con `chess-position-evaluations` por FEN normalizado (solo piezas, turno, enroques, al paso) | M3 (valor), M5 (RM, DPO) |
| `rukh-puzzles-split` | Puzles por dificultad (1000-1500 / 1500-2000 / 2000+) con split fijo | M6 (harness) |
| `rukh-pairs-dpo` | Pares (posición, jugada buena, jugada mala) desde evaluaciones: diferencia ≥ 100 cp | M5 |

Splits: por **mes** (entrenamiento con meses A-B, validación con el mes C) para evitar fugas por partidas
repetidas de los mismos jugadores; los puzles se dividen por `PuzzleId` con semilla fija.

## Tokenización (lab de M1)

Tres representaciones de la misma partida, comparadas con la misma arquitectura pequeña en el mismo
número de pasos:

1. **Vocabulario fijo de jugadas UCI**: todas las cadenas `desde-hasta[promoción]` posibles
   (~1 900 tokens; se construye enumerando pares de casillas y promociones, no observando datos).
   Ventaja: un token por jugada, secuencias cortas (≤ 200). Es la representación por defecto.
2. **SAN a nivel de carácter** (como Karvonen): vocabulario ~32 símbolos, secuencias 5-6× más largas,
   el modelo tiene que aprender la notación. Sirve para enseñar por qué la tokenización importa.
3. **BPE entrenado sobre el texto UCI** con `tokenizers` (HF): se ve qué fusiones aprende (aperturas
   enteras como un token) y cómo eso ayuda o estorba.

Tokens especiales: `<bos>`, `<eos>`, `<pad>`, resultado (`<1-0>`, `<0-1>`, `<1/2>`) y Elo de cada bando
por tramos de 100 (`<w1800>` … `<b2400>`) al principio de la secuencia; son las "instrucciones" del
modelo desde el primer día y hacen posible el condicionamiento de M4. Longitud de contexto: 200 tokens
(cubre el 95 % de las partidas ≥ 1800 Elo); las partidas más largas se truncan o se trocean con solape.

## Pipeline de datos (`rukh/src/rukh/data/`)

1. `fetch.py`: lectura remota de parquet con DuckDB, filtros, escritura local en parquet particionado
   por mes en `data/raw/` (gitignored). Manifiesto JSON con filtros, meses, conteos y hashes.
2. `uci.py`: SAN→UCI con `python-chess`, verificación de legalidad (descarta partidas corruptas),
   limpieza de comentarios. Paralelizado con `multiprocessing` (una partida ≈ 0,3 ms).
3. `tokenize.py`: las tres tokenizaciones; salida en arrays `uint16` empaquetados en `.npy`/`memmap`
   con índices de partida (formato "blocks" con longitud fija, como nanoGPT).
4. `positions.py`: FEN de cada posición de las partidas, normalizado a los cuatro campos de
   `chess-position-evaluations`, y cruce con las evaluaciones (DuckDB join) → `cp`/`mate` por posición.
5. `puzzles.py`: splits de puzles y conversión a (FEN, secuencia de jugadas UCI correcta).
6. `pairs.py`: pares DPO desde evaluaciones (mejor jugada del PV contra una jugada legal peor con
   diferencia ≥ 100 cp; se equilibra por fase de la partida).
7. `publish.py`: subida a `chorcat/` con `huggingface_hub`, dataset cards desde plantillas Jinja.

Herramientas: `python-chess` 1.999 (GPL-3, solo se usa como librería en Python; no se redistribuye),
`duckdb` 1.5.5, `polars` 1.44, `pyarrow` 25, `zstandard` 0.25, `datasets` 5.0.1, `huggingface_hub`.

## Stockfish

Binario oficial de `stockfishchess.org/download` (GPL-3, Windows AVX2), usado por UCI desde
`python-chess` (`chess.engine`). Usos: estimar Elo del modelo jugando contra Stockfish con `UCI_LimitStrength`
y `UCI_Elo` (1320-3190) o con `Skill Level`; recompensas en GRPO (evaluación a profundidad baja, 8-12,
en ~5 ms por posición); verificación de pares DPO. Las 395 M de evaluaciones públicas cubren la mayor
parte de las posiciones de apertura y medio juego frecuentes, así que Stockfish en vivo se reserva para
posiciones no cubiertas y para partidas de evaluación.

## Volumen en disco (estimado)

| Ítem | GB |
|---|---|
| Parquet filtrado (2 meses) | 2-3 |
| Tokens empaquetados (3 tokenizaciones) | 3-5 |
| Evaluaciones (solo las 4-6 particiones necesarias, o join remoto) | 8-12 |
| Puzles | 1 |
| Checkpoints y runs de MLflow | 10-20 |
| Total a reservar en `E:` | ~40 |
