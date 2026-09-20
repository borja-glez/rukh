# P5 · Alineamiento (M5) · Plan de implementación

**Objetivo:** que el modelo deje de imitar y empiece a **preferir**. Un modelo de recompensa
entrenado con Bradley-Terry, pares de preferencia generados por el propio modelo, DPO sobre ellos,
recompensas verificables con tests, GRPO, y la lección M5 con su galería de *reward hacking*.

**Spec:** `docs/spec/02-modelos-y-entrenamiento.md` (Componente 4), `docs/spec/03-curriculo.md`
(M5), `GOAL.md` (P5). Plan previo: `2026-09-20-p4-finetuning.md`.

**Criterios de aceptación (`GOAL.md`):**

1. **DPO o GRPO ≥ +50 Elo (con intervalo) sobre su base.**
2. **RM ≥ 75 %** de acierto en pares reservados.
3. Publicados con card: `chorcat/rukh-rm`, `-dpo` y `-grpo`.

---

## Cómo quedaron los criterios (al cerrar el hito)

1. **+50 Elo con intervalo:** cumplido en la estimación puntual por los tres métodos y por el
   camino que no se había probado nunca en el proyecto — enfrentamiento directo, dos direcciones,
   agrupadas. DPO off-policy **+67** (IC 45-89), DPO on-policy **+57** (IC 36-79), GRPO **+43**
   (IC 13-72). Los tres separados del cero. En la lectura estricta —extremo inferior por encima de
   50— se queda en 45, 36 y 13, y eso se dice con todas las letras.
2. **RM ≥ 75 %:** rozado. Cinco corridas dan 72,91 / 74,21 / 74,71 / 74,76 / 75,19 %. Y **dos de
   ellas son la misma semilla**, así que 1,3 puntos de esa dispersión no son la partición: son el
   suelo de reproducibilidad de la propia medición (D-113). El criterio está por debajo del ruido
   del instrumento que lo mide.
3. **Publicados con card:** sí, con la desviación de nombre de D-122.

Lo que el hito enseñó **además** de lo que pedía: que restar dos absolutos arrastra los dos ruidos
(D-110), que el triángulo de tres modelos no cierra (D-120), que alinear cuesta legalidad (D-121),
y que la recompensa que GRPO optimiza se maximiza colapsando la política (D-123 a D-125).

---

## El hallazgo que ordena el hito: la resta de dos ruidos

P4 terminó midiendo su propio instrumento y el número da miedo: **dos tiradas idénticas de la
escalera, mismo modelo y misma semilla, dieron 1498 y 1558** (D-107). El suelo de reproducibilidad
es de unos 40 Elo de una sigma.

El criterio 1 de este hito pide **+50 Elo con intervalo**. Si se mide como se ha medido hasta
ahora —correr la escalera sobre la base, correrla sobre el modelo alineado y restar— hace falta
esto:

| Instrumento | Qué mide | Partidas para afirmar +50 Elo |
|---|---|---:|
| Escalera, dos tiradas y restar | dos fuerzas absolutas | **757 por lado** (1 514 en total) |
| **Enfrentamiento directo** | la **diferencia**, una sola vez | **185 en total** |

**Ocho veces más barato**, y sin el suelo de reproducibilidad de la escalera, porque los dos
modelos juegan la misma partida y no hay un tercero cuyo humor haya que promediar.

La aritmética es la misma de `labs/m4/games_needed.py` aplicada a dos poblaciones distintas. Una
ventaja de 50 Elo es una tasa esperada de **0,5715** en un enfrentamiento directo, y para separar
0,5715 de 0,5 al 95 % bastan 185 partidas. Contra la escalera, 50 Elo son 0,0712 de tasa sobre una
base de 0,45, y separar eso pide 757 partidas **por condición**.

**La regla que ordena el hito: para medir una diferencia, mide la diferencia.** No midas dos
absolutos y los restes. Es la continuación natural de lo que P4 aprendió midiendo su propio ruido,
y es la primera cosa que se construye aquí.

### Lo que esto NO cambia

La escalera se queda **tal cual**. Sigue dando el Elo absoluto de cada etapa, que es lo que hace
comparables las once filas de `docs/benchmarks.md` y las cards ya publicadas. Cambiar el límite de
Stockfish de tiempo a nodos —que arreglaría la reproducibilidad, backlog de P6— invalidaría todos
los Elo publicados y obligaría a recalibrar los ocho peldaños. No en este hito.

---

## Lo que ya existe y no hay que rehacer

| Pieza | Estado |
|---|---|
| `rukh.train.dpo` | **funciona**: `DpoConfig` (β, lr, `nll_weight`), pérdida DPO sobre un token, validación |
| `data/pairs/dpo-prompts.parquet` | 13 838 pares **fuera de política**, equilibrados por fase (4 629 cada una), `min_delta_cp: 100` |
| `chorcat/rukh-medium-dpo` | publicado: 1529 (1470-1583), **38,8 % de puzles**, el mejor de la familia |
| `PositionEncoder` + `MultiHead` | de M3, la base del modelo de recompensa |
| `RukhForCausalLM` | de M4: `RewardTrainer`, `DPOTrainer` y `GRPOTrainer` funcionan sin escribir ninguno |
| `Player` / `play_game_with` | de M4: dos decoders pueden enfrentarse casi gratis |
| Las cuatro herramientas de medición de P4 | partidas necesarias, corrida de control, suelo de reproducibilidad, rango del instrumento |

**Lo medido hasta ahora con DPO, para tenerlo delante:** sobre `small` v3, 1425 → 1460 (+35);
sobre `medium-v4`, 1504 → 1529 (+25). Los dos con los intervalos solapados. Es decir, **el efecto
que hay que superar o medir mejor es de ese orden**, y el criterio pide 50.

---

## Track A · El instrumento antes que el experimento

### Tarea A1 · Enfrentamiento directo entre dos modelos

**Ficheros:** `src/rukh/eval/match.py`, `src/rukh/cli.py`, `tests/unit/test_match.py`.

`play_game_with(player, opponent, ...)` ya acepta cualquier `Opponent` con `choose(board)`. Basta
envolver un `DecoderPlayer` como oponente y tener dos modelos jugando.

- Colores alternos y **mismas aperturas** para los dos lados: cada apertura se juega dos veces, una
  con cada color, para que un repertorio afortunado no decida el resultado.
- Salida: tasa de puntos, diferencia de Elo `400·log10(p/(1-p))` y su intervalo por bootstrap.
- [ ] Test: con el mismo modelo en los dos lados la diferencia medida incluye el cero.
- [ ] Test: las aperturas se reparten en pares espejo y ningún par queda huérfano.
- [ ] `rukh eval match --a <ckpt> --b <ckpt> --games N`.

### Tarea A2 · La aritmética, como lab y como guardia

**Ficheros:** `labs/m5/games_needed_match.py`.

El hermano de `labs/m4/games_needed.py` para esta población: dada una ventaja en Elo, cuántas
partidas de enfrentamiento hacen falta; y al revés, qué ventaja detecta una tirada de N partidas.

- [ ] Se corre **antes** de cada medición, no después, y su número entra en el plan de la tirada.

### Tarea A3 · El control, desde el minuto uno

P4 aprendió que una métrica que sube con el tratamiento hay que medirla también sin él.

- [ ] Enfrentamiento **base contra base** (mismo checkpoint, semillas distintas) como cero medido
      del instrumento. Si eso no sale centrado en cero, nada de lo demás vale.

---

## Track B · Modelo de recompensa

### Tarea B1 · Cabeza escalar y pérdida de Bradley-Terry

**Ficheros:** `src/rukh/models/reward.py`, `src/rukh/train/reward.py`, tests.

`PositionEncoder` de M3 con una cabeza escalar sobre el vector agrupado. La pérdida es
`-log σ(r(mejor) - r(peor))`: no hay etiqueta absoluta, solo el orden.

- [ ] Test: la pérdida es invariante a sumar una constante a las dos recompensas (Bradley-Terry
      solo determina diferencias, y un RM que dependa del cero está roto).
- [ ] Test: con pares perfectamente separables la exactitud llega a 1 y la pérdida a 0.
- [ ] **Criterio 2**: exactitud ≥ 75 % en pares reservados, partidos **por partida** y no por par,
      con la misma función pura de CRC-32 que M3 (dos pares de la misma posición son casi el mismo
      par).
- [ ] Correlación de la recompensa con `cp`, Spearman además de Pearson, por la lección de M3.

### Tarea B2 · Qué NO demuestra el RM

- [ ] La lección dice en voz alta que DPO **no necesita** un RM —su referencia es implícita— y que
      el RM está aquí porque es la pieza que hace falta para entender PPO y porque es medible por
      sí sola. Publicarlo sin esa frase sería sugerir que DPO lo usa.

---

## Track C · Pares on-policy y DPO

### Tarea C1 · Pares generados por el propio modelo

**Ficheros:** `src/rukh/data/onpolicy.py`, `configs/data/pairs-onpolicy.yaml`.

Los 13 838 pares que hay son **fuera de política**: salen de evaluaciones de Lichess, no del
modelo. Un par on-policy se construye muestreando **4 jugadas del modelo** en una posición,
puntuándolas con Stockfish y quedándose con la mejor y la peor.

- [ ] La diferencia importa y hay que decirla: fuera de política enseña «esta jugada es mejor que
      esta otra»; on-policy enseña «de lo que **tú** ibas a jugar, esto era mejor que aquello». La
      segunda ataca los errores que el modelo comete de verdad.
- [ ] Test: las cuatro jugadas salen del muestreo del modelo y son legales; el par se descarta si
      las cuatro coinciden o si la diferencia no llega a `min_delta_cp`.
- [ ] Manifiesto con el predicado, como todo corte de datos del proyecto.

### Tarea C2 · DPO on-policy y la comparación de las dos fuentes

- [ ] **La medición que vale**: enfrentamiento directo `base` vs `DPO-offpolicy` vs
      `DPO-onpolicy`, con las partidas que A2 diga.
- [ ] Se publica la comparación aunque salga que on-policy no mejora: es el resultado.
- [ ] Vigilar la legalidad. D-071 midió que DPO se la come cuando no hay holgura (98,90 % en
      `small`), y en `medium` no. Si baja, se dice y se busca el β.

---

## Track D · Recompensas verificables y GRPO

### Tarea D1 · Las funciones de recompensa, con tests antes que entrenamiento

**Ficheros:** `src/rukh/train/rewards.py`, `tests/unit/test_rewards.py`.

Es el primer trabajo del hito porque es CPU pura y no depende de nada.

- **Legalidad como puerta**: una jugada ilegal vale 0 y no entra en el resto.
- **Δcp normalizado con tope**, respecto de la mejor jugada de la posición.
- **Bonus por mate**, **penalización por repetición**.
- [ ] Cada función con su test de casos conocidos: la mejor jugada saca el máximo, una ilegal saca
      cero, un mate saca el bonus, y una repetición triple penaliza.
- [ ] Test de **saturación**: ninguna función puede dar recompensa infinita ni negativa sin tope,
      porque una recompensa sin tope es una invitación al hacking.

### Tarea D2 · GRPO

**Ficheros:** `src/rukh/train/grpo.py`, `configs/train/grpo-*.yaml`.

G = 8 jugadas por posición, ventaja relativa dentro del grupo, KL contra la referencia. 300 pasos
en `tiny` en vivo, `small` de noche.

- [ ] **Galería de hacking**: se documenta cada atajo que el modelo encuentre y su corrección. Si
      no encuentra ninguno, se dice —y se sospecha del diseño de la recompensa, no del modelo.
- [ ] La misma medición que C2: enfrentamiento directo contra su base.

---

## Track E · Curso · Módulo M5

**Ficheros:** `rukh-lab/src/content/lessons/m5/01-alineamiento.mdx`, `m5.json` (módulo y
cheatsheet), glosario, figuras e islas.

Misma línea que M1-M4: qué vas a construir, el mecanismo antes que la receta, números medidos y
nunca inventados, y los criterios cumplidos y no cumplidos con todas las letras.

Figuras e islas. Las cuatro planeadas se convirtieron en seis, y dos de ellas no estaban previstas
porque la medición las pidió:

- `SubtractingNoise` — **por qué restar dos absolutos es mal instrumento**, que era el hallazgo con
  el que se abrió el hito.
- `GroupBaseline` — lo que planeaba `GrpoGroup`: un grupo con dispersión y uno plano, con la media
  del propio grupo como línea base. En barras y no sobre el tablero, porque lo que hay que ver es
  la **distancia a la media**, y un tablero la esconde.
- `RewardBands` — el acierto del reward model por banda, con la de mate desplomándose y la de diez
  pares dibujada hueca.
- `HackedShape` — la galería, pero midiendo **ventajas** y no coronas: las seis recompensas coronan
  la misma jugada, así que la figura planeada («los atajos encontrados con la posición») no tenía
  nada que enseñar. Lo que enseña es que sin suelo, una jugada ilegal puntúa por encima de una
  legal mala.
- `Intransitive` — **no estaba en el plan**. Las tres aristas de 800 partidas dejan un residuo de
  47 Elo a 2,4 σ, y eso hacía falta dibujarlo.
- `OverOptimised` — lo que planeaba `RewardCurves`, con la forma que salió: la recompensa que no se
  puede falsear sube, se dobla y baja, mientras el indicador de colapso sube monótono.

- [x] Cheatsheet y términos nuevos (Bradley-Terry, DPO, β, GRPO, ventaja, KL, RLVR, reward hacking,
      on-policy vs off-policy, enfrentamiento directo).
- [ ] `m5.json` a `live` con los resultados reales.
- [ ] `pnpm check`, `test`, `build`, E2E y Lighthouse verdes en las dos webs.

---

## Track F · Publicación y registro

- [ ] `chorcat/rukh-rm`, `chorcat/rukh-medium-dpo-onpolicy` y `chorcat/rukh-medium-grpo`.
- [ ] **Desviación a registrar:** `GOAL.md` nombra `rukh-small-dpo` y `-small-grpo`. Se trabaja
      sobre `medium-v4` por lo mismo que en P4 (D-105): es el modelo que la demo sirve y el que
      tiene margen. Si `medium` resulta demasiado lento para GRPO, se cae a `small` y se dice.
- [x] Ledger desde **D-110** (llega hasta D-125).
- [ ] `docs/benchmarks.md` regenerado con `rukh eval benchmarks`.
- [x] Runbook de alineamiento, como el de P4.

---

## Orden de ejecución y descansos

| # | Trabajo | Recurso | Descanso después |
|---|---|---|---|
| 1 | D1: funciones de recompensa con tests | CPU ligera | — |
| 2 | A1, A2: enfrentamiento directo y su aritmética | CPU ligera | — |
| 3 | A3: control base contra base | CPU + Stockfish ~30 min | 15 min |
| 4 | B1: modelo de recompensa | GPU ~20 min | 15 min |
| 5 | C1: pares on-policy | GPU + Stockfish ~1 h | 20 min |
| 6 | C2: DPO on-policy + enfrentamiento | GPU ~20 min + medición | 20 min |
| 7 | D2: GRPO | GPU 1-2 h | 20 min |
| 8 | Medición final y escalera de las etapas nuevas | GPU + Stockfish | 20 min |
| 9 | Exportación, paridad y publicación | CPU | — |
| 10 | Demo y lección | CPU | — |

**Regla que no se salta:** ningún trabajo pesado de CPU mientras corra una partida contra
Stockfish. Y 15-20 minutos de descanso entre trabajos largos.

## Qué necesito de Borja

- Nada para empezar.
- Al final: probar la demo y decidir si GRPO merece una etapa propia en ella.
