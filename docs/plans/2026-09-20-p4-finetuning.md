# P4 · Fine-tuning e instrucción (M4) · Plan de implementación

**Objetivo:** convertir el decoder en una familia de modelos afinados. Tres afinados completos
(maestros y condicionado por Elo), LoRA escrito a mano y con `peft` sobre una envoltura HF del
decoder, QLoRA de Qwen3 sobre PGN como comparación honesta con un LLM general, el selector de Elo
y los adaptadores vivos en la demo, y la lección M4.

**Spec:** `docs/spec/02-modelos-y-entrenamiento.md` (Componente 3), `docs/spec/03-curriculo.md`
(M4), `docs/spec/05-web-demo.md`, `GOAL.md` (P4). Planes previos: `2026-09-19-p2-decoder.md`,
`2026-09-19-p3-encoder.md`, `2026-09-19-elo-1200.md`.

**Criterios de aceptación (GOAL.md):**

1. Elo por condición **monótono** (1500 < 2000 < 2400, con intervalo de confianza).
2. Tabla comparativa con Qwen afinado con QLoRA sobre el mismo recorte.
3. Publicados con card: el modelo de maestros, el condicionado por Elo, los adaptadores
   `rukh-lora-*` y `rukh-qwen3-pgn-qlora`.
4. En la demo: selector "juega como 1500/2000/2400" y adaptadores intercambiables.

## Estado · 2026-09-20 18:50

Se actualiza a medida que caen las mediciones. Lo que está **medido** lleva el número; lo que está
en marcha lleva la hora prevista.

### Criterios de aceptación

| # | Criterio | Estado |
|---|---|---|
| 1 | Elo por condición monótono (1500 < 2000 < 2400 con IC) | ❌ **no se cumple**, y medido por qué (D-100) |
| 2 | Tabla comparativa con Qwen | ⏳ Qwen entrenando (arrancó 18:15) |
| 3 | Publicados con card los cinco artefactos | ⏳ código listo, exportación pendiente |
| 4 | Demo: selector de Elo + adaptadores intercambiables | ⏳ código listo, faltan los ficheros |

### Criterio 1, con detalle, porque es el que no sale

Seis condiciones, 160 partidas cada una: 1425 · 1549 · 1498 · 1538 · 1606 · 1644 Elo. Ni monótono
ni separado. La aritmética de `labs/m4/games_needed.py` dice que entre `<w1500>` y `<w2000>` harían
falta **24 420 partidas por condición** (sesenta horas de Stockfish) para separar una diferencia de
−0,013 en tasa de puntos: no es que falten partidas, es que la diferencia no está. Entre los
extremos (`<w1200>` y `<w2400>`) **sí** están separados, con 64 partidas habría bastado.

Lo que sí mueve la cabecera, medido en la misma tirada:

- **entropía analítica de la primera jugada, monótona 6 de 6**: 1,596 → 1,647 → 1,741 → 1,841 →
  1,881 → 1,992 bits, sin jugar una sola partida y sin ruido de muestreo;
- **top-1 con máximo en `<w1800>`** (51,4 → 54,1 → 53,3 %), que es donde está la media de Elo de
  las posiciones de validación: la cabecera que mejor predice es la que describe a quien jugó;
- **legalidad que baja al subir la cabecera** (100,00 → 99,70 %), el mismo signo que la entropía;
- **puzles planos** (37,0 – 38,2 %): la táctica no se mueve.

Conclusión del hito: **la cabecera mueve el estilo, no la fuerza táctica.** Está en la lección con
la tabla delante y en el ledger como D-100 y D-101.

### Qué queda, en orden

| # | Trabajo | Recurso | Estado |
|---|---|---|---|
| 1 | Barrido de 6 condiciones | GPU + Stockfish | ✅ 18:40 |
| 2 | Control: el modelo base en los dos extremos del eje | GPU + Stockfish | ⏳ ~19:20 |
| 3 | Evaluación canónica de `medium-elo` | GPU + Stockfish | ⏳ ~19:45 |
| 4 | Qwen3 QLoRA | GPU | ⏳ arrancó 18:15 |
| 5 | Descanso | — | 15-20 min |
| 6 | Maestros: escalera completa + reevaluación de las etapas publicadas | GPU + Stockfish | pendiente |
| 7 | `rukh eval qwen` por el mismo harness | GPU + Stockfish | pendiente |
| 8 | Reintento de `<w1500>` y `<w2400>` con 400 partidas | GPU + Stockfish | pendiente |
| 9 | Exportación ONNX (incluida la adaptable), paridad y publicación | CPU | pendiente |
| 10 | Demo, lección, cheatsheet y glosario | CPU | en marcha |

**Regla que se está siguiendo:** nada de trabajo pesado de CPU mientras corre una escalera. El
rival tiene 0,1 s por jugada, así que robarle CPU lo debilita y el Elo del modelo sale inflado. Las
exportaciones esperan a que la escalera esté parada.

### Ya hecho y verificado

- Corpus del tramo bajo descargado y equilibrado; `medium-elo` afinado; los doce tokens de cabecera
  muertos, entrenados.
- LoRA a mano, idéntica a `peft` paso a paso (`|delta| = 0,00e+00` en seis pasos).
- Dos adaptadores de estilo de **1,6 MB**: 99,85 % de `1. e4` y 99,88 % de `1. d4`, **sin coste**
  medible en legalidad, top-1 ni puzles.
- **Adaptadores como entradas del grafo ONNX** (D-102): con ceros el fichero es el modelo base
  exactamente; con un adaptador de 1,6 MB es ese estilo. El navegador cambia de estilo sin
  descargar otro modelo de 221 MB. Diez pruebas nuevas, incluida la ida y vuelta completa del
  fichero que descarga la página.
- Demo: selector de estilo y de Elo cableados de punta a punta (registro, protocolo, worker, panel);
  158 pruebas en verde.
- 699 pruebas de `rukh` en verde, `ruff` limpio, `pnpm check` y `pnpm test` limpios.

---

## Restricciones globales

- Las de P0-P3: commits en inglés con Conventional Commits, código y docstrings en inglés, docs en
  español, pydantic `extra="forbid"`, `pytest -m unit` en CPU y sin red en < 3 min, nada de pesos
  en git.
- Rama `p4-finetuning` desde `main` en los tres repos (P0-P3 ya están mergeados y desplegados).
- **El vocabulario no cambia.** Los 27 tokens de Elo por bando ya existen desde P1
  (`<w0600>`…`<w3200>`); afinar solo entrena los que hasta ahora estaban muertos.
- Descansos de 15-20 min entre trabajos largos de GPU o de CPU, como en P2 y P3.

---

## El hallazgo que ordena el hito

`configs/data/lichess-2025-01-02.yaml` fija `min_elo: 1800` y ese filtro se aplica a **los dos
jugadores**. El corpus entero del proyecto —los 19 M de partidas de `tokens-v4`, la Elite DB
incluida— vive entre 1800 y ~3100. `data/elo-bins/manifest.json` lo dice sin ambigüedad: su bin más
bajo es **1800**.

Consecuencia: los tokens `<w1000>`…`<w1700>` **nunca han recibido un gradiente**. Su embedding es el
de la inicialización. Pedirle al modelo publicado que "juegue como 1500" no le pide nada: le pone
delante un vector aleatorio.

Esto reinterpreta D-069 ("el condicionamiento alto no da Elo, aunque el eje funcione"). La conclusión
de aquel día —imitar a un 2400 no gana partidas sin búsqueda táctica— sigue en pie para el extremo
alto. Pero el extremo bajo, que es el que el criterio de P4 necesita y el que un jugador humano
querría de verdad, **no se había probado nunca porque no había datos**.

Por eso P4 empieza en los datos y no en el modelo. Sin partidas por debajo de 1800 el criterio 1 es
inalcanzable, y con ellas se vuelve la funcionalidad más útil de la demo: un rival que baja a tu
nivel en vez de aplastarte a 1500 Elo reales.

---

## Track A · El eje de Elo se vuelve real

### Tarea A1 · `max_elo` en el filtro de descarga

**Ficheros:** `src/rukh/data/fetch.py`, `configs/data/lichess-low.yaml`,
`tests/unit/test_fetch.py`.

`FetchConfig` gana `max_elo: int | None = None` y el SQL añade
`WhiteElo <= {max_elo} AND BlackElo <= {max_elo}` cuando está. El manifiesto lo registra como un
filtro más, así que un corpus mixto queda trazable.

- [ ] Test unitario sobre el constructor de SQL: con `max_elo` aparecen las dos condiciones, sin él
      ninguna; el manifiesto las recoge.

### Tarea A2 · Descargar y convertir el tramo bajo

**Salidas:** `data/raw-low/`, `data/uci-low/`.

`configs/data/lichess-low.yaml`: meses 2025-01 y 2025-02, `min_elo: 1000`, `max_elo: 1799`, el resto
igual que el corpus original (≥ 180 s de base, terminaciones normales, sin variantes, ≥ 20 plies).
Límite 3 000 000 de partidas. Después `rukh data uci` con `uci_dir: data/uci-low`.

Coste estimado: descarga 1-2 h (red), conversión 30-60 min (CPU, todos los núcleos). **Descanso de
20 min después.**

- [ ] Verificación: `manifest.json` con los conteos por mes; histograma del Elo medio por bins de
      100 que cubra 1000-1799 sin huecos.

### Tarea A3 · Corpus balanceado por Elo

**Ficheros:** `src/rukh/data/elo_bins.py` (varias fuentes y ancho de banda configurable),
`configs/data/pipeline-elo.yaml`, `tests/unit/test_elo_bins.py`.

`EloBinsConfig` gana `sources: list[str]` (en vez de un único `uci_dir`), `bin_width: int = 100` y
`min_bin`/`max_bin`. Se genera `data/elo-bins-v2/games.parquet` con **N partidas por banda de 200
Elo entre 1000 y 2600**, plano por construcción. Los bins de arriba tienen menos material
(2600: 17 522; 2800: 2 316), así que N se fija a lo que el bin más pobre permita sin vaciar los
demás; el manifiesto publica el reparto real y la lección enseña ese límite en vez de esconderlo.

Luego `rukh data tokenize` con `out_dir: data/tokens-elo` sobre ese parquet.

- [ ] Test unitario del SQL de bandas con un parquet sintético de tres bins.
- [ ] Verificación: conteos por banda en el manifiesto; el `.npy` empaquetado contiene los ids de
      `<w1000>`…`<w2600>` en proporciones parecidas (se comprueba contando tokens).

### Tarea A4 · Afinado: `medium-elo`

**Ficheros:** `src/rukh/train/common.py` (`init_from`), `configs/train/medium-elo.yaml`,
`tests/unit/test_train_init_from.py`.

Hoy solo existe `--resume`, que continúa una corrida (optimizador, paso, run de MLflow). Un afinado
es otra cosa: **pesos sí, estado no**. `RunConfig` gana `init_from: str | None`, que carga los pesos
y arranca un run nuevo con su propio calendario.

Receta: `lr` 1e-4 (spec, Componente 3), warmup 200, coseno hasta 0,1, 6 000 pasos ≈ 300 M tokens,
`batch` efectivo 256. ~30 min de GPU. **Descanso de 15 min.**

- [ ] Test unitario: `init_from` carga los pesos y deja el optimizador a cero; `init_from` +
      `resume` a la vez es un error de configuración.

### Tarea A5 · Medición por condición y criterio 1

**Ficheros:** `src/rukh/eval/sweep.py`, `src/rukh/cli.py` (`rukh eval sweep`),
`configs/eval/greedy-elo.yaml`, `tests/unit/test_sweep.py`.

`rukh eval sweep --model <ckpt> --elos 1200,1500,1800,2100,2400` corre la suite determinista una vez
por condición (misma temperatura, mismo `seed`, mismos rivales; solo cambia `header_elo`), escribe
una tabla Markdown, un `results.json` por condición y un
`artifacts/web/elo-conditioning.json` para el curso y la demo.

Se registra por condición: Elo con IC, legalidad sin máscara, top-1 y puzles. La legalidad por
condición es interesante por sí sola: si el modelo "juega mal a propósito", ¿lo hace jugando peor o
jugando ilegal? Es una pregunta que el hito puede responder con un número.

Coste: 5 condiciones × ~25 min ≈ 2 h de GPU + Stockfish. **Descanso de 20 min a mitad.**

- [ ] **Criterio 1**: Elo(1500) < Elo(2000) < Elo(2400). Si los intervalos se solapan, se dice, y se
      repite con más partidas por peldaño (`elo_games: 40`) solo en las tres condiciones del
      criterio antes de declarar nada.
- [ ] Si la monotonía no aparece: se documenta en el ledger con la evidencia y se busca la causa
      (¿pocas pasadas?, ¿el afinado olvidó el extremo alto?, ¿hace falta más peso del token de Elo?).
      No se declara cumplido lo que no lo esté.

### Tarea A6 · Afinado de maestros y diversidad

**Ficheros:** `configs/train/medium-masters.yaml`, `src/rukh/eval/diversity.py`,
`tests/unit/test_diversity.py`.

`medium-v4` afinado solo con la Elite DB (`lr` 1e-4, 4 000 pasos). El spec pide medir dos cosas:
"sube el Elo, baja la diversidad". La segunda métrica (**entropía de aperturas en 200 partidas
propias**) está en la tabla del spec y hoy sale `n/a`; P4 la implementa y la aplica a todas las
etapas comparables.

- [ ] Test unitario de la entropía con distribuciones conocidas (una sola apertura → 0; uniforme
      sobre k → log k).
- [ ] Verificación: tabla `medium-v4` / `medium-masters` / `medium-elo` con Elo, top-1, puzles y
      entropía de aperturas.

---

## Track B · LoRA, a mano y con `peft`

### Tarea B1 · `LoRALinear` escrito desde cero

**Ficheros:** `src/rukh/models/lora.py`, `tests/unit/test_lora.py`.

```python
class LoraConfig(BaseConfig):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    targets: tuple[str, ...] = ("attn.q", "attn.v")

class LoRALinear(nn.Module):        # envuelve un nn.Linear congelado
    def forward(self, x): return self.base(x) + self.scale * self.B(self.A(self.drop(x)))

def apply_lora(model, cfg) -> int          # devuelve parámetros entrenables
def merge_lora(model) -> None              # W <- W + BA, y quita los envoltorios
def save_adapter(model, path) -> None      # safetensors, solo A y B (~2 MB con r=8)
def load_adapter(model, path) -> None
```

- [ ] Tests: con `B` a cero la salida es idéntica al modelo base (inicialización correcta);
      `merge_lora` da exactamente los mismos logits que el modelo sin fundir (tolerancia de
      `float32`); solo A y B tienen `requires_grad`; el adaptador guardado pesa lo que dice la
      cuenta `2 · r · d · n_capas · n_objetivos`; cargar en un modelo limpio reproduce los logits.

### Tarea B2 · Entrenar adaptadores de estilo

**Ficheros:** `src/rukh/train/lora.py`, `src/rukh/cli.py` (`rukh train lora`),
`configs/train/lora-e4.yaml`, `configs/train/lora-d4.yaml`.

El eje de estilo es la **primera jugada de las blancas**, que sale gratis del corpus (`uci` empieza
por `e2e4` o `d2d4`) y además se ve en la demo: eliges el adaptador y el modelo abre distinto. La
columna `eco` da un segundo eje para la lección (un especialista en siciliana, B20-B99) sin coste de
datos.

Base congelada `medium-v4`, `r=8`, `alpha=16`, objetivos `q` y `v`, `lr` 2e-4, 1 500 pasos. ~10 min
por adaptador.

- [ ] Verificación: reparto de la primera jugada en 200 partidas propias antes y después
      (el número que demuestra que el adaptador hace algo), y Elo del modelo + adaptador para
      enseñar cuánto cuesta el estilo en fuerza.

### Tarea B3 · Envoltura HF y `peft` sobre ella

**Ficheros:** `src/rukh/models/hf.py`, `tests/unit/test_hf_wrapper.py`, `pyproject.toml` (extra
`hf`).

```python
class RukhConfig(PretrainedConfig):  model_type = "rukh"
class RukhForCausalLM(PreTrainedModel):   # envuelve MoveDecoder sin copiarlo
    def forward(self, input_ids, labels=None, **kw) -> CausalLMOutput
```

`transformers`, `peft`, `trl` y `accelerate` entran como extra `hf` (no en el núcleo: la CI de CPU
tiene un presupuesto de 3 min). Los tests que los necesitan se saltan si el extra no está.

- [ ] Tests: `save_pretrained`/`from_pretrained` ida y vuelta; los logits de `RukhForCausalLM`
      coinciden con los de `MoveDecoder` sobre la misma entrada; `peft.get_peft_model` engancha LoRA
      y cuenta los mismos parámetros entrenables que `apply_lora` con la misma `r` y los mismos
      objetivos.
- [ ] **La comparación que justifica escribir LoRA a mano:** el mismo adaptador entrenado con
      `apply_lora` y con `peft` sobre los mismos datos, misma semilla y mismo calendario, y las dos
      curvas de pérdida superpuestas. Si divergen, hay un error en la implementación propia y se
      encuentra; si coinciden, la lección puede decir que lo escrito a mano *es* LoRA.

---

## Track C · Qwen3 con QLoRA sobre PGN

### Tarea C1 · Recorte de PGN y afinado

**Ficheros:** `src/rukh/train/qwen.py`, `configs/train/qwen3-pgn.yaml`,
`src/rukh/data/pgn_text.py`.

Muestras de texto plano desde `data/uci` (SAN reconstruido con `python-chess`, que ya es
dependencia): cabecera con los dos Elo y el movetext hasta la jugada N, y la jugada N+1 como
objetivo. Mismo mes de validación que el decoder, así que la comparación es sobre el mismo recorte.

`Qwen/Qwen3-0.6B` con LoRA `r=16` sobre las proyecciones de atención. **Decisión abierta:** 4 bits
con `bitsandbytes` si funciona en sm_120 sin `nvcc`; si no, LoRA en bf16 (0,6 B parámetros caben de
sobra en 32 GB) y se registra la desviación en el ledger, con la nota de que el ahorro de memoria
que da nombre a QLoRA no se demuestra a esta escala.

### Tarea C2 · El mismo harness para los dos modelos

**Ficheros:** `src/rukh/eval/qwen_source.py`.

La comparación solo vale si la métrica es la misma. `MoveSource` gana una implementación que
prompt-ea al LLM y parsea la jugada SAN, y a partir de ahí el harness de P2 mide lo de siempre:
legalidad sin máscara, top-1, puzles y Elo contra la misma escalera.

- [ ] **Criterio 2**: tabla con `medium-v4`, `medium-elo` y `qwen3-pgn-qlora` en las cuatro
      métricas, con los parámetros de cada uno y el coste de entrenamiento. Se espera que el Qwen
      pierda por mucho; si gana, es un resultado aún más interesante y se dice igual.
- [ ] La tasa de **jugadas no parseables** es una métrica propia del Qwen y entra en la tabla: un
      LLM de texto puede contestar algo que no es una jugada, y el decoder no puede.

---

## Track D · Demo

### Tarea D1 · El selector de Elo se enciende

**Ficheros:** `rukh-web/src/lib/registry.ts`, `src/islands/ModelPanel.tsx`, `e2e/`.

El selector ya está en pantalla y desactivado con el texto "Condicionar por Elo llega en M4"
(`ModelPanel.tsx:166`). P4 publica `chorcat/rukh-medium-elo`, añade la etapa con
`eloConditioned: true` y lo enciende. Las opciones del selector son **las condiciones medidas**, no
un rango continuo: se ofrece lo que se ha comprobado que cambia algo.

- [ ] E2E: con la etapa condicionada el selector está activo; con las demás, desactivado y con la
      pista visible. Partida real en Chrome con WebGPU a 1500 y a 2400.

### Tarea D2 · Adaptadores intercambiables

**Decisión técnica pendiente de medida.** Fundir un adaptador y exportar el ONNX entero cuesta
221 MB por estilo en `medium`, que tira por tierra lo que LoRA ahorra. Se intenta primero la vía que
sí lo respeta: exportar el decoder con las matrices A y B como **entradas del grafo** (apiladas en
un tensor `(capas, objetivos, r, d)`), de modo que el adaptador viaje en ~2 MB y el navegador lo
cambie sin descargar otro modelo. Si la exportación con dynamo no lo aguanta, el plan B es una etapa
por estilo sobre `small` (41 MB en int8) y la lección explica el coste con los dos números delante.

- [ ] Paridad del camino elegido: mismas jugadas en 1 000 posiciones entre PyTorch con adaptador y
      ONNX con adaptador.
- [ ] E2E: cambiar de adaptador cambia la primera jugada del modelo.

---

## Track E · Curso · Módulo M4

**Ficheros:** `rukh-lab/src/content/lessons/m4/01-fine-tuning.mdx`,
`src/content/modules/m4.json`, `src/content/cheatsheets/m4.json`,
`src/content/glossary/terms.json`, `src/components/figures/*.astro`, `src/islands/EloDial.tsx`.

Misma línea que M1-M3: qué vas a construir, el mecanismo antes que la receta, números medidos y
nunca inventados, y los criterios cumplidos y no cumplidos dichos con todas las letras.

Figuras SVG animadas nuevas:

- `LoraDecomposition.astro` — `W + BA` con el cuello de botella `r` y el recuento de parámetros a la
  vista: por qué 2 MB pueden mover un modelo de 115 M.
- `EloAxis.astro` — el eje de condicionamiento con el agujero por debajo de 1800 del corpus viejo y
  el eje completo después. Es la figura que explica el hito entero.
- `AdapterSwap.astro` — base congelada, adaptadores que entran y salen.
- `ForgettingCurve.astro` — qué se pierde al afinar (top-1 en el recorte original) frente a qué se
  gana.

Isla nueva `EloDial.tsx`: el dial de condición con el Elo medido y su intervalo, alimentado por
`artifacts/web/elo-conditioning.json`. Es la visualización que hace tangible el criterio 1.

- [ ] Cheatsheet `m4.json` y términos nuevos en el glosario (LoRA, rango, adaptador, SFT,
      instruction tuning, olvido catastrófico, cuantización de 4 bits, QLoRA, entropía de aperturas).
- [ ] `m4.json` pasa a `status: live` con los resultados reales en `outcomes`.
- [ ] `pnpm check`, `pnpm test`, `pnpm build`, e2e y Lighthouse verdes en las dos webs.

---

## Track F · Publicación y registro

- [ ] `chorcat/rukh-medium-masters`, `chorcat/rukh-medium-elo` (pesos + ONNX fp16/int8 + card),
      `chorcat/rukh-lora-e4`, `chorcat/rukh-lora-d4`, `chorcat/rukh-qwen3-pgn-qlora`.
- [ ] **Desviación que hay que registrar:** `GOAL.md` nombra `rukh-small-masters` y
      `rukh-small-elo`. La familia se construye sobre `medium-v4` porque es el modelo que la demo
      sirve y el que tiene margen para que el eje de Elo se note; el nombre cambia con el modelo.
- [ ] `rukh publish adapter` como camino propio: un adaptador no es un modelo (no tiene ONNX ni
      vocabulario propio) y su card tiene que decir sobre qué base se monta.
- [ ] Ledger `docs/decisiones-de-ejecucion.md` desde **D-079**, empezando por el hallazgo del
      corpus 1800+.
- [ ] `docs/benchmarks.md` con la tabla completa del hito.

---

## Orden de ejecución y descansos

| # | Trabajo | Recurso | Descanso después |
|---|---|---|---|
| 1 | A1, B1, B3, A6 (entropía): código y tests | CPU ligera | — |
| 2 | A2: descarga del tramo bajo y conversión UCI | red + CPU | 20 min |
| 3 | A3: corpus balanceado + tokenización | CPU | 15 min |
| 4 | A4: afinado `medium-elo` | GPU ~30 min | 15 min |
| 5 | A5: barrido de condiciones (5 × 25 min) | GPU + Stockfish | 20 min a mitad y al final |
| 6 | A6: afinado de maestros + su evaluación | GPU ~60 min | 20 min |
| 7 | B2: dos adaptadores LoRA + medición | GPU ~30 min | 15 min |
| 8 | C1, C2: Qwen QLoRA y su evaluación | GPU ~90 min | 20 min |
| 9 | Exportación ONNX, paridad y publicación | GPU corta | — |
| 10 | D1, D2, E: demo y lección | CPU | — |

## Qué necesito de Borja

- Nada para empezar: los repos están en `main`, `HF_TOKEN` funciona (`whoami` → `chorcat`),
  Stockfish y la 5090 responden.
- Al final del hito: **decidir el merge** de `p4-finetuning` en los tres repos y **probar la demo**
  con el selector de Elo, a ser posible también en el móvil.
