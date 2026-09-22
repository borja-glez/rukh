# Alinear el decoder: preferencias, recompensas verificables y cómo medirlo (P5)

## Cuándo

Al rehacer la familia de M5 —el modelo de recompensa, los dos DPO (fuera y dentro de política) y
el GRPO—, o al construir un conjunto de pares nuevo. También cada vez que haya que comparar dos
modelos entre sí, porque el instrumento del hito sirve para cualquier par de checkpoints.

Requisitos: `uv sync --extra cu128 --extra hf --group dev`, los pares de P1 en
`data/pairs/dpo-prompts.parquet`, Stockfish en `tools/stockfish/` (o `RUKH_STOCKFISH`), y
`HF_TOKEN` solo para `publish`. Todo se ejecuta desde `rukh/`.

**Las tres reglas que no se saltan:**

1. **Nada pesado mientras corre un enfrentamiento o una escalera.** Vale lo mismo que en P4 y por
   la misma razón: el rival juega con un límite de tiempo y robarle CPU cambia el resultado. Con
   una diferencia nueva de este hito —la generación de pares on-policy y GRPO usan el motor por
   **profundidad**, así que ellos sí son reproducibles bajo carga; lo que se degrada es lo que
   compite con ellos, no ellos.
2. **La profundidad del motor se fija y se escribe en el run.** D-118 midió que el valor de una
   recompensa para una jugada fija se mueve al cambiar la profundidad, incluso en la recompensa
   sana. Dos ejecuciones con profundidades distintas no son comparables y nada en el log lo avisa.
3. **Entre trabajos largos, 15-20 minutos de descanso.**

## Comandos, en orden

| # | Paso | Comando | Tiempo | Salida |
|---|---|---|---|---|
| 1 | Cuántas partidas cuesta la medición | `uv run python labs/m5/games_needed_match.py` | segundos | la tabla de precios, en pantalla |
| 2 | Control del instrumento | `uv run rukh eval match --a <base> --b <base> --games 100` | ~1 min | debe salir 0,5000 clavado |
| 3 | Modelo de recompensa | `uv run rukh train reward --config configs/train/rm.yaml` | ~4 min | `checkpoints/rm-*/reward.pt`, `run.json` |
| 4 | Pares dentro de política | `uv run rukh data onpolicy --config configs/data/onpolicy.yaml` | ~35 min | `data/pairs-onpolicy/pairs.parquet`, `manifest.json` |
| 5 | DPO fuera de política | `uv run rukh train dpo --config configs/train/dpo-offpolicy.yaml` | ~8 min | `checkpoints/medium-v4-dpo-offpolicy/dpo.pt` |
| 6 | DPO dentro de política | `uv run rukh train dpo --config configs/train/dpo-onpolicy.yaml` | ~4 min | `checkpoints/medium-v4-dpo-onpolicy/dpo.pt` |
| 7 | La comparación que vale, **en las dos direcciones** | `uv run rukh eval match --a <dpo> --b <base> --games 400 --seed 7` y luego con `--a` y `--b` intercambiados | ~4 min cada una | `artifacts/eval/*/results.json` y `report.md` |
| 7b | Agrupar las dos direcciones | `uv run python labs/m5/pooled_match.py` | segundos | la tabla agrupada y el residuo del triángulo |
| 8 | Galería de reward hacking | `uv run python labs/m5/reward_hacking.py --depth 10` | ~2 min | las tres tablas y la de profundidades |
| 9 | GRPO, **dos tasas** | `uv run rukh train grpo --config configs/train/grpo.yaml` y otra con `--run-name medium-v4-grpo-fast` y `lr` 5e-6 | ~12 min cada una (1 500 pasos) | `checkpoints/medium-v4-grpo*/grpo.pt` |
| 9b | Cuál de las dos, enfrentándolas | `uv run rukh eval match --a <rápida> --b <lenta> --games 400 --seed 7`, y al revés | ~4 min cada una | la única comparación directa entre las dos candidatas |
| 10 | Evaluación canónica de cada etapa | `uv run rukh eval --model <ckpt> --config configs/eval/greedy.yaml --stage <nombre>` | ~26 min | `artifacts/eval/<nombre>/` |
| 11 | Tabla de benchmarks | `uv run rukh eval benchmarks` | segundos | `docs/benchmarks.md` |
| 12 | Exportar | `uv run rukh export --ckpt <ckpt> --out artifacts/onnx/<nombre> --fp16 --int8 --check-parity` | ~12 min cada uno | `model{,-fp16,-int8}.onnx`, `parity.json` |
| 13 | Publicar los modelos | `uv run rukh publish model --ckpt <ckpt> --repo chorcat/rukh-<nombre> --stage <el mismo del paso 10> --onnx artifacts/onnx/<nombre>` | minutos | repo en el Hub |
| 14 | Publicar el reward model | `uv run rukh publish reward --run checkpoints/rm-* --repo chorcat/rukh-rm` | segundos | repo con card, sin ONNX |
| 15 | Publicar el dataset on-policy | `uv run rukh data publish --name rukh-pairs-onpolicy` | minutos | dataset en el Hub |

**Antes de subir nada, mira el tamaño de la carpeta staged.** Un `model.safetensors` de 25 KB donde
debería haber 151 MB es lo que dejó un test cuyo aislamiento no funcionaba (D-129), y `ls -la` es la
única comprobación que lo habría pillado. Cuesta un segundo.

El `--stage` del paso 13 tiene que ser **el mismo** con el que se corrió la evaluación del paso 10,
o la card sale sin números: `read_eval` los busca por nombre de etapa. Y `check_eval_matches`
compara el sha del checkpoint contra el que se midió, así que poner el `--stage` de otro modelo
falla en vez de publicar los números equivocados (D-074).

El paso 2 no es ceremonia. Se corrió el día que se escribió `rukh eval match` y salió **0,975 con
999 jugadas ilegales**, que destapó dos errores reales en `infer/game.py` (D-111). Un instrumento
nuevo se estrena midiendo algo cuya respuesta ya se sabe, y este cuesta un minuto.

## Cómo se lee cada salida

### `rukh eval match`

```
score:    0.5687 (191W 73D 136L)
elo:      +48 (95 % CI 19 to 80)
verdict:  separated from zero
illegal:  441 by A, 692 by B
would need 200 games to call this edge, played 400
```

La línea que decide es la del **intervalo de Elo**, no la del punto. Si toca el cero, la
conclusión es «no se puede distinguir», y decir «+48» sin el intervalo es decir menos de lo que se
midió. El reparto de colores tiene que salir exacto a la mitad: `play_match` se niega a jugar un
número impar de partidas precisamente para que no pueda no salir.

**Y una ejecución sola no basta.** El mismo par medido con los lados intercambiados dio +26 con el
intervalo incluyendo el cero: la estimación se movió 22 puntos —ruido normal con 400 partidas— y el
**veredicto** se dio la vuelta (D-120). Se corren las dos direcciones y se agrupan con
`labs/m5/pooled_match.py`, que además calcula el residuo del triángulo cuando hay tres modelos.

La línea de `illegal` no es decoración: los dos DPO de M5 doblan la tasa de su base (D-121), y
como una propuesta ilegal se rescata con un sorteo enmascarado —que es un sorteo mejor—, esa tasa
es también un posible sesgo. `--mask` enmascara desde el primer sorteo para los dos lados y quita
el camino de rescate; es un control, no un ajuste.

### `rukh train reward`

```
accuracy:   74.76 % on held-out pairs
            77.30 % over the pairs an evaluation can decide
    100-200 cp    556 pairs   77.52 %
    ...
       mate cp    356 pairs   68.26 %
margin vs delta cp: Pearson -0.128, Spearman -0.067
        without mates: Pearson +0.058, Spearman +0.075
```

Las dos primeras líneas y las dos últimas se leen juntas o no se leen. El acierto global está
tirado hacia abajo por la banda de mate, que es un tercio de los pares, y la correlación global
**cambia de signo** al quitarla (D-119). Además, la semilla reparte la partición, así que el
titular depende de cuántos pares de mate le tocaron al conjunto de validación (D-113): comparar
dos ejecuciones con semillas distintas no dice nada del modelo.

### `rukh train grpo`

```
reward:     +0.8657 -> +0.8937 (group best)
            +0.8250 -> +0.8529 (engine best -- this is the score)
flat:       0.385 -> 0.472
illegal:    0.0000 -> 0.0000
kl:         0.12169 against the frozen start
groups:     6,743 useful, 5,257 flat
engine:     <n> analyses, <x> % from cache
```

**La segunda línea es la que puntúa, no la primera.** La primera toma `cp_best` dentro del grupo,
así que se maximiza proponiendo la misma jugada ocho veces: sube cuando el modelo colapsa (D-123).
La segunda toma la evaluación del motor con mejor juego y no se puede falsear así.

`flat` es el detector de colapso: la proporción de grupos de validación donde las ocho candidatas
eran la misma jugada. Si sube mucho mientras sube la recompensa, la recompensa se está comprando con
determinismo (D-124).

`illegal` sale 0,0000 con `restrict_to_legal: true`, que es lo normal: las candidatas se muestrean
entre jugadas legales y la puerta nunca se dispara. Para que enseñe algo hay que usar
`grpo-legality.yaml`.

Y la elección entre dos ejecuciones **no se hace con esta salida**: se hace enfrentándolas (paso 9b).
Restar lo que cada una hizo contra la base es el error que el hito entero desaconseja, y se cometió
una vez antes de darse cuenta (D-126).

## Lo que se aprendió montándolo

- **Para medir una diferencia, mide la diferencia** (D-110). La escalera repite con un suelo de
  unos 40 Elo, así que restar dos absolutos arrastra los dos ruidos. El enfrentamiento directo
  sale 8,1 veces más barato en partidas y separó del cero un +40 que la escalera dejaba solapado.
- **Un `head()` no es una muestra** (D-114). Los pares están equilibrados por fase en bloques y las
  primeras filas son 65 % aperturas frente al 33 % real.
- **Una puerta que lo que separa puede saltarse no es una puerta** (D-115). `ILLEGAL = 0,0` caía
  dentro del intervalo legal, así que una jugada imposible puntuaba por encima de una legal que
  repite.
- **El hackeo de recompensa no está en la cima** (D-116). Seis recompensas, cinco rotas, coronan la
  misma jugada: son todas monótonas en la evaluación. El daño está en la forma del grupo.
- **Una ablación cuyas condiciones difieren en dos cosas no mide ninguna** (D-117).
- **Ninguna recompensa que lea al motor es estable con la profundidad** (D-118). La sana se mueve
  0,99 entre profundidad 4 y 14; la que no tiene suelo, 47,0.
- **El triángulo no cierra** (D-120). Las tres aristas agrupadas a 800 partidas dejan un residuo de
  45 Elo con error típico 18, o sea 2,6 σ, que es lo que `labs/m5/pooled_match.py` imprime hoy
  sobre los artefactos publicados (D-120 lo registró como 47 a 2,4 σ el día que se midió; la
  conclusión es la misma y el registro no se reescribe): la fuerza no es un solo número por modelo
  cuando los emparejamientos interactúan.
  Y un veredicto binario leído del borde de un intervalo se da la vuelta con el ruido normal.
- **Alinear cuesta legalidad** (D-121). Los dos DPO doblan la tasa de propuestas ilegales de su
  base, y `nll_weight: 0.1` no lo evitó. GRPO paga menos, 1,66×. Mide el coste en la misma ejecución
  que el beneficio.
- **La recompensa que optimizas y la que publicas no son la misma** (D-123 a D-125). La del grupo se
  maximiza colapsando la política; la del motor no. Publica las dos y `flat_share` con ellas.
- **Un criterio que se cumple añadiendo partidas hasta que se cumple no era un criterio** (D-127).
  El extremo inferior quedó en 49,74 con 1 600 partidas, y no se jugaron las 56 que faltaban.
