# Retomar Rukh desde P4

Pega el bloque de `/goal` de abajo en una sesión nueva de Claude Code abierta en
`E:\work\ai\chess-lm`. Antes, haz lo que hay en "Antes de retomar".

---

## Estado al parar (2026-09-19, 12:30)

P0, P1, P2 y P3 están **construidos, revisados, corregidos y verificados en hardware real**. Los
tres repos están en la rama `p3-encoder`, sin subir a ningún sitio y sin ficheros sueltos:

| Repo | Rama | Commits sobre `main` |
|---|---|---|
| `rukh` | `p3-encoder` | 70 |
| `rukh-lab` | `p3-encoder` | 16 |
| `rukh-web` | `p3-encoder` | 11 |

Las ramas por hito siguen existiendo (`p0-scaffold`, `p1-datos`, `p2-decoder`, `p3-encoder`), cada
una encadenada sobre la anterior. Ninguna se ha mergeado a `main`.

### Lo que está medido (no estimado)

| Etapa | Parámetros | Legales sin máscara | Top-1 | Puzles | Elo (IC 95 %) |
|---|---|---|---|---|---|
| `tiny` | 5 309 952 | 94,5 % | 40,3 % | — | 495 (320-709) |
| `small` (T=0,6) | 38 971 392 | 99,4 % | 51,1 % | — | 1181 (1076-1249) |
| `small` (determinista) | 38 971 392 | **99,4 %** | 51,1 % | 22,1 % | **1359 (1293-1429)** |
| `medium` (determinista) | 115 120 128 | 99,4 % | 52,9 % | 23,9 % | 1422 (1353-1483) |
| **`medium-v4` + DPO** | 115 120 128 | **99,8 %** | 53,4 % | **38,8 %** | **1529 (1470-1583)** |

Los Elo de esta tabla **no son los que se publicaron el 2026-09-19**: aquellos (64 / 785 / 1007 /
1091) salían de un ajuste apoyado en cuatro rivales con el Elo escrito a mano y equivocado entre
428 y 581 puntos (D-070). Son las mismas partidas con la escalera medida.

| Encoder | F1 de error (ajustado) | Margen sobre heurística | ROC AUC | Spearman valor↔cp |
|---|---|---|---|---|
| `moves` (preentrenado con MMM) | 0,179 | **+9,0** | 0,738 | 0,407 |
| `squares` (desde cero) | 0,146 | +5,7 | 0,696 | 0,422 (Pearson 0,688) |

**Criterios de aceptación:** legalidad ≥ 99 % **cumplido** (99,8 %); Elo ≥ 1200 **cumplido**
(1529, IC 1470-1583; y ya lo estaba con los 1359 de `small`);
paridad ONNX ≥ 99,9 % **no cumplido** en el decoder (fp16 99,80 %, int8 95,40 %) y **cumplido** en
el encoder (100 % en las tres precisiones); F1 de error ≥ heurística + 5 **cumplido** (+9,0);
correlación de valor ≥ 0,8 **no cumplido** (0,42). Todo está documentado tal cual en
`rukh/docs/decisiones-de-ejecucion.md` (D-001 a D-057).

### Datos y artefactos en disco (`rukh/`, todo fuera de git)

- `data/raw` 4,6 GB · `data/uci` 922 MB (5 896 388 partidas, 0 ilegales) · `data/tokens` 6,9 GB
  (UCI, SAN y BPE empaquetados) · `data/positions` 330 MB (5 M posiciones) · `data/evals` 85 MB
  (488 159 posiciones con evaluación) · `data/puzzles` 22 MB (156 000 con partida real) ·
  `data/elo-bins` 63 MB · `data/elite` (541 085 partidas) · `data/pairs` (13 887 pares DPO).
- `checkpoints/`: `tiny`, `small`, `medium`, `encoder-mmm`, y las cabezas en `probe`, `full`,
  `last-n` y los cuatro puntos de la curva.
- `artifacts/onnx/{small,encoder}` con fp32, fp16 e int8; `artifacts/embeddings/positions.npy`
  (488 159 × 384) para la fase 2; `artifacts/tokenizer/` y `artifacts/web/` sincronizados a las webs.
- Las cuatro lecciones (M0-M3) están en `vigente`, con **cero** salidas inventadas.

## Antes de retomar (esto lo haces tú)

1. **Crear los tres repos en GitHub** `borja-glez/{rukh,rukh-lab,rukh-web}` y empujar `main` y las
   cuatro ramas. `gh` está autenticado como `borja-glez`, así que también puede hacerlo el agente
   si se lo pides.
2. **Decidir los merges** de `p0-scaffold`, `p1-datos`, `p2-decoder` y `p3-encoder`. Van
   encadenadas: mergear `p3-encoder` las lleva todas.
3. **DNS y Dokploy**: `rukh.borjaglez.com` (desde `rukh-web`) y `lab.rukh.borjaglez.com` (desde
   `rukh-lab`), Dockerfile en la raíz, contexto `.`, auto-deploy en `main`.
4. **`HF_TOKEN`** con permiso de escritura en `chorcat`: sin él no se publica nada. Hay siete
   artefactos listos y validados en seco (`rukh data publish --list`).
5. **Decidir qué modelo sirve la demo en móvil.** El de 8 bits pesa 43,5 MB pero cambia la jugada
   en el 4,6 % de las posiciones; el de media precisión pesa 78,8 MB y se queda a una décima del
   listón. Hoy la demo no debe publicarse con el de 8 bits por defecto (D-049).
6. **Probar la demo en tu móvil** cuando el dominio esté sirviendo (punto de intervención de P2).

---

## El `/goal` para pegar

```text
/goal Retomo el proyecto Rukh en E:\work\ai\chess-lm desde el hito P4. Lee primero
E:\work\ai\chess-lm\RETOMAR-P4.md (estado al parar y lo que ya está medido), luego GOAL.md,
CLAUDE.md y rukh/docs/decisiones-de-ejecucion.md completo (D-001 a D-057: ahí están todas las
decisiones y desviaciones, incluidas las que contradicen el spec). Los planes de los hitos
anteriores están en rukh/docs/plans/. El diseño vinculante sigue siendo rukh/docs/spec/00 a 09.

Los tres repos están en la rama p3-encoder con P0, P1, P2 y P3 hechos, revisados y verificados en
hardware real; no hay nada mergeado a main ni publicado en Hugging Face. Empieza confirmando
conmigo qué he hecho de los puntos pendientes (repos en GitHub, merges, DNS y Dokploy, HF_TOKEN,
decision sobre el modelo de movil) antes de dar por hecho que estan resueltos.

Ejecuta P4, P5 y P6 y despues A1 a A6, en orden, cumpliendo los criterios de aceptacion de cada
uno. Para cada hito: plan escrito en rukh/docs/plans/, rama pN-<slug> desde la anterior, tareas
con tests, revision de la rama, una sola ola de correcciones y parada para que yo decida el merge.
Solo parate en los puntos marcados "Borja interviene" en GOAL.md.

Verifica de verdad antes de dar nada por hecho: comandos con su salida, entrenamientos en la
5090, partidas en Chrome real con WebGPU contra el contenedor de produccion, Docker con sus
cabeceras. Nada de salidas inventadas en las lecciones: donde vaya una salida real, se ejecuta y
se pega. Registra cada decision y cada desviacion en rukh/docs/decisiones-de-ejecucion.md
continuando la numeracion en D-058. Commits en ingles siguiendo conventional commits, sin
coautoria ni referencias a herramientas ni a la sesion.

Lo que mas me importa sigue siendo el curso: tiene que servirme para aprender IA generativa y
agentica de cero a experto, con las partes agenticas (RAG, LangGraph, LangSmith, MCP, multiagente)
igual de cuidadas que las de modelado. Cada hito publica la leccion de su modulo con numeros
medidos, no estimados, y dice claramente que criterios cumple y cuales no.

Cosas que arrastra el proyecto y que P4 debe tener en cuenta: el decoder se queda en 1007 Elo
frente al liston de 1200 y esta medido que el cuello son los datos y no la capacidad (D-054), asi
que valora traer mas meses de partidas antes de gastar GPU en modelos mayores; la temperatura de
muestreo vale mas de 200 puntos de Elo (D-047), asi que toda comparacion entre etapas tiene que
fijar el muestreo; y la cuantizacion int8 del decoder cambia la jugada en el 4,6 % de las
posiciones (D-049), lo que afecta a que modelo sirve la demo.
```
