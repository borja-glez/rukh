# Runbooks · procedimientos operativos

Recetas paso a paso para tareas que se repiten y que no conviene reconstruir de memoria:
lanzar y recoger un entrenamiento largo, regenerar un dataset publicado, exportar y verificar
la paridad ONNX, publicar en Hugging Face, arrancar el bot de Lichess.

## Formato

Un fichero por procedimiento, `<verbo>-<objeto>.md` (por ejemplo `entrenar-small.md`,
`publicar-modelo.md`), con estas secciones:

1. **Cuándo** se ejecuta y qué requisitos previos hay (GPU, tokens, disco).
2. **Comandos** exactos, en orden, con los tiempos esperados.
3. **Verificación**: qué mirar para saber que ha ido bien.
4. **Si falla**: los errores conocidos y su remedio.

## Índice

- [`data-pipeline.md`](data-pipeline.md): el pipeline completo de P1 (`fetch → uci → tokenize →
  positions → evals → puzzles → pairs → elite → elo-bins → publish`), tiempos, salidas y cómo
  reanudar `evals`.
- [`afinado-y-adaptadores.md`](afinado-y-adaptadores.md): la familia de M4 (corpus plano por Elo,
  afinado condicionado, afinado de maestros, adaptadores LoRA), el barrido por condición con su
  corrida de control, la exportación del grafo que acepta adaptadores en caliente y la publicación
  de los cinco artefactos. Incluye las trampas que costaron una medición cada una.

Mientras tanto, el arranque del entorno está en el `README.md` de la raíz (`uv sync`, `rukh info`,
`scripts/get_stockfish.py`, `rukh engine check`, `rukh mlflow ui`).
