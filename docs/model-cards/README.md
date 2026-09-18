# Model cards

Copia versionada de las model cards (y dataset cards) que se publican en Hugging Face bajo
`chorcat/rukh-*`. Las cards se escriben **en inglés** (ver `docs/spec/08`, fila "Idioma") y se
generan desde MLflow con `rukh publish <run>` (P2 en adelante): métricas de la tabla única
(`docs/benchmarks.md`), datos y manifiesto, receta de entrenamiento, limitaciones y enlace a la
demo con `?stage=<id>`. Licencia de los pesos: Apache-2.0.

## Convenciones

- Un fichero por repo del Hub: `rukh-small.md`, `rukh-encoder.md`, `rukh-games-1800.md`…
- El fichero de aquí es la fuente; `rukh publish` lo sube como `README.md` del repo del Hub.
- Cada card enlaza el run de MLflow (id) y el commit que la generó.

## Índice

Todavía no hay cards; las primeras (datasets `rukh-games-1800` y `rukh-tokenizer`) llegan con P1.
