# Benchmarks · tabla única de resultados

Una fila por etapa del modelo; las columnas son las métricas del harness de evaluación
(`docs/spec/02-modelos-y-entrenamiento.md`, "Componente 5"). La tabla la regenera
`rukh eval nightly` (P6) a partir de los informes JSON de `rukh eval --model <id> --suite full`;
hasta entonces se rellena a mano al cerrar cada hito.

Cómo se mide cada columna:

| Métrica | Cómo |
|---|---|
| Legalidad sin máscara | % de jugadas legales sobre 10 000 posiciones de validación |
| Exactitud de siguiente jugada | Top-1 y top-3 contra la jugada humana en validación (por tramo de Elo) |
| Puzles | % resueltos por tramo de dificultad (1000-1500 / 1500-2000 / 2000+), 2 000 puzles por tramo |
| Elo estimado | 100 partidas contra Stockfish `UCI_Elo` en 1320/1500/1800/2000, ambos colores, con máscara de legalidad; regresión logística con intervalo de confianza |
| Δcp medio | Pérdida media de centipeones por jugada respecto a Stockfish profundidad 12 en 500 posiciones |
| Diversidad | Entropía de aperturas jugadas en 200 partidas propias |

## Resultados

| Etapa | Modelo (Hub) | Legalidad sin máscara (%) | Top-1 (%) | Top-3 (%) | Puzles 1000-1500 (%) | Puzles 1500-2000 (%) | Puzles 2000+ (%) | Elo estimado (IC 95 %) | Δcp medio | Diversidad (bits) | Run MLflow | Fecha |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

Etapas previstas (`docs/spec/02`): base tiny/small/medium, maestros (SFT), Elo-cond, LoRA, RM,
DPO, GRPO, Qwen-QLoRA y las referencias externas Karvonen-50M y Maia.
