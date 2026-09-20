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

| Etapa | Legalidad sin máscara (%) | Top-1 (%) | Top-3 (%) | Puzles 1000-1500 (%) | Puzles 1500-2000 (%) | Puzles 2000+ (%) | Elo estimado (IC 95 %) | Δcp medio | Entropía 1.ª jugada (bits) | Fecha |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| `lora-d4` | 99.8 % | 55.1 % | 83.0 % | 57.4 % | 37.5 % | 16.9 % | n/a | n/a | 0.0160 | 2026-09-20 |
| `lora-e4` | 99.8 % | 54.7 % | 83.3 % | 58.1 % | 38.2 % | 17.2 % | n/a | n/a | 0.0209 | 2026-09-20 |
| `medium-greedy` | 99.4 % | 52.9 % | 80.9 % | 37.6 % | 24.0 % | 10.2 % | 1091 (990-1194) | n/a | n/a | 2026-09-19 |
| `small` | 99.4 % | 51.1 % | 79.4 % | n/a | n/a | n/a | 785 (680-896) | n/a | n/a | 2026-09-19 |
| `small-greedy` | 99.4 % | 51.1 % | 79.4 % | 34.7 % | 21.1 % | 10.3 % | 1007 (920-1101) | n/a | n/a | 2026-09-19 |
| `tiny` | 94.5 % | 40.3 % | 67.1 % | n/a | n/a | n/a | 64 (-200-292) | n/a | n/a | 2026-09-19 |

6 etapas medidas con la misma suite. Las filas del encoder viven aparte porque no comparten una sola columna con estas; las que una medición retiró no están (`rukh eval drop`).
