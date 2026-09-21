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
| `lora-d4` | 99.8 % | 55.1 % | 83.0 % | 57.4 % | 37.5 % | 16.9 % | 1535 (1481-1582) | n/a | 0.0160 | 2026-09-21 |
| `lora-e4` | 99.8 % | 54.7 % | 83.3 % | 58.1 % | 38.2 % | 17.2 % | 1538 (1488-1604) | n/a | 0.0209 | 2026-09-21 |
| `medium-elo` | 99.7 % | 54.9 % | 82.5 % | 59.8 % | 37.1 % | 16.6 % | 1592 (1532-1650) | n/a | 1.7408 | 2026-09-21 |
| `medium-masters` | 99.7 % | 54.9 % | 82.3 % | 58.1 % | 38.7 % | 17.1 % | 1563 (1502-1630) | n/a | 1.8850 | 2026-09-21 |
| `medium-v4-dpo-greedy` | 99.8 % | 53.4 % | 80.5 % | 58.3 % | 39.9 % | 18.1 % | 1529 (1470-1583) | n/a | 1.7785 | 2026-09-20 |
| `medium-v4-dpo-onpolicy-greedy` | 99.2 % | 52.3 % | 79.9 % | 59.0 % | 40.3 % | 17.8 % | 1632 (1567-1706) | n/a | 1.7511 | 2026-09-21 |
| `medium-v4-greedy` | 99.8 % | 54.4 % | 82.2 % | 57.9 % | 38.0 % | 16.6 % | 1535 (1476-1599) | n/a | 1.7695 | 2026-09-21 |
| `medium-v4-grpo-greedy` | 99.3 % | 53.3 % | 80.3 % | 58.9 % | 39.1 % | 17.8 % | 1577 (1526-1632) | n/a | 1.7874 | 2026-09-21 |
| `qwen3-pgn-qlora` | 62.5 % | 12.5 % | n/a | 1.8 % | 0.7 % | 0.4 % | < 807 | n/a | n/a | 2026-09-21 |
| `small-greedy` | 99.4 % | 51.1 % | 79.4 % | 34.7 % | 21.1 % | 10.3 % | 1321 (1247-1383) | n/a | 1.7412 | 2026-09-20 |
| `small-v3-greedy` | 99.1 % | 52.4 % | 80.5 % | 41.0 % | 27.0 % | 12.2 % | 1425 (1367-1484) | n/a | 1.6393 | 2026-09-21 |
| `tiny-greedy` | 94.5 % | 40.3 % | 67.1 % | 14.1 % | 8.6 % | 3.9 % | 778 (479-904) | n/a | 1.7214 | 2026-09-21 |

12 etapas medidas con la misma suite. Las filas del encoder viven aparte porque no comparten una sola columna con estas; las que una medición retiró no están (`rukh eval drop`).
