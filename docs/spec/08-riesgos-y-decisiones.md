# 08 · Decisiones por defecto, riesgos y licencias

## Decisiones tomadas por defecto (cámbialas antes de arrancar si no te convencen)

| Tema | Decisión por defecto | Alternativa |
|---|---|---|
| Nombre | `rukh` (definitivo; persa para la torre del ajedrez, y Borja = *burj* = torre). Constante única en cada repo por higiene | — |
| Dominios | Jugar en `rukh.borjaglez.com`; curso en `lab.rukh.borjaglez.com` (decidido por Borja); el antiguo `lab.borjaglez.com` queda como está hasta nueva orden | — |
| Repos | Tres repos (decidido por Borja): `borja-glez/rukh` (ML), `rukh-lab` (curso), `rukh-web` (jugar); el anterior `borja-glez/lab` queda archivado | Monorepo único (descartado) |
| Código compartido entre webs | Copias sincronizadas por script (tokens CSS, tokenizador TS, vocabulario y fixtures) con test de hash en CI | Paquete npm propio (sobrecoste para dos consumidores) |
| Publicación | Todo modelo, adaptador y dataset al Hub `chorcat/rukh-*` en la colección `Rukh`, con card EN, antes de cerrar el hito; nada de pesos en git | — |
| Demo | Una pantalla sencilla y responsive desde P0 (ordenador, tablet, móvil); lo avanzado en un cajón "más" | Demo con todos los paneles a la vista (descartado: complejidad) |
| Tokenización por defecto | Vocabulario fijo UCI (~1 900 tokens) + tokens de Elo/resultado | Carácter (Karvonen) para comparar |
| Tamaño del modelo del curso | `small` 12 capas, d=512 (~40M) | `medium` opcional de noche |
| Datos | 2 meses recientes filtrados (≥ 1800, sin bullet), ≈ 3-6 M partidas | Más meses si sobra tiempo |
| Etiquetas de valor | Cruce con `chess-position-evaluations` (sin ejecutar Stockfish masivamente) | Stockfish local para lo no cubierto |
| Estimación de Elo | Partidas contra Stockfish con `UCI_Elo` (1320-3190) y `Skill Level` por debajo | Bot de Lichess (Elo real, opcional) |
| Tablero | `cm-chessboard` (MIT) + `chess.js` (BSD-2) | `chessground` (GPL-3) si se acepta GPL en la demo |
| Stockfish en navegador | No en fase 1 (GPL y 205 MB); en servidor en fase 2 | `stockfish` npm con la demo bajo GPL |
| Inferencia web | `onnxruntime-web` directo (modelo propio) | transformers.js si se convierte a GPT-2 de HF |
| LLM comentarista (fase 2) | Local (LM Studio/Ollama) en desarrollo; modelo pequeño en CPU o puente a la 5090 para la demo pública | API de pago (descartado) |
| Licencias | Código MIT; contenido del curso CC BY-NC-SA 4.0; pesos Apache-2.0; datos derivados CC0 con atribución a Lichess | — |
| Idioma | Curso ES; código, README, model cards y `/proyecto/` EN | — |
| Commits | En inglés con Conventional Commits, sin coautoría ni referencias a herramientas (nota 2026-09-18: decisión D-001 en `docs/decisiones-de-ejecucion.md`; antes "en español, imperativo") | — |

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| El `small` no llega a 1200 Elo | Más datos (meses), más pasos, `medium`; medir con IC; el objetivo del curso es el proceso, la cifra se documenta tal cual |
| Estimar Elo contra Stockfish limitado es ruidoso | ≥ 100 partidas por nivel, ambos colores, regresión logística con IC; validar con el bot de Lichess al final |
| Cruce con evaluaciones deja posiciones sin etiqueta | Stockfish local a profundidad 12 solo para las que faltan; caché SQLite |
| GRPO inestable o hackeado (tablas por repetición, jugadas pasivas) | Recompensa de progreso, penalización de repetición, KL β 0,02-0,04, `mask_truncated_completions`, galería documentada |
| `torch` cu128 vs CPU con versiones distintas en `uv` | Fijar `torch>=2.11,<2.12` desde el índice cu128 en el extra `cu128` y el mismo rango desde PyPI en `cpu`; test guardián del lock |
| Sin `nvcc` en Windows: nada de flash-attn | SDPA; `torch.compile` opcional |
| Modelos > 300 MB no caben en navegadores embebidos; wasm32 limita a 4 GB | Los modelos de este proyecto pesan 40-110 MB: no aplica; se documenta igualmente |
| Asyncify de ORT no admite llamadas concurrentes | Sesiones en serie, `run` serializado, un worker por modelo |
| Vite incrusta workers como `data:` si el literal `new Worker(new URL(...))` no está en el punto de llamada | Guardián de build que cuenta los chunks de workers |
| Dokploy usa la carpeta del Dockerfile como contexto si se deja vacío | Dockerfile en la raíz de cada repo web; Docker Context Path `.` explícito |
| Tres repos: las copias compartidas divergen | Scripts `sync:*` + test de hash en la CI de cada web; el hito que cambia el tokenizador actualiza los tres repos en la misma ola |
| La demo se complica y deja de verse bien en móvil | Pantalla única protegida por E2E en tres viewports con capturas; toda función nueva va al cajón "más" |
| Lighthouse oscila al filo de 0,95 en el runner | `numberOfRuns: 3`, reservar alturas de islas (CLS), subir informes si falla |
| Disco | Reservar ~40 GB en `E:`; `data/` y `mlruns/` fuera de git |
| Alcance se dispara (síndrome del proyecto anterior) | Cada hito cierra un módulo publicable; nada de audio/vídeo; la fase 2 no empieza hasta cerrar P6 |

## Licencias y atribuciones que hay que mostrar

- Lichess (datos CC0): atribución voluntaria en README, dataset cards y página de la demo.
- Elite DB (nikonoel): citar la fuente; origen Lichess.
- `python-chess` (GPL-3) y Stockfish (GPL-3): se ejecutan en Python del lado de Borja/servidor; no
  se redistribuyen con la demo web. Si algún día se incluye `stockfish` npm en la demo, la demo pasa a
  GPL-3.
- `lichess-bot` (AGPL-3): solo se ejecuta localmente para el bot; el motor UCI propio se publica aparte.
- Libros de Gutenberg: dominio público; conservar la cabecera de Gutenberg en el corpus publicado.
- Modelos de referencia: Karvonen (MIT), searchless_chess (Apache-2.0), Maia (GPL-3 código; solo se
  citan resultados o se ejecutan localmente para comparar).
