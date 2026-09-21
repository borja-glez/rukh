# 04 · La web del curso (`lab.rukh.borjaglez.com`)

Sitio estático que publica el curso con el sistema de diseño del portfolio, en un subdominio nuevo
(`lab.rukh.borjaglez.com`, app propia en Dokploy desde `borja-glez/rukh-lab`). El anterior
`lab.borjaglez.com` (proyecto aparcado) sigue como está hasta que Borja decida retirarlo o redirigirlo.

## Stack (el mismo que borjaglez.com, verificado)

- Astro ≥ 7.3 estático, `@astrojs/sitemap`, `@astrojs/mdx`, `@astrojs/preact` para islas
  interactivas, `astro-expressive-code` para código con temas duales. TypeScript 6 (nunca 7), pnpm 10,
  Node ≥ 24. Prettier con `prettier-plugin-astro`. GSAP solo si una figura lo necesita.
- Diseño: se copian los tokens de `E:\work\borjaglez.com\src\styles\global.css` (`--paper`, `--ink`,
  `--ink-2`, `--muted`, `--line`, `--brand`, `--accent-text`, `--event`, `--ok`, `--split-1..5`,
  `--frame-bg`, todos con `light-dark()`), las fuentes (Bricolage Grotesque + IBM Plex Mono vía Astro
  Fonts), el `Header`/`Footer`/`Rail`/`SectionHead`/`FigureFrame` y la lógica de tema (`color-scheme`,
  sin flash). Se documentan divergencias en `docs/design-system.md`. Contraste mínimo AA en ambos temas
  (Lighthouse `color-contrast` como aserción).
- Repo propio `borja-glez/rukh-lab` (ver 07). Sin backend. Lo que comparte con la demo
  (tokens CSS, tokenizador TS + vocabulario) llega por copias sincronizadas con script
  (`pnpm sync:tokenizer`, `pnpm sync:data`) y un test de hash, nunca por dependencia de workspace.

## Estructura de contenidos

```
src/content/
  modules/   m0.json … m6.json, a1.json … a6.json   (título, fase, horas, estado, resumen, artefactos)
  lessons/   <mod>/<nn>-<slug>.mdx                    (frontmatter: title, description, module, order,
                                                       duration, level, status, updated, keywords,
                                                       artifacts[{kind, label, href}], demo?)
  glossary/  terms.json                               (término, definición, módulo, alias)
  cheatsheets/ <mod>.json                             (pregunta, respuesta corta)
```

Rutas: `/` (landing: qué es, mapa de módulos con progreso local, enlace a la demo) · `/curso/` ·
`/curso/[modulo]/[leccion]` · `/glosario/` · `/cheatsheets/` (imprimible) · `/proyecto/` (página
en inglés para recruiters: arquitectura, tabla de resultados, model cards, enlaces) · `/og/[...].png`
· `404`. i18n: `es` por defecto sin prefijo; `/en/` solo landing y proyecto.

## Componentes de lección

Layout de lección con nav de módulo, prosa a 72ch, TOC por IntersectionObserver, prev/next, "marcar
completada" (localStorage `rukh:progress:*`), tiempo estimado y estado (borrador/vigente).
Componentes MDX: `Callout` (teoría justa / en el mundo real / entrevista / así se hace hoy / roto),
`Exercise` + `Solution` (desplegable), `Term` (popover al glosario), `Figure`, `Tabs`, `NotebookLink`,
`ModelBadge` (enlace a la model card), `ResultsTable` (lee `data/results.json`), `DemoEmbed`
(inserta una isla interactiva).

## Visualizaciones (islas Preact, datos precomputados por los labs)

Cada lab termina con una celda "exporta datos para la web" que escribe JSON en `rukh/artifacts/web/`
(versionado si pesa < 1 MB); `pnpm sync:data` en el repo del curso lo copia a `src/data/`:

| Módulo | Isla | Datos |
|---|---|---|
| M1 | `TokenizerPlayground`: pega un PGN, ve los tokens de las tres representaciones y sus conteos | tokenizador UCI en TS (vocabulario fijo) + BPE exportado a JSON |
| M2 | `AttentionMap`: partida corta, capa/cabeza, pesos sobre las jugadas | JSON precomputado |
| M2 | `TrainingReplay`: legalidad y Elo por checkpoint con slider | JSON de MLflow |
| M3 | `ValueBar`: partida con barra de evaluación del encoder vs Stockfish | JSON |
| M4 | `EloSelector`: misma posición, jugada preferida por `<w1500>`/`<w2000>`/`<w2400>` | JSON |
| M4 | `LoraRank`: valores singulares de ΔW por capa | JSON |
| M5 | `GrpoGroup`: G jugadas de una posición con recompensa y ventaja | JSON |
| M5 | `RewardCurves`: recompensa por función y KL | JSON |
| M6 | `Leaderboard`: tabla de etapas ordenable | `results.json` |

Las islas que ejecutan modelos en vivo (jugar contra el modelo) viven en la demo, no en el curso;
el curso enlaza a la demo con parámetros (`?stage=dpo&elo=1500`).

## Calidad y despliegue

- CI: `format:check`, `lint` (ESLint + eslint-plugin-astro), `astro check`, Vitest (helpers puros,
  registro de componentes MDX, paridad ES/EN), `build`, Playwright (curso navegable, progreso, axe),
  Lighthouse (`numberOfRuns: 3`, performance/a11y/SEO ≥ 0,95, artefacto `.lighthouseci` si falla),
  `docker build`.
- CSP: generada por Astro con hashes (`security.csp`), sin scripts inline sin hash; nginx con
  `frame-ancestors 'none'`, HSTS, `Permissions-Policy` restrictiva (el curso no usa cámara ni micro).
- Docker multi-stage `node:24-alpine` → `nginx:1.29-alpine`, Dockerfile en la raíz del repo
  (Dokploy: app desde `borja-glez/rukh-lab`, Dockerfile path `Dockerfile`, contexto `.`), `gzip_static`,
  cache inmutable en `/_astro/`. Dominio `lab.rukh.borjaglez.com` (HTTPS Let's Encrypt, auto-deploy por
  webhook en `main`).
- SEO/portfolio: OG por lección (satori + resvg), JSON-LD `Course`/`LearningResource`/`Person`,
  sitemap; enlace cruzado desde borjaglez.com al terminar la fase 1 (entrada en proyectos y sección
  de IA generativa; cambio pequeño en ese repo).

## Redacción de lecciones

Lecciones en español, tono directo, sin referencias a los cursos de origen; "teoría justa" ≤ 40 % del
texto; cada módulo abre con "qué vas a construir" y cierra con "qué has aprendido, cómo se mide y
cheatsheet". Runbook `docs/runbooks/lessons.md` con la plantilla y la lista de componentes. Las
lecciones se escriben en el mismo hito que su lab (no se dejan para el final).

> **Nota (2026-09-21, D-134):** un módulo son varias lecciones de una a dos horas, partidas por sus
> secciones (teoría · cómo se mide y lo que salió · labs), nunca una sola página de veinte mil
> palabras. La primera parte conserva el slug del módulo; la cheatsheet se muestra solo en la
> última; cada parte abre diciendo de dónde viene y a dónde va, y la parte de labs abre con un
> «punto de partida» que dice qué hay que tener en disco y el `rukh pull` que lo trae.
> Cada módulo cierra, y el siguiente abre, con `RepoTag`: el enlace a la etiqueta del repo
> `rukh` de ese día y la línea para clonarla, con el aviso de que es una fotografía.
