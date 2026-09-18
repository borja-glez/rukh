# labs

A lab is a small, focused experiment that backs one lesson of the course: it produces a number,
a curve or a comparison the lesson shows and discusses. Labs live here as thin Jupyter notebooks
or scripts that import the `rukh` package; the real logic stays in `src/rukh/` so it is tested
and reusable.

## Rules

- One folder per module (`m1-tokenization/`, `m2-decoder/`, …), one notebook or script per lab.
- A lab reads data from `data/` (gitignored) and logs to MLflow like any other run
  (`rukh.tracking.start_run`), so every figure in the course can be traced back to a run.
- Notebooks are committed with outputs cleared; heavy results are exported as data, not kept in
  the notebook.

## Exporting data to the course

The course site (`rukh-lab`) renders visualizations from static JSON. A lab exports what a
lesson needs into `artifacts/web/<module>/<name>.json`:

- Under 1 MB per file, plain JSON (numbers, strings, arrays), versioned in git.
- Each file carries a `meta` object with `run_id` (MLflow), `git_sha`, `generated_at` and the
  `rukh_version`, so the site can show provenance.
- `rukh-lab` copies these files with its `pnpm sync:data` script; nothing is imported across repos.

Labs arrive with P1 (tokenization comparison). This folder holds only this README until then.
