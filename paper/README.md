# Paper draft

`paper.tex` is the full end-to-end draft of the paper, in the IEEE conference
format (`IEEEtran`), built entirely from this repo's real results.

## How to compile (easiest: Overleaf)

1. Go to [overleaf.com](https://www.overleaf.com), **New Project → Upload Project**.
2. Upload `paper.tex` **and** the four figures from `results/figures/`
   (`model_comparison.png`, `confusion_matrices.png`, `timeseries_anomalies.png`,
   `anomaly_closeup.png`). Put the PNGs either next to `paper.tex` or keep the
   repo structure — `paper.tex` already points at `../results/figures/` via
   `\graphicspath`.
3. Set the compiler to **pdfLaTeX** and click Recompile. (It auto-runs twice to
   resolve citations/figure numbers.)

Locally instead: `pdflatex paper.tex` twice, from this `paper/` folder.

## What's real vs. placeholder

- **All numbers, tables, and figures are real** — straight from
  `results/metrics.json` and `results/figures/` (regenerable via `python main.py`).
- **All 12 citations are real and verified** — see `../REFERENCES.md`. One
  (`rotating2023`, ref [11]) has its author list marked with a `% NOTE:` to
  confirm against the published PDF before final submission.
- **Placeholders to fill in:** author affiliations, city/country, and Prof.
  Abhi Shah's email — marked `[Institution]`, `[City, Country]`, `[email]` in
  the author block.

## Before submitting

- Reformat to the **target venue's** template once Prof. Shah picks the venue
  (ROADMAP E2/E3) — if it's IEEE, this file is already in the right format.
- Confirm ref [11]'s authors.
- Have both authors read it; the prose is meant to be humanised/adjusted to
  taste.
