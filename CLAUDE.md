# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Research paper by Sarjan Shah and Prof. Abhi Shah: *Protocol Sensitivity in Unsupervised
Industrial Anomaly Detection: Threshold Transfer, Retraining Risk, and Inference Cost*,
targeting IJPHM. The first version (ICACS-2026) was rejected; the current work is the
revision. **Start from `CHECKPOINT.md`.** Layout:

- `README.md` — overview, layout, reproduce commands, data access.
- `docs/` — `ROADMAP.md`, `DEVLOG.md`, `PROGRESS.md` (logging convention below),
  `WALKTHROUGH.md` (first-version guide), `REFERENCES.md`; `docs/revision/` holds the
  revision reports 01–11.
- `data/` — `SKAB/` (tracked); `SWaT_official_xlsx/`, `external/` (HAI, WADI, converted
  SWaT) and `downloads/` are iTrust-licensed or regenerable and must never be committed.
- `src/` — shared code: `preprocessing.py`, `features.py`, `deep_baseline.py`, `drift.py`,
  `swat.py`, plus first-version `models.py`, `evaluate.py`, `visualize.py`, `persist.py`.
- `experiments/revision/` — experiments behind the current paper (`common.py`,
  `skab_revised.py`, `swat_revised.py`, `wadi_revised.py`, `hai_revised.py`,
  `drift_revised.py`, `edge_revised.py`, `figures_revised.py`).
- `experiments/initial_analysis/` — first-version scripts and `main.py`, kept as evidence.
- `results/revised/` (current) and `results/initial_analysis/` (first version).
- `paper/current/` — `paper2_revised.tex`, the IJPHM Word file, Overleaf zip and
  `make_ijphm_docx.py`; `paper/icacs_2026/` — rejected version; `paper/templates/ijphm/`.
- `apps/skab.py` — Streamlit explorer (`streamlit run apps/skab.py`).
- `docker/Dockerfile.edge2` — capped container for the cost benchmark.

## Dataset: SKAB

`data/SKAB/` holds multivariate time-series sensor readings from a testbed with an artificially induced anomaly-causing valve, organized as:

- `data/SKAB/anomaly-free/anomaly-free.csv` — baseline run with no anomalies (no `anomaly`/`changepoint` columns).
- `data/SKAB/other/*.csv` — runs with miscellaneous anomalies (14 files).
- `data/SKAB/valve1/*.csv` — runs with anomalies from valve 1 (16 files).
- `data/SKAB/valve2/*.csv` — runs with anomalies from valve 2 (4 files).

All CSVs are semicolon-delimited (`;`) with columns:
`datetime;Accelerometer1RMS;Accelerometer2RMS;Current;Pressure;Temperature;Thermocouple;Voltage;Volume Flow RateRMS[;anomaly;changepoint]`

## Logging convention

Sarjan reads/reviews/verifies; Claude drives the implementation. To make that
possible, every work session must update two files (not just one):

- `docs/PROGRESS.md` — one line per session (non-negotiable, format:
  `[DATE] [duration] what was done`).
- `docs/DEVLOG.md` — the detailed narrative: what we set out to do, the steps
  taken and why, problems hit and exactly how they were diagnosed/fixed,
  and a "plain language" outcome summary at the end of each entry. This is
  the file to read back later to understand *why* something is the way it
  is, not just *that* it changed.
- `docs/ROADMAP.md` — the forward-looking checklist: every checkpoint from "no
  code" to "submitted paper," each elaborated (what/why/depends-on), with
  checkboxes ticked as they complete. Update this whenever a checkpoint
  finishes or the plan changes — it's the single source of truth for "what's
  next."

At the end of any response involving real work, also give a short
plain-language summary in chat: what was done and what the outcome was.

## Development Commands

Always run Python commands within the activated virtual environment (`source env/bin/activate`), from the repo root.

- Current paper experiments: `python -m experiments.revision.<name>` (SWaT on official
  data: `SWAT_SOURCE=official python -m experiments.revision.swat_revised`).
- Figures: `python -m experiments.revision.figures_revised`.
- IJPHM Word manuscript: `python paper/current/make_ijphm_docx.py`.
- First-version pipeline: `python -m experiments.initial_analysis.main`.
- Streamlit app: `streamlit run apps/skab.py`.
- No local LaTeX: compile `paper/current/overleaf_paper2.zip` on Overleaf.
