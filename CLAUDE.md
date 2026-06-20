# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

SensorGuard-Research is a complete, working research pipeline (data → features
→ 6 models → evaluation → figures → persistence) for the paper "Comparative
Evaluation of Unsupervised Machine Learning Models for Multivariate Sensor
Anomaly Detection in Industrial Systems." Only the paper itself (Phase D in
`ROADMAP.md`) remains. Layout:

- `README.md` — project overview, dataset info, and how to run things.
- `WALKTHROUGH.md` — the plain-English guide to everything built and why;
  read this first to review results without reading code.
- `ROADMAP.md` / `DEVLOG.md` / `PROGRESS.md` — the living plan, the detailed
  narrative log, and the one-line-per-session index (see "Logging convention"
  below).
- `SKAB/` — the Skoltech Anomaly Benchmark (SKAB) dataset (CSV files).
- `env/` — local Python 3.14 virtual environment. Activate with `source env/bin/activate`.
- `src/` — the pipeline: `preprocessing.py` (loading/splitting), `features.py`
  (feature engineering), `models.py` (the 6 models), `evaluate.py` (scoring),
  `postprocess.py` (a rejected smoothing idea, kept for the record),
  `persist.py` (model save/load), `visualize.py` (paper figures).
- `main.py` — runs the whole pipeline end to end; regenerates
  `results/metrics.json`, `results/figures/`, and `results/models/`.
- `experiments/` — one-off ablation/decision scripts backing specific
  methodology choices (training strategy, hyperparameter sweep, temporal
  smoothing, feature ablation, extra models) — kept as evidence, not dead code.
- `skab.py` — Streamlit app for exploring the dataset and viewing model
  results (`streamlit run skab.py`).
- `results/` — official metrics, figures, and the saved model bundle.

## Dataset: SKAB

`SKAB/` holds multivariate time-series sensor readings from a testbed with an artificially induced anomaly-causing valve, organized as:

- `SKAB/anomaly-free/anomaly-free.csv` — baseline run with no anomalies (no `anomaly`/`changepoint` columns).
- `SKAB/other/*.csv` — runs with miscellaneous anomalies (14 files).
- `SKAB/valve1/*.csv` — runs with anomalies from valve 1 (16 files).
- `SKAB/valve2/*.csv` — runs with anomalies from valve 2 (4 files).

All CSVs are semicolon-delimited (`;`) with columns:
`datetime;Accelerometer1RMS;Accelerometer2RMS;Current;Pressure;Temperature;Thermocouple;Voltage;Volume Flow RateRMS[;anomaly;changepoint]`

## Logging convention

Sarjan reads/reviews/verifies; Claude drives the implementation. To make that
possible, every work session must update two files (not just one):

- `PROGRESS.md` — one line per session (non-negotiable, format:
  `[DATE] [duration] what was done`).
- `DEVLOG.md` — the detailed narrative: what we set out to do, the steps
  taken and why, problems hit and exactly how they were diagnosed/fixed,
  and a "plain language" outcome summary at the end of each entry. This is
  the file to read back later to understand *why* something is the way it
  is, not just *that* it changed.
- `ROADMAP.md` — the forward-looking checklist: every checkpoint from "no
  code" to "submitted paper," each elaborated (what/why/depends-on), with
  checkboxes ticked as they complete. Update this whenever a checkpoint
  finishes or the plan changes — it's the single source of truth for "what's
  next."

At the end of any response involving real work, also give a short
plain-language summary in chat: what was done and what the outcome was.

## Development Commands

Always run Python commands within the activated virtual environment (`source env/bin/activate`).

- **Run the full pipeline** (regenerates `results/metrics.json`, the 4
  figures, and the saved model bundle):
  ```bash
  python main.py
  ```
- **Launch the interactive Streamlit app** (dataset explorer + Model Results tab):
  ```bash
  streamlit run skab.py
  ```
- **Re-run a specific methodology experiment** (run as a module so `src/` is importable):
  ```bash
  python -m experiments.hyperparameter_sweep
  ```
