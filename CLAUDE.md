# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

SensorGuard-Research is in an early/pre-code stage: the repository currently contains dataset files, analysis/exploratory tools, and a virtual environment. It consists of:

- `README.md` — project overview, dataset info, and setup instructions.
- `SKAB/` — the Skoltech Anomaly Benchmark (SKAB) dataset (CSV files).
- `env/` — local Python 3.14 virtual environment. Activate with `source env/bin/activate`.
- `visualize.py` — CLI/interactive tool for summarizing dataset files and saving Matplotlib signal/correlation plots under `plots/`.
- `explore.ipynb` — Jupyter Notebook containing code cells for interactive, block-by-block data analysis.

## Dataset: SKAB

`SKAB/` holds multivariate time-series sensor readings from a testbed with an artificially induced anomaly-causing valve, organized as:

- `SKAB/anomaly-free/anomaly-free.csv` — baseline run with no anomalies (no `anomaly`/`changepoint` columns).
- `SKAB/other/*.csv` — runs with miscellaneous anomalies (14 files).
- `SKAB/valve1/*.csv` — runs with anomalies from valve 1 (16 files).
- `SKAB/valve2/*.csv` — runs with anomalies from valve 2 (4 files).

All CSVs are semicolon-delimited (`;`) with columns:
`datetime;Accelerometer1RMS;Accelerometer2RMS;Current;Pressure;Temperature;Thermocouple;Voltage;Volume Flow RateRMS[;anomaly;changepoint]`

## Development Commands

Always run Python commands within the activated virtual environment (`source env/bin/activate`).

- **List dataset files**:
  ```bash
  python visualize.py --list
  ```
- **Run interactive visualizer selector**:
  ```bash
  python visualize.py
  ```
- **Visualize a specific CSV file**:
  ```bash
  python visualize.py --file SKAB/valve1/0.csv
  ```
- **Visualize a file without saving plots to disk (show GUI window)**:
  ```bash
  python visualize.py --file SKAB/valve1/0.csv --no-save
  ```
