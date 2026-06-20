# SensorGuard-Research

**Paper:** "Comparative Evaluation of Unsupervised Machine Learning Models
for Multivariate Sensor Anomaly Detection in Industrial Systems"

This repository compares 6 unsupervised anomaly-detection models (Z-score,
One-Class SVM, LOF, PCA-reconstruction, Elliptic Envelope, Isolation Forest)
on the real Skoltech Anomaly Benchmark (SKAB) dataset — a multivariate
sensor time series from a water-circulation testbed with induced valve
faults. Isolation Forest is the best performer (F1 0.466); see
`results/metrics.json` for full numbers.

**New to this repo? Read [`WALKTHROUGH.md`](WALKTHROUGH.md) first** — it's
a plain-English guide to every file, every result, and every figure, written
so you can review everything without reading code.

---

## Dataset: SKAB (Skoltech Anomaly Benchmark)

The `SKAB/` directory contains multivariate time-series sensor readings
collected from a water circulation testbed, with artificially induced valve
anomalies:

- **`SKAB/anomaly-free/anomaly-free.csv`**: Baseline run under normal conditions (no label columns).
- **`SKAB/valve1/*.csv`**: 16 runs with anomalies induced at valve 1.
- **`SKAB/valve2/*.csv`**: 4 runs with anomalies induced at valve 2.
- **`SKAB/other/*.csv`**: 14 runs with miscellaneous other anomalies.

The CSVs are semicolon-delimited (`;`), 1 reading/second, with columns:

```
datetime;Accelerometer1RMS;Accelerometer2RMS;Current;Pressure;Temperature;Thermocouple;Voltage;Volume Flow RateRMS[;anomaly;changepoint]
```

`anomaly`/`changepoint` are binary labels present only on the fault-run files.

---

## Repository layout

```
src/            the pipeline: preprocessing, feature engineering, the 6 models, evaluation, persistence
main.py         runs the whole pipeline end to end
experiments/    one-off scripts behind specific methodology decisions (kept as evidence)
results/        official metrics.json, the 4 paper figures, and the saved model bundle
skab.py         Streamlit app: dataset explorer + live model-results viewer
ROADMAP.md      the checkpoint plan, ticked as work completes
DEVLOG.md       detailed narrative of every session: what/why/how, including dead ends
PROGRESS.md     one-line-per-session index
WALKTHROUGH.md  the guided tour — start here
```

---

## Getting started

### 1. Environment

A pre-configured Python 3.14 virtual environment lives in `env/`:

```bash
source env/bin/activate
```

### 2. Reproduce the results

```bash
python main.py
```

This regenerates `results/metrics.json`, the 4 figures in `results/figures/`,
and the saved model bundle in `results/models/`. Takes well under a minute.

### 3. Explore interactively

```bash
streamlit run skab.py
```

Opens a browser tab with the raw-data explorer (sensor graphs, data quality,
anomaly close-ups, correlation insights) plus a **"🏆 Model Results"** tab
showing the trained-model comparison and figures.
