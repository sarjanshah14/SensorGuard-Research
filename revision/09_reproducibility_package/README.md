# Deliverable 9: Reproducibility package

Original materials are untouched: `src/` (one backward-compatible flag added, `include_elapsed`, default = original behaviour), `experiments/*.py`, `results/*.json`, `results/figures/`, `paper/paper1.tex`, the submitted PDF. Everything revised lives in:

| Path | What |
|---|---|
| `experiments/revision/common.py` | detectors with continuous scores, threshold rules, metrics, event metrics, block bootstrap (`python -m experiments.revision.common` runs its self-check) |
| `experiments/revision/skab_revised.py` | SKAB pooled + SKAB-benchmark protocol + ablations → `results/revised/skab.json` |
| `experiments/revision/swat_revised.py` | SWaT standard protocol + original layout with same code → `results/revised/swat.json` |
| `experiments/revision/drift_revised.py` | retraining-policy study → `results/revised/drift.json` |
| `experiments/revision/edge_revised.py` + `docker/Dockerfile.edge2` | capped-container cost benchmark → `results/revised/edge.json` |
| `experiments/revision/figures_revised.py` | figures → `results/revised/figures/` |
| `paper/paper2_revised.tex` | revised manuscript |

## Environment
- Host: Apple M4, 16 GB, macOS; Python 3.14 venv `env/`.
- Key packages: numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, scikit-learn 1.9.0, pyod 3.6.7, numba 0.68.1, torch 2.12.1 (MPS on host). Exact list: `requirements.txt`.
- Container: `python:3.12-slim-bookworm`, linux/arm64, same library versions, torch CPU wheel.

## Data
- SKAB: included in `SKAB/` (from github.com/waico/SKAB; cite Katser & Kozitsin 2020, DOI 10.34740/KAGGLE/DSV/1693952).
- SWaT: **not redistributable**. Request the official release from iTrust, SUTD (https://itrust.sutd.edu.sg/itrust-labs_datasets/). The study used the Kaggle re-upload `vishala28/swat-dataset-secure-water-treatment-system` v3 (downloaded automatically via `kagglehub` on first run). `experiments/revision/swat_revised.py:load_standard` documents and asserts how the standard split is reconstructed from that copy.

## Commands
```bash
source env/bin/activate
python -m experiments.revision.common          # self-check
python -m experiments.revision.skab_revised    # ~2 min
python -m experiments.revision.swat_revised    # ~40 min (LSTM-AE + bootstrap dominate)
python -m experiments.revision.drift_revised   # ~60 min
docker build -f docker/Dockerfile.edge2 -t sensorguard-edge2 docker/
docker run --rm --cpus=1 --memory=1g -v "$PWD":/app \
  -v ~/.cache/kagglehub/datasets/vishala28/swat-dataset-secure-water-treatment-system/versions/3:/swat \
  -e SWAT_DIR=/swat sensorguard-edge2           # ~10 min
python -m experiments.revision.figures_revised
```

## Seeds and determinism
Isolation Forest: seeds 0-9 reported as mean ± sd; other classical detectors deterministic (Elliptic Envelope MCD seed 0). LSTM-AE: seeds 0-2 on SKAB, seed 0 on SWaT; MPS kernels are not bit-exact, so re-runs can differ in the third decimal. Bootstrap: numpy default_rng(0), B = 500 (SKAB) / 200 (SWaT).
