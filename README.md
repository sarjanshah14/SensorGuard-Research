# Protocol Sensitivity in Unsupervised Industrial Anomaly Detection

Sarjan Shah and Prof. Abhi Shah, Department of Information Technology, Lok Jagruti Kendra University.

Paper: *Protocol Sensitivity in Unsupervised Industrial Anomaly Detection: Threshold Transfer, Retraining Risk, and Inference Cost* (target: International Journal of Prognostics and Health Management, IJPHM).

Ten unsupervised detectors (z-score, PCA reconstruction, Elliptic Envelope, Isolation Forest, One-Class SVM, LOF, ECOD, COPOD, HBOS, LSTM autoencoder) are compared on four industrial benchmarks (SKAB, SWaT, WADI, HAI) under documented protocols. The study measures how the split definition, threshold rule, sensor set and retraining data change the conclusions, and the per-reading cost in a 1-CPU / 1 GB container.

**New here? Start with `CHECKPOINT.md`.** For a plain-language tour, see the explainer page linked there.

## Layout

```
README.md  CHECKPOINT.md  CLAUDE.md  requirements.txt
docs/                 ROADMAP, DEVLOG, PROGRESS, WALKTHROUGH, REFERENCES
docs/revision/        revision reports 01-11 (audit, literature, novelty, change log, venues)
data/SKAB/            SKAB benchmark (public, tracked)
data/SWaT_official_xlsx/  official iTrust SWaT files (licensed, NOT tracked)
data/external/        HAI, WADI, converted SWaT (NOT tracked)
data/downloads/       original zip downloads (NOT tracked)
src/                  shared pipeline code (features, loaders, deep baseline, drift score)
experiments/revision/ experiments behind the current paper
experiments/initial_analysis/  scripts of the first (ICACS-2026) version, kept as evidence
results/revised/      results and figures of the current paper
results/initial_analysis/  results of the first version
paper/current/        current paper: .tex, IJPHM Word file, Overleaf zip, Word build script
paper/icacs_2026/     rejected ICACS-2026 version and its template
paper/templates/ijphm/  IJPHM Word templates
apps/skab.py          Streamlit explorer for SKAB and the first-version results
docker/               capped-container image for the cost benchmark
archive/SensorGuard/  old hackathon repo (reference only, not tracked)
```

## Reproduce

```bash
source env/bin/activate
python -m experiments.revision.common                    # metric self-check
python -m experiments.revision.skab_revised              # ~2 min
SWAT_SOURCE=official python -m experiments.revision.swat_revised   # ~15 min
SWAT_SOURCE=official python -m experiments.revision.drift_revised  # ~70 min
python -m experiments.revision.hai_revised               # ~15 min
python -m experiments.revision.wadi_revised              # ~20 min
python -m experiments.revision.figures_revised
python paper/current/make_ijphm_docx.py                  # IJPHM Word manuscript
```

Cost benchmark (Docker): see `docs/revision/09_reproducibility_package/README.md`.

## Data access

- **SKAB**: public (github.com/waico/SKAB), included in `data/SKAB/`.
- **HAI 21.03**: public (github.com/icsdataset/hai); place `hai-21.03/` in `data/external/hai/`.
- **SWaT (A1 & A2, Dec 2015) and WADI (A2, 2019)**: request from iTrust, SUTD (https://itrust.sutd.edu.sg/itrust-labs_datasets/). Licensed: do not redistribute or commit. Put the SWaT `.xlsx` files in `data/SWaT_official_xlsx/` and the WADI folder in `data/external/wadi/`.
