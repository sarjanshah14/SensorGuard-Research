# Paper drafts

Papers are numbered so future work can be added alongside:

- **`paper2_revised.tex`** — **the current paper (2026-10-28)**, rewritten after the
  ICACS-2026 rejection from re-run experiments (`results/revised/`). Venue-neutral
  `article` class, author-year references (IJPHM uses APA). Upload
  `overleaf_paper2.zip` (regenerable: copy the .tex + `results/revised/figures/*.png`
  into a `figures/` folder) to Overleaf, compile with pdfLaTeX. See
  `../revision/08_change_log.md` for every change and `../CHECKPOINT.md` for what is left.

- **`paper1.tex`** — the current, complete paper, formatted for **ICACS 2026 /
  Springer LNNS** in the **Springer LLNCS** class (`llncs.cls`): six-model
  comparison across **two** datasets (SKAB + SWaT), the dataset-dependent-winner
  finding, a lightweight LSTM-autoencoder deep baseline, concept-drift
  adaptation, and the edge/constrained-hardware deployability result. Written in
  the impersonal academic voice (no first-person "we/our"). Built entirely from
  this repo's real results. Superseded by `paper2_revised.tex` (kept as the submitted, rejected version).
- `paper.tex` — the earlier single-dataset draft (SKAB only, first-person
  voice, IEEE format). Kept for history; superseded by `paper1.tex`. The IEEE
  version of `paper1.tex` is preserved in git history (before the LLNCS reformat).

## How to compile (easiest: Overleaf)

1. Go to [overleaf.com](https://www.overleaf.com), **New Project → Upload
   Project**, and upload **`overleaf_paper1.zip`** (in this folder). It is
   self-contained.
2. The ZIP already bundles everything: `paper1.tex`, the Springer class
   (`llncs.cls`) and bib style (`splncs04.bst`), and all **12 figures**. Because
   `llncs.cls` is bundled, it compiles even if the Overleaf project template
   lacks it. `paper1.tex` sets `\graphicspath{{figures/}{./}}`, so figures next
   to the `.tex` are found.
3. Set the compiler to **pdfLaTeX** and Recompile (it runs twice to resolve
   citations/figure numbers). The bibliography is a manual `thebibliography`, so
   **no BibTeX pass is needed**.

The 12 figures used are:
- `model_comparison.png`, `confusion_matrices.png`, `timeseries_anomalies.png`,
  `anomaly_closeup.png`  (SKAB)
- `swat_model_comparison.png`  (SWaT comparison)
- `cross_dataset_comparison.png`  (winner-flip)
- `deep_recon_error.png`, `deep_training_loss.png`  (deep LSTM-AE baseline)
- `swat_drift_motivation.png`, `swat_drift_adaptation.png`, `drift_adaptation.png`
  (concept drift)
- `edge_benchmark.png`  (deployability)

Local compile is only possible with a TeX distribution installed (MacTeX /
TeX Live): `pdflatex paper1.tex` twice from the `overleaf_paper1/` folder.

## What's real vs. placeholder

- **All numbers, tables, and figures are real** — from `results/metrics.json`,
  `results/swat_metrics.json`, `results/swat_drift.json`,
  `results/concept_drift.json`, `results/deep_baseline.json`,
  `results/edge_benchmark.json`, and the figures in `results/figures/` (all
  regenerable via `python main.py`, `python -m experiments.swat_comparison`,
  `... .swat_drift`, `... .concept_drift`, `... .deep_baseline`,
  `... .edge_benchmark`, and `... .make_figures`).
- **The two dataset-excerpt tables** (`tab:skabhead`, `tab:swathead`) are real
  rows read from the source CSVs. SWaT's absolute `Timestamp` column is shown as
  a relative time index `t` (seconds) on purpose, so the collection date is not
  surfaced in the paper.
- **All 28 citations are real and verified** — see `../REFERENCES.md` for the
  original 12-entry pool; the list was later expanded to 28 (SWaT, concept drift,
  KS test, MCC/Powers, LSTM, Adam, PyTorch, NumPy, pandas, DBSCAN, ADWIN,
  Sakurada autoencoders, WADI). `rotating2023`'s author list was confirmed
  against the published Heliyon record on 2026-07-16. Nothing outstanding.
- **SWaT citation note:** the data was obtained via a public Kaggle re-upload for
  development; cite the official iTrust/SUTD source (`goh2016`/`mathur2016`) and
  observe its terms for any formal submission.
- **Author block is filled:** Sarjan Shah and Abhi Shah, Department of
  Information Technology, Lok Jagruti University, Vadodara, Gujarat, India.

## Before submitting

- Confirm `rotating2023`'s authors against the published PDF.
- Add ORCID iDs to the author block if available (`\orcidID{...}` in LLNCS).
- Have both authors read it; the prose is meant to be humanised/adjusted to taste.
- If the venue changes from ICACS/Springer, reformat to that venue's template.
