# Deliverable 10: Final independent assessment

Date: 2026-10-28. No acceptance is guaranteed or predicted.

## What improved
- **Correctness:** the SWaT split error was found and fixed; inconsistent threshold calibration replaced by two explicit rules applied to every detector; Elliptic Envelope now actually robust; the invalid in-container accuracy number removed; latency measured per reading.
- **Rigour:** threshold-free metrics, event recall, detection delay, block-bootstrap CIs, seeds, paired differences, three added baselines, both SKAB protocols, ablations.
- **Honesty of claims:** every original headline claim was re-tested; four were withdrawn because they did not survive (dataset-specific winners, deep winner on SWaT, label-free adaptation benefit, sub-ms latency).
- **Literature:** 39 references; the argument is built on the 2021-2024 benchmark/critique literature the original ignored.
- **Reproducibility:** one command per experiment, results JSON, self-checking metric code, Docker image for the cost benchmark.

## What the paper now demonstrates
1. On SWaT's standard split, ten detectors are statistically tied in ranking ability (AUROC 0.825-0.858).
2. Protocol choices (split semantics, training protocol, threshold rule) change MCC by up to ~0.8 and change the winner.
3. Normal-week → attack-week distribution shift breaks label-free thresholds for LOF, PCA and the LSTM-AE but not for z-score, OCSVM, IF.
4. Genuine normal operation drifts enough to break a static Isolation Forest; retraining helps only with clean (labelled) data; drift triggering ≈ periodic retraining at the same budget; label-free retraining fails.
5. Single-reading costs span µs to >100 ms in a 1-CPU / 1 GB container.

## What still limits the contribution
- **Novelty is modest and methodological.** Protocol-sensitivity critiques exist (Kim 2022, Wu & Keogh 2023, Liu & Paparrizos 2024, Garg 2021). Our specific angles are narrower. A selective ML venue may judge it incremental.
- **SWaT results rest on a reconstructed re-upload** (24 of 25 continuous sensors). Must be rerun on the official iTrust release before submission. This is the single most important pending item.
- **Two water datasets only.**
- **Drift study is one stream**, no CIs across streams, two detectors, one window length.
- **Container ≠ device.**
- **Literature search** was web-search based; papers were not read in full (only abstracts/records). Authors must read the key ~6 papers in full before submission (listed in `03_literature_review.md` §1).
- **No local LaTeX engine**: the manuscript has not been compiled; compile on Overleaf and fix any errors.

## Reviewer criticisms: status
| Criterion | Status |
|---|---|
| References | Addressed (39 verified, argument-driven) |
| Novelty | Partly addressed (reframed honestly; still modest) |
| Depth | Addressed (CIs, seeds, baselines, protocols, drift baselines) |
| Critical qualities | Addressed (withdrawn claims, threats to validity) |
| Technical correctness | Addressed, pending official-SWaT confirmation |
| Theory | Partly addressed (MCC/F1 and prevalence analysis, KS effect size; no formal analysis) |
| Language | Rewritten; needs a human read-through |
| Clarity of conclusions | Addressed (RQs answered directly) |
| Scholarly quality | Improved; depends on full reading of key papers |
| Topic relevance | Unchanged (was fine) |

## Must do before submission (in order)
1. Request official SWaT from iTrust (https://itrust.sutd.edu.sg/itrust-labs_datasets/) with the university email; rerun `experiments.revision.swat_revised` and `drift_revised` on it; update numbers.
2. Prof. Shah reads the revised paper and approves the reframing (it changes the research question).
3. Compile on Overleaf; convert to the target venue template (IJPHM: its LaTeX template, APA references).
4. Read in full: Audibert 2022, Sarfraz 2024, Kim 2022, Garg 2021, Li 2019 (MAD-GAN), Deng & Hooi 2021; adjust any sentence that misstates them.
5. Optional but valuable: a third dataset from another machine class (HAI is public).

## Was the work original vs. revised kept separate?
Yes. Original code/results/paper unchanged (one backward-compatible flag added to `src/features.py`). All revised material is under `experiments/revision/`, `results/revised/`, `paper/paper2_revised.tex`, `revision/`.

## Note on the SKAB drift experiment
The original SKAB drift result (MCC 0.00 → 0.106) was not carried into the revised paper: the static detector had no skill even on the regime it was trained on (MCC −0.027), and retraining used ground-truth labels, so the experiment cannot show adaptation. This is a negative result recorded here, not hidden.
