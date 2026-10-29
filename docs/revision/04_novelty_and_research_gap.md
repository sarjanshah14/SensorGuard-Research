# Deliverable 4: Research-gap and novelty report (final, after experiments)

## Original claims vs. what survives

| # | Original claim | Closest prior work | Overlap | Status after re-analysis |
|---|---|---|---|---|
| 1 | "Best detector is dataset-dependent" (IF on SKAB, EE on SWaT) | Audibert 2022; Schmidl 2022 | Already the headline of Audibert 2022 | **Withdrawn as a contribution.** And the specific winners did not survive: on SWaT's standard split all 10 detectors are statistically tied in AUROC (0.825-0.858, overlapping CIs); EE drops from MCC 0.766 to 0.008 (label-free). |
| 2 | "Deep model best on SWaT (MCC 0.84), below chance on SKAB" | Audibert 2022; Sarfraz 2024 | Known pattern | **Withdrawn.** LSTM-AE AUROC 0.852 ties the classical detectors on SWaT; its threshold fails under the normal-week to attack-week shift (MCC 0.134 LF, 0.266 VM). Below chance on SKAB pooled confirmed (0.442). |
| 3 | "Use MCC, not F1" | Chicco 2020; Garg 2021 | Known | Kept as method, cited, not claimed. |
| 4 | "Static detector degrades under genuine drift; KS-triggered retraining restores skill (MCC 0.24 → 0.59)" | Gama 2014; Lu 2018; Rabanser 2019 | Mechanism known | Original evidence invalid (attack-period stream, oracle labels, no baseline). **Re-tested**: see `results/revised/drift.json` and paper §6. |
| 5 | "Edge-deployable, < 1 ms per reading" | MLPerf Tiny 2021 | Method known | Original latency was batch-amortised; **re-measured** single-reading latency in capped container (`results/revised/edge.json`). Supporting evidence only. |

## Strongest defensible contribution
An empirical, methodological re-examination showing **how much protocol choices decide which lightweight unsupervised detector "wins"** on two public industrial benchmarks, with evidence:
1. **Split semantics:** a widely downloaded SWaT re-upload stores the attack period's normal rows in `normal.csv`; using it as-is inflates AUROC (0.84-0.93 vs 0.83-0.86) and reorders detectors (HBOS first → last).
2. **Threshold rule:** a single contamination tuned on one detector and shared made the z-score look useless on SWaT (MCC 0.18); per-model calibration gives 0.84 on the same data. Under label-free thresholds, normal-week → attack-week shift breaks LOF/PCA/LSTM-AE (68-73% false alarms) but not z-score/OCSVM/IF.
3. **Training protocol:** on SKAB, pooled training gives chance-level detection for all detectors; SKAB's per-run protocol gives run-level AUROC up to 0.79 and inverts the ranking.
4. **Retraining policy and label assumption** (drift study) and **measured single-reading cost** complete the deployment picture.

This is narrower than the original paper's claims and is honest about being empirical/methodological, not algorithmic.

## Revised research questions
- **RQ1** Under standard, documented protocols with uncertainty, do lightweight detectors differ in their ability to rank anomalies on SKAB and SWaT? → *Mostly no: ties within CIs on both.*
- **RQ2** How much do split definition, training protocol and threshold rule change measured performance and rankings? → *Enough to change the winner and to move MCC by up to 0.8.*
- **RQ3** Does drift-triggered retraining beat periodic or continuous retraining, and what does selecting retraining data without labels cost? → *see §6 of the paper.*
- **RQ4** What does each detector cost per reading under a 1-CPU / 1 GB cap? → *see §7.*

## Remaining novelty risks (stated plainly)
- Protocol-sensitivity papers exist (Kim 2022; Wu & Keogh 2023; Liu & Paparrizos 2024; Garg 2021). Our angle (split semantics of a re-upload, threshold-calibration rule, retraining-label assumption) is narrower; a reviewer may still judge it incremental.
- Our literature search was web-search based, not a database-level systematic review; a closer prior study may exist (especially on SWaT normal-vs-attack-week distribution shift, which some papers mention — e.g. pump state differences between SWaT train and test noted in ICS literature found in search #29).
- Two datasets, both water systems; no physical device.
- The SWaT findings come from a re-upload reconstructed to the standard split; must be confirmed on the official iTrust release.
