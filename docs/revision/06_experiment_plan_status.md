# Deliverable 6 (status): what was run

Plan and rationale: `01_02_04_05_06_stage1_audit.md` §5. Status as of 2026-10-28:

| ID | Experiment | Status | Output |
|---|---|---|---|
| E1 | SWaT standard protocol (+ original layout with same code) | **Performed** | `results/revised/swat.json` |
| E2 | Consistent per-model thresholds (label-free, val-MCC, val-F1) + AUROC/AUPRC | **Performed** | skab.json, swat.json |
| E3 | IF 10 seeds; LSTM-AE 3 seeds (SKAB) / 1 (SWaT); block bootstrap CIs; paired differences | **Performed** | skab.json, swat.json |
| E4 | Retraining policies × data selection on SWaT genuine normal week + attack period (IF, EE) | **Performed** | `results/revised/drift.json` |
| E5 | Single-reading latency, throughput, size, load time, peak RSS in 1-CPU/1 GB ARM64 container | **Performed** | `results/revised/edge.json` |
| E6 | ECOD, COPOD, HBOS baselines | **Performed** | all JSONs |
| E7 | Ablations: elapsed-time feature, features before/after dropping rows | **Performed** | skab.json `ablation_auroc` |
| E8 | SKAB reference (per-run, first 400 rows) protocol | **Performed** | skab.json `skab_benchmark_protocol` |
| E9 | Third dataset (HAI 21.03) | **Performed** | `results/revised/hai.json` |
| E10 | Official SWaT from iTrust | **Pending — requires Sarjan/Prof. Shah to request** | — |
| — | VUS-PR, range-based P/R | Not done (stated as limitation) | — |
| — | Drift study with multiple streams / CIs, ADWIN, threshold-only adaptation | Not done (limitation) | — |

"Independently reproduced": all revised experiments were run once by the scripts listed; none has been rerun by a second person. Original-paper table values were verified against the original JSON files (Stage 1).
