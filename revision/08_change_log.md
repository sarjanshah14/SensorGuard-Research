# Deliverable 8: Change log (original → revised)

Original: `Shah_ICACS2026_Unsupervised_Anomaly_Detection.pdf` / `paper/paper1.tex` (unchanged).
Revised: `paper/paper2_revised.tex`. Revised numbers: `results/revised/*.json`. Original numbers: `results/*.json` (unchanged).

## A. Scientific changes

| # | Original passage / claim | Problem found | Revised approach | Evidence | New experiment? |
|---|---|---|---|---|---|
| S1 | SWaT "normal block of 395,298 rows… a further block in which seven channels were not recorded is dropped" (§3.2) | That block is the normal rows of the **attack period**; the dropped block is the genuine 7-day normal run (duplicated in the re-upload) | Reconstructed standard split: train on normal run (minus 21,600 start-up rows), test on full 449,919-row attack period | Raw-file inspection (session 23); MAD-GAN / GDN protocol | Yes (E1) |
| S2 | "about 1.44 million rows" | Duplicates in re-upload | 496,800 normal + 449,919 attack-period rows | Timestamp de-duplication | No |
| S3 | SWaT winner Elliptic Envelope MCC 0.766; LSTM-AE best 0.841 | Artefacts of split + shared threshold | All 10 detectors tied in AUROC (0.825-0.858); EE label-free MCC 0.008; LSTM-AE 0.134 (LF) / 0.266 (VM); z-score 0.716 (LF) best | `swat.json` | Yes |
| S4 | Contamination 0.35 "interior optimum" (SKAB), 0.05 shared (SWaT) | SKAB tuned by F1 on pre-leak-fix features; SWaT tuned on IF only and shared | Per-model thresholds by two explicit rules (label-free 99th pct; validation-MCC); F1 rule only as sensitivity | `common.py` | Yes (E2) |
| S5 | Threshold-dependent metrics only, single run | No uncertainty, no ranking metric | AUROC, AUPRC, event recall, delay; block-bootstrap 95% CIs (files / hours); IF 10 seeds; LSTM-AE 3 seeds (SKAB); paired differences | `skab.json`, `swat.json` | Yes (E3) |
| S6 | SKAB: IF best (MCC 0.147) | Pooled protocol near chance; SKAB's own protocol not reported | Pooled: best AUROC 0.575 [0.504, 0.645], ties; SKAB per-run protocol: run AUROC up to 0.794, ranking inverted | `skab.json` | Yes (E8) |
| S7 | Six detectors | Missing standard fast baselines | Added ECOD, COPOD, HBOS (PyOD) | — | Yes (E6) |
| S8 | Elliptic Envelope "robust MCD" | `support_fraction=1.0` disables robustness | Default (robust) MCD | `common.py` | Yes |
| S9 | `time_since_start` feature; features built after dropping fault rows | Train/test range mismatch; windows bridge removed rows | Removed; features built on full recordings. Ablation: ≤0.02 AUROC change | `skab.json` ablation | Yes (E7) |
| S10 | "SWaT normal operation drifts, FP 8% → 100%" | Stream was attack-period normal rows | Streamed genuine normal week: static IF FP 12% (first day) → 76% (last day); EE stable (0.9%) | `drift.json` | Yes (E4) |
| S11 | "Lightweight KS adaptation restores skill (MCC 0.24→0.59), label-free statistic" | Retraining used ground-truth labels; no baselines | Compared static / KS / periodic (matched budget) / always × oracle / predicted-normal / all-rows. Oracle: KS 0.488 ≈ periodic 0.456 < always 0.537 (static 0.167). Label-free: collapse (MCC ≈ 0) or absorbed attacks (0.19-0.30) | `drift.json` | Yes (E4) |
| S12 | SKAB drift (MCC 0.00 → 0.106) | Static had no skill even on seen regime; oracle labels | Dropped from revised paper (not interpretable); recorded here and in Deliverable 10 | `results/concept_drift.json` | No |
| S13 | "Inference below 1 ms per reading (IF 0.0019 ms)" | Batch time / rows, not single-reading latency | Single-reading p50/p95/p99 over 1,000 calls in capped container: IF ≈ 2.3-2.8 ms; ECOD/COPOD > 100 ms | `edge.json` | Yes (E5) |
| S14 | "Container reproduces SWaT ordering (EE 0.81 vs IF 0.66)" | Test set contained training rows | Removed; container measures cost only | `edge_revised.py` docstring | Yes |
| S15 | "Raspberry-Pi-class" / "edge-deployable" | Overstates a capped Docker container | "one-CPU, 1 GB ARM64 container on an Apple M4 host; not a physical device; latencies are lower bounds" | — | No |
| S16 | Mechanism "SWaT's 51 correlated sensors favour covariance models" | Untested; 25 sensors used | Withdrawn; replaced by measured threshold-transfer behaviour | `swat.json` | No |
| S18 | Two datasets, both water | Reviewer depth/generalisation; "both water systems" limitation | Added HAI 21.03 (turbine, boiler, water treatment; HIL testbed), same protocol. Ranking differs from SWaT (PCA/LOF lead 0.93/0.92; EE 0.66); label-free thresholds transfer (FAR 0.4-1.6%); only z-score and OCSVM dependable on both SWaT and HAI | `hai.json` | Yes (E9) |
| S19 | "Periodic retraining with the same number of retrains as KS" (counts actually differed: 74 vs 80, 52 vs 10) | Period was set from KS's successful refits; skipped refits (<1,000 rows) were not distinguished from attempts | Attempts vs refits reported separately; periodic swept over P = 1,2,3,4,6,12,24,48 to give a budget curve. IF/oracle: KS 0.488 (74 refits) ≈ periodic at 120 refits; KS checks cost 96 s vs 19 s for every-batch refits | `drift.json` (old: `drift_v1_unmatched.json`) | Yes |
| S20 | Peak RSS 1.10–1.13 GB inside a 1 GB container; 533 MB floor for every detector | Forked child processes count pages shared with the parent | Spawned processes; baseline RSS + per-model peak increase (VmHWM reset) + cgroup memory.peak/events. Container peaked at the 1 GB limit (3,127 reclaim events, 0 OOM kills) only for ECOD/COPOD | `edge.json` (old: `edge_v1_forked.json`) | Yes |
| S21 | Title "Evaluation Protocol Decides the Winner…lightweight…" | Too assertive; "lightweight" contradicted by ECOD/COPOD memory | "Protocol Sensitivity in Unsupervised Industrial Anomaly Detection: Threshold Transfer, Retraining Risk, and Inference Cost" | external review | No |
| S22 | "Our search found no study comparing drift-triggered with periodic retraining" | Prior work exists (Poenaru-Olaru et al., CAIN 2024; Dasari 2026 preprint) | Cited; contribution narrowed to label assumption in retraining-data selection on ICS data | literature search | No |
| S23 | Wide tables broken across the two Word columns | Layout | Tables and figures placed after the references in a one-column section (allowed by IJPHM template); table captions below tables | Word build | No |
| S17 | Contribution: dataset-dependent winner; deep not always better | Already established (Audibert 2022; Sarfraz 2024) | Reframed: protocol sensitivity + retraining-label risk + measured cost; prior work credited | `04_novelty_and_research_gap.md` | No |

## B. Editorial changes

| # | Original | Revised | Reason |
|---|---|---|---|
| E1 | Title "Comparative and Adaptive Evaluation… A Cross-Dataset and Edge-Deployability Study" | "Evaluation Protocol Decides the Winner: Threshold Stability, Retraining Risk and Cost of Lightweight Unsupervised Anomaly Detectors for Industrial Sensor Data" ("re-examination" dropped: the earlier version was never published) | Matches the actual contribution; drops "cross-dataset" (no transfer was tested) and "edge" (no device) |
| E2 | 28 references, many tool/1950s-2000s method cites | 39 references; added 2018-2024 benchmark, critique, metric, drift and edge literature; tool cites moved to availability statement | Reviewer reference score 4/10 |
| E3 | Related Work = list of methods | Organised as argument: detectors → what benchmarks agree on → where evaluations go wrong → drift → cost → gap | Reviewer theory/depth scores |
| E4 | Equations definitional only | MCC equation now used to explain F1 of flag-everything (2π/(1+π) = 0.46 on SKAB) and precision's dependence on prevalence | Theory score |
| E5 | Abstract 330 words, results repeated in four places | Single statement per finding; RQs answered once in Conclusion | Language/clarity scores |
| E6 | "roughly halves false alarms (73% → 24%)" | Not carried over (finding withdrawn); arithmetic noted (that was −67%) | Accuracy |
| E7 | Figure captions said "six models" with seven-row tables | All captions describe exactly what is plotted | Accuracy |
| E8 | SWaT "public re-upload… should be cited… for any formal release" (TODO-like sentence) | Dataset section states exact copy, its defects, reconstruction, and that results need confirmation on the official release | Integrity |
| E9 | LLNCS format | Venue-neutral `article` + author-year citations (IJPHM uses APA); swap template when venue confirmed | Venue change |
| E10 | No limitations structure | "Threats to validity" section with eight specific items | Critical-qualities score |
