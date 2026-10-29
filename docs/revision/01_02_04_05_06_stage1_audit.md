# Revision Stage 1: Diagnosis, Audit, Reviewer Matrix, Novelty, Experiment Plan

Manuscript audited: `Shah_ICACS2026_Unsupervised_Anomaly_Detection.pdf` (ICACS-2026 Paper ID 580, rejected 56/100 and 49/100).
Audit date: 2026-10-28. Auditor: Claude (Opus 5.5), reading the PDF, all code in `src/` and `experiments/`, all `results/*.json`, and the raw SWaT/SKAB files.
Status legend: VERIFIED (checked against code/data), REPRODUCED (re-run), PENDING, UNRESOLVED.

---

## 0. Artifact inventory

| Artifact | Present | Notes |
|---|---|---|
| Submitted PDF | yes | repo root |
| LaTeX source | yes | `paper/paper1.tex` (LLNCS) |
| Pipeline code | yes | `src/*.py`, `main.py` |
| Experiment scripts | yes | `experiments/*.py` (11 scripts) |
| Result files | yes | `results/metrics.json`, `swat_metrics.json`, `deep_baseline.json`, `concept_drift.json`, `swat_drift.json`, `edge_benchmark.json` |
| Figures | yes | `results/figures/*.png` (13) |
| SKAB data | yes | `SKAB/` (35 CSVs, 46,860 rows) |
| SWaT data | yes (unofficial) | Kaggle re-upload `vishala28/swat-dataset-secure-water-treatment-system` v3, cached under `~/.cache/kagglehub` |
| Official SWaT (iTrust) | **no** | must be requested from iTrust/SUTD |
| Reviewer comments | **no** | only numeric scores were provided |
| Docker config | yes | `docker/Dockerfile.edge` |
| Random seeds | fixed single seed 42 | no repeated runs anywhere |

Every number in the paper's tables was checked against `results/*.json`. **All match** (VERIFIED). The problems below are in experimental design and data handling, not arithmetic.

---

## 1. Deliverable 1: Initial research diagnosis

### What the manuscript actually demonstrates
1. On SKAB, under one specific pooled-training protocol, six classical detectors reach near-chance discrimination (best MCC 0.147, Isolation Forest).
2. F1 can reward a degenerate "flag-almost-everything" detector, and MCC/balanced accuracy expose it. This is correct but already established in the literature (Garg et al. 2021; Kim et al. 2022).
3. A pipeline of lightweight detectors fits inside a 1-CPU / 1 GB ARM64 Docker container on an Apple-Silicon host, with batch-amortised inference below 1 ms per row and peak RSS 484 MB.
4. A clean, readable, reproducible code base with one documented and removed leak (wall-clock features).

### What it claims beyond that (gaps between claim and evidence)
| Claim | Problem |
|---|---|
| SWaT results (Elliptic Envelope MCC 0.766; LSTM-AE 0.841) | Computed on a **mis-identified data split** (see A1). Not comparable to the literature and not the protocol described in the paper. |
| "SWaT normal operation drifts; false alarms rise 8% to 100%" | The streamed "normal" data is the normal-labelled rows **of the attack period**, interleaved with 36 attacks. The drift may be attack after-effects, not normal operational drift. Unsupported as stated. |
| "Label-free" drift adaptation | Only the trigger is label-free. Retraining selects rows using ground-truth labels (`anomaly == 0`). |
| Adaptation "restores skill" | No comparison to periodic retraining; on SWaT the detector retrained in 25 of 34 batches, so it is effectively periodic. |
| "Cross-dataset" findings | Two separate within-dataset evaluations, not cross-dataset generalisation. The sensor-correlation mechanism is offered without a test. |
| "Inference below 1 ms per reading" | Measured as batch time divided by batch size (vectorised throughput), not single-reading latency. One run, no repeats. |
| "Constrained run reproduces the SWaT ordering" | The in-container SWaT test set contains the same normal rows the models were trained on (`experiments/edge_benchmark.py:129-136`). Invalid. |
| "about 1.44 million rows" (SWaT) | Inflated by duplicated rows in the re-upload. Unique rows ≈ 946,719 (496,800 normal + 449,919 attack-period). |

### What the paper does well
- Leakage awareness (per-file causal features, documented clock-feature leak).
- Correctly identifies the F1 degeneracy and reports MCC/balanced accuracy.
- Honest limitations paragraph and a fully reproducible code base.
- Resource measurements under explicit CPU/RAM caps (cgroup-verified 1.0 CPU).

### Likely causes of the scores (evidence-based inference; reviewers gave no comments)
- **References 4/10 (both):** 28 refs; only ~4 are domain papers from 2019 or later. Tool citations (NumPy, pandas, PyTorch, Adam, Docker in *Linux Journal*) pad the list. None of the central benchmarking/critique literature is cited (Audibert 2022, Wu & Keogh 2021, Kim 2022, Schmidl 2022, Garg 2021, Sarfraz 2024, Liu & Paparrizos 2024). No modern SWaT method (GDN, USAD, TranAD) is cited although the paper evaluates on SWaT.
- **Novelty 4-5:** "No single model wins across datasets" and "deep is not always better" are already the headline findings of Audibert et al. 2022 and Sarfraz et al. 2024. Without engaging them, the paper appears unaware of them.
- **Depth / critical qualities 4-5:** single seed, single split, no uncertainty, no threshold-free metrics, no event-wise metrics, drift without baselines, SKAB numbers far below SKAB's own published leaderboard (Conv-AE F1 0.78) with no explanation.
- **Theory 5:** equations are definitional (z-score, KS, MCC) and not used to explain results.
- **Language 5-6:** long, repetitive paragraphs; claims repeated in abstract, intro, discussion and conclusion.

---

## 2. Deliverable 5: Methodological audit (with evidence)

### A. Data provenance and preprocessing

**A1. SWaT split is mis-identified (CRITICAL, VERIFIED).**
Evidence from the raw Kaggle files:
- `normal.csv` = 1,387,098 rows. The 395,298-row "fully recorded" block that the paper trains on spans **2015-12-28 10:00 to 2016-01-02 14:59**, which is the **attack-period recording**. It has exactly 36 gaps (54,738 s total), and those gaps are where the 54,621 rows of `attack.csv` sit. 395,298 + 54,621 = 449,919, the size of the official attack file. Zero timestamp overlap between the two.
- The remaining 991,800 rows (2015-12-22 16:00 to 12-28 09:59) are the **genuine normal run, duplicated** (495,000 duplicate timestamps; 496,800 unique), with 7 channels blank (MV101, AIT201, MV201, P201, P202, P204, MV303). The paper discarded this block.
- `attack.csv` contains only attack-labelled rows.

Consequences:
1. The model was trained on normal intervals **between attacks**, including post-attack recovery periods. Training ends 2016-01-01 04:35, while validation attacks span 12-28 to 12-31 and test attacks span 12-31 to 01-02. Training data and evaluation attacks are interleaved in time.
2. Test anomaly rate is 31.5% (59,295 test-normal + 27,311 attack rows), versus 12.1% in the standard protocol.
3. Rolling features bridge the removed attack gaps in the normal block, and bridge separate attacks in the attack-only block (each attack's first rows inherit the previous attack's tail).
4. The paper's descriptions ("a long run of normal operation followed by a run with 36 attacks"; "a further block in which seven channels were not recorded is dropped") misdescribe what was used.
5. The SWaT row count "~1.44 million" is wrong (duplicates).

Standard protocol in the literature (e.g. Li et al. 2019 MAD-GAN, arXiv:1809.04758; Deng & Hooi 2021): train on the 7-day normal run (496,800 rows), drop the first 21,600 rows (start-up stabilisation), test on the full 449,919-row attack-period file in time order.
**Action:** re-run all SWaT experiments under the standard protocol (Category C). With this copy only 24 continuous sensors are usable (AIT201 is blank in the normal run). Request official SWaT from iTrust (Category D).

**A2. `time_since_start` creates a train/test distribution mismatch (MEDIUM, VERIFIED, effect PENDING).**
SKAB: training values reach 9,960 s (anomaly-free run), while test tails restart at 0 and never exceed 291 s. SWaT: training reaches ~276,000 s and every test segment restarts at 0. The feature is not causal-meaningful at test time. **Action:** ablate.

**A3. Rolling windows span removed rows (LOW-MEDIUM, VERIFIED).** SKAB training features are built *after* dropping anomalous rows (`main.py:59-67`), so 5 s/30 s windows and first differences join pre-fault and post-fault normal rows across the removed fault. **Action:** compute features on the full run, then drop anomalous rows.

**A4. SKAB protocol differs from the SKAB benchmark (HIGH for credibility).** The SKAB leaderboard (github.com/waico/SKAB) reports outlier-detection F1 0.78 (Conv-AE, MSET) with per-run evaluation. This paper pools 35 runs with different operating baselines into one model, which likely explains near-chance scores. The paper never mentions the leaderboard. **Action:** run a per-file protocol variant as defined in the SKAB repository notebooks (protocol to be confirmed from the notebooks before running), report both, and explain.

### B. Fairness of comparisons and thresholds

**B1. Inconsistent threshold rules (HIGH, VERIFIED).**
- SKAB classical: contamination 0.35 chosen by **F1** (`experiments/hyperparameter_sweep.py:64`), on a feature set that still contained the leaky clock features (sweep predates their removal, DEVLOG session 7). The paper argues F1 is the wrong selector.
- SWaT classical: chosen by **MCC**, but tuned on Isolation Forest only and copied to all models (`experiments/swat_comparison.py:89-104`), which disadvantages OCSVM/LOF/PCA/z-score.
- LSTM-AE: threshold tuned per model by **MCC** (`src/deep_baseline.py:143`).
- Contamination 0.35 means each model is calibrated to flag 35% of *normal training rows*: high false-alarm rate by construction.

**B2. Threshold calibration uses labelled anomalies (MEDIUM).** All validation selection uses attack/fault labels. That is semi-supervised threshold calibration, which the paper must state; a label-free alternative (quantile of validation-normal scores) should be reported alongside.

**B3. Single run, no uncertainty (HIGH).** Seed 42 only. Isolation Forest and the LSTM-AE are stochastic. MCC gaps such as 0.147 vs 0.086 are untested. Test rows are autocorrelated, so CIs must use **block bootstrap by file (SKAB, 34 files) or by attack event (SWaT)**, not i.i.d. rows.

### C. Metrics
- Only point-wise, threshold-dependent metrics. Missing: AUROC, AUPRC (or VUS-PR, Paparrizos 2022; Liu & Paparrizos 2024 recommend VUS-PR), event-wise recall / detection delay, false alarms per hour.
- MCC/balanced accuracy choice is sound and should stay.
- Fig. 1/5 captions say "six models"; tables contain seven rows.

### D. Model-specific
- Elliptic Envelope is fitted on 10 whitened PCA components with `support_fraction=1.0`. That turns off MCD robustness (it becomes a plain Gaussian fit). The paper describes it as robust MCD. **Action:** correct the text or re-run with the default support fraction.
- LOF stores the full training set (13.9 MB); this is included in the size measurement (good).
- LSTM-AE: per-row score is the MSE of the causal window ending at that row; scores are reconstruction errors, not probabilities (paper is fine here). SKAB validation MCC 0.178 vs test −0.045 shows a validation/test protocol mismatch (validation = whole held-out files, test = file tails).

### E. Cross-dataset
Separate train/test within each dataset. No transfer. Rename to "multi-dataset comparison". The causal story (sensor correlation favours covariance models) is a hypothesis; a cheap test is to compare per-dataset mean absolute inter-sensor correlation and the gain of Elliptic Envelope vs Isolation Forest under controlled sensor subsets (optional).

### F. Concept drift
- SKAB: retraining rows filtered by ground-truth labels (`experiments/concept_drift.py:205`). Static model has MCC −0.027 on the *seen* regime too, so there is no skill to "restore". Whole-stream recall drops 0.679 → 0.459 (not reported in paper). SKAB drift result is not interpretable as recovery.
- SWaT: stream is normal rows of the attack period (A1). All streamed rows are known-normal by construction (oracle). 25 retrains in 34 batches. Evaluation pairs each batch with the same 3,000 attack rows: fine as a controlled probe.
- Missing baselines: periodic retraining (same budget), always-retrain, ADWIN/Page-Hinkley trigger, and label-free retraining (all recent rows, or rows predicted normal) to expose contamination risk.
- Adaptation tested only on Isolation Forest, not on the SWaT winner.
- Prediction precedes retraining in both scripts: **causal, no future leakage** (VERIFIED, good).

### G. Edge claims
- Environment: Docker on Apple-Silicon macOS host, `--cpus=1 --memory=1g`, linux/arm64, python 3.12-slim. This is a **resource-capped container, not an edge device**. The paper says so in Sec. 4.8, but the abstract/title wording ("Raspberry-Pi-class", "edge-deployability") overstates it.
- Latency = total batch predict time / rows (`edge_benchmark.py:95-97`). Single-row latency (the real-time case) will be much higher because of per-call Python/sklearn overhead. No repeats, no cold start, no model-load time.
- Peak memory = `ru_maxrss` of the whole run (reasonable; report as process peak RSS).
- SWaT in-container MCC evaluates on training rows (invalid, remove or fix).
- No energy measurement; do not claim energy efficiency.

### H. Numerical consistency
- All table values match JSON (VERIFIED).
- "roughly halves false alarms" (73% → 24%) is a 67% relative reduction: wording wrong (understates).
- "51 SWaT sensors move together" (Sec 5.3) but only 25 were used.
- "36 labelled attacks": the re-upload's attack rows form 35 contiguous segments (two attacks adjacent); official count 36 (some versions list 41). Cite the source count.

---

## 3. Deliverable 2: Reviewer-response matrix

| Reviewer concern (R1/R2) | Likely underlying issue | Evidence in manuscript | Required action | New experiment? | Verification | Status |
|---|---|---|---|---|---|---|
| Topic relevance 8/7 | Fit fine; framing too broad | Title has 3 topics | Narrow title/RQs | No | Re-read | PENDING |
| Scholarly quality 6/5 | Claims exceed evidence; SWaT mis-split | A1, F, G | Fix data, qualify claims | Yes (E1) | Re-run, ledger | PENDING |
| Language 6/5 | Repetition, long paragraphs | Abstract 330 words; results repeated 4× | Rewrite tightly | No | Word counts, read-through | PENDING |
| Use of theory 5/5 | Equations definitional, not explanatory | Eqs 1-3 | Add score/threshold theory, base-rate effect on precision, KS effect size vs sample size, why covariance models suit correlated plants | No | Each equation used in a result | PENDING |
| Novelty 5/4 | Main findings already known | No citation of Audibert 2022 / Sarfraz 2024 | Reframe to protocol-sensitivity + deployment-cost study; position against prior work | Yes (E2, E4) | Novelty table | PENDING |
| Technical correctness 6/6 | SWaT split, threshold inconsistency, drift oracle labels, latency definition | A1, B1, F, G | Fix all; report both protocols | Yes (E1-E5) | Re-run | PENDING |
| Critical qualities 5/4 | No uncertainty, no baselines for drift, SKAB leaderboard gap unexplained | B3, F, A4 | CIs, drift baselines, SKAB-protocol run | Yes | Block-bootstrap CIs | PENDING |
| Clarity of conclusions 6/5 | Four "outcomes" each overstated | Conclusion | Answer RQs directly, scoped | No | Claims ledger | PENDING |
| Depth 5/4 | 1 seed, point metrics only, no modern baselines | Tables 3-4 | AUROC/AUPRC/VUS-PR, event metrics, seeds, PyOD baselines (ECOD, COPOD, HBOS, kNN) | Yes | Tables with CIs | PENDING |
| References 4/4 | Dated, padded, misses core literature | Ref list | Rebuild around argument; verify every entry | No | Citation audit (`revision/LITERATURE.md`) | IN PROGRESS |

---

## 4. Deliverable 4 (initial): Novelty audit

| Current claimed contribution | Closest prior work | Already known? | Remaining gap | Verdict |
|---|---|---|---|---|
| Best detector is dataset-dependent | Audibert et al. 2022 (16 methods, 5 datasets: no family wins consistently); Schmidl et al. 2022 (71 algorithms) | Yes | None on its own | Drop as headline; keep as supporting observation |
| Deep is not always better | Audibert 2022; Sarfraz et al. 2024 (simple baselines match SOTA) | Yes | None | Drop as headline |
| F1 rewards flag-everything; use MCC | Garg et al. 2021 (trivial detectors); Kim et al. 2022 (PA inflation) | Yes | None | Keep as methodology, cite prior work |
| Leakage-free protocol | Wu & Keogh 2021, Kim 2022 (benchmark flaws) | Partly | Concrete demonstration that dataset layout of a public re-upload changes conclusions | **Candidate contribution** (protocol sensitivity) |
| Drift adaptation on commodity hardware | Gama 2014, Lu 2018 reviews; streaming AD (Ahmad 2017) | Mechanism known | Cost-vs-benefit of trigger vs periodic retraining, with label-free retraining, for lightweight ICS detectors, measured on capped hardware | **Candidate contribution** if E4 is done properly |
| Edge deployability | MLPerf Tiny (Banbury 2021) defines latency/energy benchmarking | Method known | Joint accuracy + resource table for classical detectors with single-reading latency | Supporting, not headline |

**Strongest defensible contribution (proposed):** an empirical, deployment-oriented audit showing how protocol choices (data-split definition, threshold-calibration rule, metric, and the label assumption in retraining) change the conclusions drawn about lightweight unsupervised detectors on SKAB and SWaT, together with their measured resource costs. This is methodological/empirical, not algorithmic. It must be stated as such.

**Novelty risks:** protocol-sensitivity studies exist (Kim 2022; Wu & Keogh 2021; Liu & Paparrizos 2024). The gap is narrow: lightweight ICS detectors, threshold-calibration and retraining-label assumptions, plus resource costs. A deeper search (Stage 2) must check for close duplicates before any "gap" wording.

### Proposed research questions
- **RQ1.** Under a standard, leakage-controlled protocol, how do lightweight classical detectors and a small LSTM-AE compare on SKAB and SWaT, measured with threshold-free metrics (AUROC, AUPRC/VUS-PR) and threshold-dependent metrics (MCC, event recall, false alarms per hour) with block-bootstrap uncertainty?
- **RQ2.** How much do conclusions change with protocol choices: SWaT split definition (re-upload layout vs standard), threshold rule (F1 vs MCC vs label-free quantile), and point vs event metrics?
- **RQ3.** Does KS-triggered retraining outperform periodic and always-retrain baselines at equal or lower retraining cost, and what happens when retraining data is selected without labels?
- **RQ4.** What are single-reading latency, batch throughput, model-load time, model size and peak memory for each detector in a 1-CPU/1 GB ARM64 container?

---

## 5. Deliverable 6: Prioritised experiment plan

| ID | Category | What | Why | Compute (Mac) | Status |
|---|---|---|---|---|---|
| E1 | C | **SWaT standard protocol**: train on genuine 7-day normal run (drop first 21,600 rows; 24 continuous sensors, AIT201 excluded), test on full 449,919-row attack-period file in time order; features computed on the full continuous recording | Fixes A1; makes SWaT comparable with literature | ~15-30 min (OCSVM dominates; can subsample OCSVM training with a stated cap) | PENDING |
| E2 | B/C | **Consistent thresholds + threshold-free metrics**: per-model threshold by (a) validation MCC, (b) label-free 99th/99.5th percentile of validation-normal scores; report AUROC, AUPRC, VUS-PR (if `vus` package installs) | Fixes B1, B2, C | minutes | PENDING |
| E3 | C | **Uncertainty**: 10 seeds for stochastic models (IF, LSTM-AE 5 seeds); block bootstrap CIs (SKAB by file, SWaT by attack event + normal blocks); paired bootstrap for top-2 differences | Fixes B3 | minutes-1 h | PENDING |
| E4 | C | **Drift redo**: on SWaT genuine normal run streamed forward (and attack file), compare static / periodic (same retrain count) / always / KS-trigger / Page-Hinkley trigger; retraining data (i) oracle-normal (ii) all recent rows (iii) rows predicted normal; for IF and Elliptic Envelope | Fixes F | ~30 min | PENDING |
| E5 | C | **Edge redo**: single-row latency p50/p95/p99 over 1,000 calls, batch throughput, model load (cold) time, 5 repeats, fixed SWaT in-container evaluation | Fixes G | ~10 min + Docker | PENDING |
| E6 | C | **Extra lightweight baselines** via PyOD: ECOD, COPOD, HBOS, kNN (new dependency `pyod`) | Depth; standard fast baselines | minutes | PENDING |
| E7 | B | Ablations: `time_since_start` on/off; features computed before dropping anomalous rows (A2, A3); Elliptic Envelope support_fraction default | Correctness | minutes | PENDING |
| E8 | C | SKAB per-file protocol (as defined in waico/SKAB notebooks) to compare with the leaderboard | Fixes A4 | minutes | PENDING |
| E9 | D | Third dataset (HAI, public GitHub) | Generalisation beyond water | hours | OPTIONAL |
| E10 | D | Official SWaT from iTrust (all 25 sensors, v1) | Provenance | external request | Sarjan/Prof. Shah to request |

Interpretation rules (fixed before running, to avoid outcome-driven framing):
- If E1 changes the SWaT ranking, the paper reports both protocols and makes the sensitivity itself a finding (RQ2).
- If KS-trigger does not beat periodic retraining at equal cost in E4, the drift section reports that plainly and the contribution becomes "drift is real and costly; simple periodic retraining suffices".
- If label-free retraining degrades detection (contamination), report it as the key deployment risk.
- No result is dropped because it is unfavourable.

---

## 6. Evidence ledger (key claims)

| # | Claim (paper location) | Evidence | Verifiable? | Type | Action |
|---|---|---|---|---|---|
| L1 | SKAB 46,860 rows, 35 files (Sec 3.1) | Counted | yes | data | keep (VERIFIED) |
| L2 | SKAB train 28,368 / test 7,503, 30.0% anomalous (Sec 4.2) | Counted | yes | data | keep (VERIFIED) |
| L3 | SWaT ~1.44 M rows (Sec 3.2, Concl.) | Re-upload duplicates | yes | data | **correct to ~0.95 M unique** |
| L4 | SWaT normal block 395,298 rows is "normal operation" (Sec 3.2) | It is attack-period normal rows | yes | data | **correct; re-run (E1)** |
| L5 | SKAB IF best, MCC 0.147 (Table 3) | metrics.json | yes | result | re-run with consistent thresholds + CIs (E2/E3) |
| L6 | SWaT EE MCC 0.766, LSTM-AE 0.841 (Table 4) | swat_metrics.json | yes, but wrong split | result | **re-run (E1)** |
| L7 | Contamination 0.35 "interior optimum" (Sec 4.5) | F1-selected, pre-leak-fix | yes | method | **re-tune (E2)** |
| L8 | Static FP 8% → 100% "genuine temporal drift" (Sec 5.4) | Attack-period stream | yes | result/inference | **re-run on genuine normal (E4); qualify** |
| L9 | Adaptation MCC 0.24 → 0.59 (Table 6) | swat_drift.json | yes | result | re-run with baselines (E4) |
| L10 | Label-free statistic triggers recovery (Sec 6) | Trigger yes, retraining uses labels | yes | inference | **qualify** |
| L11 | Inference < 1 ms per reading (Sec 5.5) | Batch-amortised | yes | result | **re-measure single-row (E5)** |
| L12 | Container reproduces SWaT ordering (Sec 5.5) | Evaluated on training rows | yes | result | **remove or fix (E5)** |
| L13 | Peak memory 484 MB (Sec 5.5) | ru_maxrss | yes | result | keep with definition |
| L14 | EE uses robust MCD (Sec 4.3) | support_fraction=1.0 | yes | method | **correct text or re-run (E7)** |
| L15 | "Deep detectors do not consistently outperform" (Sec 2) | cites [3,4]; Audibert 2022 is the right source | yes | citation | **re-cite** |
| L16 | False alarms "roughly halve" 73→24% | arithmetic | yes | wording | **fix (−67%)** |
