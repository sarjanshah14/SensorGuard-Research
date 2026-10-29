# SensorGuard-Research — End-to-End Gameplan

Living checklist from "no code" to "submitted paper." Checkpoints are
sequential within a phase; phases can overlap (e.g. paper writing can start
once Phase B results stabilize, in parallel with Phase C). Check boxes get
ticked as work completes — see `DEVLOG.md` for the detailed why/how behind
each one, and `PROGRESS.md` for the one-line-per-session index.

---

## Phase A — Data Foundation

### [x] A1. Verified SKAB data-loading pipeline
**What:** `src/preprocessing.py` — `load_anomaly_free`, `load_skab_all`,
`chronological_split`.
**Why it's a separate checkpoint:** Everything downstream (features,
models, evaluation) silently inherits any bug here. Getting it *verified*,
not just written, before building on top of it was the whole point of
session 1.
**Done:** Loads all 35 CSVs correctly (9,401 anomaly-free rows + 34 fault
runs / 37,459 rows). Split is per-source-file chronological 80/20 — caught
and fixed a real bug where filenames collided across `valve1`/`valve2`/
`other` folders before this was correct. Verified via `notebooks/01_explore.ipynb`.

### [x] A2. Reviewed SensorGuard (old hackathon repo) for ideas
**What:** Read `anomaly_ml.py`, `model_training.py`,
`generate_sample_anomalies.py` — not ported, just mined for ideas, per
Sarjan's instruction that 1.5-year-old code needs review, not reuse.
**Done:** Kept hour/day-of-week + rolling-stat feature ideas and the
deviation-magnitude anomaly taxonomy (useful for the paper's discussion
section). Rejected its data leakage habits and hardcoded hyperparameters.

---

## Phase B — Modeling

### [x] B1. Feature engineering
**What:** `src/features.py` — 43 causal features (raw value, rolling
mean/std @5s/30s, delta, per sensor; plus hour/day-of-week/time-since-start),
computed per source experiment so windows never cross between independent
SKAB runs.
**Done.**

### [x] B2. Four models + evaluation harness
**What:** `src/models.py` (Z-score, One-Class SVM, LOF, IsolationForest
behind one shared interface, identical scaled feature matrix, no labels
used in training) + `src/evaluate.py` (precision/recall/F1/confusion
matrix/timing → `results/metrics.json`) + `main.py` (end-to-end glue).
**Done.** First real numbers: IsolationForest F1 0.334 > LOF 0.247 >
One-Class SVM 0.254 > Z-score 0.230 — directionally right (IsolationForest
should win, per the paper's hypothesis) but all 4 are weaker than they
should be.

### [x] B3. Fix the training-contamination problem
**What:** Right now `main.py` trains on `anomaly-free.csv` (100% normal) +
the chronological 80% of every fault CSV pooled together — and that pooled
fault-train slice is itself ~37% anomalous, so the combined training set
ends up ~28% anomalous. Unsupervised/novelty methods (IsolationForest,
One-Class SVM, LOF) assume training data is *mostly* normal; 28%
contamination violates that assumption and is the most likely reason all 4
models' absolute scores are mediocre.
**Why it matters:** This is a methodology correctness issue, not a tuning
nicety — a reviewer would flag "you trained an unsupervised novelty
detector on data that's 28% anomalous" as a real flaw. Fixing it before
writing the Methodology section avoids rewriting that section later.
**Concrete plan:**
1. Re-run with training restricted to `anomaly-free.csv` only (drop the
   pooled fault-train slice from training entirely) — see if scores
   improve. This is the "textbook" unsupervised setup for SKAB.
2. As a second variant, try training on `anomaly-free.csv` + only the
   rows in the fault-train slice where `anomaly == 0` (i.e., filter out
   the known-anomalous rows from the fault CSVs' training portion, instead
   of dropping the fault CSVs' normal rows entirely — keeps more "normal
   under realistic operating conditions" data than anomaly-free alone).
3. Compare both variants' precision/recall/F1 against the current
   (contaminated) baseline already in `results/metrics.json` — keep
   whichever is most defensible, and record the comparison in `DEVLOG.md`
   since it's a methodology decision the paper needs to justify.
**Resolved:** Tried all 3 variants (`experiments/training_strategy_comparison.py`).
Training on `anomaly-free.csv` alone looked best by raw F1 (~0.46) but was a
false win — recall hit ~100% because the models started flagging almost
every fault-experiment row as anomalous (the clean baseline run's "normal"
doesn't match the fault rigs' "normal" operating conditions). The winner is
`anomaly-free.csv` + only the normal-labeled rows from each fault CSV's
80% training slice (0% contamination, more realistic operating diversity
than anomaly-free alone). `main.py` now uses this by default. New numbers:
IsolationForest F1 0.350 (up from 0.334), all 4 models improved, no
degenerate "flag everything" behavior. See `DEVLOG.md` session 3 for the
full comparison table.

### [x] B4. Hyperparameter sweep
**What:** `contamination`/`nu` (was fixed at 0.1) and LOF's `n_neighbors`/
Z-score's `threshold` were carried over as nominal defaults, never tuned
for SKAB.
**Resolved:** `experiments/hyperparameter_sweep.py` carves out a
validation slice via `file_holdout_split` (whole experiments held out, not
a time-slice — see note below) and sweeps each hyperparameter. First
attempt at the validation split used a time-slice instead and produced a
98%-anomalous validation set (SKAB runs have a sustained-fault block in
their back half; the slice landed inside it) — switched to holding out
whole files, which gave a representative 38% anomaly rate. Sweep results
required judgment, not blind argmax: IsolationForest's contamination sweep
had a genuine interior peak (0.35), but One-Class SVM's nu and LOF's
contamination swept all the way to sklearn's 0.5 ceiling without ever
turning over — the same "flag almost everything" failure mode from B3 —
and Z-score's top scorer (threshold=0.5) had recall=100%, the exact
signature of flagging everything. Adopted contamination=0.35 (from
IsolationForest's real peak) across all 3 contamination/nu-based models
for consistency, kept LOF's n_neighbors=20 (sweep gains beyond it were
marginal/noisy), and set Z-score threshold=2.0 (best non-degenerate
value). Numbers at this checkpoint: IsolationForest F1
0.350 -> 0.496, Z-score 0.248 -> 0.410, One-Class SVM 0.279 -> 0.410, LOF
0.257 -> 0.381 — all 4 improved, none show the flag-everything signature.
**NOTE: revised down by B5, which removed leaking wall-clock features —
see B5 for the final official numbers (IsolationForest F1 0.466).**

---

### [x] B5. Investigate legitimate score improvements (+ catch a leak)
**What:** Sarjan asked whether we could honestly raise the scores and, if
so, frame the before/after as part of the paper. Tried two principled
ideas; the honest outcome was a wash-to-slightly-down, and the process
caught a data leak.
**Resolved:**
1. **Temporal smoothing** (`src/postprocess.py`, `experiments/temporal_smoothing.py`):
   real faults are sustained, so isolated 1-2s anomaly predictions should be
   removable false alarms. Tuned `min_run` on the validation files. It
   helped IsolationForest on *validation* (k=8) but *lowered* its test F1
   (0.496 → 0.478, recall fell more than precision rose) and hurt the other
   3 outright. **Rejected** — the false alarms are sustained, not blips.
2. **Wall-clock features** (`experiments/feature_ablation.py`): removing
   `hour`/`day_of_week` lowered IsolationForest F1 by 0.03. That drop is the
   tell that they were **leaking** — the test set is the last 20% of the
   *same* files used in training, so within a file `hour` is ~constant and
   acts as a file-identifier the model exploits to recall that file's
   fault-propensity. **Removed them** (`build_features(include_clock=False)`
   is now the default); the honest, leakage-free numbers are lower but
   trustworthy.
**Official leakage-free test results (regenerated into results/metrics.json):**
IsolationForest F1 **0.466** (P 0.362, R 0.656) > Z-score 0.411 > One-Class
SVM 0.406 > LOF 0.378. IsolationForest is still the clear winner — the
paper's central claim holds. **Conclusion: no legitimate score increase was
available; the rigorous path slightly lowers the headline number, and we
keep it. Modeling is done.** (Minor noted caveat: contamination=0.35 was
tuned on the 43-feature set; kept as-is since it's a coarse knob — a full
re-tune on 41 features is optional future work.)

### [x] B6. Add two higher-ranked classical models (PCA-recon, Elliptic Envelope)
**What:** Sarjan asked whether any model ranked above Isolation Forest is
cost-free, feasible, and could top it. The only feasible (scikit-learn, no
GPU, in-scope) candidates were PCA reconstruction-error and Elliptic
Envelope (robust Mahalanobis), both of which model sensor *correlations*.
**Resolved (`experiments/extra_models.py`, both added to `src/models.py`):**
- Elliptic Envelope's covariance was singular on our 41 partly-collinear
  features (slogdet blew up); fixed by PCA-whitening to 10 components first
  — a standard, numerically-sound Mahalanobis setup.
- Both tuned fairly (contamination on the validation files). Even at their
  own validation-best (0.5), **neither beat Isolation Forest** on the test
  set. Their theoretical edge comes from modeling raw sequences/correlations
  that our engineered tabular features don't expose.
- **Decision:** keep both as comparison points (not as the proposed model),
  turning this into a **6-model** study. `visualize.py` updated (bar chart +
  2×3 confusion grid). Official results regenerated with all 6 sharing
  contamination=0.35.
**Final official 6-model test F1:** IsolationForest **0.466** > Elliptic
Envelope 0.418 > Z-score 0.411 > One-Class SVM 0.406 > LOF 0.378 > PCA-recon
0.370. IsolationForest is still the clear winner; the broader comparison
makes the paper more rigorous. **Modeling is now genuinely done.**

## Phase C — Results, Visualization, and the Streamlit App

### [x] C1. Paper-quality figures
**What:** `src/visualize.py` (distinct from the existing `visualize.py`
CLI) with 4 functions, all saving 300 DPI PNGs to `results/figures/`:
`plot_model_comparison` (grouped bar, precision/recall/F1 × 4 models),
`plot_confusion_matrices` (2×2 heatmap grid), `plot_time_series_with_anomalies`
(per-sensor traces, true-anomaly shading, predicted-anomaly dots), and
`plot_anomaly_closeup` (±30s around the first changepoint, with the
changepoint drawn as a vertical line). `main.py` now generates all 4 at
the end of a run, using IsolationForest's predictions and an
auto-selected showcase test file (`select_showcase_file`: prefers a run
with a changepoint and a balanced anomaly mix — picked `valve1/7`).
**Done.** Verified all 4 render correctly. Note for later polish: the
close-up's window can land entirely inside a sustained-anomaly region
(fully shaded) depending on the file; the changepoint line + flow-rate
jump still illustrate the transition. `run_all_models` gained a
`return_predictions=True` option so figures reuse the fitted predictions
instead of re-fitting.

### [x] C2. Surface results inside `skab.py`'s Streamlit app
**What:** A new "🏆 Model Results" tab in the existing Streamlit explorer.
**Done.** The tab reads `results/metrics.json` and shows: headline metrics
(best model + its precision/recall/F1) with a plain-English gloss, the full
4-model comparison table (best F1 row highlighted), and the 4 paper PNGs
from `results/figures/` each with a plain-English caption. Falls back to a
"run main.py" prompt if results don't exist yet. Also corrected the now-
stale train/test split table in the older "🤖 Model-Ready" tab, which had
described an outdated (and backwards) split that contradicted the actual
pipeline. Verified the app launches cleanly (HTTP 200, no tracebacks) and
unit-tested the table/figure logic.

### [x] C3. Model persistence
**What:** Save fitted models via `joblib` for reproducibility.
**Done.** `src/persist.py` bundles all 6 fitted models + the StandardScaler
+ the feature names into a single `results/models/anomaly_models.joblib`.
`main.py` writes it every run (`run_all_models` gained `return_models=True`).
Verified: reloading the bundle and scoring the test set reproduces
IsolationForest F1 0.466 with no retraining. The `.joblib` (~18 MB, mostly
One-Class SVM's support vectors) is git-ignored as a regenerable artifact;
the folder is kept via `.gitkeep`.

---

## Phase F — Research Contribution: Concept-Drift Adaptation

Added after reviewer-style feedback flagged the plain 6-model comparison as
not a research contribution. Turns the study's own winning model into the
subject of a new question: what happens when a deployed detector meets an
operating regime it was never trained on? Runs entirely on the Mac, no extra
hardware, on data already in the repo. Does **not** touch `main.py` or the
official `results/metrics.json`.

### [x] F1. Drift detector (`src/drift.py`)
**What:** A streaming drift score + `DriftDetector`. **Done.** Uses the KS
two-sample **D-statistic** (mean across features), not the p-value: with
thousands of rows per batch the p-value saturates and flags 100% of files
regardless of real change. The D-statistic is a bounded, sample-size-robust
effect size. `DriftDetector.check(window)` returns `(drifted, score)` and
`update_reference()` re-anchors after a retrain.

### [x] F2. Concept-drift experiment (`experiments/concept_drift.py`)
**What:** Stream test files by fault family (`other → valve1 → valve2`);
the initial IsolationForest trains **only on "other"** so valve1/valve2 are
genuinely unseen. Two tracks scored batch-by-batch: **static** (never
retrains) vs **adaptive** (retrains on a sliding window of the most recent
normal-labeled batches when drift fires). **Done.** Run via
`python -m experiments.concept_drift`; saves `results/concept_drift.json`.

### [x] F3. Make the result real, not a rigged or null finding
**What:** Get a genuine, honest positive result. **Done — the key methodology
work.** Diagnosed why the first runs tied: the static model *flags everything*
on unseen regimes (recall 1.0, precision ≈ base rate), so F1 rewards a
skill-less baseline and there's nothing to "recover." Fixes: (1) score by
**MCC + balanced accuracy** (which collapse to 0 / 0.5 for flag-everything),
not F1; (2) **select contamination by MCC** on the other-only holdout (→0.05,
selective) rather than F1 (→0.35, the B4 flag-everything trap); (3) let
retraining recalibrate to the new regime (sliding window, no offset pinning).
Result: on the unseen valve regimes the static model has **zero skill (MCC
0.000, balanced acc 0.500)** while adaptive **restores skill (MCC 0.106,
balanced acc 0.558)** and cuts false alarms (precision 0.285→0.336).

### [x] F4. Headline figure (`plot_drift_adaptation` in `src/visualize.py`)
**What:** Skill (balanced accuracy) over the regime-shifting stream, static vs
adaptive, with fault-family bands, retrain markers, and the 0.5 chance line.
**Done.** `results/figures/drift_adaptation.png`: static flatlines on the 0.5
floor across all of valve1/valve2; adaptive climbs to 0.6–0.75.

### [ ] F5. Fold concept-drift into the paper
**What:** New subsection (methodology + results + the figure) framing the
contribution: a deployed static detector silently degrades to chance under
regime shift; lightweight CPU-only drift adaptation restores real skill.
**Next.** Reframe the paper's narrative around this so it's no longer "just
a comparison." Also still open from the edge-pivot memo: surface the
latency/memory numbers already in `metrics.json`.

---

## Phase G — Second Dataset (SWaT): generalisation + real-drift replication

Added to answer the biggest reviewer critique ("single dataset, doesn't
generalise"). Uses the genuine SWaT (Secure Water Treatment, iTrust/SUTD,
Dec 2015) testbed via a Kaggle re-upload. Full data, no downsampling, all on
the Mac. Does not touch SKAB results.

### [x] G1. Verify + load SWaT honestly (`src/swat.py`)
**Done.** Confirmed real SWaT (51 vars @1 Hz + Normal/Attack label). Kept only
the fully-recorded 395,298-row normal block (a 991,800-row block has 7 blank
sensors — discarded, not imputed) + 54,621 attack rows. Features use the 25
continuous sensors with the unchanged SKAB rolling-stats recipe (26 discrete
actuators excluded), so the two datasets are methodologically identical.

### [x] G2. 6-model comparison on SWaT (`experiments/swat_comparison.py`)
**Done — honest winner-flip.** Same protocol as main.py; contamination
validation-tuned to 0.05 by MCC (flag-everything rejected); leakage-safe
held-out test. **Elliptic Envelope wins (MCC 0.766) > One-Class SVM (0.634) >
Isolation Forest (0.473)** — IF, the SKAB winner, is only 3rd. Reported as-is.
Finding: the best classical detector is dataset-dependent (correlation
structure pays off on SWaT's 51 richly-correlated sensors).

### [x] G3. Concept drift on SWaT — REAL temporal drift (`experiments/swat_drift.py`)
**Done — bigger win than SKAB.** SWaT normal genuinely drifts over the ~4.5-day
run (a static detector's false-alarm rate on normal climbs 8% → 60–100%).
Static vs adaptive over 34 batches (25 retrains): MCC 0.242 → 0.594, balanced
accuracy 0.630 → 0.826, false alarms 73% → 24%, attack recall 0.99 → 0.90.
Figure `results/figures/swat_drift_adaptation.png`. This is *real* drift, not a
proxy — it answers the "your drift is simulated" limitation directly.

### [ ] G4. Decide paper reframing + fold SWaT in
**Needs author input.** The winner-flip changes the paper's headline from
"Isolation Forest wins" to "best detector is dataset-dependent; rigorous
multi-dataset evaluation + drift adaptation matter." Run by Prof. Shah before
rewriting §V/§"why correlation models don't win"; then add a SWaT subsection
(comparison table + the SWaT drift figure/result) alongside the SKAB sections.

---

## Phase H — Edge/Docker Deployability Benchmark

Answers the "tested in only one environment" critique and backs the paper's
"deployable on cheap edge hardware" claim — Mac-only, no Raspberry Pi, per
Sarjan's constraint. Uses Docker to impose Pi-class limits on the Mac.

### [x] H1. Constrained-container benchmark (`experiments/edge_benchmark.py`, `docker/Dockerfile.edge`)
**Done.** Ran the full pipeline inside a Docker container capped to **1 CPU,
1 GB RAM, ARM64** (same CPU family as a Raspberry Pi). Measures per-model train
time, inference latency (ms/sample), on-disk size, plus the drift adaptation's
on-device costs (KS check ~60 ms/batch, retrain ~0.11 s), peak RAM, and the
effective cgroup CPU limit. Supporting code: `src.swat.load_swat_bounded` +
`SWAT_DIR` env override (loads SWaT in-container without network / without
holding the full CSV in RAM).

**Result:** everything fits with headroom. Inference 0.0012–0.24 ms per reading
(IsolationForest/EllipticEnvelope ~2 µs); model sizes 1 KB–14 MB (leads:
EllipticEnvelope 285 KB, IsolationForest 1.1 MB); **peak RAM 484 MB of the
1024 MB cap**; 25 s total. `results/edge_benchmark.json`. Honest scope: matches
a Pi's constraints + ARM instruction set, not its slower per-core speed, so
latency is an optimistic lower bound ("constrained ARM container", not a
physical Pi).

### [ ] H2. Fold the deployability numbers into the paper
**Next (with the SWaT fold-in, G4).** Add a short "Deployability / edge" table
(latency, model size, RAM footprint, retrain cost under Pi-class limits) — this
also closes the long-open edge-pivot gap (latency/memory numbers were in
metrics.json but never surfaced in the paper body). Optionally note a real
Raspberry Pi as future work for true on-device timing.
**Done in `paper1.tex`** (Table tab:edge + Fig fig:edge).

---

## Phase I — Deep Sequential Baseline (reviewer request)

Both external peer reviews (session 19) asked for a deep baseline to prove the
classical-vs-deep accuracy trade-off. Done for real, on the Mac.

### [x] I1. LSTM autoencoder (`src/deep_baseline.py`, `experiments/deep_baseline.py`)
**Done.** PyTorch 2.12 (Apple MPS). Lightweight single-layer LSTM AE over
30-step windows of raw scaled sensors; reconstruction-error score; threshold
tuned on validation by MCC (anti-degenerate), same discipline as classical;
evaluated per row on the identical SKAB + SWaT test splits. `torch`/`kagglehub`
pinned in requirements.txt.

### [x] I2. Honest split result (`results/deep_baseline.json`)
**Done — robust across 8 and 25 epochs.** SKAB: the AE FAILS (MCC −0.045, below
chance and below every classical model). SWaT: the AE WINS outright (MCC 0.841 >
Elliptic Envelope 0.766). Extends "best detector is dataset-dependent" to deep
models, more extremely. Cost: needs PyTorch runtime + GPU, ~4–15× slower
inference than the classical leaders, so classical stays the edge choice.

### [x] I3. Fold the deep baseline into the paper
**Done.** LSTM-AE row in both tables (top+bold on SWaT), a Deep Sequential
Baseline methodology subsection, and updates to abstract / contributions /
Related Work / cross-dataset / Discussion / Conclusion / edge note.
`cross_dataset_comparison.png` regenerated to include it. `overleaf_paper1.zip`
rebuilt. Note: drift adaptation on the deep model is left as future work.

---

## Phase D — Paper Writing

Per Sarjan's own ordering: easiest sections first, Abstract/Intro last.
**A complete end-to-end first draft now exists at `paper/paper.tex`** (IEEE
`IEEEtran` conference format). All sections below are drafted in it; what
remains is review, humanising the prose, filling author placeholders, and
reformatting to the final venue's template (E3). Boxes are ticked as
"drafted"; final sign-off happens with the authors.

### [x] D1. Dataset section
Describe SKAB: testbed, sensors, fault types, label semantics.
**Drafted** in `paper/paper.tex` §III with exact corpus figures.

### [x] D2. Methodology section
Feature engineering (41 leakage-free features), the 6 models, the
chronological per-file split, the 0%-contamination training set (B3), the
whole-file validation split, and the contamination=0.35 choice (B4).
**Drafted** in §IV.

### [x] D3. Results section
**Drafted** in §V: the 6-model table (sorted by F1), all 4 figures embedded
from `results/figures/`, the confusion-matrix error analysis (IF catches
1477/2252, 2608 false alarms), and the "why correlation-based models don't
win" analysis. Numbers pulled directly from `results/metrics.json`.

### [x] D4. Related Work
8–10 citations on sensor anomaly detection.
**Done:** 12 real, verified citations gathered into `REFERENCES.md`
(IEEE-formatted, each checked against publisher/arXiv/DOI), and the Related
Work prose (§II) is drafted around them. One author list ([11]) still to
confirm before final submission.

### [x] D5. Discussion & Limitations
**Drafted** in §VI: modest absolute precision stated plainly, the wall-clock
leak documented (F1 0.496→0.466), single-dataset/single-fault-family caveat,
tabular-vs-sequence-models note, coarse contamination caveat.

### [x] D6. Abstract + Introduction
**Drafted:** abstract (~200 words) + Introduction (§I) with explicit
contribution list, written last to match the body.

---

## Phase E — Submission Logistics

### [x] E1. Paper title decided and ready to send
"Comparative Evaluation of Unsupervised Machine Learning Models for
Multivariate Sensor Anomaly Detection in Industrial Systems" — confirmed.
### [x] E2. Prof. Abhi Shah finds venue/conference
Venue chosen: **ICACS 2026** (2nd Int. Conf. on AI, Communication Technologies &
Smart Cities, CHARUSAT), published in **Springer LNNS, Scopus-indexed**. Paper
submission deadline **2026-07-10**.
### [x] E3. Format paper to venue's template
`paper1.tex` reformatted from IEEEtran into the conference's **Springer LLNCS**
class (`llncs.cls` + `splncs04.bst`), with Prof. Shah's review fixes folded in
(Index Terms→Keywords, SWaT date dropped, jargon simplified, dataset-excerpt
tables added). `overleaf_paper1.zip` bundles the class so it is self-contained.
### [ ] E4. Final submission
Depends on Prof. Shah's final review + humanising pass, then upload by 2026-07-10.

---

## Phase R — Revision after ICACS rejection (2026-10-28)

### [x] R1. Audit paper vs code/data (`revision/01_02_04_05_06_stage1_audit.md`)
Found the SWaT split error, inconsistent thresholds, label-using drift retraining, batch-amortised latency.
### [x] R2. Literature review (`revision/03_literature_review.md`) — 39 verified references
### [x] R3. Re-run all experiments under fixed protocols (`experiments/revision/`, `results/revised/`)
### [x] R4. Rewrite paper (`paper/paper2_revised.tex`), change log, final assessment, venue strategy
### [x] R5. Rerun on official SWaT (iTrust) + WADI A2 — done 2026-10-30
### [x] R6. Prof. Shah approved framing; IJPHM Word manuscript built; code public on GitHub
### [ ] R7. Submit to IJPHM (+ arXiv preprint)

---

## Right now

See `CHECKPOINT.md` (top section). The revised paper exists; the remaining steps (R5-R7) need people, not code.
