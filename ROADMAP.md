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

### [ ] C3. (Optional/low-priority) Model persistence
**What:** Save fitted models via `joblib` to `results/models/` for
reproducibility. Not required for the paper's results, just nice-to-have.

---

## Phase D — Paper Writing

Per Sarjan's own ordering: easiest sections first, Abstract/Intro last.

### [ ] D1. Dataset section
Describe SKAB: testbed, sensors, fault types, label semantics. Mostly
factual — can be drafted now, doesn't depend on B3/B4 settling.

### [ ] D2. Methodology section
Describe feature engineering, the 4 models, the train/test split — and
explicitly the contamination decision from B3 (this section needs B3
finished first, since "how we built the training set" is exactly what B3
is fixing).

### [ ] D3. Results section
Numbers + figures from B3/B4/C1. Depends on those being finalized.

### [ ] D4. Related Work
8–10 citations from Google Scholar/arXiv on sensor anomaly detection.
Independent of the code — can be done in parallel with anything above.

### [ ] D5. Discussion & Limitations
Honest about: single dataset (SKAB only), no live deployment, future work.
Can reuse the SensorGuard anomaly-taxonomy definitions (Drift/Spike/
Dropout/Noise/Calibration Error) here as framing for "what kinds of faults
exist beyond what SKAB labels."

### [ ] D6. Abstract + Introduction
Written last, once D1–D5 exist to summarize.

---

## Phase E — Submission Logistics

### [x] E1. Paper title decided and ready to send
"Comparative Evaluation of Unsupervised Machine Learning Models for
Multivariate Sensor Anomaly Detection in Industrial Systems" — confirmed.
### [ ] E2. Prof. Abhi Shah finds venue/conference
Blocked on him, not on us — title (E1) is what he needed to start.
### [ ] E3. Format paper to venue's template
Depends on E2 (template depends on which venue).
### [ ] E4. Final submission
Depends on D1–D6 and E3.

---

## Right now

**Next task: start Phase D — paper writing.** All of the modeling (B) and
results/visualization (C1-C2) work is done; C3 (model persistence) is
optional/low-priority. The highest-leverage remaining work is drafting the
paper, easiest sections first: D1 (Dataset) and D4 (Related Work) don't
depend on anything else and can start immediately; D2 (Methodology) and D3
(Results) can pull directly from DEVLOG.md + results/metrics.json + the
figures.
