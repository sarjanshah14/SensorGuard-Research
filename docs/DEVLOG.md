# Dev Log

Detailed, narrative record of every work session: what we set out to do, the
steps taken, problems hit and how they were resolved, decisions made and why,
and a plain-language summary of the outcome. `PROGRESS.md` stays a one-line-
per-session index; this file is the long version to read back later.

---

## Session 1 — 2026-06-19 — Bootstrap the data-loading pipeline

**Goal:** Get a clean, verified way to load SKAB into pandas before building
anything else on top of it ("checkpoint by checkpoint", per Sarjan's brief).

**Steps taken:**
1. Surveyed the repo as it actually was (not assumed): `SKAB/` already had
   all 35 CSVs at root, `skab.py`/`visualize.py`/`explore.ipynb` already
   existed from earlier ad-hoc work, `env/` already had every package
   needed (pandas, sklearn, streamlit, etc.) — no installs required.
2. Asked clarifying questions before writing code: where to get the
   `SensorGuard` reference repo (it didn't exist locally), whether to
   restructure the existing `skab.py`/`visualize.py`/`explore.ipynb`, whether
   to move `SKAB/` into a `data/` folder, and whether to set up git.
3. Based on answers: cloned `SensorGuard` (git-ignored, reference only),
   left the existing files where they were, kept `SKAB/` at root, and set up
   a `.gitignore` + first real commit.
4. Wrote `src/preprocessing.py`: `load_skab_single`, `load_anomaly_free`,
   `load_skab_all`, `chronological_split`.
5. Wrote `notebooks/01_explore.ipynb` to sanity-check loading + plot one
   run with anomalies shaded.

**Issue faced:** First version of `chronological_split` grouped fault CSVs
by filename (`source_file`) to split each run 80/20 independently — but
`valve1/`, `valve2/`, and `other/` all reuse filenames like `0.csv`, `1.csv`.
That meant `valve1/0.csv` and `valve2/0.csv` both got `source_file="0"` and
were silently merged into one group during the split, mixing two unrelated
physical experiments together. Caught it by checking `n_source_files == 24`
when 34 files were expected.

**Fix:** Namespaced `source_file` as `f"{folder}/{name}"` (e.g.
`"valve1/0"` vs `"valve2/0"`) so every run stays distinct. Re-ran the check:
34 distinct source files, confirmed.

**Outcome (plain language):** We can now reliably load all the SKAB data
into pandas, correctly split each experiment's fault data 80% train / 20%
test by time (never shuffled, never mixed with a different experiment), and
the anomaly-free baseline file is kept separate as training-only data. This
was verified by actually running it, not just writing it. Committed to git.

---

## Session 2 — 2026-06-19 — Feature engineering + the 4 models

**Goal:** Build the actual ML comparison: Z-score, One-Class SVM, LOF,
IsolationForest, on an identical feature matrix, scored on precision/recall/
F1/timing.

**Steps taken:**
1. Per Sarjan's instruction, read the old `SensorGuard` hackathon code
   *only for ideas* (via a research subagent, not for copy-pasting) — it's
   ~1.5 years old and used fully synthetic `random.uniform()` sensor data,
   so it needed review rather than reuse.
   - **Kept**: hour-of-day / day-of-week features, rolling mean/std as a
     general idea, and the deviation-magnitude anomaly taxonomy (Drift/
     Spike/Dropout/Noise/Calibration Error) as useful definitions for the
     paper's discussion section later.
   - **Explicitly rejected**: baseline computed from only the first 5
     readings (data leakage risk), evaluating the drift model on its own
     training data (no held-out set), randomly-assigned severity labels,
     and hardcoded magic numbers (`contamination=0.1` typed inline with no
     explanation) scattered through the code.
2. Wrote `src/features.py`: per-sensor raw value + rolling mean (5s, 30s) +
   rolling std (5s) + delta, plus hour/day-of-week/time-since-start — 43
   features total. Computed **per source experiment**, never letting a
   rolling window cross from the end of one SKAB run into the start of
   another (they're independent physical experiments with independent
   clocks — bleeding across that boundary would manufacture a fake
   correlation between unrelated runs).
3. Wrote `src/models.py`: Z-score, One-Class SVM (rbf, nu), LOF
   (`novelty=True` — required to predict on held-out data at all, since
   LOF's default mode only supports scoring the data it was fit on),
   IsolationForest — all behind one `train_*`/`predict_*` interface, all
   fit on the same `StandardScaler`-scaled matrix, none of them using
   labels during training (kept genuinely unsupervised, not just
   "unsupervised-flavored").
4. Wrote `src/evaluate.py`: precision/recall/F1/confusion matrix plus
   train/inference timing, dumped to `results/metrics.json`.
5. Wrote `main.py` to wire it together: train = `anomaly-free.csv` (all of
   it, assumed 100% normal from its filename) + the chronological 80% of
   each fault CSV; test = the remaining 20% of fault CSVs only — never the
   anomaly-free file, since it has no real ground-truth labels to score
   against.
6. Ran it. It worked on the first full run (no bugs this time) — see
   results below.

**Result:**

| model | precision | recall | f1 |
|---|---|---|---|
| Z-score | 0.291 | 0.190 | 0.230 |
| One-Class SVM | 0.286 | 0.229 | 0.254 |
| LOF | 0.353 | 0.190 | 0.247 |
| IsolationForest | 0.358 | 0.313 | **0.334** |

**Open issue, not yet fixed:** the pooled training set is ~28% anomalous.
Unsupervised anomaly detectors assume training data is *mostly* normal —
28% contamination is high for that assumption, and is the most likely
reason all 4 models' absolute scores are mediocre. IsolationForest still
won, which matches the paper's hypothesis, but the numbers should improve
once this is addressed (candidate fix: train on `anomaly-free.csv` alone,
or down-weight/exclude labeled-anomalous rows from the fault-CSV training
slice — not yet tried).

**Outcome (plain language):** All 4 models now actually run, on the same
features, the same split, and get scored the same way. IsolationForest
came out on top, which is what the paper wants to show — but the scores
are mediocre across the board because the training data has more
anomalies mixed in than these methods are designed to tolerate. That's
the next thing to fix, not a dead end.

---

## Session 3 — 2026-06-19 — Fixing the training-contamination problem (ROADMAP B3)

**Goal:** Session 2 ended with training data that was ~28% anomalous —
too dirty for unsupervised novelty detectors, which assume training data
is mostly normal. Try to fix it and see if scores improve.

**Steps taken:**
1. Wrote `experiments/training_strategy_comparison.py` — reuses the
   existing `src/` building blocks to test 3 ways of building the
   training set, all evaluated against the exact same test set:
   - **pooled** (session 2's baseline): anomaly-free.csv + every row of
     the fault CSVs' 80% training slice, anomalies included (~28% dirty).
   - **anomaly_free_only**: anomaly-free.csv alone (0% dirty, but it's a
     single clean run — less variety).
   - **filtered**: anomaly-free.csv + only the rows from the fault CSVs'
     training slice that are labeled normal (0% dirty, more variety than
     anomaly_free_only since it includes "normal" moments from the actual
     fault rigs, not just the one pristine baseline rig).
2. Ran it (`python -m experiments.training_strategy_comparison` — needed
   an `__init__.py` in `experiments/` and to run as a module so `src/` is
   importable; running the script directly failed with
   `ModuleNotFoundError: No module named 'src'` since Python only adds the
   script's own folder to the import path, not the repo root).

**Issue faced:** `anomaly_free_only` looked like the best option at first
glance — F1 jumped to ~0.46 for 3 of the 4 models. But recall was ~0.99–1.0
across the board, which is a tell: a model that flags almost every single
row as anomalous gets near-perfect recall *for free* (it can't miss an
anomaly if it calls everything an anomaly), while precision lands right
around the test set's actual anomaly rate (~0.30) purely by chance. Checked
this by looking at recall, not just F1 — F1 alone would have hidden this.
**Why it happened:** anomaly-free.csv is one pristine, steady-state run.
The fault rigs' "normal" periods (when no fault is occurring) still differ
from that pristine run's specific operating point — different flow rate,
pressure, etc. A model trained only on the pristine run learns too narrow a
definition of "normal," so it flags the fault rigs' legitimately-normal
moments as anomalies too, not just the real faults.

**Decision:** Use **filtered** as the new default in `main.py`. It keeps
training data 100% free of labeled anomalies (the methodologically correct
thing for an "unsupervised" claim) while including realistic operating
variety from the actual fault rigs, which avoids the "flag everything"
failure mode.

**Result (filtered, now in `results/metrics.json`):**

| model | precision | recall | f1 | vs. session 2 f1 |
|---|---|---|---|---|
| Z-score | 0.304 | 0.210 | 0.248 | 0.230 → 0.248 |
| One-Class SVM | 0.299 | 0.262 | 0.279 | 0.254 → 0.279 |
| LOF | 0.349 | 0.203 | 0.257 | 0.247 → 0.257 |
| IsolationForest | 0.358 | 0.342 | **0.350** | 0.334 → 0.350 |

**Outcome (plain language):** Cleaning up the training data (removing the
labeled-anomalous rows instead of leaving them mixed in) made every single
model better, with no downsides. We also nearly fooled ourselves with a
training setup that looked even better on paper (higher F1) but was
actually just crying wolf on almost every reading — worth remembering as a
trap when judging "is this score actually good" later in the project, not
just for this one decision.

---

## Session 4 — 2026-06-20 — Hyperparameter sweep (ROADMAP B4)

**Goal:** `contamination`/`nu` (0.1), LOF's `n_neighbors` (20), and
Z-score's `threshold` (3.0) were all carried-over nominal defaults, never
actually tuned for SKAB. Tune them properly.

**Steps taken:**
1. Wrote `experiments/hyperparameter_sweep.py`. To tune hyperparameters
   without leaking the test set into the final reported numbers, split
   `fault_train` (already held out from `fault_test`) a second time into
   a fit set (for training candidates) and a validation set (for scoring
   them) — `fault_test` itself stays untouched until the very end.

**Issue #1 — bad validation split design:** First attempt split
`fault_train` again chronologically (last 20% of it). Result: the
validation set came out **98% anomalous**. Diagnosed by checking one
file's anomaly flag decile-by-decile over its own timeline — confirmed
this is real SKAB behavior, not a bug: each run tends to go normal ->
fault induced and sustained for a contiguous middle-to-late chunk -> often
back to normal right at the end. The "last 20% of the first 80%" slice
landed almost exactly inside that sustained-fault block for most files.
**Fix:** Added `file_holdout_split()` to `src/preprocessing.py` — instead
of slicing time within files, it holds out entire experiment files for
validation, keeping each one's natural normal/anomalous mix intact.
Re-ran: validation anomaly rate became 38%, much closer to the test set's
30% and far more trustworthy.

**Issue #2 — first grid was too narrow:** Swept contamination/nu over
`[0.01...0.25]` and every single model's "best" value landed on `0.25`,
the edge of the grid — a tell that the true optimum was outside the range
tested. **Fix:** widened the grid to cover the full range sklearn allows
(`contamination`/`nu` up to 0.5), and Z-score's threshold down to 0.5.

**Issue #3 — "best by F1" isn't always trustworthy:** With the wider grid,
IsolationForest's contamination sweep showed a genuine interior peak at
0.35 (F1 rose, then fell past it — a real signal). But One-Class SVM's nu
and LOF's contamination kept climbing all the way to sklearn's hard limit
of 0.5 without ever turning over, and Z-score's top scorer (threshold=0.5)
had recall exactly 100%. Both are the same "flag almost everything" trap
from session 3, just reappearing inside a hyperparameter sweep instead of
a training-data choice. **Decision:** rejected the raw argmax for those
three. Adopted IsolationForest's real peak (contamination=0.35) as a
shared anomaly-rate assumption across all 3 contamination/nu-based models
(consistent with the original brief's "identical assumption across
models" spirit), kept LOF's `n_neighbors=20` (gains beyond it in the sweep
were small and noisy, and pushing it much higher works against LOF's
whole premise of looking at *local* neighborhoods), and set Z-score's
threshold to 2.0 (the best value before recall pins to 100%).

**Final result, on the real (never-touched-until-now) test set:**

| model | before tuning | after tuning |
|---|---|---|
| Z-score | 0.248 | 0.410 |
| One-Class SVM | 0.279 | 0.410 |
| LOF | 0.257 | 0.381 |
| IsolationForest | 0.350 | **0.496** |

**Outcome (plain language):** All 4 models got meaningfully better, and
none of them are cheating by just flagging everything (we checked this
time, having been burned once already). IsolationForest is now clearly
ahead of the other 3, by a wider margin than before — a stronger result
for the paper's central claim. We also hit the same "model is just crying
wolf" trap twice more during this session (once in the validation split,
once in the sweep itself), which is worth remembering as a recurring
thing to check for, not a one-off.

---

## Session 5 — 2026-06-20 — Paper-quality figures (ROADMAP C1)

**Goal:** Now that the numbers are tuned and stable, produce the figures
the paper needs, saved as high-resolution (300 DPI) PNGs.

**Steps taken:**
1. Added a `return_predictions=True` option to
   `evaluate.run_all_models` so the figure code can reuse the predictions
   the models already made, instead of re-fitting them just to draw plots.
2. Wrote `src/visualize.py` (a new file — kept separate from the older
   `visualize.py` CLI tool at repo root) with 4 plotting functions:
   - **model_comparison**: grouped bar chart, precision/recall/F1 with one
     bar per model.
   - **confusion_matrices**: a 2×2 grid of heatmaps, one per model,
     showing how many normal/anomalous rows each got right vs. wrong.
   - **time_series_with_anomalies**: for one example test run, every
     sensor's signal over time, with the true anomaly stretches shaded red
     and the model's predicted anomalies marked as orange dots.
   - **anomaly_closeup**: the same idea zoomed into ±30 seconds around the
     moment a fault begins (the "changepoint"), with a red dashed line at
     that moment.
3. Wired all 4 into `main.py` so they regenerate on every pipeline run,
   driven by IsolationForest's predictions (the paper's proposed model)
   and an auto-picked example file.

**Decisions worth noting:**
- **Which example file to show:** rather than hardcode one, added
  `select_showcase_file()` — it prefers a test run that actually contains
  a changepoint (so the close-up has a real transition to draw) and, among
  those, the most balanced normal/anomaly mix so both show up in the plot.
  It picked `valve1/7`.
- **Row alignment safety:** the figures attach each model's predictions
  back onto the test data as a column, which only works if the prediction
  array lines up row-for-row with the test rows. It does (because of how
  the data is grouped before features are built), but I added an assertion
  in `main.py` so that if a future change ever breaks that alignment, it
  fails loudly instead of silently drawing wrong dots.

**Verified:** ran `main.py`, opened all 4 PNGs, confirmed they're correct
and readable. The confusion matrix confirms the story numerically —
IsolationForest catches 1709 of 2252 real anomalies (the best of the 4),
while keeping false alarms comparable to the others.

**Outcome (plain language):** We now auto-generate the 4 charts the paper
will use, every time we run the pipeline, at print quality. They visually
back up the headline result: IsolationForest is the best of the 4 at
actually catching faults. One small cosmetic caveat noted in ROADMAP for
later — the zoomed-in close-up can sometimes be fully inside a fault
stretch (so the whole background is shaded), depending on which file is
auto-picked; it still clearly shows the fault's physical signature (a jump
in flow rate right at the changepoint line).

---

## Session 6 — 2026-06-20 — Results inside the Streamlit app (ROADMAP C2)

**Goal:** Make the model results reviewable visually inside the existing
`skab.py` Streamlit explorer, instead of only in the terminal or the raw
`results/metrics.json` file (Sarjan's stated preference for clarity).

**Steps taken:**
1. Added a new "🏆 Model Results" tab to `skab.py`. It reads
   `results/metrics.json` and shows:
   - Headline metric cards (best model + its precision/recall/F1) with a
     plain-English explanation of what those numbers mean.
   - The full 4-model comparison table, with the best-F1 row highlighted.
   - The 4 paper figures (the exact PNGs from `results/figures/`), each
     with a plain-English caption explaining how to read it.
   - A graceful "run `python main.py` first" message if results don't
     exist yet.
2. Kept it additive — didn't restructure the existing tabs or loaders,
   reused the app's existing style conventions.

**Issue found while doing this:** the older "🤖 Model-Ready" tab had a
"Train / Test Split Strategy" table that was now factually wrong — it
predated our actual pipeline and described a different (and backwards)
split: "Train = anomaly-free only; Validation = 20% of fault files; Test =
80% of fault files." Our real split is Train = anomaly-free + normal rows
of the first 80% of each fault file, Validation = whole held-out files (for
tuning only), Test = last 20% of each fault file. **Fix:** rewrote that
table to match what the code actually does, with pointers to the DEVLOG
sessions that explain why. Left a stale doc like that in place would have
risked it leaking into the paper as a wrong methodology description.

**Verification:** `python -m py_compile skab.py` (syntax), launched the app
headless and confirmed it serves HTTP 200 with no tracebacks in the log,
and unit-tested the risky bits outside Streamlit (the pandas Styler that
highlights the best row, the metrics.json parse, and the 4 figure paths) —
all good.

**Outcome (plain language):** You can now open the Streamlit app
(`streamlit run skab.py`), click the new "Model Results" tab, and see the
whole story in one place — which model won, the full comparison table, and
all 4 charts with explanations of how to read them — without touching the
terminal. Also caught and fixed an out-of-date description elsewhere in the
app that would have contradicted our real method.

---

## Session 7 — 2026-06-20 — Trying to legitimately raise the score, and a leak we caught (ROADMAP B5)

**Goal:** Sarjan asked the right question: can we *legitimately* improve the
scores, and if so, write that up as part of the paper ("score was X, it was
lacking because Y, we did Z, now it's W")? His condition: only adopt a
change if it genuinely, honestly helps — otherwise wrap up, what we have is
already a valid comparison.

**Attempt 1 — temporal smoothing of predictions.**
- Reasoning: real faults last many seconds; an isolated 1-2 second
  "anomaly" prediction in the middle of normal operation is almost always a
  false alarm. So requiring a prediction to persist for >= `min_run` seconds
  before we believe it should remove false alarms and raise precision.
- Built `src/postprocess.py` (`smooth_min_run`, applied per source file so
  runs never merge across experiments) and `experiments/temporal_smoothing.py`,
  tuning `min_run` on the validation files (never the test set).
- Result: it helped IsolationForest on *validation* (best at min_run=8) but
  on the actual test set it *lowered* its F1 (0.496 -> 0.478) — it cut
  recall (0.76 -> 0.61) more than it raised precision (0.37 -> 0.39) — and
  it hurt the other 3 models outright (their best was "no smoothing").
- Interpretation: the false alarms are NOT short blips; they're sustained
  stretches where the models misread normal operation. Smoothing can't fix
  that. **Rejected.**

**Attempt 2 — removing wall-clock features (and what it exposed).**
- Reasoning: `hour` and `day_of_week` could be *spurious*. Built
  `experiments/feature_ablation.py` to compare with vs. without them.
- Result: removing them *lowered* IsolationForest F1 by 0.03 (0.496 ->
  0.466). The other 3 barely moved.
- The important realization: that 0.03 was **leakage**, not skill. Our test
  set is the last 20% (by time) of the *same* fault files used for training.
  Within a single short file the hour barely changes, so `hour` effectively
  becomes an ID for "which experiment this is" — and because train and test
  come from the same files, the model can use that ID to recall how
  fault-prone each file is. That inflates the test score in a way that would
  evaporate on any genuinely new machine/recording.
- **Decision: remove them.** `build_features` now defaults to
  `include_clock=False` (41 features instead of 43). `time_since_start` is
  kept — elapsed seconds within a run is a legitimate within-experiment
  signal, not a wall-clock ID.

**Regenerated the official results** (`python main.py`) with the
leakage-free feature set. New committed numbers:

| model | F1 (leaky, old) | F1 (honest, official) |
|---|---|---|
| Z-score | 0.410 | 0.411 |
| One-Class SVM | 0.410 | 0.406 |
| LOF | 0.381 | 0.378 |
| IsolationForest | 0.496 | **0.466** |

IsolationForest is still the clear winner, so the paper's central claim is
unchanged.

**Outcome (plain language):** We genuinely tried to push the score up two
different honest ways. Neither worked — and the second attempt actually
revealed that a small part of our earlier score was the model "cheating" by
noticing *when* each experiment was recorded rather than detecting faults.
We removed that crutch, so our official IsolationForest score came down a
little (about 0.50 to 0.47), but it's now a number we can fully defend to a
reviewer. So per Sarjan's condition, we are NOT able to legitimately raise
the score — and we're wrapping up the modeling here. The leak we caught is
itself a good, honest detail to mention in the paper's methodology. Two
small ablation scripts (`experiments/temporal_smoothing.py`,
`experiments/feature_ablation.py`) are kept as the evidence behind these
decisions.

---

## Session 8 — 2026-06-20 — Two more models: can anything beat Isolation Forest? (ROADMAP B6)

**Goal:** Sarjan asked which models rank *above* Isolation Forest in theory
are also cost-free and feasible for us, and whether any would actually top
it. The only feasible candidates (scikit-learn, no GPU, no scope change)
were **PCA reconstruction-error** and **Elliptic Envelope** (robust
Mahalanobis distance) — both model the correlation *between* sensors, which
is the structure Isolation Forest doesn't directly use.

**Steps taken:**
1. Added `train/predict_pca_recon` and `train/predict_elliptic_envelope` to
   `src/models.py`, and `experiments/extra_models.py` to tune each one's
   contamination on the validation files and evaluate on the locked test set
   alongside the original 4.

**Issue hit:** Elliptic Envelope exploded numerically — 26 MB of
`slogdet` divide-by-zero / overflow warnings. Cause: our 41 features are
partly redundant (each sensor's rolling-mean tracks its raw value), so the
covariance matrix is singular and the Mahalanobis distance it relies on is
undefined. **Fix:** PCA-whiten to 10 decorrelated components first, then fit
Elliptic Envelope there — a standard, numerically-stable way to do
Mahalanobis on collinear data. Re-ran clean.

**Result — neither beat Isolation Forest.** Even given their own
validation-best setting, on the test set: IsolationForest F1 0.466, Elliptic
Envelope 0.435, PCA-recon 0.411 (the rest below). With the shared
contamination=0.35 used in the official pipeline: IsolationForest 0.466 >
Elliptic Envelope 0.418 > Z-score 0.411 > One-Class SVM 0.406 > LOF 0.378 >
PCA-recon 0.370. Interpretation: their theoretical advantage comes from
modeling raw sensor *sequences*, but we feed engineered tabular features, so
that advantage doesn't materialize here.

**Decision:** keep both anyway, as extra comparison points (not as the
proposed model). This upgrades the study from 4 models to **6**, which is
more rigorous and more convincing — "we tested six classical methods and
Isolation Forest won" is a stronger sentence than "we tested four." Updated
`src/visualize.py` for 6 models (wider bar chart, 2×3 confusion grid) and
regenerated the official `results/metrics.json` + figures. Verified both
figures render correctly.

**Outcome (plain language):** We tried the two best *feasible* models that,
on paper, should beat Isolation Forest. Neither did on our data — and we
now understand why (they need the raw time-signal, which our feature setup
doesn't hand them). Rather than throw that away, we kept them in the
comparison, so the paper now compares six methods with Isolation Forest
still on top. Along the way we hit and fixed a real numerical breakdown in
Elliptic Envelope (singular covariance), which is itself a small, honest
methods detail. **Modeling is now genuinely finished — next stop is writing
the paper.**

---

## Session 9 — 2026-06-20 — Saving the trained models (ROADMAP C3)

**Goal:** Make a run reproducible without retraining — save the fitted
models to disk so they can be reloaded and used directly.

**Steps taken:**
1. Gave `run_all_models` a `return_models=True` option so it can hand back
   the fitted model objects (not just their scores/predictions).
2. Wrote `src/persist.py` (`save_artifacts` / `load_artifacts`). It bundles
   everything needed to score new data into one file
   (`results/models/anomaly_models.joblib`): all 6 fitted models, the
   `StandardScaler` they were trained with (so new data gets scaled the same
   way), and the feature-name list (so you know what to feed them).
3. Wired the save into `main.py` — it now writes the bundle on every run.
4. Git-ignored the `.joblib` file (it's ~18 MB and fully regenerable by
   `python main.py`), but kept the `results/models/` folder via a `.gitkeep`
   so the structure is preserved.

**Verification:** loaded the saved bundle in a fresh Python process, rebuilt
the test set, scaled it with the *saved* scaler, and predicted with the
*saved* Isolation Forest — got F1 0.466, exactly matching the committed
result, with zero retraining. So the persistence round-trips correctly.

**Outcome (plain language):** Every time the pipeline runs, it now also
saves the trained models to a single file. Anyone (including future-us) can
reload that file and reproduce the exact predictions instantly, without
re-running the training. The file itself isn't stored in git because it's
big and can always be regenerated. With this, **every code checkpoint in the
project is complete** — all that remains is writing the paper.

---

## Session 10 — 2026-06-20 — Bug fix: Model Results tab going blank / crashing on non-anomaly-free data

**Goal:** Sarjan started actually using the live Streamlit app
(`streamlit run skab.py`) to review results, per the WALKTHROUGH guide, and
found a real bug: with the sidebar's "Experiment type" dropdown on
"anomaly-free" everything worked, but switching it to "ALL", "other",
"valve1", or "valve2" made the "🏆 Model Results" tab show nothing, and
switching to "other" specifically crashed with
`ValueError: underlying array is read-only` at a line in the *Insights* tab
(`np.fill_diagonal(corr_diff.values, 0)`).

**Diagnosis:** Two things compounding into one confusing symptom:
1. `compute_correlation_matrices()` is wrapped in `@st.cache_data`.
   Streamlit marks the numpy buffer behind any cached return value
   read-only, to stop one part of the app from accidentally corrupting
   another part's cached copy. `corr_diff = (anom_corr - normal_corr).abs()`
   inherits that read-only-ness from its inputs, so the old code's in-place
   `np.fill_diagonal(corr_diff.values, 0)` — which tries to zero the
   diagonal *in place* — crashed.
2. This code path only runs when there are anomalies to compare against
   normal data — i.e. every dropdown choice except "anomaly-free" (which
   has none and skips it entirely). That's exactly the symptom: works only
   on anomaly-free.
3. Why the *later* "Model Results" tab went blank too, not just "Insights":
   Streamlit re-runs the **entire script top-to-bottom** on every
   interaction, including every `with tab:` block, regardless of which tab
   is visually open. An unhandled crash earlier in the script (Insights)
   halts everything below it in that run — including the Model Results
   tab's code — even though the two tabs are otherwise unrelated.

**Fix:** Instead of mutating `corr_diff` in place, copy it to a fresh
writable numpy array first, zero the diagonal there, then rebuild a
DataFrame from that copy for the rest of the calculation:
```python
off_diag = corr_diff.to_numpy(copy=True)
np.fill_diagonal(off_diag, 0)
off_diag_df = pd.DataFrame(off_diag, index=corr_diff.index, columns=corr_diff.columns)
max_break = off_diag_df.max().max()
max_pair = off_diag_df.stack().idxmax()
```

**Verification:** An initial attempt to verify by importing `skab.py` as a
module (to simulate selecting "other" in the sidebar) was unreliable —
importing it runs the *entire* script as a side effect, which isn't a real
isolated test. Instead, reproduced the exact mechanism standalone: wrapped
a DataFrame-returning function in `@st.cache_data`, called it, and
confirmed the returned array's `.flags.writeable` is `False` — matching the
real cause. Confirmed the *old* code raises the identical
`ValueError: underlying array is read-only` against that read-only data,
and the *new* code runs cleanly and produces the correct max-break value.
`python -m py_compile skab.py` also passes.

**Outcome (plain language):** The dashboard had a real bug: a chart's code
was trying to modify data that Streamlit's caching system marks
"protected." That crash, happening inside the Insights tab, was silently
also breaking the separate Model Results tab, because Streamlit re-runs the
whole page on every click no matter which tab you're looking at — so one
bug in one tab took down an unrelated one. Fixed by working on a private
copy of the data instead of the protected original. The app should now work
correctly with every "Experiment type" option in the sidebar, not just
anomaly-free.

---

## Session 11 — 2026-06-20 — Bug fix #2: "ALL" mode freezing the whole app

**Goal:** Sarjan reported the app was *still* not working — specifically
that the sidebar's "ALL" option (the literal default on page load) made
the page "go away" with nothing loading, while every other option
("anomaly-free", "other", "valve1", "valve2") loaded fine. He asked me to
actually check it myself rather than just reason about the code, and to
refactor whatever was still broken.

**Why session 10's fix didn't catch this:** that session fixed a *crash*
(an exception with a traceback). This was a *freeze* — no exception, no
error message, just the page never finishing. A silent hang doesn't show
up by reading the code; it has to be measured.

**How I actually verified it this time, instead of guessing:**
1. First tried Streamlit's real test harness (`streamlit.testing.v1.AppTest`,
   which runs the actual script and lets you click widgets exactly like a
   browser would, server-side). The very first `at.run()` — which defaults
   to "ALL" because it's the first item in the dropdown — timed out past
   120 seconds. That alone proved "ALL" really does hang, on the very first
   load, independent of anything else.
2. Measured the "ALL" dataframe directly: concatenating all 35 SKAB CSVs
   and sorting by datetime gives 46,860 rows with **1,014 duplicate
   timestamps** — different physical experiments were recorded with
   overlapping wall-clock times, so sorting by absolute time interleaves
   rows from unrelated runs. Running `anomaly_intervals()` (which detects
   anomaly start/end by watching the `anomaly` column flip) on that
   interleaved mess produced **231 "episodes"** and **130 "changepoints"**
   — almost all of them artifacts of jumping between different
   experiments mid-stream, not real fault episodes.
3. Timed the actual rendering cost: building one sensor's Plotly chart with
   those 231 shaded regions took **7.6 seconds**, almost entirely from the
   231 `add_vrect` calls (the same chart with zero shading took 36
   *milliseconds* — vrects are the entire cost). The Sensor Graphs tab does
   this once per sensor — 8 sensors × ~360 total shapes (231 vrects + 130
   vlines) — which is roughly a minute of pure rendering time before that
   tab can even finish, well before the page can move on to the Model
   Results tab the user actually wanted to see.

**Fix (`skab.py`):**
- In the Sensor Graphs tab, when "ALL" is selected, stop passing the
  (meaningless, exploded) interval/changepoint lists into the per-sensor
  charts — render the raw signal with no shading instead. Updated the
  existing warning message to explain *why* (not just "select a file for
  cleaner plots," but the real reason: merging 35 unrelated experiments by
  wall-clock time produces nonsensical boundaries, and rendering them all
  is what was slow).
- The live one-line summary banner above the tabs had the same problem —
  it would report a "longest episode" computed from those same bogus
  intervals for "ALL." Fixed it to show total anomaly seconds without that
  misleading per-episode claim when "ALL" is selected.

**Verification (re-ran with the real harness, not just inspection this
time):**
- `AppTest` initial load (which is "ALL" by default): **120+ second
  timeout → 1.2 seconds.**
- Drove all 5 "Experiment type" options (`anomaly-free`, `other`,
  `valve1`, `valve2`, `ALL`) through the real script end-to-end: all 5
  load in under 1 second, zero exceptions, all 7 tabs render.
- Specifically opened the Model Results tab after selecting `valve1` to
  confirm it still shows the real metrics (IsolationForest F1 0.466) —
  it does, since that tab never depended on the sidebar selection at all.

**Outcome (plain language):** The first fix (session 10) solved a crash;
this was a second, different bug — a *freeze*, not a crash, so it didn't
show up by reading the code, only by actually timing it. The cause:
selecting "ALL" merges 35 separate experiments and lays them out by clock
time, but those experiments were recorded with overlapping clocks, so the
merge interleaves unrelated runs and makes the "where do anomalies
start/stop" logic invent hundreds of fake boundaries. Drawing all of those
as shaded regions on 8 charts was the actual slow part — about a minute of
pure rendering — which is why "ALL" (the option you see by default when you
open the app) looked like it never loads, while every other option (which
only loads one real experiment at a time) was always fine. Fixed by simply
not shading those fake boundaries for the combined view. Confirmed with a
real automated run of the app, not just a code read-through: the page that
used to hang past 2 minutes now loads in just over 1 second, for every
single sidebar option.

---

## Session 12 — 2026-06-20 — Cleaning the repo for external review

**Goal:** Sarjan is about to hand this repo to Prof. Abhi Shah to clone and
read. He asked to strip out old/unnecessary stuff and keep only what
matters — i.e., make sure a first-time outside reader isn't tripped up by
leftover scaffolding from before the real pipeline existed.

**What I removed and why:**
- `explore.ipynb` (root) and `notebooks/01_explore.ipynb` — both pre-date
  (or were one-off sanity checks from session 1 of) the actual `src/`
  pipeline; neither is imported by any pipeline code, and `WALKTHROUGH.md`'s
  file map never mentions them — they'd already been functionally
  forgotten. Confirmed via `grep` that no `.py` file references either.
- `visualize.py` (root) — a standalone pre-pipeline CLI for ad-hoc plotting,
  fully superseded by `src/visualize.py` (the real paper-figure generator
  driven by `main.py`) and by `skab.py`'s own Sensor Graphs tab. Also not
  referenced by any pipeline code.
- `jupyter` from `requirements.txt` — no notebooks remain in the repo, so
  the dependency was dead weight.

**What I deliberately did NOT touch:** `experiments/*.py` (the ablation/
decision scripts) — these are referenced throughout `DEVLOG.md` and
`WALKTHROUGH.md` as the actual evidence behind methodology choices
(training strategy, hyperparameter sweep, the rejected smoothing idea, the
leak fix, the extra-models test). Removing them would have deleted the
proof behind several claims this log makes. Also left `DEVLOG.md`,
`PROGRESS.md`, and `ROADMAP.md` untouched even where they mention the
deleted files — those are a historical record of what we did and when, not
current documentation, and editing history to match the present would make
them dishonest.

**What I updated instead of deleting:** `CLAUDE.md`'s "Project state"
section still said the repo was in an "early/pre-code stage" — wildly
stale, since the entire 6-model pipeline, results, and Streamlit app were
already done. Rewrote it (and its "Development Commands" section, which
only listed the now-deleted `visualize.py`) to describe what's actually
here. Rewrote `README.md` the same way — it previously only described the
dataset and the old CLI/notebook tools, with no mention of the actual
research, the 6 models, or the results; it's now a short front door that
points straight at `WALKTHROUGH.md`.

**Verification:** `python -m py_compile` on `main.py`, `skab.py`, every
`src/*.py`, and every `experiments/*.py` — all pass. Re-grepped for any
remaining `.py` references to the three deleted files — none found.

**Outcome (plain language):** Removed three files that were leftover from
before the real project existed (an old notebook, a duplicate plotting
script, and a one-off test notebook) — none of them were used by any of the
actual research code, so deleting them changes nothing about how the
pipeline runs. Also rewrote the two "front door" docs (`README.md` and
`CLAUDE.md`) since they were still describing the project as barely
started, which would have confused your professor on first read. Left the
experiment scripts and the DEVLOG/PROGRESS/ROADMAP history alone — those
are exactly the kind of "boring but important" evidence a reviewer should
see, not clutter.

---

## Session 13 — 2026-06-21 — Gathering real citations (ROADMAP D4)

**Goal:** Before writing the paper, assemble a pool of *real, verifiable*
references — Sarjan asked me to do my part (find citations) ahead of Prof.
Shah adding his, and to explain what citations are.

**What I did:** Searched the web for each work we actually rely on and
confirmed each against its publisher/arXiv/DOI rather than citing from
memory (fabricated references are the fastest route to desk-rejection).
Collected 12 into `REFERENCES.md`, IEEE-formatted, grouped by purpose
(dataset, surveys, the 6 models, application context, the library) with a
section-by-section map of which reference supports which claim. Verified the
official SKAB citation directly from the dataset's own repository. Flagged
one (the rotating-machinery review) whose exact author list I want to
confirm against the published PDF before submission. Noted honestly that the
z-score baseline has no single canonical primary paper (it's a textbook
method) rather than inventing one.

**Outcome (plain language):** The paper now has a verified set of real
references to build on — no made-up sources, each one checked. Also
explained to Sarjan what a citation is and why they have to be real.

---

## Session 14 — 2026-06-21 — Drafting the full paper end-to-end (ROADMAP D1–D6)

**Goal:** Sarjan asked for a complete end-to-end paper draft, which he'll
then humanise himself.

**Steps taken:**
1. Pulled the exact figures the paper needed straight from the pipeline
   rather than from memory: re-ran the data split to confirm the training
   set is 28,368 rows (9,401 anomaly-free + 18,967 normal-labelled fault
   rows) and the test set is 7,503 rows with 2,252 anomalies (30.0%); read
   `results/metrics.json` for every model's precision/recall/F1 and
   confusion matrix; confirmed the feature count is 41 from `features.py`.
2. Wrote `paper/paper.tex` in the IEEE `IEEEtran` conference format — the
   format we'd already agreed on — with all seven sections plus abstract and
   keywords: Introduction (with an explicit 4-point contribution list),
   Related Work, Dataset, Methodology, Results (a 6-model table + all four
   figures embedded from `results/figures/` + a confusion-matrix error
   analysis + an analysis of why the correlation-based models don't win),
   Discussion & Limitations, and Conclusion. All 12 references from
   `REFERENCES.md` are included as a `thebibliography` block.
3. Wrote `paper/README.md` with Overleaf/pdfLaTeX compile steps and an
   explicit list of what's real (every number, table, figure, citation)
   versus placeholder (author affiliations/emails, and the ref [11] author
   check).

**Decisions worth noting:**
- **Honest framing throughout, by design.** The abstract and Discussion
  state plainly that absolute precision is modest (~0.36, ~2 in 3 alarms
  false), that this is normal for real industrial benchmarks, and that the
  contribution is the *comparison*, not a state-of-the-art detector. The
  wall-clock leak (F1 0.496 → 0.466) is documented as a feature of the
  rigour, not hidden. This is the version a reviewer trusts.
- **Kept it self-contained / low-dependency.** Used only standard IEEE
  packages and removed the one `siunitx` command (`\SI{1}{\hertz}` →
  `1\,Hz`) so it compiles on a vanilla pdfLaTeX/Overleaf setup with no extra
  package installs.
- **Author block left as placeholders** (`[Institution]`, `[City, Country]`,
  `[email]`) for Sarjan and Prof. Shah to fill — I only know Sarjan's email,
  and inventing affiliations would be wrong.

**Verification:** No local LaTeX toolchain here (`pdflatex` not installed),
so I could not produce the PDF myself — flagged that the user compiles on
Overleaf. I instead verified the inputs: re-derived the dataset counts from
code, cross-checked every reported metric against `results/metrics.json`,
and grepped the `.tex` to confirm no remaining non-standard package
commands. The prose, numbers, and references are all consistent with the
repo's actual results.

**Outcome (plain language):** There is now a complete, submittable-quality
first draft of the whole paper — every section, every number, every figure,
and a dozen real references — written from our actual results, not invented.
It's honest about what the results are and aren't, which is exactly what
makes it credible. What's left is human work, not code: you and Prof. Shah
read and humanise it, fill in your affiliations, confirm one reference's
authors, and (once he picks a venue) reformat to that venue's template.
I couldn't compile the PDF here because there's no LaTeX installed on this
machine — that step happens on Overleaf, per `paper/README.md`.

---

## [2026-06-30] Session 15 — Concept-drift adaptation: a real new contribution (ROADMAP D-extension)

**What we set out to do.** External reviewer-style feedback flagged the
paper's core as "just a 6-model comparison" — rigorous but not a research
contribution. After weighing options (edge/Raspberry-Pi deployment, temporal
features, MLOps), Sarjan chose **concept-drift adaptation**, with one hard
constraint: everything runs on his MacBook, no extra hardware, using data
already in the repo. Goal: build a detector that notices when incoming
sensor data stops matching what it learned and retrains itself, and show it
beats a static model when the operating regime shifts. Plan approved as
`zesty-fluttering-piglet.md`.

**How drift is simulated.** SKAB has three fault families (`other`,
`valve1`, `valve2`) — physically different failure modes, i.e. different
distributions. We stream the test files one at a time in family order
(`other → valve1 → valve2`) so the regime visibly shifts partway through.
Crucially, the initial model is trained **only on anomaly-free + the "other"
family's normal rows**, so valve1/valve2 are genuinely unseen until the
stream reaches them (unlike `main.py`, whose per-file 80/20 split leaks the
early 80% of every valve file into training). New code: `src/drift.py`
(detector) + `experiments/concept_drift.py` (the experiment), reusing the
existing feature/model/eval building blocks. Nothing in `main.py` or the
official `results/metrics.json` was touched.

**Problems hit and exactly how they were diagnosed and fixed.** This is the
important part — the result did *not* work at first, and the path to making
it work was the actual research.

1. **KS *p-value* saturates.** The first drift detector counted features
   with KS-test p < 0.05. With thousands of rows per batch, p-values
   collapse to ~0 for any tiny wobble, so it flagged 100% of validation
   files at every threshold — useless. *Fix:* use the KS *D-statistic* (max
   CDF distance, bounded [0,1], an effect size that's sample-size-robust) as
   the drift score instead. Confirmed it then jumps cleanly at the
   `other→valve1` boundary and stays low within a family.

2. **Static and adaptive tied (F1 0.435 ≈ 0.438), and Sarjan rightly
   refused to accept "we tried it, it failed, here's an honest negative."**
   He insisted: find *why* it fails and fix it. So we diagnosed instead of
   shrugging. Two mechanical causes were found and fixed:
   (a) IsolationForest's `contamination` re-derives its decision threshold
   as a quantile of whatever training scores it sees, so naive refitting
   silently resets the alarm rate every retrain; (b) retraining on *all*
   accumulated history drowns the new regime's few hundred rows under
   thousands of old "other" rows, so the trees barely move. Fixed (b) with a
   sliding window (`deque(maxlen=5)` — retrain only on recent batches, let
   the model forget the old regime).

3. **The real root cause — F1 was measuring the wrong thing.** Even after
   those fixes the numbers barely moved, so we inspected the per-batch
   behaviour directly. The static model had **recall = 1.0 on 30 of 34
   files** with precision pinned at the ~0.25 anomaly base rate: it was
   *flagging almost everything*. There was no degradation for the adaptive
   model to "recover," because a flag-everything detector never misses and
   never improves. The contamination=0.35 the F1-maximising sweep had picked
   was itself the near-degenerate "flag everything" trap this project
   already rejected back in B4. The fix had three coordinated parts:
   - **Score by skill, not F1.** Switched the headline metric to **MCC and
     balanced accuracy**, which (unlike F1) collapse to 0 / 0.5 for a
     flag-everything predictor. F1 actively *rewards* the degenerate
     baseline here, so it was the wrong scorecard.
   - **Select contamination by MCC, not F1.** On the other-only holdout,
     F1 rises monotonically toward the degenerate end (0.32→0.59) while MCC
     peaks at a *selective* contamination (0.05) and collapses to ~0 at the
     flag-everything end. MCC-based selection principledly picks 0.05 — a
     detector that can actually discriminate.
   - **Let retraining recalibrate.** Dropped the offset-pinning from fix
     2(a): on a genuine regime shift we *want* IsolationForest to re-anchor
     "normal" to the new regime, which is exactly what cuts the false-alarm
     flood.

**The result (honest and positive).** On the **unseen valve regimes** —
where drift actually happens — the static detector has **zero discriminative
skill: MCC = 0.000, balanced accuracy = 0.500** (literally chance; it can
only flag everything). Drift-triggered retraining **restores real skill:
MCC = 0.000 → 0.106, balanced accuracy = 0.500 → 0.558**, and cuts the
false-alarm rate (precision 0.285 → 0.336). Over the whole stream, MCC goes
from −0.011 (worse than chance) to +0.049. The headline figure
(`results/figures/drift_adaptation.png`) makes it unmistakable: the static
line sits flat on the 0.5 chance floor across all of valve1/valve2 while the
adaptive line climbs to 0.6–0.75. Artifacts: `src/drift.py`,
`experiments/concept_drift.py`, `results/concept_drift.json`, the new figure,
and `mcc`/`balanced_accuracy` added to `src/evaluate.evaluate` (additive;
`main.py`'s 6-model results unaffected).

**Outcome (plain language).** We turned the weakest critique of the paper —
"it's just a comparison" — into a genuine contribution. We show that the
paper's own winning model, deployed and left alone, silently degrades to
*worthless* (no better than a coin flip) the moment the machine enters an
operating mode it wasn't trained on — and that a lightweight, CPU-only,
runs-on-your-Mac drift detector that retrains on recent data restores real
detection skill and roughly halves false alarms. Critically, this is an
*honest* positive result, not a rigged one: the win only appears once you
score with a metric (MCC) that refuses to credit the "alarm on everything"
cheat, which is the same anti-degenerate discipline the project has used
since B4. The static baseline genuinely looks fine on F1 (0.43) and is
genuinely skill-less (MCC 0) at the same time — and explaining *why* both are
true is the paper's new story.

---

## [2026-07-01] Session 16 — Second dataset (SWaT): generalisation check + real-drift replication

**What we set out to do.** Add a second real dataset to kill the single
biggest reviewer critique ("only one dataset, doesn't generalise"). Sarjan
downloaded the genuine **SWaT** (Secure Water Treatment, iTrust/SUTD, Dec
2015) via a Kaggle re-upload (kagglehub id
`vishala28/swat-dataset-secure-water-treatment-system`). Instruction from
Sarjan was emphatic: **be honest, no cheating/transparency, run as much data
as possible (don't heavily trim), and simplify at the end.**

**Data verification + honest cleaning.** Confirmed it's the real SWaT: 51
process variables at 1 Hz + a `Normal/Attack` label. normal.csv = 1,387,098
rows but is two concatenated runs — a **395,298-row block with all 51 sensors**
and a 991,800-row block where 7 sensors (MV101, AIT201, MV201, P201, P202,
P204, MV303) are entirely blank. We **kept only the fully-recorded block** for
training rather than dropping those 7 columns (present at test time) or
imputing ~1M fabricated values. attack.csv = 54,621 rows, all `Attack`, no
NaNs. New code: `src/swat.py` (loader + the 25 continuous-sensor list; the 26
discrete actuators are excluded from features so the SKAB rolling-stats recipe
applies unchanged and the two datasets are methodologically identical).

**Stage 1 — 6-model comparison (`experiments/swat_comparison.py`).** Same
protocol as main.py: train on normal only; tune the shared contamination on a
**validation split by MCC, rejecting flag-everything** (it chose 0.05; MCC fell
monotonically toward the degenerate high-contamination end, just like SKAB);
report on a held-out test set (normal_test + attack_test), leakage-safe and
chronological. **Ran on the COMPLETE dataset, no downsampling** (we first
timed One-Class SVM/LOF at ds=5/2/1; full data = 276,708 train rows, OCSVM
~332 s, whole run ~494 s — acceptable for a one-off, so we used everything per
Sarjan's instruction). Result on the SWaT test set (sorted by MCC):

| model | MCC | bAcc | F1 | P | R |
|---|---|---|---|---|---|
| **EllipticEnvelope** | **0.766** | 0.881 | 0.839 | 0.846 | 0.833 |
| One-Class SVM | 0.634 | 0.838 | 0.755 | 0.653 | 0.896 |
| IsolationForest | 0.473 | 0.754 | 0.657 | 0.546 | 0.823 |
| Z-score | 0.180 | 0.567 | 0.511 | 0.349 | 0.955 |
| LOF | 0.105 | 0.520 | 0.489 | 0.324 | 0.994 |
| PCA-recon | 0.002 | 0.501 | 0.461 | 0.316 | 0.858 |

**The honest, important finding: the winner FLIPPED.** On SKAB, Isolation
Forest won; on SWaT it is **third**, beaten decisively by the correlation-based
**Elliptic Envelope** and by One-Class SVM. This contradicts the original
paper's headline ("Isolation Forest is best") and its "why correlation-based
models don't win" section. We are **not** hiding or spinning this. It is a
genuine, useful result: *no single classical detector is best across datasets;
the winner is dataset-dependent.* It is also mechanistically sensible — SWaT
has 51 richly-correlated sensors, so a model that exploits inter-sensor
covariance (Elliptic Envelope) shines, whereas SKAB has only 8. The paper's
thesis should shift from "Isolation Forest wins" to "the best detector is
dataset-dependent, so rigorous multi-dataset evaluation matters" — a stronger,
more defensible claim that reinforces the drift contribution. **This is a
narrative change for Sarjan + Prof. Shah to sign off on, not something to
rewrite unilaterally.**

**Stage 2 — concept drift on SWaT (`experiments/swat_drift.py`), with REAL
temporal drift.** First measured whether SWaT's *normal* operation drifts: a
detector trained on the first 50k normal rows flags 8% of early normal
(by design) but climbs to flagging **60–100% of *later* normal** as anomalous
— the sensor baselines genuinely move over the ~4.5-day run (KS drift 0.13 →
0.47). This is *real* concept drift, not SKAB's fault-family proxy, and
directly answers the "your drift is simulated" weakness. Built the static-vs-
adaptive stream: train on first 50k normal; stream the rest of normal forward
in 10k batches; each batch paired with a fixed held-out attack sample (so both
false alarms and detection are scored, and any skill change is attributable to
the drifting normal); adaptive retrains on a sliding window of recent normal
when the KS detector fires (25/34 batches). Result (mean over stream):

| | static | adaptive |
|---|---|---|
| MCC | +0.242 | **+0.594** |
| balanced accuracy | 0.630 | **0.826** |
| precision | 0.312 | **0.583** |
| false-alarm rate on normal | **0.734** | **0.244** |
| recall on attacks | 0.993 | 0.896 |

So adaptive retraining **cuts false alarms from 73% to 24%**, **more than
doubles MCC (0.24→0.59)**, and **keeps attack detection at 90%**. This is a
*bigger, cleaner* drift win than SKAB (0.00→0.11), on genuine temporal drift.
Figure: `results/figures/swat_drift_adaptation.png` (static decays to the 0.5
chance floor; adaptive holds at 0.8–0.9). Artifacts: `results/swat_metrics.json`,
`results/swat_drift.json`, two SWaT figures.

**Transparency notes (no cheating).** (1) Full dataset used, no downsampling.
(2) contamination validation-tuned by MCC with flag-everything rejected, never
tuned on test. (3) The winner-flip is reported as-is, against our own prior
headline. (4) The blank-sensor block was discarded honestly, not imputed.
(5) The one constructed element — pairing each drift batch with a fixed attack
sample to score detection — is documented in the script and does not affect the
false-alarm trajectory (the core symptom), which is normal-only.

**Outcome (plain language).** The second dataset did its job and then some. It
shows our methodology generalises, it honestly overturns the "one model wins"
story into the stronger "it depends on the data" story, and — best of all — it
reproduces the concept-drift contribution on *real* drift even more
convincingly than the original dataset (false alarms 73%→24%, skill doubled,
detection preserved). Everything ran on the full data on the Mac. The one thing
needing human judgement is how to reframe the paper's headline given the
winner-flip; flagged for Sarjan and Prof. Shah.

---

## [2026-07-02] Session 17 — Edge/Docker deployability benchmark (constrained ARM container)

**What we set out to do.** Address the remaining "tested in only one
environment" critique and make the paper's "deployable on cheap edge hardware"
claim concrete — under Sarjan's hard constraint of using only his Mac (no
Raspberry Pi). Approach: instead of buying hardware, run the whole pipeline
inside a **Docker container capped to Raspberry-Pi-class limits (1 CPU core,
1 GB RAM, ARM64)**. Sarjan's M4 is ARM, so the container runs the *same CPU
family* as a real Pi; we cap cores + RAM to Pi levels. Honest scope stated up
front: this reproduces a Pi's *constraints + instruction set*, not its slow
per-core speed, so latency figures are an optimistic lower bound labelled
"constrained ARM container", not "measured on a physical Pi".

**What was built.** `docker/Dockerfile.edge` (python:3.12-slim ARM64 + only
the compute libs, pinned to host versions; code + data mounted at run time);
`experiments/edge_benchmark.py` (per-model train time, inference latency in
ms/sample, on-disk model size, F1/MCC as a sanity check; plus the drift
adaptation's on-device costs — KS-check latency and one retrain; plus peak RAM
and the effective cgroup CPU limit). Added `SWAT_DIR` env override and
`load_swat_bounded()` to `src/swat.py` so SWaT loads inside the container
without kagglehub/network and without holding the full 1.4 M-row CSV in RAM.

**Problems hit + fixes (all real, all transparent).**
1. `docker build` timed out pulling the base image (a `| tail` pipe hid the
   real failure). Fix: pull `python:3.12-slim-bookworm` separately first.
2. Build then crashed the Docker daemon mid-install (socket vanished) — the
   combined install of numpy+scipy+pandas+sklearn spiked the builder VM's
   memory. Fix: restarted Docker Desktop (`open -a Docker`, waited for the
   daemon) and split the install into separate cached RUN layers.
3. First container run → **exit 137 (OOM-killed), no output**: the SWaT bench
   called `load_swat()`, which reads the entire 384 MB normal.csv (~1.5 GB in
   RAM) before slicing. Fix: `load_swat_bounded(n_normal, n_attack)` reads only
   the needed rows via pandas `nrows`; also ran with `PYTHONUNBUFFERED=1` so
   output survives a kill.
4. SWaT bench then errored (`RangeIndex has no total_seconds`) — the bounded
   loader didn't set the datetime index build_features needs. Fixed.
5. `os.cpu_count()` reported the host's 10 cores, not the cap — misleading.
   Fixed to read `/sys/fs/cgroup/cpu.max` (reports 1.0, honest).

**Result (inside 1 CPU / 1 GB / ARM64).** Everything ran cleanly. Inference
latency per reading (ms): IsolationForest 0.0019, EllipticEnvelope 0.0017,
Z-score ~0, PCA-recon 0.013, LOF 0.090, One-Class SVM 0.24 — all
sub-millisecond, the two lead models ~2 µs. Model sizes: Z-score 1 KB,
EllipticEnvelope 285 KB, PCA 8 KB, IsolationForest 1.1 MB, One-Class SVM
3.4 MB, LOF 14 MB. Drift adaptation on-device: KS check ~60 ms/batch, one
retrain ~0.11 s. SWaT (bounded 30k/20k) inference: IF 0.0012 ms/sample
(MCC 0.66), EllipticEnvelope 0.0040 ms/sample (MCC 0.81 — still ahead of IF,
consistent with the full SWaT finding). **Peak RAM 484 MB of the 1024 MB cap;
whole run 25 s.** Saved `results/edge_benchmark.json`.

(Note: the SKAB MCCs here are lower than the headline numbers because the edge
benchmark reuses the default contamination 0.35 rather than re-tuning — it is a
speed/memory/footprint test, and the MCC column is only a sanity check that the
constrained box produces the same maths as the host. It does.)

**Outcome (plain language).** The deployability claim is now backed by a real
experiment: the full method — both datasets, all six models, and the
self-healing drift adaptation — runs comfortably inside Raspberry-Pi-class
limits (one core, under half a gigabyte of RAM), detecting each sensor reading
in well under a millisecond, with the winning models only a few hundred KB to
~1 MB and an on-device retrain costing ~0.1 s. It is not a physical Pi (the
M4 core is faster), and that is stated honestly, but it directly answers "does
this fit on small/cheap hardware?" — yes, with large headroom. Everything ran
on the Mac via Docker, no extra hardware, exactly within Sarjan's constraint.

---

## [2026-07-02] Session 17b — Full figure set (Prof. Shah: more original data-figures = lower plagiarism risk)

**Why.** Prof. Shah advised that the more credible, self-generated figures the
paper has, the lower the plagiarism/originality risk (they're our own results,
not copied). So the goal was to make sure every claim has a figure.

**What was added.** Three new plotting functions in `src/visualize.py`
(`plot_cross_dataset_comparison`, `plot_edge_benchmark`,
`plot_swat_drift_motivation`) + `experiments/make_figures.py` to generate the
cross-cutting figures from the saved JSONs. Also **re-ran `main.py` so
`results/metrics.json` now includes MCC/balanced accuracy** for SKAB (it was
generated before those metrics existed), giving both datasets one shared
yardstick.

**New figures (all from our own data):**
1. `cross_dataset_comparison.png` — each model's MCC on SKAB vs SWaT side by
   side; shows the winner-flip cleanly (Isolation Forest tallest on SKAB,
   Elliptic Envelope tallest on SWaT; every model scores higher on SWaT).
2. `edge_benchmark.png` — two-panel deployability chart (inference latency +
   model size, log-scaled) with the "1.0 CPU, peak 484 MB of 1024 MB" caption.
3. `swat_drift_motivation.png` — the false-alarm rate on normal climbing 8% →
   ~100% over ~70 h alongside the rising KS drift score; the visual case for
   why adaptation is needed. Trajectory saved to
   `results/swat_drift_motivation.json`.

**Figure inventory now (11 total):** SKAB — model_comparison, confusion_matrices,
timeseries_anomalies, anomaly_closeup, drift_adaptation. SWaT —
swat_model_comparison, swat_confusion_matrices, swat_drift_adaptation.
Cross-cutting — cross_dataset_comparison, edge_benchmark, swat_drift_motivation.
Every major claim (comparison on both datasets, the winner-flip, drift
adaptation on both, the drift motivation, and edge deployability) now has its
own figure. The only symmetry gap is a SWaT per-sensor time-series illustration
(SKAB has one), skipped because SWaT's 51 sensors don't plot cleanly.

**Outcome (plain language).** The paper now has a complete, self-made picture
for every result — nothing borrowed. That both strengthens the paper visually
and, per Prof. Shah, keeps originality high.

---

## [2026-07-03] Session 20 — Deep sequential baseline (LSTM autoencoder)

**What we set out to do.** Both external reviews flagged the same gap: the paper
argues classical detectors are a low-compute alternative to deep models but
never trained a deep model to prove the accuracy trade-off. Sarjan chose to
close it for real (not just defend by citation). Constraint unchanged: it all
runs on his Mac.

**Setup.** Installed PyTorch (torch 2.12.1 — wheels exist for Python 3.14; Apple
MPS/GPU available). Built `src/deep_baseline.py`: a lightweight LSTM
autoencoder. Input is a sliding window of 30 consecutive *raw* scaled sensor
readings (8 channels on SKAB, 25 on SWaT) — deliberately giving the deep model
the temporal structure the tabular features suppress. A single-layer LSTM
encoder (64 hidden) compresses the window, a matching LSTM decoder reconstructs
it, and a row's anomaly score is the reconstruction MSE of the causal window
ending at that row. Trained on normal windows only; the decision threshold is
chosen on the validation split by MCC while rejecting flag-everything cut-offs —
the exact discipline used for the classical contamination — so the comparison is
fair. Evaluated per row on the identical SKAB and SWaT test splits.
`experiments/deep_baseline.py` runs both datasets and saves
`results/deep_baseline.json`.

**Fairness pass.** First run (8 epochs, stride 5) had the SKAB loss still
falling, so a reviewer could cry "under-trained." Re-ran at 25 epochs with
denser windows (stride 2 → 12k SKAB windows, converged to MSE ~0.06). The
result held, so it is not an artefact of under-training.

**Result (honest, robust, and a genuine finding).**
- **SKAB (8 sensors):** the AE *fails* — MCC **−0.045**, below chance and below
  every classical detector (best classical: Isolation Forest 0.147).
- **SWaT (25 sensors):** the AE *wins outright* — MCC **0.841**, F1 0.881,
  precision 0.974, beating the best classical (Elliptic Envelope 0.766).
- **Why:** deep needs dimensionality + data volume. SWaT gives it 25 correlated
  channels and hundreds of thousands of normal windows to learn a faithful
  reconstruction; SKAB gives it 8 channels and ~12k windows, too little to learn
  a normal manifold that generalises to subtle valve faults.
- **Cost:** ~58k params / 0.2 MB on disk (comparable to classical), BUT it needs
  the whole PyTorch runtime (far larger than scikit-learn), benefits from a GPU
  for training, and its per-reading inference is ~4–15× slower than the classical
  leaders. So it was *not* run inside the 1 GB edge container, and classical
  detectors remain the pragmatic edge choice.

**How it strengthens the paper (rather than breaking it).** It does not
contradict the thesis — it *extends* it. "The best detector is dataset-dependent"
now holds for deep models too, more extremely: best of all on SWaT, worst of all
on SKAB. And it directly answers both reviewers with a measured number instead
of a citation. Reported completely straight: on SWaT the deep model beats our own
classical champion, and we say so (LSTM-AE is the top, bolded row of the SWaT
table). Nothing was tuned to protect the classical story.

**Paper integration.** Added the LSTM-AE row to both comparison tables (bottom
on SKAB; top + bold on SWaT, where it leads precision/F1/MCC/bAcc). Added a
"Deep Sequential Baseline" methodology subsection. Updated the abstract, the
contribution bullets, Related Work (no longer "future work" — we ran it), the
cross-dataset subsection (new paragraph on the deep split + its cost), the
Discussion (first point + the tabular-vs-sequence limitation), the Conclusion,
and the edge subsection (why the deep model is excluded from the container run).
Regenerated `cross_dataset_comparison.png` to include LSTM-AE (SWaT 0.84 tallest,
SKAB −0.05 the only below-zero bar). Pinned `torch==2.12.1` and
`kagglehub==1.0.2` in requirements.txt. Verified the `.tex` (5 tables, 10
figures, all cites/refs/environments balanced, 0 em-dashes) and rebuilt
`paper/overleaf_paper1.zip`.

**Outcome (plain language).** We actually built and trained the deep model the
reviewers wanted, on the Mac, and reported the truth: it is the single best
detector on the big, sensor-rich dataset and completely useless on the small
one. That is a sharper, more honest version of the paper's core message, and it
closes the last major reviewer critique with real evidence rather than a promise.
The one thing left undone by choice: the drift adaptation was studied only for
the classical detectors, not the deep model — noted as future work.

---

## Session 21 — 2026-07-05 — Reformat to Springer LLNCS + Prof. Shah's review fixes

**Goal:** A venue was chosen — **ICACS 2026** (CHARUSAT), published in **Springer
LNNS (Scopus-indexed)**, submission deadline **2026-07-10**. The conference
supplied a template folder (`Word+Latex-Template/`). Prof. Shah reviewed the
IEEE-format `paper1.tex` and asked for five changes: rename "Index Terms" to
"Keywords" (and keep only ~6), the SWaT 2015 date reads as too old, put the whole
paper into the conference's template, simplify a couple of hard-to-read terms
(e.g. "anomaly=0", the Kolmogorov–Smirnov statistic), and attach the `head()` of
each dataset for readers.

**Steps taken:**
1. **Inspected the template.** The `Word+Latex-Template/` folder's LaTeX package
   (`Springer Lecture Notes in Computer Science.zip`) extracts to the standard
   Springer **LLNCS** class: `llncs.cls`, `splncs04.bst`, and `samplepaper.tex`.
   Read the sample to learn the exact conventions (title/`\titlerunning`,
   `\author`+`\inst`+`\institute`+`\email`, `\keywords{...}` **inside** the
   abstract, captions-above-tables, `\doi`, `\begin{credits}` with
   `\discintname`, `splncs04` bib style).
2. **Dataset-age decision (with Sarjan).** Confirmed the exact row counts from
   the files: SKAB 46,860 rows / 8 sensors / 2020; SWaT 1,441,719 rows (normal
   1,387,098 + attack 54,621) / 51 channels / 2015. The "SWaT is only ~50k lines"
   worry was a misread of the attack run — SWaT is actually one of the largest
   ICS benchmarks. Newer same-domain options exist (HAI 2020–2022, WADI 2017) but
   swapping would mean re-running the entire SWaT half (comparison, drift, deep
   baseline, edge, all figures) five days before the deadline. Decision: **keep
   SWaT, drop the date mention** (Prof. Shah's own fallback), and add a scale
   clause. HAI/WADI noted as paper-2 work.
3. **Front matter → LLNCS.** `\documentclass[conference]{IEEEtran}` →
   `\documentclass[runningheads]{llncs}`; removed `\IEEEoverridecommandlockouts`;
   added `\usepackage[T1]{fontenc}`; replaced the `\IEEEauthorblockN/A` block
   with `\author{Sarjan Shah\inst{1} \and Abhi Shah\inst{1}}`, `\authorrunning`,
   `\titlerunning`, and `\institute{... \email{...}}`.
4. **Keywords.** Deleted the `\begin{IEEEkeywords}` block and moved the 6 terms
   into `\keywords{... \and ...}` inside the abstract — LLNCS renders this as
   "**Keywords:**" automatically, which is exactly Prof. Shah's rename.
5. **Dropped the date.** Rewrote the SWaT paragraph to remove "The December 2015
   collection" and instead state the ~1.44M-row scale (defusing the "old/small"
   read). Verified no visible "2015/2016/December" remains anywhere outside the
   bibliography (the only hits left are inside citation *keys* like
   `\cite{goh2016}`, which don't render).
6. **Simplified jargon.** "the `anomaly`$=0$ rows" → "the rows labelled normal
   (that is, with no fault present)". For the KS D-statistic, kept the term and
   equation but added a plain sentence: it measures how far the current data has
   moved from the training data on a 0-to-1 scale (0 = identical, 1 = no
   overlap), and rewrote the p-value paragraph in plainer words.
7. **Head-of-dataset tables.** Added two small excerpt tables from real rows:
   `tab:skabhead` (SKAB, 4 sensors + `anomaly`, datetime kept — 2020 is fine) and
   `tab:swathead` (SWaT, 4 of 51 channels + label). **Important:** SWaT's real
   `Timestamp` literally reads `28/12/2015`, so showing it would have
   re-introduced the very date we just removed — replaced it with a relative
   time index `t` (seconds). Referenced both tables from their subsections.
8. **Single-column floats + bibliography.** LLNCS is single-column, so all 5
   `figure*` → `figure`, `width=0.86\textwidth` and `\columnwidth` →
   `\textwidth`. Reformatted all 18 references from IEEE style to Springer
   (`splncs04`) style (surname-first authors, bold volume, `\doi{}`), and added a
   `\begin{credits}` competing-interests declaration (Springer now expects one).
9. **Packaging.** Bundled `llncs.cls` + `splncs04.bst` into `overleaf_paper1/`
   alongside the new `paper1.tex` and the 12 figures, rebuilt
   `overleaf_paper1.zip` (self-contained, so it compiles even if a fresh Overleaf
   project lacks the class). Updated `paper/README.md` (LLNCS, 12 figures, upload
   the ZIP, no BibTeX pass needed).

**Problem hit / how resolved:** there is **no TeX engine on this Mac**
(`pdflatex`/`tectonic`/`latexmk` all absent), so a local compile was impossible.
Rather than guess, verified structurally with scripts: every `\ref`/`\cite`
resolves to a `\label`/`\bibitem` (0 undefined, 0 unused), all environments
balance (1 document, 1 abstract, 7 tables, 12 figures, 3 equations, 1 credits,
1 thebibliography), `\keywords` sits inside the abstract, 0 em-dashes, and no
leftover IEEE commands. Overleaf is the intended compile target and the bundled
class makes the ZIP self-contained.

**Plain-language outcome:** the paper is now in the exact Springer template the
conference wants, with every one of Prof. Shah's notes handled — "Keywords"
instead of "Index Terms", the old-looking 2015 date gone (SWaT kept, now
described by its ~1.44M-row size), the confusing terms explained in plain words,
and a sample-rows table for each dataset so any reader can see the raw data.
`overleaf_paper1.zip` is ready to upload and compile on Overleaf. What's left is
human-side: Prof. Shah's final read + Sarjan's humanising pass, then submit by
July 10. Only real caveat logged honestly: it was verified by structure, not by a
local compile, because this machine has no LaTeX installed.

---

## Session 22 — 2026-10-28 — ICACS rejection, flaw analysis, venue research, checkpoint

**Goal:** React to the ICACS 2026 rejection (Paper ID 580; reviewer totals 56 and 49, pass mark 60; scores only, no comments), work out why, find free venues, and make the project resumable.

**Steps:** Extracted the final PDF text and read it fully; mapped low scores (references 4, novelty 4-5, depth 4-5, critical qualities 4-5) to concrete weaknesses (see CHECKPOINT.md). Searched for no-fee venues: IJPHM (no fees, diamond OA, Scopus/ESCI) is the best fit; TMLR, Elsevier hybrid journals, IEEE INDIN/CASE 2027 are alternatives, with deadlines partly unverified. Wrote CHECKPOINT.md with status, flaws, prioritised TODO and venues. Added the zip, overleaf folder and conference template to .gitignore (regenerable/third-party) and committed everything else.

**Plain language:** The paper was rejected mainly for weak/dated references, thin novelty and shallow rigor. There is a concrete plan to fix those and resubmit to a free journal; CHECKPOINT.md lets any new chat pick up from here.

---

## Session 23 — 2026-10-28 — Revision Stage 1: audit after rejection

**Goal:** Sarjan pasted a large "paper rescue" master prompt (11 deliverables: diagnosis, reviewer matrix, literature review, novelty audit, methodological audit, experiment plan, revised manuscript, change log, reproducibility package, final assessment, venue strategy). Stage 1 = understand and audit before rewriting anything.

**Steps:** Read every pipeline file and experiment script, cross-checked all paper numbers against results JSON (all match), then inspected the raw SWaT re-upload. Found that `normal.csv`'s 395,298-row "fully recorded" block spans 28 Dec 2015 - 2 Jan 2016 with 36 gaps exactly where the 54,621 attack rows sit (395,298 + 54,621 = 449,919 = the official attack file). So the model was trained on normal rows *between attacks*, and the real 7-day normal run (present twice, with 7 channels blank) was thrown away. Literature (MAD-GAN, GDN) confirms the standard protocol: train on the 496,800-row normal run minus the first 21,600 start-up rows, test on the full attack file (~12% attacks). Also found: SKAB contamination 0.35 was chosen by F1 on the old leaky feature set; SWaT contamination tuned on IF only and shared; drift retraining filters rows by ground-truth labels; SWaT drift retrained 25/34 batches (effectively periodic); edge latency is batch-amortised; the in-container SWaT MCC was computed on training rows; Elliptic Envelope runs with support_fraction=1.0 (no MCD robustness); time_since_start differs in range between train and test. Ran 20 literature searches and verified ~17 key references (Audibert 2022, Kim 2022, Wu & Keogh, Schmidl 2022, Sarfraz 2024, TSB-AD, Garg 2021, VUS, ADBench, PyOD, Lu 2018, NAB, MLPerf Tiny, HAI, MAD-GAN, GDN, SKAB leaderboard).

**Plain language:** The arithmetic in the paper is right, but the SWaT half was trained on the wrong slice of data, and several "wins" (drift adaptation, edge speed) were measured in ways that flatter them. The headline findings ("no single model wins", "deep isn't always better") are already published by others, which explains the low novelty score. Everything is written up in `revision/AUDIT.md` with a ranked experiment plan; next session runs the SWaT fix first.

---

## Session 24 — 2026-10-28 (overnight) — Revision stages 2-4: re-run everything, rewrite the paper

**Goal:** Sarjan asked for the whole master-prompt revision to be completed overnight, plus an Artifact explaining before/after.

**Steps:** (1) Built `experiments/revision/common.py`: every detector returns a continuous score; two threshold rules applied to all models (label-free 99th pct of held-out normal; validation-MCC), F1 rule for sensitivity only; AUROC/AUPRC, event recall, delay; block bootstrap. Self-check passes. (2) SKAB: pooled protocol fixed (features before dropping rows, no elapsed time), both thresholds, 500-resample file bootstrap, IF 10 seeds, LSTM-AE 3 seeds, SKAB per-run protocol (from waico/SKAB `core/utils.py`: first 400 rows per file), ablations. (3) SWaT: rebuilt the standard split from the re-upload (de-duplicated normal week; merged attack period = 449,919 rows, 12.1%); 24 sensors (AIT201 blank); also scored the original layout with the same code. (4) Drift: 244 hourly batches (normal week then attack week), static/KS/periodic/always × oracle/pred-normal/all × IF/EE. (5) Edge: new `Dockerfile.edge2` (pyod, numba, CPU torch); per-model child processes, single-reading latency.

**Problems hit:** torch in the venv was missing `libtorch_cpu.dylib` (reinstalled 2.12.1). Docker was not installed; Sarjan installed Docker Desktop mid-run. First image had a 0-byte `typing_extensions.py` (corrupt layer, exit 135 bus error) → rebuilt with `--no-cache`. The fitted PyOD wrappers held lambdas and could not be pickled for the size benchmark → drop the factory after fit. Figure label collisions fixed after looking at each render.

**Results (plain language):** On SWaT done properly, all ten detectors are equally good at ranking attacks; the "winners" in the old paper came from how the data were split and how thresholds were set. Simple detectors (z-score, OCSVM, IF) kept their false alarms under control without labels; LOF, PCA and the deep model did not. Real normal operation does drift, but retraining only helps if you know which data are normal; without labels it fails. Real per-reading latency is ~1,000× the old number. The paper was rewritten around this ("Protocol choices decide the winner"), with 39 verified references. What is left needs people: official SWaT data, Prof. Shah's approval, Overleaf compile.

---

## Session 25 — 2026-10-29 — HAI as third dataset; IJPHM Word manuscript

**Goal:** add a non-water benchmark, settle the title, and produce the manuscript in the IJPHM template (Sarjan supplied the three IJPHM .docx templates; Prof. Shah approved the new framing).

**Steps:** downloaded HAI 21.03 (sparse git checkout, 179 MB, `external_data/`, gitignored); wrote `experiments/revision/hai_revised.py` with the same protocol as SWaT. Two runs were killed (exit 137, out of memory) inside PyOD's ECOD: it rescores test rows jointly with all stored training rows (850k × 275 features). Fixed by fitting ECOD/COPOD on a 100k-row subsample and scoring in 50k-row chunks (documented as a small source of score variation). For Word output, pandoc with the IJPHM template as reference doc, then a post-processor that maps pandoc's styles to the template's `phm_*` styles (headings auto-number), rebuilds the title block, adds the CC BY licence note, and sets one-column title/abstract + two-column body. Bugs fixed on the way: pandoc writes `" />"` (regex missed it); a greedy section-properties replacement deleted the body.

**Plain language:** On HAI the detectors are *not* tied and a different group wins (PCA and LOF), and thresholds learned from normal data work fine there. So "who wins" depends on the plant as well as the protocol; the steady performers across both industrial plants were the two simplest methods. The paper now has three datasets and a Word file in the journal's own template.

---

## Session 26 — 2026-10-29 — Response to external pre-submission review

**Goal:** an external read of the Word manuscript (6.8/10 potential, 4/10 readiness) raised four blockers: official SWaT, inconsistent memory numbers (1.1 GB RSS in a 1 GB container), mismatched retraining budgets, broken table layout; plus methods detail, synthesis table, repository URL, title.

**Diagnosis:** both numeric issues were real bugs. (1) The periodic policy's interval was derived from the KS policy's *successful* refits, while refits are skipped when the selected window has under 1,000 rows, so budgets did not match (52 vs 10 with predicted-normal selection). (2) Edge children were *forked*; a forked child's RSS includes pages shared with the parent that the cgroup charges once, inflating per-model memory and creating the 533 MB floor.

**Fixes:** drift now records attempts and refits and sweeps P = 1..48 to give a budget curve; edge uses spawned processes, reports baseline RSS, per-model peak increase (VmHWM reset after loading) and the cgroup's memory.peak/events. Word build moves tables/figures after the references into a one-column section and puts table captions below. Added implementation table, synthesis table, Poenaru-Olaru et al. (CAIN 2024) as closest prior work, new title.

**Plain language:** the two numbers the reviewer doubted were measurement mistakes, now fixed: the container really was capped at 1 GB (the heavy ECOD/COPOD runs pushed it to the limit without crashing), and with fair budgets a drift trigger saves some retrains but checking for drift costs more than simply retraining every hour. Only the official SWaT data and a public code link remain.

---

## Session 27 — 2026-10-30 — Official SWaT and WADI, repository clean-up, public code

**Goal:** close the external reviewer's top blocker (official SWaT), add WADI, make the code public, tidy the folder, and explain the whole project from zero.

**Data:** the first iTrust download failed (OneDrive throttling, "TooManyRequests"; only network logs arrived). Re-downloading just the three Physical files worked: Normal_v1 (495,000 rows), Attack_v0 (449,919 rows, 54,621 attacks incl. 37 labelled "A ttack"). WADI A2 (2019) arrived intact (784,571 normal / 172,803 attack rows; 4 empty columns; 2 trailing empty rows that first made the sensor filter reject everything).

**Findings:** with the re-upload's 24 sensors, official and re-upload results agree to within 0.001 AUROC, so the reconstruction was faithful. Adding AIT201 (blank in the re-upload) leaves ranking tied but breaks the z-score and One-Class SVM label-free thresholds; analysers AIT402/AIT201/AIT50x leave their training ranges in the attack week. WADI shows the same mechanism: 2B_AIT_002_PV out of range on 55.5% of normal test rows gives six detectors an identical ~56% false-alarm rate; dropping three analysers brings it to ~1%.

**Problems:** a "wait until WADI finishes" chain matched its own command line with pgrep and never started; replaced with direct sequential runs. The Mac restarted mid-drift-run and lost 50 minutes of work, so drift_revised now saves a partial file after every group and resumes, and runs under caffeinate.

**Clean-up:** docs/, data/, experiments/{revision,initial_analysis}, results/{revised,initial_analysis}, paper/{current,icacs_2026,templates}, apps/, archive/. All paths updated and tested; licensed data excluded by .gitignore and verified absent from all of git history before the public push.

**Plain language:** the official data confirmed the corrected story and sharpened it: what decides whether a detector is usable is mostly whether its alarm threshold survives drifting chemical sensors, not how well it ranks attacks. The paper is now final apart from a human read-through, and the code is public.

## Session 28 — 2026-10-30 — Second external review: readable tables, reconciled numbers

**Goal:** the second review (readiness 3/10, "do not submit this version yet") found broken tables in the rendered DOCX, missing numbers in Table 9, and several method/result mismatches. Fix the presentation first, then reconcile the text with the result files; no new experiments.

**Rendering:** installed LibreOffice to render the DOCX to PDF and inspect it, as the reviewer did. Bisecting showed two separate causes. (1) The IJPHM template has no `Table` or `Compact` styles, which pandoc's tables reference; LibreOffice then dissolved every table into one-cell-per-line text. Swapping styles.xml for pandoc's fixed it, so the builder now copies the missing pandoc styles into the template's styles. (2) The new plain-text maths turned `\%` inside `$...$` into a bare `%`, a LaTeX comment, which silently deleted the rest of the line (this is what emptied the Table 9 cells and dropped the summary table entirely in one build). All simple inline maths (nu, k=20, 10^-3, arrows, <=, P=1...) is now written as Unicode text; only the three real formulas remain Word equations. Tables get full width, fixed layout, column widths weighted by content, 8 pt text, rules, rows that do not split, and "Table n." / "Figure n." caption labels. Result: 19 pages instead of 37, every table checked visually.

**Text reconciled against results/revised/*.json:** SKAB now says IF (0.504-0.645) and COPOD (0.507-0.621) narrowly exclude 0.5, IF's seed range is 0.550-0.603, and no detector is better than the next (paired diff CI includes 0). Methods state that tables use the seed-0 model, bootstrap CIs cover test-block variability only, seed spread is separate, and that paired differences were computed only for the top two per benchmark (so "statistically indistinguishable" for all ten was softened to "within 0.045 AUROC, overlapping intervals"). Drift method now describes the P sweep, attempts vs refits and CPU accounting; the missing P=4 column was added from drift_official.json; the trigger result is bounded to one stream. Table 8 gains p99 (from edge.json), the memory column is "Peak RSS increase (MB)", and the caption says the benchmark used the re-upload's 24 sensors (120 features). The abstract's "<12%" claim names COPOD, ECOD, HBOS and Isolation Forest. WADI ablation is described as post hoc and diagnostic.

**WADI timing check:** the time column holds only mm:ss. The Row counter and per-day counts (86,400 per full day) show the attack file is gap-free at 1 Hz; the normal file has one 264,000-row gap (29 Sep to 2 Oct, about 73 h) removed in the release, inside the training part. Stated in the Datasets section.

**Plain language:** the research did not change. The Word file now shows every number in proper tables (the earlier "missing" numbers were a conversion bug), and the text now claims exactly what the evidence supports, no more.

**Addendum (guidelines check):** the IJPHM author guidelines require the small-caps journal header from page 2 and the first-author licence note. Rebuilding the section breaks had dropped the template's header/footer references; they are now on every section, with a distinct first page (journal/ISSN/year footer, no header). LibreOffice needed the distinct-first-page flag on the two-column body section as well, as in the template. Plain language: the file now looks like an IJPHM paper on every page.
