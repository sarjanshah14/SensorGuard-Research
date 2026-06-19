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
