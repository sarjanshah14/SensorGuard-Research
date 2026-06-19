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
