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
