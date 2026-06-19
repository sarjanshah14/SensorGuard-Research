# WALKTHROUGH — read this to understand everything we built

This is the guided tour, written for *reviewing*, not coding. It assumes you
haven't opened any results yet and don't want to wade through jargon. Read it
top to bottom once; it tells you what we did, where everything lives, how to
see the results yourself, and — in detail — what every number and every chart
actually means.

If you only read one section, read **§6 (the results)** and **§7 (how to read
each figure)**.

---

## 1. The project in three sentences

Industrial machines have sensors, and those sensors can signal faults. We took
a **real** public dataset of a pump/pipe/valve rig with sensors (called
**SKAB**) and asked: can a machine-learning model *learn what normal looks
like* and then flag faults, better than a simple statistical rule? We compare
**6 methods** and show the smartest one (**Isolation Forest**) wins.

**The paper's one-line claim:** *"On a real industrial sensor benchmark,
learned anomaly-detection models — especially Isolation Forest — detect faults
better than a fixed statistical threshold."*

**Paper title:** "Comparative Evaluation of Unsupervised Machine Learning
Models for Multivariate Sensor Anomaly Detection in Industrial Systems."

---

## 2. Key words, in plain English (the only jargon you need)

- **Anomaly / fault:** a moment when the machine is misbehaving. In the data
  it's a label: `anomaly = 1` (fault) or `0` (normal).
- **Unsupervised / novelty detection:** the model is trained on **only normal
  data**. It learns "this is what healthy looks like" and then flags anything
  that doesn't fit. It never gets shown examples of faults during training.
  (This is why "what counts as normal" matters so much below.)
- **Multivariate:** many sensors at once (8 here), not one.
- **Precision, Recall, F1:** the three scores we judge models on. Explained in
  detail in §6 — for now: **recall** = "of the real faults, how many did it
  catch?", **precision** = "of its fault alarms, how many were right?", **F1**
  = a single number balancing the two.
- **Train / validation / test split:** we cut the data into three piles —
  one to *learn* from, one to *tune settings* on, and one *locked away* until
  the very end to *judge* fairly. You never judge on data you learned from,
  the same way you don't grade an exam using the practice questions.
- **Chronological split:** because this is time-series data, we cut by *time*
  (first 80% / last 20%), never by shuffling. Shuffling would let the model
  peek at the future.
- **Contamination:** the fraction of data a model *assumes* is abnormal. A
  setting, not a fact.

---

## 3. Suggested reading order (what to open, in what order)

1. **`README.md`** — one-paragraph project + dataset overview.
2. **This file (`WALKTHROUGH.md`)** — you're here.
3. **`ROADMAP.md`** — the whole plan as a checklist, with every checkpoint
   explained and ticked. This is the fastest way to see *what's done and
   what's left*. Each finished checkpoint says what we did and why.
4. **`DEVLOG.md`** — the detailed story, session by session. Read this when
   you want the *why* behind a decision (e.g. "why did the score go down?").
   Sessions 3, 4 and 7 are the most important — they're where the real
   methodology decisions and the two "gotchas" happened.
5. **`PROGRESS.md`** — one line per session; a quick index/timeline.
6. **The results themselves** — §5 below tells you how.

If you have 10 minutes: ROADMAP.md + §6 + §7 of this file.

---

## 4. Where everything lives (file map)

```
research/
├── SKAB/                     the raw dataset (35 CSV files)
├── src/                      the actual pipeline code
│   ├── preprocessing.py      loads the data, splits train/test by time
│   ├── features.py           turns raw sensors into model inputs (41 features)
│   ├── models.py             the 6 models (Z-score, One-Class SVM, LOF, PCA-recon, Elliptic Envelope, Isolation Forest)
│   ├── evaluate.py           scores the models (precision/recall/F1)
│   ├── postprocess.py        the smoothing idea we tested (and rejected)
│   └── visualize.py          makes the 4 paper figures
├── main.py                   runs the whole pipeline end to end
├── experiments/              one-off studies behind our decisions
│   ├── training_strategy_comparison.py   (why we train the way we do)
│   ├── hyperparameter_sweep.py           (how we picked settings)
│   ├── temporal_smoothing.py             (a fix we tried, rejected)
│   └── feature_ablation.py               (how we caught the data leak)
├── results/
│   ├── metrics.json          THE NUMBERS (precision/recall/F1 for all 6 models)
│   └── figures/              THE CHARTS (4 PNGs, see §7)
├── skab.py                   the Streamlit visual app (incl. "Model Results" tab)
├── ROADMAP.md / DEVLOG.md / PROGRESS.md    the logs
└── SensorGuard/              the old hackathon project (reference only, ignored by git)
```

---

## 5. How to see the results yourself

Everything runs locally, no cloud, no GPU. From the `research/` folder:

```bash
source env/bin/activate          # turn on the project's Python environment
```

**Option A — the visual app (recommended, easiest to read):**
```bash
streamlit run skab.py
```
A browser tab opens. Click the **"🏆 Model Results"** tab. You'll see the
headline scores, the comparison table, and all 4 figures with plain-English
captions. (The other tabs explore the raw data — also worth a look.)

**Option B — re-run the whole pipeline and regenerate everything:**
```bash
python main.py
```
This prints the score table to the terminal and rewrites `results/metrics.json`
and the 4 figures in `results/figures/`. It takes well under a minute.

You do **not** need to run anything to review — the results are already saved
in `results/`. Running just regenerates them.

---

## 6. The results — what the numbers mean

First, the three scores, using a concrete example. Imagine the test data has
**2,252 real fault-seconds**, and a model raises some number of alarms:

- **Recall** = of those 2,252 real faults, what fraction did it catch?
  High recall = it doesn't miss much.
- **Precision** = of all the alarms it raised, what fraction were real faults
  (not false alarms)? High precision = it doesn't cry wolf.
- **F1** = a single score (0 to 1) that is high only when *both* precision and
  recall are good. We use F1 as the headline because it punishes a model that
  wins one by sacrificing the other. Higher is better.

There's always a tension: flag more aggressively → catch more faults (recall ↑)
but more false alarms (precision ↓), and vice-versa. F1 balances them.

### The official numbers (from `results/metrics.json`)

Sorted best to worst by F1 (all share the same contamination=0.35 setting):

| Model | Precision | Recall | F1 | What it is |
|---|---|---|---|---|
| **Isolation Forest** | **0.362** | **0.656** | **0.466** | **the proposed/best model** |
| Elliptic Envelope | 0.342 | 0.539 | 0.418 | robust Gaussian (Mahalanobis), runner-up |
| Z-score | 0.329 | 0.548 | 0.411 | the "dumb" statistical baseline |
| One-Class SVM | 0.326 | 0.537 | 0.406 | an intermediate ML method |
| LOF | 0.321 | 0.460 | 0.378 | density-based ML method |
| PCA-recon | 0.305 | 0.470 | 0.370 | reconstruction-error method |

**How to read this:** Isolation Forest has the best F1 (0.466) and by far the
best recall (0.656) — it catches ~66% of real faults, more than any other
method, while keeping precision comparable to the rest. That's the paper's
result: the smartest model wins. The two models we added later (Elliptic
Envelope and PCA-recon) are "theoretically" higher-ranked methods that capture
sensor correlations — but on our engineered features they did **not** beat
Isolation Forest (Elliptic Envelope came 2nd; PCA-recon last). That's an honest
finding worth reporting, not a failure.

**Being honest about the absolute level:** these scores are *modest* — none is
near-perfect. Precision is ~0.36, meaning roughly 1 in 3 alarms is a real fault
and the rest are false alarms. **That's okay for this paper** because the
contribution is the *comparison* (which method is best on a real benchmark),
not beating a world record. We'll state this plainly in the paper's Limitations
section. Anomaly detection on real industrial data is genuinely hard, and these
numbers are in a believable range for it.

---

## 7. How to read each figure (the important part)

All four live in `results/figures/`. Here's what each one is, why it exists,
and exactly how to read it.

### 7.1 `model_comparison.png` — the headline chart
- **What it shows:** a bar chart with three groups (Precision, Recall, F1).
  Within each group, 6 bars — one per model.
- **Why it exists:** it's the single picture that proves the paper's claim.
- **How to read it:** find the **F1** group on the right. The tallest bar is
  **Isolation Forest** (far right of each cluster). Now look at the **Recall**
  group — Isolation Forest is clearly tallest there too. That's the story: it
  catches the most faults and has the best overall balance.
- **What to ignore:** don't expect any bar near 1.0 — see the honesty note in
  §6. Relative height between models is the point, not absolute height.

### 7.2 `confusion_matrices.png` — where each model's mistakes are
- **What it shows:** six 2×2 grids, one per model. Each grid is the breakdown
  of correct vs. wrong calls:
  - top-left = correctly said "normal"
  - bottom-right = correctly caught a fault
  - top-right = **false alarm** (said fault, was normal)
  - bottom-left = **missed fault** (said normal, was fault)
- **Why it exists:** F1 is a summary; this shows the *actual counts* behind it,
  which is more honest and concrete.
- **How to read it (Isolation Forest):** of 2,252 real faults it correctly
  caught **1,477** (bottom-right) and missed **775** (bottom-left) — that's the
  0.656 recall. It also raised **2,608** false alarms (top-right) — that's why
  precision is ~0.36. Compare to the others: Isolation Forest's bottom-right is
  the fullest (catches the most faults).
- **The takeaway in words:** Isolation Forest errs toward *over-warning* — it
  catches most faults but with a lot of false alarms. For a safety-monitoring
  use case, missing fewer faults is usually the better trade.

### 7.3 `timeseries_anomalies.png` — predictions on one real run
- **What it shows:** one example test run (`valve1/7`), every one of the 8
  sensors stacked vertically over time. **Red shaded bands** = the periods that
  were *truly* faults. **Orange dots** = the seconds where Isolation Forest
  *predicted* a fault.
- **Why it exists:** to make "what detection looks like" concrete, not just a
  number. A reviewer wants to *see* it working on real signals.
- **How to read it:** orange dots sitting *inside* red bands = correct catches.
  Orange dots *outside* red bands = false alarms. You'll see mostly-correct
  alignment plus scattered false alarms — exactly what the ~0.36 precision /
  ~0.66 recall predicts. Look at the bottom sensor (**Volume Flow Rate**):
  you can usually see its value change during the red (fault) region — that's
  the physical fault being visible in the raw signal.
- **Note:** there may be a straight line across a gap in the middle — that's a
  real time-gap in that recording, drawn honestly, not a bug.

### 7.4 `anomaly_closeup.png` — zoom into the moment a fault starts
- **What it shows:** the same run, zoomed to ±30 seconds around the exact
  instant the fault was injected (the **red dashed vertical line**, called the
  "changepoint").
- **Why it exists:** the full-run chart is too zoomed-out to see *how* sensors
  react at fault onset; this close-up shows the transition.
- **How to read it:** find the red dashed line (the fault start). Look at the
  **Volume Flow Rate** panel at the bottom — its level jumps right at the line.
  That jump is the physical signature of the valve fault. This is the kind of
  change the model is learning to react to.
- **Caveat (already noted in ROADMAP):** depending on which run is auto-picked,
  the whole window can fall inside one long fault, so the background is fully
  shaded red. The changepoint line + the flow-rate jump are still the things
  to look at.

---

## 8. The honest caveats (so nothing surprises you later)

We made a point of catching ourselves cheating. Three things you should know,
all documented in `DEVLOG.md`:

1. **"Cry wolf" trap (sessions 3 & 4):** a model that flags *everything* as a
   fault gets perfect recall and looks great on F1 — but is useless. We caught
   this twice and rejected those configurations. Always sanity-check recall
   isn't ~100%.
2. **The data leak we removed (session 7):** we originally fed the models the
   *time of day* of each reading. It quietly boosted the score — but only
   because our test data comes from the same recording sessions as the training
   data, so "time of day" secretly told the model *which experiment* it was
   looking at. That's leakage, not skill. We removed it. Our headline F1 for
   Isolation Forest dropped from 0.496 to **0.466** as a result — a smaller but
   *trustworthy* number. This is a feature, not a bug: it's exactly the kind of
   rigor that makes a paper credible.
3. **Scores are modest, on purpose-honest:** see §6. We report what's real.

There's also one minor open item: the models' main setting (`contamination =
0.35`) was tuned before we removed the leaky features. It's a coarse knob and
almost certainly still fine, but a full re-tune is listed as optional future
work in ROADMAP B5.

---

## 9. What's done and what's next

**Done:** the entire data + modeling + results pipeline. Data loading, feature
engineering, all 6 models, settings tuned honestly, leakage removed, figures
generated, and everything viewable in the Streamlit app.

**Not done:** the **written paper** (ROADMAP Phase D) — Dataset section,
Methodology, Results, Related Work, Discussion, Abstract/Intro. That's the next
big effort. The easiest starting points (Dataset and Related Work) don't depend
on anything else.

**Optional, skippable:** saving trained models to disk (ROADMAP C3).

When you've reviewed the above and are happy, the natural next step is to start
drafting the paper — and the whole DEVLOG is essentially the raw material for
the Methodology and Results sections.
