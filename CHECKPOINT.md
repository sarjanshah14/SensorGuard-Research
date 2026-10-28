# CHECKPOINT: resume here (last updated 2026-10-28)

Paste this to a fresh Claude session: "Read CHECKPOINT.md, CLAUDE.md, then continue."

## What this project is
Paper by Sarjan Shah + Prof. Abhi Shah (Lok Jagruti University): unsupervised
anomaly detection on industrial sensor data. 6 classical detectors + 1 LSTM-AE on
SKAB and SWaT, concept-drift adaptation, Docker "Pi-class" edge benchmark.
Current paper: `paper/paper1.tex` (Springer LLNCS). Final submitted PDF:
`Shah_ICACS2026_Unsupervised_Anomaly_Detection.pdf`. Code/results all done and
reproducible (see CLAUDE.md, README.md; `python main.py`, `python -m experiments.<name>`).

## >>> CURRENT STAGE (read first) <<<
Revision project (after ICACS rejection) — ALL STAGES DONE 2026-10-28 except items needing humans.
- 2026-10-29: HAI 21.03 added as 3rd dataset (`experiments/revision/hai_revised.py`, `results/revised/hai.json`, data in `external_data/hai/` gitignored). Title changed to "Evaluation Protocol Decides the Winner: Threshold Stability, Retraining Risk and Cost of Lightweight Unsupervised Anomaly Detectors for Industrial Sensor Data". IJPHM Word version: `paper/Shah_IJPHM_Evaluation_Protocol_Decides_the_Winner.docx` built by `python3 paper/make_ijphm_docx.py` from the .tex + `ijphm/ijPHM-FullPaperTemplate.docx`. 43 refs (3 IJPHM papers added).
- Revised paper: `paper/paper2_revised.tex` (+ `paper/overleaf_paper2/` with figures). Venue-neutral article, author-year refs (IJPHM uses APA). NOT compiled (no LaTeX locally) — compile on Overleaf.
- Deliverables: `revision/` 01..11 (stage-1 audit file holds 01/02/05/06 plan; 03 lit review; 04 novelty; 06 status; 08 change log; 09 repro package; 10 final assessment; 11 venues).
- Revised code: `experiments/revision/` (common, skab_revised, swat_revised, drift_revised, edge_revised, figures_revised) + `docker/Dockerfile.edge2`. Results: `results/revised/*.json`, figures in `results/revised/figures/`. Originals untouched.
- Before/after explainer Artifact: https://claude.ai/artifact/92o9bRhq862TJwb8u59sGN (private; share via its Share menu).
- Headline revised findings: SWaT standard split -> 10 detectors tied (AUROC 0.825-0.858); label-free MCC best z-score 0.716; original winners collapse (EE 0.008, LSTM-AE 0.134). SKAB pooled ~chance; SKAB per-run protocol run-AUROC up to 0.79. Drift: real normal week breaks static IF (1% -> 40-94% after ~48h); retraining helps only with labelled-normal data; KS ≈ periodic; label-free retraining fails. IF single-reading latency 2.3-2.8 ms (orig claimed 0.002).
- 2026-10-28: Prof. Shah APPROVED the new framing; Sarjan compiles on Overleaf himself. Target IJPHM: https://papers.phmsociety.org/index.php/ijphm/about/submissions
- HUMAN TODO: (1) request official SWaT + WADI from iTrust (https://itrust.sutd.edu.sg/itrust-labs_datasets/), rerun swat_revised + drift_revised; HAI (public, github.com/icsdataset/hai) can be added meanwhile; (3) IJPHM template; (4) read 6 key papers in full (revision/03 §1); (5) optional HAI dataset.

## Status
- ICACS 2026 (Paper ID 580): **REJECTED** 2026-10-28. Scores 56 and 49 of 100
  (pass = 60). No written comments. Weakest: references 4/10 (both), novelty 4-5,
  depth 4-5, critical qualities 4-5, language 5-6.
- Plan: strengthen, then submit to a free venue.

## Inferred flaws (from reading the PDF)
1. Refs: 28, mostly pre-2012 classics + tool cites (NumPy, pandas, PyTorch, Adam, Docker); almost no 2020-26 AD papers.
2. Only classical + one tiny LSTM-AE; no USAD/TranAD/Anomaly-Transformer, no PyOD (HBOS/COPOD/ECOD/kNN/DeepSVDD).
3. Single run: no seeds, no CIs, no significance tests; only threshold-dependent metrics (no AUROC/AUPRC).
4. Drift study only on Isolation Forest (not the SWaT winner Elliptic Envelope, not the LSTM-AE); no periodic-retrain or ADWIN baseline; SKAB drift recovery tiny (MCC 0.106).
5. SKAB best MCC 0.147 (near chance). SWaT thresholds tuned using attack labels in validation (not purely unsupervised).
6. Edge claim is an ARM Docker container, not a real Raspberry Pi.
7. Both datasets are water systems. SWaT from Kaggle re-upload (limitations sentence reads like a TODO; request official iTrust access).
8. Table captions say "six models" but tables hold seven. Prose needs a humanising pass.

## TODO (in priority order)
1. Rigor: AUROC/AUPRC, 5-10 seeds + CIs, significance test, threshold chosen without attack labels variant.
2. Baselines: PyOD set (HBOS, COPOD, ECOD, kNN, Deep SVDD), USAD, TranAD.
3. Data: add HAI (public GitHub) + one non-water set (bearing/rotating machinery); request SWaT officially.
4. Drift: run on winner + LSTM-AE; add periodic-retrain and ADWIN baselines; threshold ablation.
5. Edge: NO Raspberry Pi and no time (confirmed 2026-10-28). Reword everywhere as "Pi-class resource-constrained ARM64 container emulation"; keep the optimistic-latency caveat; real-Pi run stays future work.
6. Rewrite refs to ~35-40, >=50% from 2020-26; drop tool cites. Fix captions, SWaT sentence, humanise language (impersonal voice, no "we").
7. Update stale docs: WALKTHROUGH.md, README.md, CLAUDE.md, ROADMAP.md "Right now" (all still SKAB-only).

## Target venues (free = no author fee; conferences still charge registration)
- **IJPHM** (PHM Society): no submission/article fees, diamond OA, Scopus/ESCI/DOAJ. Top pick. Check current author instructions (site has stale pages).
- TMLR: free (unverified), but wants ML novelty. Elsevier hybrid journals (Eng. Appl. of AI, Computers in Industry): free on subscription route; confirm fees.
- IEEE INDIN 2027 (Lisbon, 10-14 May 2027, deadline not found); IEEE CASE 2027 (Linz, Aug; deadline ~1 Mar 2027 per one listing, verify).
- arXiv preprint in parallel. Avoid MDPI / IEEE Access (APCs).

## LOW-EFFORT PATH (chosen constraint: no Pi, little time; Sarjan has ~30 min/day)
Claude does the work; Sarjan only reviews. Cheapest changes with best score gain, in order:
A. Text only (no experiments): rewrite refs (drop tool cites, add 2020-26 AD papers, verified real), fix captions, reword edge as emulation, fix SWaT sentence, humanise language, retitle to IJPHM scope (health monitoring).
B. Quick scripts (minutes of compute on the Mac): AUROC/AUPRC for all models, 5 seeds with CIs, add PyOD baselines (HBOS/COPOD/ECOD/kNN), drift on the SWaT winner (Elliptic Envelope) + periodic-retrain baseline.
C. Skip for now: HAI/bearing datasets, USAD/TranAD, real Pi, official SWaT request (mention as limitation/future work).
Then submit to IJPHM, post arXiv preprint.

## Open questions for Sarjan
Is Prof. Shah OK with a journal target (IJPHM)? Official SWaT access? (Pi: answered, none.)

## Working rules (see CLAUDE.md + memory)
Claude drives code; Sarjan reviews. Every session: update PROGRESS.md (one line), DEVLOG.md (narrative), ROADMAP.md, and this file. End chats with a plain-English summary. Paper voice is impersonal (no "we/our"). No TeX engine locally: compile on Overleaf (`paper/overleaf_paper1.zip`, regenerable, not in git).
