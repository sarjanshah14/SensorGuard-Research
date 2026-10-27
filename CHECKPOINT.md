# CHECKPOINT: resume here (last updated 2026-10-28)

Paste this to a fresh Claude session: "Read CHECKPOINT.md, CLAUDE.md, then continue."

## What this project is
Paper by Sarjan Shah + Prof. Abhi Shah (Lok Jagruti University): unsupervised
anomaly detection on industrial sensor data. 6 classical detectors + 1 LSTM-AE on
SKAB and SWaT, concept-drift adaptation, Docker "Pi-class" edge benchmark.
Current paper: `paper/paper1.tex` (Springer LLNCS). Final submitted PDF:
`Shah_ICACS2026_Unsupervised_Anomaly_Detection.pdf`. Code/results all done and
reproducible (see CLAUDE.md, README.md; `python main.py`, `python -m experiments.<name>`).

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
5. Edge: real Raspberry Pi run if Sarjan has one (ASK), else reword as emulation.
6. Rewrite refs to ~35-40, >=50% from 2020-26; drop tool cites. Fix captions, SWaT sentence, humanise language (impersonal voice, no "we").
7. Update stale docs: WALKTHROUGH.md, README.md, CLAUDE.md, ROADMAP.md "Right now" (all still SKAB-only).

## Target venues (free = no author fee; conferences still charge registration)
- **IJPHM** (PHM Society): no submission/article fees, diamond OA, Scopus/ESCI/DOAJ. Top pick. Check current author instructions (site has stale pages).
- TMLR: free (unverified), but wants ML novelty. Elsevier hybrid journals (Eng. Appl. of AI, Computers in Industry): free on subscription route; confirm fees.
- IEEE INDIN 2027 (Lisbon, 10-14 May 2027, deadline not found); IEEE CASE 2027 (Linz, Aug; deadline ~1 Mar 2027 per one listing, verify).
- arXiv preprint in parallel. Avoid MDPI / IEEE Access (APCs).

## Open questions for Sarjan
Does he have a Raspberry Pi? Is Prof. Shah OK with a journal target? Official SWaT access?

## Working rules (see CLAUDE.md + memory)
Claude drives code; Sarjan reviews. Every session: update PROGRESS.md (one line), DEVLOG.md (narrative), ROADMAP.md, and this file. End chats with a plain-English summary. Paper voice is impersonal (no "we/our"). No TeX engine locally: compile on Overleaf (`paper/overleaf_paper1.zip`, regenerable, not in git).
