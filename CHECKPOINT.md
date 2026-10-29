# CHECKPOINT: resume here (last updated 2026-10-30)

Paste this to a fresh Claude session: "Read CHECKPOINT.md, CLAUDE.md, then continue."

## What this project is
Paper by Sarjan Shah + Prof. Abhi Shah (Lok Jagruti University), revised after the
ICACS-2026 rejection and targeting IJPHM: "Protocol Sensitivity in Unsupervised
Industrial Anomaly Detection: Threshold Transfer, Retraining Risk, and Inference Cost".
10 detectors × 4 datasets (SKAB, SWaT official, WADI A2 official, HAI 21.03).

## >>> CURRENT STAGE (read first) <<<
- 2026-10-30 (session 28): second external review addressed. Word file rebuilt: tables readable, Table 9 numbers restored, captions labelled, 19 pages; checked in a LibreOffice PDF render (`paper/current/Shah_IJPHM_preview_LibreOffice.pdf`). SKAB/HAI/drift/resource text reconciled with the JSON results. Next: Sarjan opens the .docx in Word, reads it with Prof. Shah, submits.
- 2026-10-30: folder reorganised (see README.md layout). Official SWaT (iTrust A1&A2 Dec 2015) and WADI A2 obtained and run. Kaggle SWaT copy matches official with the same 24 sensors (AUROC within 0.001); the 25th sensor AIT201 (missing from the copy) drifts and breaks z-score/OCSVM label-free thresholds (z FAR 1.4% -> 42%). WADI: 3 drifting analysers cause 56% false alarms for 6 detectors; removing them -> ~1%.
- Paper: `paper/current/paper2_revised.tex`; Word: `paper/current/Shah_IJPHM_Protocol_Sensitivity_Industrial_Anomaly_Detection.docx` (`python paper/current/make_ijphm_docx.py`). 45 refs.
- DONE 2026-10-30: drift study on official SWaT (`results/revised/drift_official.json`); paper drift section updated. Paper is FINAL pending Sarjan/Prof. Shah read-through. Code public at https://github.com/sarjanshah14/SensorGuard-Research (link in Data Availability).
- PENDING (Sarjan): final Word layout check + read with Prof. Shah; submit to IJPHM https://papers.phmsociety.org/index.php/ijphm/about/submissions
- Explainers: https://claude.ai/artifact/GPYx5MBE1RYdqSZ9FJiRun (from-zero explainer), https://claude.ai/artifact/92o9bRhq862TJwb8u59sGN (before/after). Both private.
- Reports: `docs/revision/` 01-11. Logs: `docs/PROGRESS.md`, `docs/DEVLOG.md`, `docs/ROADMAP.md`.

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
