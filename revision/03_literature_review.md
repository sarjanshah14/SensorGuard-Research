# Deliverable 3: Literature Review Report

Search dates: 2026-10-28 (two rounds). Searcher: Claude (Opus 5.5) using a general web search tool and direct page fetches.

## 1. Method and honest coverage statement

- **Platforms actually used:** a general web search engine (results surface publisher pages, DOI resolvers, arXiv, PMLR, NeurIPS/VLDB/AAAI proceedings, PubMed Central, dblp-style indexes), plus direct fetches of GitHub (SKAB repository and its source code) and the PHM Society site. Google Scholar, Scopus, IEEE Xplore and Semantic Scholar APIs were **not** queried directly; their records appeared only when the search engine surfaced them.
- **Volume:** 32 queries; roughly 300 result records were returned and screened by title/snippet; about 45 landing pages, abstracts, or proceedings records were examined; **0 full papers were read end to end**. Two primary code sources were read in full (SKAB `core/utils.py` and the SKAB isolation-forest notebook).
- **Consequence:** every "main point" below comes from an abstract, a proceedings landing page, or a publisher record (marked *abstract*). Experimental details of cited papers are not claimed unless stated in those records. Before submission, the authors should read in full at least: Audibert 2022, Sarfraz 2024, Kim 2022, Garg 2021, Li 2019 (MAD-GAN) and Deng & Hooi 2021.
- **Inclusion:** peer-reviewed or widely cited preprints on (A) unsupervised multivariate/time-series AD, (B) SKAB/SWaT/ICS datasets, (C) multi-dataset comparisons, (D) concept drift, (E) edge/resource evaluation, (F) evaluation methodology, (G) reproducibility critiques.
- **Exclusion:** patents; retracted papers; vendor blogs; papers whose bibliographic record could not be confirmed; APC/fee pages; secondary summaries (used only to locate primaries).

## 2. Search log

| # | Round | Query (abridged) | Area | Kept |
|---|---|---|---|---|
| 1 | 1 | Audibert "Do deep neural networks contribute…" | C | audibert2022 |
| 2 | 1 | Kim "Towards a rigorous evaluation of TSAD" | F | kim2022 |
| 3 | 1 | Wu & Keogh "benchmarks are flawed" | G | wu2023 |
| 4 | 1 | Schmidl "comprehensive evaluation" PVLDB | C | schmidl2022 |
| 5 | 1 | Sarfraz "Quo vadis" ICML 2024 | G | sarfraz2024 |
| 6 | 1 | Liu & Paparrizos TSB-AD | F/G | liu2024 |
| 7 | 1 | SKAB GitHub (fetch + `core/utils.py`) | B | skab (protocol: first 400 rows/file train) |
| 8 | 1 | SWaT 449,919 / start-up removal | B | li2019madgan |
| 9 | 1 | Garg "evaluation of AD and diagnosis in MTS" | C/F | garg2021 |
| 10 | 1 | Paparrizos "Volume under the surface" | F | paparrizos2022 |
| 11 | 1 | ADBench; PyOD | A | han2022, zhao2019 |
| 12 | 1 | Lu "Learning under concept drift: a review" | D | lu2018 |
| 13 | 1 | Ahmad NAB Neurocomputing | D | ahmad2017 |
| 14 | 1 | Banbury MLPerf Tiny | E | banbury2021 |
| 15 | 1 | Shin HAI 1.0 | B | shin2020 |
| 16 | 1 | Deng & Hooi GDN; TranAD | A | deng2021 |
| 17-20 | 1 | venues and fees | — | (Deliverable 11) |
| 21 | 2 | USAD KDD 2020 | A | audibert2020 |
| 22 | 2 | TranAD PVLDB | A | tuli2022 |
| 23 | 2 | ECOD TKDE; COPOD | A | li2022ecod (COPOD DOI not found → cite via PyOD only) |
| 24 | 2 | Chicco & Jurman MCC | F | chicco2020 |
| 25 | 2 | Tatbul range precision/recall | F | tatbul2018 |
| 26 | 2 | Rabanser "Failing loudly" | D | rabanser2019 |
| 27 | 2 | Goldstein & Uchida PLoS ONE; HBOS | A | goldstein2016 |
| 28 | 2 | Hundman telemanom KDD 2018 | A | hundman2018 |
| 29 | 2 | drift + ICS retraining policies (extended) | D | no head-to-head periodic-vs-triggered study on SWaT found; closest: turbofan drift benchmark (venue not verifiable, excluded) |
| 30 | 2 | lightweight AD on Raspberry Pi / edge | E | arXiv 2512.19383 survey (preprint, cite with care); no standard Pi benchmark of classical detectors found |
| 31 | 2 | IJPHM author guidelines (fetch) | — | Deliverable 11 |
| 32 | 2 | SKAB isolation-forest notebook (fetch) | B | contamination 0.0005; 3-sample rolling median |

## 3. Verified references

All entries below had authors, title, venue and year confirmed on a publisher, proceedings, DOI or arXiv record. "Abstract" = evidence level for the main point.

### Foundational methods and datasets (retained from the original paper)
| Key | Citation | DOI / URL |
|---|---|---|
| chandola2009 | Chandola, Banerjee, Kumar. Anomaly detection: a survey. ACM Comput. Surv. 41(3), 2009 | 10.1145/1541880.1541882 |
| breunig2000 | Breunig et al. LOF. SIGMOD 2000 | 10.1145/342009.335388 |
| scholkopf2001 | Schölkopf et al. Estimating the support of a high-dimensional distribution. Neural Comput. 13(7), 2001 | 10.1162/089976601750264965 |
| rousseeuw1999 | Rousseeuw, Van Driessen. Fast MCD. Technometrics 41(3), 1999 | 10.1080/00401706.1999.10485670 |
| liu2008 | Liu, Ting, Zhou. Isolation forest. ICDM 2008 | 10.1109/ICDM.2008.17 |
| shyu2003 | Shyu et al. Principal component classifier. ICDM Workshop 2003 | (proceedings; no DOI) |
| malhotra2016 | Malhotra et al. LSTM-based encoder-decoder for multi-sensor AD. ICML AD Workshop 2016 | arXiv:1607.00148 |
| gama2014 | Gama et al. A survey on concept drift adaptation. ACM Comput. Surv. 46(4), 2014 | 10.1145/2523813 |
| bifet2007 | Bifet, Gavaldà. ADWIN. SDM 2007 | 10.1137/1.9781611972771.42 |
| massey1951 | Massey. KS test. JASA 46(253), 1951 | 10.1080/01621459.1951.10500769 |
| matthews1975 | Matthews. Biochim. Biophys. Acta 405(2), 1975 | 10.1016/0005-2795(75)90109-9 |
| goh2016 | Goh et al. SWaT dataset. CRITIS 2016 | 10.1007/978-3-319-71368-7_8 |
| mathur2016 | Mathur, Tippenhauer. SWaT testbed. CySWater 2016 | 10.1109/CySWater.2016.7469060 |
| skab | Katser, Kozitsin. SKAB. Kaggle 2020 | 10.34740/KAGGLE/DSV/1693952 |
| pedregosa2011 | Pedregosa et al. Scikit-learn. JMLR 12, 2011 | jmlr.org/papers/v12/pedregosa11a |

(DOIs for the retained entries were confirmed in sessions 13-21; see `REFERENCES.md`.)

### Benchmark, critique and closely related studies (new)
| Key | Citation | DOI / URL | Main point (*abstract*) |
|---|---|---|---|
| audibert2022 | Audibert, Michiardi, Guyard, Marti, Zuluaga. Do deep neural networks contribute to multivariate time series anomaly detection? Pattern Recognition 132:108945, 2022 | 10.1016/j.patcog.2022.108945 | 16 conventional, ML and deep methods on 5 datasets; no family consistently best |
| sarfraz2024 | Sarfraz, Chen, Layer, Peng, Koulakis. Position: Quo vadis, unsupervised time series anomaly detection? ICML 2024, PMLR 235:43461-43476 | proceedings.mlr.press/v235/sarfraz24a | simple baselines match SOTA deep models; point-adjusted F1 flawed |
| garg2021 | Garg, Zhang, Samaran, Savitha, Foo. An evaluation of anomaly detection and diagnosis in multivariate time series. IEEE TNNLS, 2021 | 10.1109/TNNLS.2021.3105827 | existing metrics cannot separate good from trivial detectors; scoring function matters more than model |
| schmidl2022 | Schmidl, Wenig, Papenbrock. Anomaly detection in time series: a comprehensive evaluation. PVLDB 15(9):1779-1797, 2022 | 10.14778/3538598.3538602 | 71 algorithms; multivariate harder on average |
| kim2022 | Kim, Choi, Choi, Lee, Yoon. Towards a rigorous evaluation of time-series anomaly detection. AAAI 36(7):7194-7201, 2022 | 10.1609/aaai.v36i7.20680 | point adjustment lets random scores look state-of-the-art |
| wu2023 | Wu, Keogh. Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. IEEE TKDE 35(3):2421-2429, 2023 | 10.1109/TKDE.2021.3112126 | triviality, unrealistic density, mislabels, run-to-failure bias |
| liu2024 | Liu, Paparrizos. The elephant in the room: towards a reliable time-series anomaly detection benchmark. NeurIPS 37 (D&B), 2024 | proceedings.neurips.cc (vol. 37, pp. 108231-108261) | TSB-AD; VUS-PR most reliable measure |
| paparrizos2022 | Paparrizos et al. Volume under the surface. PVLDB 15(11):2774-2787, 2022 | 10.14778/3551793.3551830 | threshold-free, range-aware AUC |
| tatbul2018 | Tatbul, Lee, Zdonik, Alam, Gottschlich. Precision and recall for time series. NeurIPS 31, 2018 | arXiv:1803.03639 | range-based precision/recall |
| chicco2020 | Chicco, Jurman. The advantages of the MCC over F1 score and accuracy. BMC Genomics 21:6, 2020 | 10.1186/s12864-019-6413-7 | MCC more informative than F1 under imbalance |
| goldstein2016 | Goldstein, Uchida. A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data. PLoS ONE 11(4):e0152173, 2016 | 10.1371/journal.pone.0152173 | 19 algorithms, 10 datasets |
| han2022 | Han, Hu, Huang, Jiang, Zhao. ADBench. NeurIPS 35 (D&B), 2022 | arXiv:2206.09426 | 30 algorithms, 57 datasets |
| zhao2019 | Zhao, Nasrullah, Li. PyOD. JMLR 20(96):1-7, 2019 | jmlr.org/papers/v20/19-011 | toolkit used for ECOD/COPOD/HBOS |
| li2022ecod | Li, Zhao, Hu, Botta, Ionescu, Chen. ECOD. IEEE TKDE 35(12):12181-12193 | 10.1109/TKDE.2022.3159580 | parameter-free tail-probability detector |
| audibert2020 | Audibert et al. USAD. KDD 2020 | 10.1145/3394486.3403392 | adversarially trained AE on SWaT and others |
| tuli2022 | Tuli, Casale, Jennings. TranAD. PVLDB 15(6):1201-1214, 2022 | 10.14778/3514061.3514067 | transformer AD; reports F1 gains on 6 datasets |
| deng2021 | Deng, Hooi. Graph neural network-based anomaly detection in multivariate time series. AAAI 2021 | arXiv:2106.06947 | GDN on SWaT and WADI (venue confirmed via AAAI record still pending) |
| li2019madgan | Li et al. MAD-GAN. ICANN 2019 | arXiv:1809.04758 | SWaT: 496,800 normal, 449,919 attack-period rows; first 21,600 rows removed (from search snippet of the paper) |
| hundman2018 | Hundman et al. Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. KDD 2018 | 10.1145/3219819.3219845 | LSTM prediction-error AD; dynamic thresholds |
| shin2020 | Shin, Lee, Yun, Kim. HAI 1.0. USENIX CSET 2020 | usenix.org/conference/cset20/presentation/shin | HIL ICS dataset (future 3rd dataset) |
| ahmed2017 | Ahmed, Palleti, Mathur. WADI. CySWater 2017 | 10.1145/3055366.3055375 | WADI (retained) |

### Concept drift and shift detection (new)
| Key | Citation | DOI / URL | Main point (*abstract*) |
|---|---|---|---|
| lu2018 | Lu, Liu, Dong, Gu, Gama, Zhang. Learning under concept drift: a review. IEEE TKDE 31(12):2346-2363 | 10.1109/TKDE.2018.2876857 | drift detection / understanding / adaptation |
| rabanser2019 | Rabanser, Günnemann, Lipton. Failing loudly. NeurIPS 32, 2019 | arXiv:1810.11953 | two-sample tests for shift detection; univariate KS on reduced representations is competitive |
| ahmad2017 | Ahmad, Lavin, Purdy, Agha. Unsupervised real-time anomaly detection for streaming data. Neurocomputing 262:134-147, 2017 | 10.1016/j.neucom.2017.04.070 | streaming AD; NAB |

### Edge / resource evaluation (new)
| Key | Citation | DOI / URL | Main point (*abstract*) |
|---|---|---|---|
| banbury2021 | Banbury et al. MLPerf Tiny benchmark. NeurIPS D&B 2021 | arXiv:2106.07597 | accuracy, latency and energy measured together; includes an AD task |

### Dropped from the original bibliography (reason)
Ester 1996 (DBSCAN; tangential), Harris 2020 (NumPy), McKinney 2010 (pandas), Kingma 2015 (Adam), Paszke 2019 (PyTorch), Merkel 2014 (Docker, *Linux Journal*), Hochreiter 1997 (only needed if LSTM internals are described; kept as one-line method cite), Das 2023 (weakly related review), Belay 2023 and Chalapathy 2019 (retained only where a survey claim needs them), Powers 2011 (superseded by Chicco 2020 for the MCC argument). Software is acknowledged in a reproducibility statement, not as references.

## 4. Critical comparison matrix (most relevant studies)

| Study | Problem | Datasets | Methods compared | Metrics | Key finding (*abstract*) | Limitations noted | Relevance to us | Full text read? |
|---|---|---|---|---|---|---|---|---|
| Audibert 2022 | Do deep models help MTS AD? | 5 public MTS incl. SWaT | 16: conventional, ML, deep | F1-family | No family wins consistently | — (not read) | Directly overlaps our "no single winner" claim → we must not present it as new | No |
| Sarfraz 2024 | State of unsupervised TSAD | standard MTS benchmarks | simple baselines vs SOTA deep | non-PA metrics | Simple/linear baselines match deep models | position paper | Supports using classical detectors; overlaps our deep-vs-classical point | No |
| Garg 2021 | Evaluate MTS AD and diagnosis | SWaT, WADI, others | many | composite F-score | Trivial detectors fool common metrics; scoring function matters | — | Supports our MCC/flag-everything argument; our threshold-rule sensitivity echoes "scoring matters" | No |
| Kim 2022 | Evaluation rigour | SWaT, WADI, others | SOTA deep | PA, PA%K | PA inflates results | — | We use no PA; cite to justify point + event metrics | No |
| Wu & Keogh 2023 | Benchmark quality | Yahoo, NAB, NASA | — | — | Benchmarks flawed | univariate focus | Motivates our dataset-provenance check | No |
| Liu & Paparrizos 2024 | Reliable benchmark | TSB-AD 1,000+ series | 40 methods | VUS-PR | VUS-PR most reliable | — | Justifies threshold-free metrics | No |
| Li 2019 (MAD-GAN) | GAN-based MTS AD | SWaT, WADI | GAN vs baselines | P/R/F1 | — | — | Source for the standard SWaT split | No (snippet) |
| SKAB repo | Benchmark definition | SKAB | Conv-AE, MSET, T²+Q, LSTM-AE, IF… | F1, FAR, MAR, NAB | Conv-AE F1 0.78; IF F1 0.29 | per-file protocol | Our pooled protocol is non-standard → report both | Yes (code) |
| Lu 2018 / Gama 2014 | Drift adaptation | — | — | — | Detect → understand → adapt | — | Frames our drift section | No |
| Rabanser 2019 | Shift detection | image/tabular | two-sample tests | detection power | univariate tests competitive | high- vs low-data | Supports KS-based drift score | No |
| MLPerf Tiny 2021 | Edge benchmarking | 4 tasks incl. AD | — | accuracy/latency/energy | standard measurement protocol | — | Our edge section must report single-inference latency, not batch averages | No |

## 5. Established methodological pitfalls (and where our original paper stood)

| Pitfall | Source | Original paper | Revised paper |
|---|---|---|---|
| Point-adjusted F1 inflation | Kim 2022; Sarfraz 2024 | not used (good) | not used |
| Metrics that reward trivial detectors | Garg 2021; Chicco 2020 | addressed with MCC (good) | MCC + AUROC/AUPRC + event recall |
| Threshold-dependent comparisons only | Paparrizos 2022; Liu 2024 | yes (problem) | threshold-free AUROC/AUPRC added |
| Inconsistent threshold selection across models | Garg 2021 ("scoring matters") | yes (F1 on SKAB, MCC on SWaT, IF-only tuning) | one rule per model, two rules reported |
| Non-standard / mis-identified splits | Wu & Keogh 2023 | SWaT split mis-identified | standard split + sensitivity analysis |
| Single run, no uncertainty | Schmidl 2022 (multi-dataset), general practice | yes | block bootstrap + seeds |
| Treating autocorrelated rows as i.i.d. | general time-series practice | n/a (no CIs) | blocks = files / hours |
| Ignoring the dataset's own benchmark protocol | SKAB repo | yes | per-file SKAB protocol reported |

## 6. Open problems that genuinely remain (as far as this search shows)

1. **Retraining policy under drift for lightweight ICS detectors.** No study found compares drift-triggered, periodic and continuous retraining at matched cost on SWaT, nor quantifies the label assumption in retraining data selection. (Search #29 found related drift work in ICS but no head-to-head comparison. This is a search-limited statement, not proof of absence.)
2. **Protocol sensitivity of public re-uploads.** Critiques address labels and metrics (Wu & Keogh 2023; Kim 2022) but this search found no report that a widely downloaded SWaT re-upload changes the train/test semantics. Needs a targeted check before claiming.
3. **Joint accuracy-and-cost reporting for classical detectors in capped environments.** MLPerf Tiny defines the measurement approach for neural models on micro-controllers; no equivalent for classical multivariate detectors on industrial data was found.

## 7. Implications for the research design

- The "no universal winner" and "deep is not always better" points are **supporting context**, cited to Audibert 2022 and Sarfraz 2024, not contributions.
- The contribution moves to (1) protocol sensitivity (split definition, threshold rule, metric) and (2) the cost/benefit of retraining policies with and without labels, plus (3) measured resource cost.
- Threshold-free metrics, event-level recall, block-bootstrap CIs and the official SKAB protocol are required.
