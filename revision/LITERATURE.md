# Literature Review Log (Deliverable 3, in progress)

Search date: 2026-10-28. Tool: Claude Code WebSearch (web search engine, US results) + WebFetch of primary pages. No direct API access to Google Scholar / Scopus.
**Honest coverage statement:** Stage 1 ran 20 targeted searches and read ~25 result pages (abstracts, publisher/arXiv landing pages, one GitHub README). No full papers have yet been read end to end. "Verified" below means bibliographic details were confirmed on a publisher, proceedings, arXiv or DOI page. Findings marked (abstract) come from abstracts or landing pages only.

## Search log

| # | Query (abridged) | Purpose | Kept |
|---|---|---|---|
| 1 | Audibert "Do deep neural networks contribute to MTS AD" | classical vs deep benchmark | Audibert 2022 |
| 2 | Kim "Towards a rigorous evaluation of TSAD" | metric critique | Kim 2022 |
| 3 | Wu & Keogh "benchmarks are flawed" | benchmark critique | Wu & Keogh 2021/23 |
| 4 | Schmidl "comprehensive evaluation" PVLDB | large benchmark | Schmidl 2022 |
| 5 | Sarfraz "Quo Vadis" ICML 2024 | simple baselines critique | Sarfraz 2024 |
| 6 | Liu & Paparrizos "Elephant in the room" TSB-AD | benchmark + VUS-PR | Liu 2024 |
| 7 | SKAB GitHub (fetch) | dataset protocol + leaderboard | SKAB repo |
| 8 | SWaT 449,919 / stabilisation | standard SWaT protocol | Li 2019 MAD-GAN (arXiv:1809.04758) |
| 9 | Garg "evaluation of AD and diagnosis in MTS" | MTS benchmark, trivial detectors | Garg 2021 |
| 10 | Paparrizos "Volume under the surface" | threshold-free range metric | Paparrizos 2022 |
| 11 | ADBench / PyOD | tabular AD benchmark + toolkit | Han 2022, Zhao 2019 |
| 12 | Lu et al. "Learning under concept drift: a review" | drift theory | Lu 2018 |
| 13 | Ahmad et al. NAB Neurocomputing | streaming AD | Ahmad 2017 |
| 14 | Banbury MLPerf Tiny | edge benchmarking method | Banbury 2021 |
| 15 | Shin HAI 1.0 | candidate 3rd dataset | Shin 2020 |
| 16 | Deng & Hooi GDN / TranAD | modern SWaT methods | GDN (arXiv 2106.06947), TranAD venue unverified |
| 17-20 | venue + fee searches (see CHECKPOINT.md) | venue strategy | IJPHM |

Exclusions so far: patents, retracted papers, unrelated APC pages, secondary review blogs (used only to locate primaries).

## Verified references (candidates for the new bibliography)

| Key | Citation | DOI / URL | Year | Role | Main point (source) |
|---|---|---|---|---|---|
| audibert2022 | Audibert, Michiardi, Guyard, Marti, Zuluaga. Do deep neural networks contribute to multivariate time series anomaly detection? *Pattern Recognition* 132, 108945 | 10.1016/j.patcog.2022.108945 | 2022 | closest prior work | 16 methods, 5 datasets; no family consistently best (abstract) |
| kim2022 | Kim, Choi, Choi, Lee, Yoon. Towards a rigorous evaluation of time-series anomaly detection. *AAAI* 36(7):7194-7201 | 10.1609/aaai.v36i7.20680 | 2022 | metric critique | point-adjust inflates F1; random scores look SOTA (abstract) |
| wu2023 | Wu, Keogh. Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. *IEEE TKDE* 35(3):2421-2429 | 10.1109/TKDE.2021.3112126 | 2021 (online) / 2023 | benchmark critique | triviality, unrealistic density, mislabels, run-to-failure bias (abstract) |
| schmidl2022 | Schmidl, Wenig, Papenbrock. Anomaly detection in time series: a comprehensive evaluation. *PVLDB* 15(9):1779-1797 | 10.14778/3538598.3538602 | 2022 | large benchmark | 71 algorithms; multivariate harder than univariate (abstract) |
| sarfraz2024 | Sarfraz, Chen, Layer, Peng, Koulakis. Position: Quo vadis, unsupervised time series anomaly detection? *ICML*, PMLR 235:43461-43476 | proceedings.mlr.press/v235/sarfraz24a | 2024 | critique | simple baselines match SOTA; PA-F1 flawed (abstract) |
| liu2024 | Liu, Paparrizos. The elephant in the room: towards a reliable time-series anomaly detection benchmark. *NeurIPS 37 Datasets & Benchmarks*, 108231-108261 | proceedings.neurips.cc (2024) | 2024 | benchmark | TSB-AD; VUS-PR most reliable measure (abstract) |
| garg2021 | Garg, Zhang, Samaran, Savitha, Foo. An evaluation of anomaly detection and diagnosis in multivariate time series. *IEEE TNNLS* | 10.1109/TNNLS.2021.3105827 | 2021 | MTS benchmark | composite F-score; scoring function matters more than model (abstract) |
| paparrizos2022 | Paparrizos, Boniol, Palpanas, Tsay, Elmore, Franklin. Volume under the surface. *PVLDB* 15(11):2774-2787 | 10.14778/3551793.3551830 | 2022 | metric | threshold-free range-aware AUC (abstract) |
| han2022 | Han, Hu, Huang, Jiang, Zhao. ADBench: anomaly detection benchmark. *NeurIPS 35 Datasets & Benchmarks* | arXiv:2206.09426 | 2022 | tabular AD benchmark | 30 algorithms, 57 datasets (abstract) |
| zhao2019 | Zhao, Nasrullah, Li. PyOD: a Python toolbox for scalable outlier detection. *JMLR* 20(96):1-7 | jmlr.org/papers/v20/19-011 | 2019 | toolkit (if PyOD baselines added) | |
| lu2018 | Lu, Liu, Dong, Gu, Gama, Zhang. Learning under concept drift: a review. *IEEE TKDE* 31(12):2346-2363 | 10.1109/TKDE.2018.2876857 | 2018 (vol. 2019) | drift theory | detection / understanding / adaptation framework (abstract) |
| ahmad2017 | Ahmad, Lavin, Purdy, Agha. Unsupervised real-time anomaly detection for streaming data. *Neurocomputing* 262:134-147 | 10.1016/j.neucom.2017.04.070 | 2017 | streaming AD | NAB benchmark; online HTM (abstract). Volume/pages to confirm |
| banbury2021 | Banbury et al. MLPerf Tiny benchmark. *NeurIPS Datasets & Benchmarks* | arXiv:2106.07597 | 2021 | edge evaluation method | accuracy + latency + energy, includes AD task (abstract) |
| shin2020 | Shin, Lee, Yun, Kim. HAI 1.0: HIL-based augmented ICS security dataset. *USENIX CSET '20* | usenix.org/conference/cset20/presentation/shin | 2020 | candidate dataset | |
| li2019madgan | Li, Chen, Jin, Shi, Goh, Ng. MAD-GAN: multivariate anomaly detection for time series data with GANs. *ICANN 2019* | arXiv:1809.04758 | 2019 | SWaT protocol source | 496,800 normal / 449,919 attack rows; first 21,600 rows dropped (from search snippet; confirm in full text) |
| deng2021 | Deng, Hooi. Graph neural network-based anomaly detection in multivariate time series. *AAAI 2021* | arXiv:2106.06947 | 2021 | modern SWaT/WADI baseline | venue to confirm |
| skab | Katser, Kozitsin. Skoltech Anomaly Benchmark (SKAB). Kaggle | 10.34740/KAGGLE/DSV/1693952 | 2020 | dataset | leaderboard: Conv-AE F1 0.78, IF F1 0.29 (README, fetched) |

Retain from current bibliography (already verified in sessions 13-21): Chandola 2009, Breunig 2000 (LOF), Schölkopf 2001 (OCSVM), Rousseeuw 1999 (MCD), Liu 2008 (IF), Malhotra 2016 (LSTM-ED), Gama 2014, Bifet 2007 (ADWIN), Goh 2016 + Mathur 2016 (SWaT), Matthews 1975 / Chicco 2020 (MCC, to add), Massey 1951 (KS), Pedregosa 2011 (sklearn).
Candidates to drop: Ester 1996 (DBSCAN, tangential), NumPy, pandas, Adam, Docker/Linux Journal, Hochreiter 1997 (keep only if LSTM is described), Das 2023 (weak relevance; replace with a PHM-specific review).

## Still to search (Stage 2)
USAD (Audibert KDD 2020), TranAD (Tuli PVLDB 2022, verify), Anomaly Transformer (Xu ICLR 2022), OmniAnomaly (Su KDD 2019), ECOD (Li TKDE 2022), COPOD (Li ICDM 2020), Chicco & Jurman 2020 (MCC vs F1), Hundman 2018 (NASA SMAP/MSL), drift-aware AD for ICS (search needed), drift detectors Page-Hinkley (Page 1954), Rabanser 2019 "Failing loudly", edge AD on Raspberry Pi / microcontrollers (search needed), PHM reviews in IJPHM (venue fit), block bootstrap for time series (Künsch 1989 / Politis & Romano 1994), Lavin & Ahmad 2015 NAB, Tatbul 2018 range-based precision/recall (NeurIPS).
Target: screen 100+ records, read the ~15 most relevant in full.
