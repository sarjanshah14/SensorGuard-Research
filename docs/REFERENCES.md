# References — verified real citations for the paper

Every entry below is a real, published, verifiable work (I checked each one
against its publisher / arXiv / DOI — none are invented). They're written in
**IEEE reference style** (the format the IEEE conference template expects),
numbered so we can cite them in-text as `[1]`, `[2]`, … when we write.

The "Use it for" note says which section/claim each one backs. This was the
original 12-entry pool.

> **Update (2026-07-16):** the paper now cites **28** references (Prof. Shah
> asked for 25–30). The 16 added beyond this pool, all real and verified, are:
> SWaT (`goh2016`, `mathur2016`), WADI (`ahmed2017`), concept drift (`gama2014`,
> `bifet2007`), KS test (`massey1951`), metrics (`matthews1975`, `powers2011`),
> deep learning (`hochreiter1997`, `sakurada2014`, `kingma2015`, `paszke2019`),
> tooling (`harris2020`, `mckinney2010`, `merkel2014`), and DBSCAN (`ester1996`).
> The authoritative list is the bibliography in `paper/paper1.tex`.

> ✅ All references verified. [11] (rotating-machinery review) author list was
> confirmed against the published Heliyon/Cell record on 2026-07-16: Oguzhan Das,
> Duygu Bagci Das, Derya Birant — matches the bibitem `Das, O., Das, D.B.,
> Birant, D.` Nothing outstanding.

---

## Dataset

**[1]** I. D. Katser and V. O. Kozitsin, "Skoltech Anomaly Benchmark (SKAB),"
Kaggle, 2020. doi: 10.34740/KAGGLE/DSV/1693952.
*Use it for:* the Dataset section — this is the dataset itself; **must** be
cited (it's a condition of using SKAB).

## Foundational surveys (frame the problem in Intro / Related Work)

**[2]** V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey,"
*ACM Computing Surveys*, vol. 41, no. 3, pp. 1–58, Jul. 2009.
*Use it for:* Introduction + Related Work — the canonical survey; defines the
problem and the taxonomy of approaches. Almost every AD paper cites this.

**[3]** R. Chalapathy and S. Chawla, "Deep learning for anomaly detection: A
survey," *arXiv:1901.03407*, 2019.
*Use it for:* Related Work — covers the deep-learning side, which lets us
position our work as "classical methods on a real benchmark" vs. the heavier
deep models.

**[4]** M. A. Belay, S. S. Blakseth, A. Rasheed, and P. Salvo Rossi,
"Unsupervised anomaly detection for IoT-based multivariate time series:
Existing solutions, performance analysis and future directions," *Sensors*,
vol. 23, no. 5, art. 2844, 2023. doi: 10.3390/s23052844.
*Use it for:* Related Work — recent, directly on-topic (unsupervised,
multivariate, industrial/IoT); also a model for how to benchmark methods.

## The 6 models we compare (cite each where we introduce it in Methodology)

**[5]** B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C.
Williamson, "Estimating the support of a high-dimensional distribution,"
*Neural Computation*, vol. 13, no. 7, pp. 1443–1471, 2001.
*Use it for:* the **One-Class SVM** description.

**[6]** M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, "LOF:
Identifying density-based local outliers," in *Proc. ACM SIGMOD Int. Conf.
Management of Data*, 2000, pp. 93–104.
*Use it for:* the **Local Outlier Factor (LOF)** description.

**[7]** F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc.
8th IEEE Int. Conf. Data Mining (ICDM)*, 2008, pp. 413–422.
*Use it for:* the **Isolation Forest** description — our best-performing /
proposed model, so this is a key citation.

**[8]** P. J. Rousseeuw and K. Van Driessen, "A fast algorithm for the minimum
covariance determinant estimator," *Technometrics*, vol. 41, no. 3,
pp. 212–223, 1999.
*Use it for:* the **Elliptic Envelope** description (it's MCD-based robust
covariance — this is the underlying algorithm).

**[9]** M.-L. Shyu, S.-C. Chen, K. Sarinnapakorn, and L. Chang, "A novel
anomaly detection scheme based on principal component classifier," in *Proc.
IEEE ICDM Workshop on Foundations and New Directions of Data Mining*, 2003,
pp. 172–179.
*Use it for:* the **PCA-reconstruction** model description.

> **Z-score baseline:** there's no single canonical paper for simple z-score
> thresholding — it's a textbook statistical method. We'll introduce it as a
> standard statistical baseline and lean on survey [2] (which categorizes
> statistical/parametric methods) rather than inventing a primary citation.

## Application context (motivates "why industrial sensor faults matter")

**[10]** P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, and G.
Shroff, "LSTM-based encoder-decoder for multi-sensor anomaly detection,"
in *Proc. ICML Anomaly Detection Workshop*, 2016. *arXiv:1607.00148*.
*Use it for:* Related Work + Discussion — the classic deep reconstruction
approach; good contrast for "future work could try sequence models."

**[11]** O. Das, D. B. Das, and D. Birant, "Machine learning for fault analysis
in rotating machinery: A comprehensive review," *Heliyon*, vol. 9, no. 6,
art. e17584, 2023. (✅ authors confirmed against Heliyon/Cell record 2026-07-16.)
*Use it for:* Introduction — motivates predictive maintenance / industrial
fault detection as the real-world driver.

## Tools (cite in Methodology, for reproducibility)

**[12]** F. Pedregosa et al., "Scikit-learn: Machine learning in Python,"
*Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
*Use it for:* Methodology — every model we use is from scikit-learn; citing it
is standard practice and aids reproducibility.

---

## Quick map: which reference goes in which section

- **Introduction:** [2], [11], [4] (problem + motivation), [7] (our headline method)
- **Related Work:** [2], [3], [4], [10] (and contrast classical vs. deep)
- **Methodology:** [5], [6], [7], [8], [9] (one per model), [12] (the library)
- **Dataset:** [1]
- **Discussion/Limitations:** [10], [3] (what deeper models could add)

If we want a few more for Related Work depth, good candidates to add later are
a SKAB-specific evaluation paper and one more recent benchmark study — I can
pull those when we draft that section.

---

## Update (2026-07-06): expanded to 28 references

Prof. Shah asked for at least 25–30 references (the paper had 18). Added **10
more**, each verified against a real source (ACM/SIAM/Nature/dblp/MIT Press) and
given a genuine in-text `\cite{}` in `paper1.tex`, not just dumped in the
bibliography:

**[19]** S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural
Computation*, vol. 9, no. 8, pp. 1735–1780, 1997. — the LSTM architecture used
in the deep baseline (Methodology, `sec:deepmethod`).

**[20]** D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization,"
*ICLR*, 2015. — the optimizer used to train the LSTM autoencoder.

**[21]** A. Paszke et al., "PyTorch: An imperative style, high-performance deep
learning library," *NeurIPS*, vol. 32, pp. 8024–8035, 2019. — the framework the
deep baseline is implemented in.

**[22]** M. Sakurada and T. Yairi, "Anomaly detection using autoencoders with
nonlinear dimensionality reduction," *MLSDA*, pp. 4–11, 2014. — earlier
autoencoder-based reconstruction detector work (Related Work, deep-learning
paragraph), precedes the LSTM-encoder-decoder line already cited.

**[23]** A. Bifet and R. Gavaldà, "Learning from time-changing data with
adaptive windowing," *SDM*, pp. 443–448, 2007. — ADWIN, a second concept-drift
detection method alongside the KS-based approach used (Related Work, drift
paragraph).

**[24]** C. M. Ahmed, V. R. Palleti, and A. P. Mathur, "WADI: A water
distribution testbed for research in the design of secure cyber physical
systems," *CySWater*, pp. 25–28, 2017. — the newer, same-domain benchmark noted
as future work (Discussion) after the SWaT-date conversation.

**[25]** M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, "A density-based
algorithm for discovering clusters in large spatial databases with noise,"
*KDD*, pp. 226–231, 1996. — DBSCAN, the density-based clustering lineage behind
LOF (Related Work).

**[26]** D. M. W. Powers, "Evaluation: From precision, recall and F-measure to
ROC, informedness, markedness and correlation," *J. Mach. Learn. Technol.*,
vol. 2, no. 1, pp. 37–63, 2011. — backs the choice of MCC/balanced accuracy
under class imbalance (Evaluation Metrics).

**[27]** C. R. Harris et al., "Array programming with NumPy," *Nature*,
vol. 585, no. 7825, pp. 357–362, 2020. — a tool used throughout (Methodology).

**[28]** W. McKinney, "Data structures for statistical computing in Python,"
*SciPy*, pp. 56–61, 2010. — pandas, a tool used throughout (Methodology).

All 28 references are cited in strict order of first appearance in the text
(reference [1] is the first one cited, [28] the last), per Prof. Shah's
citation-order fix.
