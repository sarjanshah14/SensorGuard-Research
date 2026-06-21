# References — verified real citations for the paper

Every entry below is a real, published, verifiable work (I checked each one
against its publisher / arXiv / DOI — none are invented). They're written in
**IEEE reference style** (the format the IEEE conference template expects),
numbered so we can cite them in-text as `[1]`, `[2]`, … when we write.

The "Use it for" note says which section/claim each one backs. We won't
necessarily cite all of them — this is the pool to draw from (a 6-page IEEE
paper typically lands ~12–18 references).

> ⚠️ One still to verify before final submission: [11] (rotating-machinery
> review) — I have the journal and year but want to confirm the exact author
> list against the published PDF before it goes in. Everything else is
> confirmed.

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

**[11]** *Machine learning for fault analysis in rotating machinery: A
comprehensive review*, *Heliyon*, 2023. (⚠️ confirm exact authors before use.)
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
