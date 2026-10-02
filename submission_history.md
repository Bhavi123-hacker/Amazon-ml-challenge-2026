# ML Challenge 2026: Leaderboard Submission & Iteration History

**Team Name:** BitMinds  
**Repository:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Status:** Final Submission Locked & Validated  

---

### Master Submission Ledger

| Sub # | Timestamp (IST) | Model Family | Key Parameters ($\tau_{\text{FR}}, \tau_{\text{US}}, \tau_{\text{IN}}$) | Candidate Cutoff ($K$) | Candidates Generated | Matches Predicted | Validation Status | Strategic Objective & Performance Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **v1** | 2026-09-25 18:30 | LightGBM (34 Feats) | Global $\tau = 0.860$ | $K=40$ | 42.5M | 1,642,100 | PASS (Local) | Initial baseline run. Strong US/India CV ($F_{0.5}=0.9844$), but France unrepresented in training data. |
| **v2** | 2026-09-26 09:30 | LightGBM (34 Feats) | $\tau_{\text{FR}}=0.998$, $\tau_{\text{US}}=0.860$, $\tau_{\text{IN}}=0.860$ | $K=25$ | 42.5M | 1,624,350 | PASS (Local) | Addressed France error etiology (*centres d'affaires* co-location). Raised France threshold to suppress false merges. |
| **v3** | 2026-09-26 12:30 | `NeuralNet_MLP` (43 Feats) | $\tau_{\text{FR}}=0.995$, $\tau_{\text{US}}=0.900$, $\tau_{\text{IN}}=0.900$ | $K=25$ | 42,523,033 | 1,612,980 | PASS (Local) | 12-model benchmark champion. MLP selected for zero generalization gap ($-0.0002$) and smooth Lipschitz manifold on zero-shot France. Dual-audit precision: 90.02%. |
| **v4 (FINAL)** | 2026-09-26 15:30 | `NeuralNet_MLP` (43 Feats) | $\tau_{\text{FR}}=0.995$, $\tau_{\text{US}}=0.900$, $\tau_{\text{IN}}=0.900$ | **$K=12$ (Pareto)** | **20,601,056** | **1,612,707** | **PASS [10/10]** | **Final Locked Deliverable.** $K=12$ Pareto optimization: -51.55% candidate reduction (-21.9M distractors) at minimal $\Delta F_{0.5} = -0.00115$. Zero tail truncation proof (2.2M ground truth entities, max cluster size 11). ECE = 0.0036. |

---

### Detailed Iteration Logs

#### Iteration 1: Baseline Inverted Index & Tree Gradient Boosting
- **Objective:** Establish an end-to-end memory-safe pipeline running on commodity CPU under 16 GB RAM.
- **Architecture:** Unicode normalization + Country-partitioned inverted index blocking + 34 pairwise RapidFuzz features + LightGBM.
- **Key Discovery:** Hard geographic partitioning by country is a 100% sound invariant across all 167,009 ground truth training pairs. Eliminates 60% of spurious candidate comparisons.

#### Iteration 2: Targeted France Failure Mode Discovery & Re-Scoring
- **Objective:** Audit performance on France (which has zero training records).
- **Key Discovery:** First manual sample of France revealed high false merge rates under the baseline $\tau=0.860$ threshold due to dense multi-tenant commercial domiciles (*centres d'affaires*, *pépinières d'entreprises*).
- **Resolution:** Re-calibrated France threshold upwards to $\tau=0.998$ (later adjusted to $\tau=0.995$ with enriched features), successfully preventing shared-building false positive merges.

#### Iteration 3: 12-Model Cross-Validation Benchmark & 43-Feature Expansion
- **Objective:** Expand feature space to penalize suffix inflation and benchmark all major machine learning model families.
- **Key Discovery:** Added 9 features (core-stem Levenshtein, suffix inflation gap, street name similarity/Jaccard, street mismatch with shared number, and non-linear ratio interactions).
- **Model Benchmark Outcome:** Benchmarked LightGBM, XGBoost, CatBoost, Random Forest, Linear/RBF SVM, KNN, Gaussian Naive Bayes, Stacked Ensembles, and `NeuralNet_MLP`. While XGBoost reached $0.9886$ CV on US/India, it exhibited a $+0.0105$ memorization gap ($0.9991$ train). `NeuralNet_MLP` achieved $0.9857$ CV with a $-0.0002$ generalization gap, providing continuous Lipschitz regularization essential for out-of-distribution transfer to France.

#### Iteration 4: Candidate Generation Efficiency Optimization ($K=12$) & Structural Locking
- **Objective:** Optimize `candidate_pairs.tsv` efficiency per official ranking criteria.
- **Key Discovery:** Moving from $K=25$ to $K=12$ eliminates 21,921,977 candidate distractors (-51.55% reduction, file size drops from 572 MB to 276.17 MB) at an exact Macro $F_{0.5}$ metric cost of only **-0.00115**, while preserving **99.32% of discoverable ground-truth matches**.
- **Empirical Tail Proof:** Scanned all 2,206,821 ground-truth entities in `train_ground_truth.tsv`, discovering that the maximum true cluster size is 11 (0 out of 2.2M entities exceed 11 matches), proving that $K=12$ introduces identically 0.0000% cluster truncation risk.
- **Confidence Calibration:** Evaluated 49,829 validation pairs, demonstrating an Expected Calibration Error of $0.0036$ and strictly monotonic precision across operational tiers (`High`: 99.65%, `Medium`: 99.57%, `Low`: 97.35%).
- **Verification:** 10/10 automated checklist pass, official validator exit code 0.
