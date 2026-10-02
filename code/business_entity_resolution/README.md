# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

A high-precision, memory-efficient two-stage Entity Resolution system designed to resolve noisy commercial business records across heterogeneous sources (Source 1 reference entities vs. Source 2 & 3 candidates) under the competition's macro-averaged $F_{0.5}$ metric.

---

## 1. System Architecture

The pipeline implements a two-stage cascade architecture specifically optimized for high-precision entity resolution ($F_{0.5}$ weights Precision $2\times$ over Recall) and scalable candidate efficiency:

```
+-----------------------------------------------------------------------------------+
|                                   Raw Datasets                                    |
|                 Source 1 (Reference) | Source 2 & 3 (Candidates)                  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 0: Script-Aware Multilingual Normalization                                  |
| - Custom diacritic stripping (0x0300-0x036F) preserving Indic vowel matras        |
| - Legal entity form isolation & canonicalization (SAS, SARL, Pvt, Ltd, LLC, Inc)  |
| - Regex PIN / Postal Code, Street Number, and City extraction                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 1: Multi-Channel Inverted Index Blocking & K=12 Pareto Truncation           |
| - Strict Geographic Invariant: Country(S1) == Country(Cand) (60% space reduction) |
| - Inverted Index Channels: Sorted First-2-Tokens, Double Metaphone, Address Street|
|   Numbers + PIN/City, Significant 4-grams, and Acronym signatures                 |
| - Pareto-optimal candidate truncation at K=12 (51.55% candidate reduction,        |
|   11.89 avg candidates/entity, zero tail mass truncated on ground-truth clusters) |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 2: Enriched 43-Dimensional Feature Engineering                              |
| - Standard Name Metrics (12): Levenshtein, Jaro-Winkler, Token Sort/Set, Jaccard  |
| - Core Business Stem Features (4): Distinctive core stem distances & suffix gap   |
| - Address & Street Features (13): Edit distance, street name similarity/Jaccard,   |
|   and street-mismatch-with-shared-number collision flags                          |
| - Structural & Interaction Terms (10): Sim ratio (core_name/addr), rank, score    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 3: Regularized Neural Network Classifier (NeuralNet_MLP)                    |
| - Model: StandardScaler + MLPClassifier(hidden_layers=(64, 32), alpha=0.01, ReLU) |
| - Smooth Lipschitz-continuous manifold over continuous string distance metrics    |
| - 5-Fold Stratified Group CV (CV Macro F0.5: 0.9857, Overfitting Gap: -0.0002)    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 4: Dual-Verified Per-Country Decision Thresholding                          |
| - France: tau = 0.995 (suppresses shared commercial domicile / Mode 1 errors)     |
| - US & India: tau = 0.900 (balances high precision with recall)                   |
| - Strict candidate subset enforcement: matches <= candidates in 100.0% of rows    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 5: Final Packaging & Structural Verification                                |
| - 1,732,544 rows strictly aligned with test_source1.tsv line-by-line              |
| - 10/10 automated checklist verification PASS                                     |
| - Official validate_submission.py PASS (Exit Code 0)                              |
+-----------------------------------------------------------------------------------+
```

---

## 2. Directory Structure

```
code/business_entity_resolution/
├── README.md                                  # End-to-end reproduction guide
├── requirements.txt                           # Pinned exact Python dependencies
├── verify_checklist.py                        # 10/10 automated submission verification script
├── models/
│   ├── best_model_pipeline.pkl                # Serialized StandardScaler + MLPClassifier
│   ├── config.json                            # Model hyperparameters and feature specs
│   ├── single_source_of_truth_metrics.json    # Consolidated master repository of all metrics
│   ├── blocking_ablation_results.json         # 6-channel empirical blocking ablation report
│   ├── ground_truth_cluster_distribution.json # 2.2M ground-truth cluster size distribution
│   ├── confidence_calibration_empirical.json  # Empirical probability calibration report (ECE)
│   ├── determinism_verification_report.json   # 30,000-entity byte-identical determinism proof
│   └── stress_testing_report.json             # Adversarial, short-name, and homonym stress tests
└── src/
    ├── __init__.py                            # Package initializer
    ├── normalize.py                           # Multilingual Unicode normalization & regex
    ├── blocking.py                            # Multi-channel inverted index blocking engine
    ├── features.py                            # 43-dimensional pairwise feature engineering
    ├── evaluate.py                            # Exact Macro F0.5 competition metric
    ├── train_and_benchmark.py                 # 5-fold CV 12-model benchmarking pipeline
    ├── fit_and_save_pipeline.py               # Serializes final StandardScaler + MLP pipeline
    ├── generate_final_predictions.py          # Full test set inference and country stitching
    ├── optimize_candidate_pairs.py            # Applies K=12 Pareto candidate truncation
    ├── generate_prediction_confidence.py      # Produces prediction_confidence.tsv artifact
    ├── audit_heuristics.py                    # 5-mode heuristic error audit engine
    ├── final_full_population_audit.py         # Full-population 6.57M pair audit runner
    ├── ablate_blocking_channels.py            # Channel-by-channel blocking ablation study
    ├── analyze_cluster_tail_risk.py           # 2.2M ground-truth cluster size analysis
    ├── evaluate_confidence_calibration.py     # Probability calibration & ECE evaluator
    └── verify_determinism.py                  # End-to-end multi-threaded determinism verifier
```

---

## 3. Reproduction Instructions (Step-by-Step)

### Prerequisites
- Python 3.10, 3.11, or 3.12
- 16 GB RAM recommended
- Working directory: repository root (`d:/amazolml`)

### Step 1: Install Pinned Dependencies
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### Step 2: Train and Serialize the Production Model Pipeline
To execute the 5-fold cross-validation and serialize the winning `NeuralNet_MLP` pipeline (`StandardScaler` + `MLPClassifier(64, 32)`):
```bash
python code/business_entity_resolution/src/fit_and_save_pipeline.py
```
*Outputs generated:*
- `models/best_model_pipeline.pkl`

### Step 3: Run Full Test Set Candidate Generation & Inference
To execute multi-channel blocking, extract 43 pairwise features, and apply calibrated operating thresholds ($\tau_{\text{France}}=0.995, \tau_{\text{US}}=0.900, \tau_{\text{India}}=0.900$) across the 1.73M test records:
```bash
python code/business_entity_resolution/src/generate_final_predictions.py
```
*Outputs generated in `output/temp_work/`:*
- `results_France.tsv`, `results_US.tsv`, `results_India.tsv`

### Step 4: Apply Pareto K=12 Candidate Truncation & Stitch Master Files
To reduce candidate bloat by 51.55% and assemble the final master submission files:
```bash
python code/business_entity_resolution/src/optimize_candidate_pairs.py
```
*Master submission outputs generated in `output/`:*
- `output/matching_results.tsv` (108,890,087 bytes, 1,732,544 rows)
- `output/candidate_pairs.tsv` (289,588,044 bytes, 1,732,544 rows)

### Step 5: Generate Supplementary Prediction Confidence Metadata
To generate the per-entity uncertainty quantification artifact:
```bash
python code/business_entity_resolution/src/generate_prediction_confidence.py
```
*Output generated in `output/`:*
- `output/prediction_confidence.tsv` (60,762,156 bytes, 1,732,544 rows)
- `models/prediction_confidence_summary.json`

### Step 6: Validate Final Submission Files
Run the 10/10 automated self-verification checklist and the official competition validator:
```bash
# 1. 10/10 Automated Checklist
python code/business_entity_resolution/verify_checklist.py

# 2. Official Competition Validator
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
*Expected Validator Output:*
```
ML Challenge 2026 — submission validator
  test dir: student_resource/dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (119837 empty, 1612707 non-empty).
  candidate_pairs.tsv: 1732544 rows (239 empty, 1732305 non-empty).
PASS — no blocking issues found. Safe to submit.
```

---

## 4. Key Engineering & Empirical Highlights

1. **Script-Aware Normalization**: Preserves Indic vowel matras (`0x0900+`) while stripping Latin/French accents (`0x0300-0x036F`), ensuring semantic preservation across multilingual records.
2. **K=12 Pareto Efficiency Frontier**: Cuts candidate comparisons by **51.55%** (-21.9 million pairs) at an exact $F_{0.5}$ cost of only **-0.00115**, retaining **99.32%** of findable ground-truth matches.
3. **Zero Ground-Truth Tail Truncation**: A full-population scan of all 2,206,821 ground-truth entities proved the maximum true cluster size is 11 matches (99.78% have $\le 8$). Zero entities exceed 12 matches (0.0000%), proving that $K=12$ carries zero structural cluster truncation risk.
4. **Calibrated Probability Tiers**: Out-of-fold ground truth validation demonstrated an **Expected Calibration Error (ECE) of 0.0036** and **Brier score of 0.0021**, with strictly monotonic precision across operational tiers (`High`: 99.65%, `Medium`: 99.57%, `Low`: 97.35%).
5. **Deterministic Multi-Threaded Execution**: Verified 100.0% byte-for-byte identical outputs across repeated runs over 30,000 entities with 0 discrepancies.
