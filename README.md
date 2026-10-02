# Business Entity Resolution — Amazon ML Challenge 2026

A two-stage entity resolution pipeline that matches noisy business records across three
independent data sources (US, India, and France) using blocking-based candidate generation
and a gradient-boosted classifier, optimized for the competition's precision-weighted
Macro F0.5 metric.

## Problem

Given business records from three sources — Source 1 (a deduplicated reference set) and
Source 2 / Source 3 (noisy candidate sources) — identify every Source 2/3 record that refers
to the same real-world business as each Source 1 entity. Records share no common identifiers
and contain realistic noise: abbreviations, legal-suffix variants, typos, transliterations,
and inconsistent or missing address components. The test set includes France, a country with
**zero representation in the training data**, requiring the pipeline to generalize across an
unseen address/naming convention.

## Approach

**1. Normalization** — Non-destructive parsing into derived fields (`name_norm`, `addr_norm`,
street number, street name, city, PIN/region), with explicit handling for legal-suffix
variants, abbreviation expansion, and script-aware Unicode normalization that avoids
corrupting non-Latin (Devanagari, Tamil) text.

**2. Blocking / candidate generation** — A multi-channel inverted index (sorted name tokens,
Double Metaphone phonetic keys, address composite keys, frequency-capped significant tokens,
acronyms) reduces the ~1.7 × 10¹³ possible pairwise comparisons down to a per-entity candidate
set capped at K ≤ 12–25, chosen via a recall-vs-efficiency sweep against training ground truth.
Blocking quality is evaluated per-channel (unique catch rate vs. candidate volume contributed)
since `candidate_pairs.tsv` is itself part of the evaluation criteria.

**3. Feature engineering** — ~40+ pairwise features per candidate: string-similarity metrics
(Levenshtein, Jaro-Winkler, token sort/set, Jaccard) on both name and address, core-stem
similarity (legal suffixes stripped), street-name-specific similarity, structural flags
(PIN/city/street-number agreement), and interaction terms. Deliberately excludes
blocking-rank-derived features to avoid the model learning retrieval rank instead of genuine
similarity.

**4. Classification** — Multiple model families (LightGBM, XGBoost, CatBoost, Random Forest,
Logistic Regression, a small regularized MLP) benchmarked under 5-fold Stratified Group
cross-validation (grouped by `source1_entity_id` to prevent leakage), selected by real held-out
Macro F0.5 with explicit attention to the train/validation generalization gap.

**5. Per-country threshold calibration** — US and India thresholds are tuned directly against
labeled ground truth. France, having no labels, is calibrated via match-rate/singleton-rate
sanity checks against the training distribution and repeated, randomly-sampled manual audits
against raw text.

**6. Targeted precision guards** — Several specific, evidence-driven post-processing rules
address failure modes found through direct auditing rather than generic threshold tuning:
- *Street-number consistency guard* — rejects matches between adjacent-but-distinct addresses
  on the same street (e.g. 1828 vs 1831 Fauver Ave) that soft string similarity scores highly.
- *Compound door-number handling* (India) — prevents hierarchical plot/door identifiers
  (e.g. `1/5448` vs `1/5459`) from being tokenized in a way that causes false number overlap.
- *Multi-tenant collision guard* (France) — rejects matches where two distinct businesses
  share a building/street number but have divergent core names, a common pattern in dense
  urban commercial centers.
- *Placeholder-name guard* — records with missing/placeholder names bypass the standard
  classifier and are matched only under a stricter full-address-agreement rule, after
  confirming (via training-data inspection) that model training on these records would learn a
  spurious "missing name ⇒ match" correlation.

## Methodology discipline

Every reported metric in this project is tied to an actual command run against actual data.
France, having no ground truth, is never scored via a direct F0.5/precision/recall computation —
only via repeated, randomly-seeded manual audits against raw source text. Models and guards are
validated against known edge cases (zero-padding, address prefixes, range numbers, OCR digit
confusion) before full-scale deployment, and every full-scale regeneration is followed by a
fresh audit on the literal submitted file rather than trusting a proxy validation result.

## Repository structure

```
output/
  matching_results.tsv       # final entity matches (leaderboard-scored)
  candidate_pairs.tsv        # blocking candidate set (efficiency-scored)
code/business_entity_resolution/
  src/
    normalize.py             # non-destructive field normalization
    blocking.py               # multi-channel candidate generation
    features.py                # pairwise feature engineering
    train.py                    # model training and cross-validation
    infer.py                     # full-scale scoring and guard application
    evaluate.py                  # Macro F0.5 computation
    audit_heuristics.py           # multi-mode error classification
  README.md                       # reproduction instructions
  requirements.txt                 # pinned dependencies
Documentation_template.md           # full methodology write-up
```

## Reproducing the pipeline

```bash
pip install -r code/business_entity_resolution/requirements.txt

python -m src.normalize --input dataset/ --output data_processed/
python -m src.blocking --input data_processed/ --output output/candidate_pairs.tsv
python -m src.train --candidates output/candidate_pairs.tsv --ground-truth dataset/train/train_ground_truth.tsv
python -m src.infer --model models/champion.pkl --candidates output/candidate_pairs.tsv --output output/matching_results.tsv
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Constraints honored

- No modification of raw dataset files; all processing happens on derived copies.
- No external data, APIs, or geocoding lookups.
- Final model is MIT/Apache-2.0 licensed and well under the 8B-parameter limit.

## Known limitations

- France's precision cannot be directly measured (no training labels); all France-specific
  claims are manual-audit estimates with real sampling uncertainty.
- Recall at full test scale is estimated, not directly measured, since no test-set ground truth
  exists; reported F0.5 projections should be read as ranges, not guarantees.
