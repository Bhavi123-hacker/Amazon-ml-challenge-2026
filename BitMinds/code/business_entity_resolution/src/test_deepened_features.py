"""Section 3: Feature Engineering Deepening and Failure-Mode Counter-Features.

Tests whether adding targeted counter-features directly mitigating the dominant failure modes
(Mode 1: Co-located suite divergence; Mode 3: Franchise/chain city divergence) yields a measurable
CV Macro F0.5 improvement over the baseline 43-feature NeuralNet_MLP model:
  1. suite_exact_match: Exact match on extracted suite / unit / floor indicators
  2. same_bldg_diff_suite: Shared street number & street name, but explicitly conflicting suite
  3. brand_match_city_mismatch: High core-name similarity with completely divergent city / street
  4. co_location_divergence_penalty: High address similarity but low core-name similarity
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.evaluate import compute_macro_f05
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.train_and_benchmark import build_training_dataset

SUITE_REGEX = re.compile(
    r"\b(?:ste|suite|unit|apt|apartment|fl|floor|bldg|building|bat|bâtiment|etg|étage|porte|bureau|flat|shop|plot)\s*#?\s*([a-z0-9\-]+)\b",
    re.IGNORECASE,
)


def extract_suite(addr: str) -> str:
    """Extract suite / unit / apartment / floor identifier from address."""
    if not addr or not isinstance(addr, str):
        return ""
    m = SUITE_REGEX.search(addr)
    return m.group(1).lower() if m else ""


def add_deepened_features_df(df_pairs: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Compute and append the 4 targeted failure-mode features."""
    df = df_pairs.copy()

    # Extract suites
    s1_suites = [extract_suite(a) for a in df["addr_norm_s1"]] if "addr_norm_s1" in df.columns else ["" for _ in range(len(df))]
    cand_suites = [extract_suite(a) for a in df["addr_norm_cand"]] if "addr_norm_cand" in df.columns else ["" for _ in range(len(df))]

    # If raw strings are available
    suite_exact_match = []
    same_bldg_diff_suite = []
    brand_match_city_mismatch = []
    co_location_divergence_penalty = []

    for i in range(len(df)):
        s_s1 = s1_suites[i]
        s_c = cand_suites[i]

        # 1. Suite Exact Match
        s_match = 1.0 if (s_s1 and s_c and s_s1 == s_c) else 0.0
        suite_exact_match.append(s_match)

        # 2. Same building, different suite (Mode 1 Counter)
        st_num_match = df["street_number_match"].iloc[i]
        st_jaccard = df["st_name_jaccard"].iloc[i]
        diff_suite = 1.0 if (st_num_match == 1.0 and st_jaccard >= 0.40 and s_s1 and s_c and s_s1 != s_c) else 0.0
        same_bldg_diff_suite.append(diff_suite)

        # 3. Brand match, city mismatch (Mode 3 Counter)
        core_sim = df["core_name_tok_sort"].iloc[i]
        city_m = df["city_exact_match"].iloc[i]
        addr_sim = df["addr_token_sort"].iloc[i]
        franchise_mismatch = 1.0 if (core_sim >= 0.80 and city_m == 0.0 and st_jaccard == 0.0 and addr_sim < 0.40) else 0.0
        brand_match_city_mismatch.append(franchise_mismatch)

        # 4. Co-location divergence penalty (Mode 1 Counter)
        coloc_penalty = 1.0 if (addr_sim >= 0.80 and core_sim < 0.50) else 0.0
        co_location_divergence_penalty.append(coloc_penalty)

    df["suite_exact_match"] = suite_exact_match
    df["same_bldg_diff_suite"] = same_bldg_diff_suite
    df["brand_match_city_mismatch"] = brand_match_city_mismatch
    df["co_location_divergence_penalty"] = co_location_divergence_penalty

    new_cols = [
        "suite_exact_match",
        "same_bldg_diff_suite",
        "brand_match_city_mismatch",
        "co_location_divergence_penalty",
    ]
    return df, new_cols


def run_cv_experiment(
    df_pairs: pd.DataFrame,
    feature_cols: List[str],
    gt_map: Dict[str, Set[str]],
    desc: str,
) -> Dict[str, Any]:
    """Run 5-Fold Stratified Group CV with exact Macro F0.5."""
    print(f"\n--- Running 5-Fold Group CV: {desc} ({len(feature_cols)} features) ---", flush=True)
    t0 = time.time()

    X = df_pairs[feature_cols].values
    y = df_pairs["label"].values
    groups = df_pairs["source1_entity_id"].values

    gkf = GroupKFold(n_splits=5)
    scaler = StandardScaler()

    fold_f05 = []
    fold_prec = []
    fold_rec = []
    fold_train_f05 = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr = scaler.fit_transform(X[train_idx])
        X_v = scaler.transform(X[val_idx])
        y_tr, y_v = y[train_idx], y[val_idx]

        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.01,
            learning_rate_init=0.003,
            max_iter=50,
            early_stopping=True,
            random_state=42,
        )
        mlp.fit(X_tr, y_tr)

        probs_v = mlp.predict_proba(X_v)[:, 1]
        probs_tr = mlp.predict_proba(X_tr)[:, 1]

        # Validation evaluation
        val_df = df_pairs.iloc[val_idx].copy()
        val_df["pred_prob"] = probs_v

        best_f05, best_p, best_r = 0.0, 0.0, 0.0
        for tau in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
            val_preds = {
                s1_id: set(grp[grp["pred_prob"] >= tau]["candidate_entity_id"])
                for s1_id, grp in val_df.groupby("source1_entity_id")
            }
            val_gt = {s1_id: gt_map[s1_id] for s1_id in val_preds}
            f, p, r = compute_macro_f05(val_gt, val_preds)
            if f > best_f05:
                best_f05, best_p, best_r = f, p, r

        fold_f05.append(best_f05)
        fold_prec.append(best_p)
        fold_rec.append(best_r)

        # Train F0.5 on sample
        tr_df = df_pairs.iloc[train_idx].copy()
        tr_df["pred_prob"] = probs_tr
        unique_tr = tr_df["source1_entity_id"].unique()
        sample_tr = set(unique_tr[:300])
        tr_sample_df = tr_df[tr_df["source1_entity_id"].isin(sample_tr)]
        tr_preds = {
            s1_id: set(grp[grp["pred_prob"] >= 0.50]["candidate_entity_id"])
            for s1_id, grp in tr_sample_df.groupby("source1_entity_id")
        }
        tr_gt = {s1_id: gt_map[s1_id] for s1_id in tr_preds}
        tr_f, _, _ = compute_macro_f05(tr_gt, tr_preds)
        fold_train_f05.append(tr_f)

    mean_f05 = float(np.mean(fold_f05))
    mean_p = float(np.mean(fold_prec))
    mean_r = float(np.mean(fold_rec))
    mean_tr = float(np.mean(fold_train_f05))
    gap = mean_tr - mean_f05
    elapsed = time.time() - t0

    print(f"  Result: CV F0.5 = {mean_f05:.4f} | Prec = {mean_p:.4f} | Rec = {mean_r:.4f} | Gap = {gap:+.4f} | Time = {elapsed:.1f}s", flush=True)

    return {
        "description": desc,
        "n_features": len(feature_cols),
        "cv_f05": mean_f05,
        "cv_precision": mean_p,
        "cv_recall": mean_r,
        "train_f05": mean_tr,
        "overfitting_gap": gap,
        "training_time_s": round(elapsed, 1),
    }


def main():
    print("=== Section 3: Feature Engineering Deepening Experiment ===", flush=True)

    # 1. Build training dataset (2,000 S1 entities)
    df_pairs, gt_map = build_training_dataset("student_resource/dataset", n_s1_samples=2000, top_k_candidates=25)
    print(f"Dataset loaded: {len(df_pairs):,} pairs.", flush=True)

    # Need candidate address strings for suite extraction
    # The pairs dataframe has addr_norm, let's load cand dict map for candidate addresses
    print("Enriching candidate address strings for suite feature extraction...", flush=True)
    # Load candidate records
    needed_cands = set(df_pairs["candidate_entity_id"])
    cands_rows = []
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("student_resource/dataset/train", src)
        df_src = pd.read_csv(p, sep="\t", keep_default_na=False)
        m = df_src[df_src["entity_id"].isin(needed_cands)]
        if len(m) > 0:
            cands_rows.append(m)
    df_cands_lookup = pd.concat(cands_rows, ignore_index=True).drop_duplicates("entity_id")
    cand_addr_map = dict(zip(df_cands_lookup["entity_id"], df_cands_lookup["business_address"]))
    df_pairs["addr_norm_cand"] = [cand_addr_map.get(cid, "") for cid in df_pairs["candidate_entity_id"]]

    # Load s1 addresses
    df_s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", keep_default_na=False)
    s1_addr_map = dict(zip(df_s1["entity_id"], df_s1["business_address"]))
    df_pairs["addr_norm_s1"] = [s1_addr_map.get(sid, "") for sid in df_pairs["source1_entity_id"]]

    # 2. Add deepened features
    df_enriched, new_cols = add_deepened_features_df(df_pairs)
    print(f"Extracted 4 targeted counter-features: {new_cols}")

    # 3. Experiment A: Baseline 43 features
    res_base = run_cv_experiment(df_enriched, FEATURE_COLUMNS, gt_map, "Baseline (43 Enriched Features)")

    # 4. Experiment B: Deepened 47 features
    deepened_cols = FEATURE_COLUMNS + new_cols
    res_deep = run_cv_experiment(df_enriched, deepened_cols, gt_map, "Deepened (47 Features with Suite/Franchise Counters)")

    # 5. Compare
    delta_f05 = res_deep["cv_f05"] - res_base["cv_f05"]
    print("\n" + "=" * 90)
    print("SECTION 3: FEATURE ENGINEERING DEEPENING RESULTS")
    print("=" * 90)
    print(f"{'Feature Set':<35} | {'N Features':<12} | {'CV F0.5':<10} | {'Precision':<10} | {'Recall':<10} | {'F0.5 Delta':<12}")
    print("-" * 90)
    print(f"{res_base['description']:<35} | {res_base['n_features']:<12} | {res_base['cv_f05']:<10.4f} | {res_base['cv_precision']:<10.4f} | {res_base['cv_recall']:<10.4f} | {'Baseline (0.0)':<12}")
    print(f"{res_deep['description']:<35} | {res_deep['n_features']:<12} | {res_deep['cv_f05']:<10.4f} | {res_deep['cv_precision']:<10.4f} | {res_deep['cv_recall']:<10.4f} | {delta_f05:+.4f}")
    print("=" * 90)

    decision = "ADOPT DEEPENED FEATURES" if delta_f05 > 0.0005 else "KEEP BASELINE (DISCARD ADDED COMPLEXITY)"
    print(f"\nDECISION RULE (>+0.0005 F0.5): {decision}")
    print(f"Rationale: Delta F0.5 = {delta_f05:+.4f}. Standing rule prohibits unverified complexity that does not pay off.")

    experiment_report = {
        "baseline_43_features": res_base,
        "deepened_47_features": res_deep,
        "delta_f05": delta_f05,
        "decision": decision,
        "new_feature_names": new_cols,
    }

    out_path = "models/deepened_features_experiment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)
    print(f"Saved feature deepening experiment to {out_path}")


if __name__ == "__main__":
    main()
