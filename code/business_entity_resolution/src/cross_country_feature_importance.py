"""Task 4: Cross-Country Feature Importance and Generalization Analysis.

Computes permutation feature importance of the final NeuralNet_MLP model separately on
US validation pairs and India validation pairs, comparing feature ranking concordance
and proving why the model generalizes to France zero-shot.
"""

import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from src.blocking import BlockingIndex
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df


def build_country_val_data(
    target_country: str,
    n_s1: int = 1200,
    top_k: int = 15,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build a balanced validation dataset for a specific country."""
    print(f"\nBuilding validation dataset for {target_country} ({n_s1} S1 entities)...", flush=True)
    t0 = time.time()

    # Load S1 for this country
    s1_path = "student_resource/dataset/train/train_source1.tsv"
    s1_rows = []
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["country"] == target_country]
        if len(sub) > 0:
            s1_rows.append(sub)
        if sum(len(x) for x in s1_rows) >= n_s1 * 3:
            break
    df_s1_full = pd.concat(s1_rows, ignore_index=True)

    # Load Ground Truth
    gt_path = "student_resource/dataset/train/train_ground_truth.tsv"
    df_gt = pd.read_csv(gt_path, sep="\t", nrows=10000, keep_default_na=False)
    gt_map = {}
    for _, row in df_gt.iterrows():
        s1 = row["source1_entity_id"].strip()
        m_str = row["matched_entity_ids"].strip()
        if m_str:
            gt_map[s1] = {x.strip() for x in m_str.split(",") if x.strip()}

    # Filter S1 entities that exist in ground truth
    df_s1 = df_s1_full[df_s1_full["entity_id"].isin(gt_map.keys())].head(n_s1)
    df_s1 = apply_normalization_df(df_s1)
    target_s1_ids = set(df_s1["entity_id"])
    print(f"Selected {len(df_s1):,} {target_country} S1 entities with ground truth matches.", flush=True)

    needed_cands = set()
    for s1_id in target_s1_ids:
        needed_cands.update(gt_map.get(s1_id, set()))

    # Load candidate records
    cands_rows = []
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("student_resource/dataset/train", src)
        for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
            m = chunk[chunk["entity_id"].isin(needed_cands)]
            cands_rows.append(m)
            if len(cands_rows) <= 2:
                cands_rows.append(chunk.head(20000))
            if sum(len(x[x["entity_id"].isin(needed_cands)]) for x in cands_rows) >= len(needed_cands):
                break
    df_cands = apply_normalization_df(pd.concat(cands_rows, ignore_index=True).drop_duplicates("entity_id"))
    cand_dict_map = {row["entity_id"]: row for row in df_cands.to_dict("records")}
    print(f"Loaded {len(df_cands):,} candidate records for blocking.", flush=True)

    # Build Blocking Index
    indexer = BlockingIndex()
    indexer.build(df_cands)

    # Extract pairs
    pair_rows = []
    for s1_rec in df_s1.to_dict("records"):
        s1_id = s1_rec["entity_id"]
        true_m = gt_map.get(s1_id, set())

        scored = indexer.retrieve_candidates_for_entity(s1_rec, top_k=top_k)
        cand_ids = {cid for cid, _ in scored}
        for tm in true_m:
            if tm in cand_dict_map and tm not in cand_ids:
                scored.append((tm, 0.0))

        for rank, (cid, b_score) in enumerate(scored):
            if cid not in cand_dict_map:
                continue
            cand_rec = cand_dict_map[cid]
            feats = extract_pair_features(s1_rec, cand_rec, rank=rank, blocking_score=b_score)
            feats["label"] = 1 if cid in true_m else 0
            pair_rows.append(feats)

    df_pairs = pd.DataFrame(pair_rows)
    print(f"Generated {len(df_pairs):,} pairs for {target_country} ({df_pairs['label'].sum():,} positives, {len(df_pairs)-df_pairs['label'].sum():,} negatives) in {time.time()-t0:.1f}s.")

    X = df_pairs[FEATURE_COLUMNS].values
    y = df_pairs["label"].values
    return X, y, df_pairs


def main():
    print("=== Task 4: Cross-Country Permutation Feature Importance ===", flush=True)

    # 1. Load trained model pipeline
    model_path = "models/best_model_pipeline.pkl"
    print(f"Loading trained model pipeline from {model_path} ...", flush=True)
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # 2. Build US and India datasets
    X_us, y_us, df_us = build_country_val_data("US", n_s1=1000, top_k=15)
    X_in, y_in, df_in = build_country_val_data("India", n_s1=1000, top_k=15)

    # 3. Compute Permutation Importance
    print("\nComputing Permutation Feature Importance for US (5 repeats, ROC-AUC metric)...", flush=True)
    t0 = time.time()
    perm_us = permutation_importance(pipeline, X_us, y_us, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
    print(f"US importance computed in {time.time()-t0:.1f}s.", flush=True)

    print("\nComputing Permutation Feature Importance for India (5 repeats, ROC-AUC metric)...", flush=True)
    t0 = time.time()
    perm_in = permutation_importance(pipeline, X_in, y_in, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
    print(f"India importance computed in {time.time()-t0:.1f}s.", flush=True)

    # 4. Aggregate & Rank
    us_means = perm_us.importances_mean
    in_means = perm_in.importances_mean

    feat_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_us": us_means,
        "importance_india": in_means,
    })

    feat_df_us_sorted = feat_df.sort_values(by="importance_us", ascending=False).reset_index(drop=True)
    feat_df_in_sorted = feat_df.sort_values(by="importance_india", ascending=False).reset_index(drop=True)

    # Spearman rank correlation across all 43 features
    rho, p_val = spearmanr(us_means, in_means)
    print(f"\nCross-Country Feature Rank Correlation (Spearman rho across all {len(FEATURE_COLUMNS)} features): {rho:.4f} (p-value: {p_val:.2e})")

    # Side-by-side Top-10 Table
    print("\n" + "=" * 105)
    print("TASK 4: TOP-10 FEATURE IMPORTANCE COMPARISON (US vs INDIA)")
    print("=" * 105)
    print(f"{'Rank':<5} | {'US Top Feature':<32} | {'US Importance':<15} | {'India Top Feature':<32} | {'India Importance':<16}")
    print("-" * 105)
    top_10_list = []
    for r in range(10):
        u_f = feat_df_us_sorted.iloc[r]["feature"]
        u_i = feat_df_us_sorted.iloc[r]["importance_us"]
        i_f = feat_df_in_sorted.iloc[r]["feature"]
        i_i = feat_df_in_sorted.iloc[r]["importance_india"]
        print(f"#{r+1:<4} | {u_f:<32} | {u_i:<15.5f} | {i_f:<32} | {i_i:<16.5f}")
        top_10_list.append({
            "rank": r + 1,
            "us_feature": u_f,
            "us_importance": float(u_i),
            "india_feature": i_f,
            "india_importance": float(i_i),
        })
    print("=" * 105)

    results_data = {
        "spearman_rank_correlation": float(rho),
        "spearman_p_value": float(p_val),
        "top_10_side_by_side": top_10_list,
        "all_features_us_ranked": feat_df_us_sorted.to_dict("records"),
        "all_features_india_ranked": feat_df_in_sorted.to_dict("records"),
    }

    out_path = "models/cross_country_feature_importance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved feature importance report to {out_path}")


if __name__ == "__main__":
    main()
