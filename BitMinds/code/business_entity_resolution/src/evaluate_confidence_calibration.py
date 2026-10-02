"""Task 1: Empirical Probability Calibration & Confidence Tier Validation.

Evaluates the empirical precision, reliability curve, and Expected Calibration Error (ECE)
across the confidence tiers (High, Medium, Low) using the trained NeuralNet_MLP pipeline
on out-of-fold ground-truth entities.
"""

import json
import os
import pickle
import sys
import time
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.blocking import BlockingIndex
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df


def load_ground_truth(gt_path: str, max_rows: int = 2500) -> Dict[str, Set[str]]:
    """Load ground truth mapping source1_entity_id -> set of matched_entity_ids."""
    df_gt = pd.read_csv(gt_path, sep="\t", nrows=max_rows, keep_default_na=False)
    gt_map = {}
    for _, row in df_gt.iterrows():
        s1 = row["source1_entity_id"].strip()
        m_str = row["matched_entity_ids"].strip()
        gt_map[s1] = {x.strip() for x in m_str.split(",") if x.strip()} if m_str else set()
    return gt_map


def run_calibration_evaluation():
    data_dir = "student_resource/dataset/train"
    model_path = "models/best_model_pipeline.pkl"
    output_json = "models/confidence_calibration_empirical.json"

    print("Loading trained model pipeline...", flush=True)
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # 1. Load ground truth
    n_s1 = 2000
    print(f"Loading {n_s1} ground-truth entities...", flush=True)
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    gt_map = load_ground_truth(gt_path, max_rows=n_s1)
    target_s1_ids = set(gt_map.keys())

    needed_s2 = set()
    needed_s3 = set()
    for matches in gt_map.values():
        for m in matches:
            if m.startswith("S2-"):
                needed_s2.add(m)
            elif m.startswith("S3-"):
                needed_s3.add(m)

    # 2. Load S1 records
    print("Loading Source 1 records...", flush=True)
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s1_rows = []
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=100000, keep_default_na=False):
        matched = chunk[chunk["entity_id"].isin(target_s1_ids)]
        if len(matched) > 0:
            s1_rows.append(matched)
        if sum(len(x) for x in s1_rows) == len(target_s1_ids):
            break
    df_s1 = pd.concat(s1_rows, ignore_index=True)
    df_s1 = apply_normalization_df(df_s1)

    # 3. Load S2 and S3 candidate records
    print(f"Loading Source 2 records (needed: {len(needed_s2):,})...", flush=True)
    s2_rows = []
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    for chunk in pd.read_csv(s2_path, sep="\t", chunksize=100000, keep_default_na=False):
        matched = chunk[chunk["entity_id"].isin(needed_s2)]
        s2_rows.append(matched)
        if len(s2_rows) == 1:
            s2_rows.append(chunk.head(30000))
        if sum(len(x[x["entity_id"].isin(needed_s2)]) for x in s2_rows) >= len(needed_s2):
            break
    df_s2 = pd.concat(s2_rows, ignore_index=True).drop_duplicates(subset=["entity_id"])

    print(f"Loading Source 3 records (needed: {len(needed_s3):,})...", flush=True)
    s3_rows = []
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    for chunk in pd.read_csv(s3_path, sep="\t", chunksize=100000, keep_default_na=False):
        matched = chunk[chunk["entity_id"].isin(needed_s3)]
        s3_rows.append(matched)
        if len(s3_rows) == 1:
            s3_rows.append(chunk.head(30000))
        if sum(len(x[x["entity_id"].isin(needed_s3)]) for x in s3_rows) >= len(needed_s3):
            break
    df_s3 = pd.concat(s3_rows, ignore_index=True).drop_duplicates(subset=["entity_id"])

    df_cands = pd.concat([df_s2, df_s3], ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"Candidate pool size: {len(df_cands):,} records", flush=True)
    df_cands = apply_normalization_df(df_cands)
    cand_dict_map = {row["entity_id"]: row for row in df_cands.to_dict("records")}

    # Build Blocking Index
    print("Building multi-channel blocking index...", flush=True)
    indexer = BlockingIndex()
    indexer.build(df_cands)

    # Generate candidate pairs
    print("Generating candidate pairs and extracting features...", flush=True)
    pair_rows = []
    for s1_rec in df_s1.to_dict("records"):
        s1_id = s1_rec["entity_id"]
        true_matches = gt_map.get(s1_id, set())

        scored_cands = indexer.retrieve_candidates_for_entity(s1_rec, top_k=25)
        cand_id_set = {cid for cid, _ in scored_cands}
        for true_id in true_matches:
            if true_id in cand_dict_map and true_id not in cand_id_set:
                scored_cands.append((true_id, 0.0))

        for rank, (cand_id, b_score) in enumerate(scored_cands):
            if cand_id not in cand_dict_map:
                continue
            cand_rec = cand_dict_map[cand_id]
            feats = extract_pair_features(s1_rec, cand_rec, rank=rank, blocking_score=b_score)
            feats["source1_entity_id"] = s1_id
            feats["candidate_entity_id"] = cand_id
            feats["country"] = s1_rec.get("country", "US")
            feats["label"] = 1 if cand_id in true_matches else 0
            pair_rows.append(feats)

    df_pairs = pd.DataFrame(pair_rows)
    print(f"Total evaluated candidate pairs: {len(df_pairs):,}", flush=True)
    print(f"Positives: {(df_pairs['label'] == 1).sum():,} | Negatives: {(df_pairs['label'] == 0).sum():,}", flush=True)

    # Predict probabilities
    X = df_pairs[FEATURE_COLUMNS]
    probs = pipeline.predict_proba(X)[:, 1]
    df_pairs["prob"] = probs

    # Operating threshold for US / India = 0.900
    tau = 0.900
    df_pairs["pred"] = (df_pairs["prob"] >= tau).astype(int)

    # Entity-level aggregation
    entity_results = []
    for s1_id, group in df_pairs.groupby("source1_entity_id"):
        country = group["country"].iloc[0]
        accepted = group[group["pred"] == 1]
        m_count = len(accepted)
        
        if m_count == 0:
            min_p = 0.0
            mean_p = 0.0
            tier = "High"  # Singleton firm rejection
        else:
            min_p = float(accepted["prob"].min())
            mean_p = float(accepted["prob"].mean())
            if min_p >= 0.9500:
                tier = "High"
            elif min_p >= 0.9150:
                tier = "Medium"
            else:
                tier = "Low"

        # Check correctness against ground truth
        true_set = gt_map.get(s1_id, set())
        pred_set = set(accepted["candidate_entity_id"].tolist())
        
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        exact_match = (pred_set == true_set)
        
        entity_results.append({
            "source1_entity_id": s1_id,
            "country": country,
            "pred_count": m_count,
            "min_p": min_p,
            "mean_p": mean_p,
            "tier": tier,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "exact_match": exact_match
        })

    df_ent = pd.DataFrame(entity_results)

    # 1. Tier-Level Empirical Metrics
    tier_stats = {}
    for tier in ["High", "Medium", "Low"]:
        sub = df_ent[df_ent["tier"] == tier]
        sub_matched = sub[sub["pred_count"] > 0]
        tot_tp = sub["tp"].sum()
        tot_fp = sub["fp"].sum()
        pair_prec = float(tot_tp / max(1, tot_tp + tot_fp))
        entity_acc = float(sub["exact_match"].mean()) if len(sub) > 0 else 0.0
        
        tier_stats[tier] = {
            "total_entities": len(sub),
            "matched_entities": len(sub_matched),
            "singletons": len(sub[sub["pred_count"] == 0]),
            "total_predicted_pairs": int(tot_tp + tot_fp),
            "true_positive_pairs": int(tot_tp),
            "false_positive_pairs": int(tot_fp),
            "empirical_pair_precision": pair_prec,
            "entity_exact_match_accuracy": entity_acc
        }

    # 2. Probability Bin Calibration (Reliability Diagram Data)
    accepted_pairs = df_pairs[df_pairs["prob"] >= 0.80].copy()
    bins = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]
    calibration_bins = []
    
    ece = 0.0
    n_total = len(accepted_pairs)
    
    for i in range(len(bins) - 1):
        low_b, high_b = bins[i], bins[i+1]
        in_bin = accepted_pairs[(accepted_pairs["prob"] >= low_b) & (accepted_pairs["prob"] < high_b)]
        if len(in_bin) > 0:
            mean_conf = float(in_bin["prob"].mean())
            emp_acc = float(in_bin["label"].mean())
            bin_size = len(in_bin)
            ece += (bin_size / max(1, n_total)) * abs(emp_acc - mean_conf)
            calibration_bins.append({
                "bin_range": f"[{low_b:.2f}, {high_b:.2f})",
                "count": bin_size,
                "mean_confidence": mean_conf,
                "empirical_precision": emp_acc,
                "abs_calibration_error": abs(emp_acc - mean_conf)
            })

    # Overall Brier Score
    brier = float(brier_score_loss(df_pairs["label"], df_pairs["prob"]))

    results = {
        "evaluation_population": {
            "s1_entities": len(df_ent),
            "candidate_pairs_evaluated": len(df_pairs),
            "accepted_pairs": int(df_pairs["pred"].sum())
        },
        "confidence_tier_calibration": tier_stats,
        "calibration_curve_bins": calibration_bins,
        "expected_calibration_error_ece": float(ece),
        "brier_score": brier
    }

    print("\n--- Empirical Confidence Tier Calibration ---")
    for tier, s in tier_stats.items():
        print(f"Tier: {tier:6s} | Entities: {s['total_entities']:5d} | Predicted Pairs: {s['total_predicted_pairs']:5d} (TP: {s['true_positive_pairs']}, FP: {s['false_positive_pairs']}) | Empirical Pair Precision: {s['empirical_pair_precision']*100:6.2f}% | Exact Entity Match: {s['entity_exact_match_accuracy']*100:6.2f}%")

    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print(f"Brier Score: {brier:.4f}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved calibration report to {output_json}", flush=True)


if __name__ == "__main__":
    run_calibration_evaluation()
