"""Analyze Ground-Truth Cluster Size Distribution & Tail Risk for K=12 Candidate Truncation.

This script scans the entire 2,206,821 rows of train_ground_truth.tsv and train_source1.tsv
to empirically evaluate whether K=12 candidate cutoff artificially truncates legitimate large
entity clusters or if real-world ground truth clusters are bounded well below K=12.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

def analyze_cluster_distribution():
    data_dir = "student_resource/dataset/train"
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(data_dir, "train_source1.tsv")

    print(f"Reading {s1_path} to map entity_id -> country...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"])
    country_map = dict(zip(df_s1["entity_id"], df_s1["country"]))
    del df_s1

    print(f"Reading {gt_path} to analyze cluster sizes...", flush=True)
    
    counts_all = []
    counts_by_country = {"India": [], "US": []}
    
    total_entities = 0
    total_pairs = 0
    empty_entities = 0

    with open(gt_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            
            if matched_str:
                m_list = [x for x in matched_str.split(",") if x.strip()]
                k = len(m_list)
            else:
                k = 0
            
            total_entities += 1
            total_pairs += k
            counts_all.append(k)
            
            country = country_map.get(s1_id, "Unknown")
            if country in counts_by_country:
                counts_by_country[country].append(k)
            
            if k == 0:
                empty_entities += 1

    counts_arr = np.array(counts_all, dtype=np.int32)
    matched_mask = counts_arr > 0
    matched_counts = counts_arr[matched_mask]

    print("\n--- Full Population Summary ---")
    print(f"Total S1 entities: {total_entities:,}")
    print(f"Total Ground-Truth Pairs: {total_pairs:,}")
    print(f"Empty / Singleton entities (0 matches): {empty_entities:,} ({empty_entities / total_entities * 100:.2f}%)")
    print(f"Matched entities: {len(matched_counts):,} ({len(matched_counts) / total_entities * 100:.2f}%)")

    # Frequency histogram of match counts
    hist = Counter(counts_all)
    print("\n--- Match Count Frequency Distribution (All Entities) ---")
    for k in sorted(hist.keys())[:25]:
        cnt = hist[k]
        pct = cnt / total_entities * 100
        cum_cnt = sum(hist[i] for i in range(k + 1))
        cum_pct = cum_cnt / total_entities * 100
        print(f"  k = {k:2d}: {cnt:10,d} entities ({pct:6.2f}%) | Cumulative: {cum_cnt:10,d} ({cum_pct:6.2f}%)")

    # Percentiles for matched entities
    percentiles = [50, 75, 90, 95, 98, 99, 99.5, 99.9, 99.99, 100]
    p_vals = np.percentile(matched_counts, percentiles)
    print("\n--- Percentiles for Matched Entities (k > 0) ---")
    for p, val in zip(percentiles, p_vals):
        print(f"  p{p:<5}: {val:.2f} matches")

    # Tail analysis above K=12
    n_above_12 = np.sum(counts_arr > 12)
    pct_above_12 = n_above_12 / total_entities * 100
    pct_matched_above_12 = n_above_12 / len(matched_counts) * 100
    pairs_above_12 = np.sum(counts_arr[counts_arr > 12])
    pairs_lost_if_capped_at_12 = np.sum(np.maximum(0, counts_arr - 12))
    pct_pairs_lost_if_capped = pairs_lost_if_capped_at_12 / total_pairs * 100

    print("\n--- Tail Risk Evaluation for K=12 ---")
    print(f"Entities with > 12 matches: {n_above_12:,} out of {total_entities:,} ({pct_above_12:.4f}% of all entities)")
    print(f"Entities with > 12 matches as % of matched: {pct_matched_above_12:.4f}%")
    print(f"Entities covered with <= 12 matches: {total_entities - n_above_12:,} ({(total_entities - n_above_12) / total_entities * 100:.4f}%)")
    print(f"Total ground truth pairs in clusters > 12: {pairs_above_12:,}")
    print(f"Exact pair truncation if clusters were capped at 12: {pairs_lost_if_capped_at_12:,} pairs ({pct_pairs_lost_if_capped:.4f}% of total ground truth pairs)")

    # Per country breakdown
    country_stats = {}
    for c, arr in counts_by_country.items():
        c_np = np.array(arr, dtype=np.int32)
        c_matched = c_np[c_np > 0]
        c_above_12 = np.sum(c_np > 12)
        c_lost = np.sum(np.maximum(0, c_np - 12))
        c_tot_pairs = np.sum(c_np)
        country_stats[c] = {
            "total_entities": len(c_np),
            "matched_entities": len(c_matched),
            "total_pairs": int(c_tot_pairs),
            "mean_cluster_size": float(np.mean(c_matched)) if len(c_matched) > 0 else 0,
            "max_cluster_size": int(np.max(c_np)) if len(c_np) > 0 else 0,
            "p95": float(np.percentile(c_matched, 95)) if len(c_matched) > 0 else 0,
            "p99": float(np.percentile(c_matched, 99)) if len(c_matched) > 0 else 0,
            "entities_above_12": int(c_above_12),
            "pct_entities_above_12": float(c_above_12 / len(c_np) * 100),
            "pairs_lost_at_12": int(c_lost),
            "pct_pairs_lost_at_12": float(c_lost / max(1, c_tot_pairs) * 100),
            "coverage_at_12_pct": float((len(c_np) - c_above_12) / len(c_np) * 100)
        }

    results = {
        "total_population_entities": total_entities,
        "total_ground_truth_pairs": total_pairs,
        "singletons": empty_entities,
        "singleton_pct": float(empty_entities / total_entities * 100),
        "matched_entities": len(matched_counts),
        "mean_matched_cluster_size": float(np.mean(matched_counts)),
        "median_matched_cluster_size": float(np.median(matched_counts)),
        "max_cluster_size": int(np.max(counts_arr)),
        "percentiles_matched": {f"p{p}": float(v) for p, v in zip(percentiles, p_vals)},
        "k12_tail_risk": {
            "entities_with_gt_greater_than_12": int(n_above_12),
            "pct_of_all_entities_greater_than_12": float(pct_above_12),
            "pct_of_matched_entities_greater_than_12": float(pct_matched_above_12),
            "population_coverage_at_k12_pct": float((total_entities - n_above_12) / total_entities * 100),
            "pairs_truncated_at_k12": int(pairs_lost_if_capped_at_12),
            "pair_retention_at_k12_pct": float((1.0 - pct_pairs_lost_if_capped / 100.0) * 100)
        },
        "per_country_breakdown": country_stats
    }

    out_file = "models/ground_truth_cluster_distribution.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote full cluster distribution analysis to {out_file}", flush=True)

if __name__ == "__main__":
    analyze_cluster_distribution()
