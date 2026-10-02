"""Evaluate blocking candidate recall at different K cutoffs and blocking score thresholds.
"""

import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import numpy as np
import pandas as pd

from src.blocking import BlockingIndex
from src.normalize import apply_normalization_df


def main():
    print("=== Step 1 & 2: Profiling Blocking Recall @ K on Training Ground Truth ===", flush=True)
    t0 = time.time()
    
    # Load 3,000 ground truth entities
    gt_path = "student_resource/dataset/train/train_ground_truth.tsv"
    df_gt = pd.read_csv(gt_path, sep="\t", nrows=3000, keep_default_na=False)
    gt_map = {}
    needed_cands = set()
    for _, row in df_gt.iterrows():
        s1 = row["source1_entity_id"].strip()
        m_str = row["matched_entity_ids"].strip()
        m_set = {x.strip() for x in m_str.split(",") if x.strip()} if m_str else set()
        gt_map[s1] = m_set
        for m in m_set:
            needed_cands.add(m)

    target_s1_ids = set(gt_map.keys())
    print(f"Loaded {len(gt_map):,} S1 entities ({sum(len(v) for v in gt_map.values()):,} true match pairs).", flush=True)

    # Load S1
    s1_rows = []
    for chunk in pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(target_s1_ids)]
        if len(m) > 0:
            s1_rows.append(m)
        if sum(len(x) for x in s1_rows) == len(target_s1_ids):
            break
    df_s1 = pd.concat(s1_rows, ignore_index=True)
    df_s1 = apply_normalization_df(df_s1)

    # Load candidates
    cands_rows = []
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join("student_resource/dataset/train", src)
        for chunk in pd.read_csv(path, sep="\t", chunksize=100000, keep_default_na=False):
            m = chunk[chunk["entity_id"].isin(needed_cands)]
            cands_rows.append(m)
            if len(cands_rows) <= 2:
                # Add distractor negatives
                cands_rows.append(chunk.head(25000))
            if sum(len(x[x["entity_id"].isin(needed_cands)]) for x in cands_rows) >= len(needed_cands):
                break
    df_cands = pd.concat(cands_rows, ignore_index=True).drop_duplicates(subset=["entity_id"])
    df_cands = apply_normalization_df(df_cands)
    print(f"Built candidate pool of {len(df_cands):,} records.", flush=True)

    # Build BlockingIndex
    indexer = BlockingIndex()
    indexer.build(df_cands)
    print("BlockingIndex built.", flush=True)

    # Evaluate recall @ K
    k_values = [3, 5, 8, 10, 12, 15, 20, 25]
    recall_hits = {k: 0 for k in k_values}
    score_cutoffs = [0.0, 30.0, 40.0, 50.0]
    score_hits = {sc: 0 for sc in score_cutoffs}
    total_true_pairs = sum(len(matches) for matches in gt_map.values())

    # Measure channel contributions and overlap
    channel_hits = {
        "tok2": 0,
        "phone": 0,
        "postal": 0,
        "addr": 0,
        "sigtoken": 0,
    }

    t_eval = time.time()
    for s1_rec in df_s1.to_dict("records"):
        s1_id = s1_rec["entity_id"]
        true_matches = gt_map.get(s1_id, set())
        if not true_matches:
            continue

        scored_cands = indexer.retrieve_candidates_for_entity(s1_rec, top_k=25)
        cand_ids = [cid for cid, score in scored_cands]

        for k in k_values:
            top_k_ids = set(cand_ids[:k])
            recall_hits[k] += len(true_matches & top_k_ids)

        for sc in score_cutoffs:
            filt_ids = set(cid for cid, score in scored_cands if score >= sc)
            score_hits[sc] += len(true_matches & filt_ids)

    print("\n" + "=" * 75)
    print("BLOCKING RECALL @ K PROFILE (Training Ground Truth)")
    print("=" * 75)
    print(f"Total True Pairs Evaluated: {total_true_pairs:,}\n")
    print(f"{'Cutoff (K)':<12} | {'Captured True Pairs':<20} | {'Recall':<10} | {'Candidates/Entity':<18}")
    print("-" * 75)
    for k in k_values:
        rec = recall_hits[k] / total_true_pairs
        print(f"Top-{k:<8} | {recall_hits[k]:<20,} | {rec*100:<9.2f}% | {k:<18}")

    print("\n" + "=" * 75)
    print("BLOCKING RECALL BY MINIMUM COMPOSITE BLOCKING SCORE (at K=25)")
    print("=" * 75)
    for sc in score_cutoffs:
        rec = score_hits[sc] / total_true_pairs
        print(f"Score >= {sc:<5.1f} | Captured: {score_hits[sc]:<6,} | Recall: {rec*100:.2f}%")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
