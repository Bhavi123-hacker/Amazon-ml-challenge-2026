"""Benchmark realistic candidate selection rules on ground truth with FULL distractors.

Discovers the EXACT decision rule that maximizes Macro F0.5 on entities with full distractor sets.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.blocking import BlockingIndex
from src.evaluate import compute_macro_f05
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df

def main():
    print("=" * 80)
    print("BENCHMARKING DECISION RULES ON GROUND TRUTH WITH REALISTIC DISTRACTORS")
    print("=" * 80)

    # 1. Load Champion Model
    with open("overnight_mission/models/champion_step3_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 2. Sample 1,500 entities from held-out split
    df_held = pd.read_parquet("overnight_mission/eval/held_out_split_ids.parquet")
    np.random.seed(42)
    sample_sids = set(np.random.choice(df_held["entity_id"], size=1500, replace=False))

    # Load S1 records
    print("Loading S1 sample...", flush=True)
    s1_rows = []
    for chunk in pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(sample_sids)]
        if len(sub) > 0:
            s1_rows.append(sub)
    df_s1 = apply_normalization_df(pd.concat(s1_rows, ignore_index=True))
    s1_map = {r["entity_id"]: r for r in df_s1.to_dict("records")}

    # Load ground truth
    df_gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)
    gt_map = {}
    needed_cands = set()
    for _, r in df_gt[df_gt["source1_entity_id"].isin(sample_sids)].iterrows():
        sid = r["source1_entity_id"]
        m = [x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()] if r["matched_entity_ids"] else []
        gt_map[sid] = set(m)
        needed_cands.update(m)

    # Load large candidate pool: true matches + 200,000 realistic distractors
    print("Loading large candidate pool (true matches + 200,000 distractors)...", flush=True)
    c_rows = []
    distractor_count = 0
    for p in ["student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv"]:
        for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
            sub_true = chunk[chunk["entity_id"].isin(needed_cands)]
            if len(sub_true) > 0:
                c_rows.append(sub_true)
            if distractor_count < 100000:
                c_rows.append(chunk.head(25000))
                distractor_count += len(chunk.head(25000))

    df_cands = apply_normalization_df(pd.concat(c_rows, ignore_index=True).drop_duplicates("entity_id"))
    c_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
    print(f"Loaded {len(c_map):,} candidates in index.", flush=True)

    # Build Index
    indexer = BlockingIndex()
    indexer.build(df_cands)

    # Retrieve 20 candidates per entity and score them
    print("Retrieving and scoring 20 candidates per entity...", flush=True)
    scored_pairs = {}
    for sid, r1 in s1_map.items():
        ret = indexer.retrieve_candidates_for_entity(r1, top_k=20)
        feat_matrix = []
        c_list = []
        for cid, _ in ret:
            rc = c_map.get(cid)
            if rc:
                feats = extract_pair_features(r1, rc)
                feat_matrix.append([feats[col] for col in FEATURE_COLUMNS])
                c_list.append(cid)
        if feat_matrix:
            probs = model.predict_proba(np.array(feat_matrix, dtype=np.float32))[:, 1]
            # sort descending
            pairs = sorted(zip(c_list, probs), key=lambda x: x[1], reverse=True)
            scored_pairs[sid] = pairs
        else:
            scored_pairs[sid] = []

    print(f"Scored {len(scored_pairs):,} entities.", flush=True)

    # Test Decision Rules
    rules = [
        # (Name, function(pairs) -> list of accepted cids)
        ("1. Fixed tau = 0.75 (Old Baseline)", lambda pairs: [cid for cid, p in pairs if p >= 0.75][:10]),
        ("2. Fixed tau = 0.85", lambda pairs: [cid for cid, p in pairs if p >= 0.85][:10]),
        ("3. Fixed tau = 0.90", lambda pairs: [cid for cid, p in pairs if p >= 0.90][:10]),
        ("4. Fixed tau = 0.95", lambda pairs: [cid for cid, p in pairs if p >= 0.95][:10]),
        ("5. Fixed tau = 0.98", lambda pairs: [cid for cid, p in pairs if p >= 0.98][:10]),
        ("6. Fixed tau = 0.99", lambda pairs: [cid for cid, p in pairs if p >= 0.99][:10]),
        ("7. Dynamic Gap: p >= 0.80 & (p1 - p <= 0.05)",
         lambda pairs: [cid for cid, p in pairs if p >= 0.80 and (pairs[0][1] - p <= 0.05)][:10] if pairs else []),
        ("8. Dynamic Gap: p >= 0.85 & (p1 - p <= 0.03)",
         lambda pairs: [cid for cid, p in pairs if p >= 0.85 and (pairs[0][1] - p <= 0.03)][:10] if pairs else []),
        ("9. Dynamic Ratio: p >= 0.80 & (p / p1 >= 0.95)",
         lambda pairs: [cid for cid, p in pairs if p >= 0.80 and (p / max(1e-6, pairs[0][1]) >= 0.95)][:10] if pairs else []),
        ("10. Largest Dropoff Knee: p >= 0.75, cut at max delta p",
         lambda pairs: _cut_at_knee(pairs, min_p=0.75)),
        ("11. Largest Dropoff Knee: p >= 0.85, cut at max delta p",
         lambda pairs: _cut_at_knee(pairs, min_p=0.85)),
        ("12. High-Precision Cap: p >= 0.90, max matches = 4",
         lambda pairs: [cid for cid, p in pairs if p >= 0.90][:4]),
        ("13. High-Precision Cap: p >= 0.95, max matches = 4",
         lambda pairs: [cid for cid, p in pairs if p >= 0.95][:4]),
    ]

    print("\n" + "=" * 95)
    print(f"{'Decision Strategy':<45} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Avg Matches'}")
    print("=" * 95)

    results = []
    for name, rule_fn in rules:
        preds = {}
        total_m = 0
        for sid in s1_map:
            pairs = scored_pairs.get(sid, [])
            accepted = rule_fn(pairs)
            preds[sid] = set(accepted)
            total_m += len(accepted)
        f05, p, r = compute_macro_f05(gt_map, preds)
        avg_m = total_m / len(s1_map)
        results.append((name, f05, p, r, avg_m))
        print(f"{name:<45} | {f05:<12.6f} | {p*100:>8.2f}% | {r*100:>8.2f}% | {avg_m:.2f}")

    print("=" * 95, flush=True)

def _cut_at_knee(pairs, min_p=0.75):
    valid = [(cid, p) for cid, p in pairs if p >= min_p]
    if not valid:
        return []
    if len(valid) == 1:
        return [valid[0][0]]
    # Find largest drop between adjacent probabilities
    deltas = [valid[i][1] - valid[i+1][1] for i in range(len(valid) - 1)]
    max_idx = int(np.argmax(deltas))
    # If the largest drop is significant (>= 0.05), cut there
    if deltas[max_idx] >= 0.05:
        return [c for c, _ in valid[:max_idx + 1]]
    return [c for c, _ in valid[:4]]

if __name__ == "__main__":
    main()
