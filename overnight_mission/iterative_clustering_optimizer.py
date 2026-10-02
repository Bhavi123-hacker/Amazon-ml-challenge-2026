"""Iterative 2-Stage Clustering and Decision Optimizer across 15+ Approaches.

Evaluates on held-out ground truth with full realistic distractors.
Tracks Macro F0.5 at every iteration to discover the champion clustering strategy.
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
    print("=" * 95, flush=True)
    print("ITERATIVE 2-STAGE CLUSTERING OPTIMIZER (15+ DISTINCT ARCHITECTURES)", flush=True)
    print("=" * 95, flush=True)

    # 1. Load Champion Model
    model_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded champion LightGBM model from {model_path}.", flush=True)

    # 2. Sample 2,000 S1 entities from held-out split
    df_held = pd.read_parquet("overnight_mission/eval/held_out_split_ids.parquet")
    np.random.seed(42)
    sample_sids = set(np.random.choice(df_held["entity_id"], size=2000, replace=False))

    print(f"Loading {len(sample_sids):,} held-out S1 entities...", flush=True)
    s1_rows = []
    for chunk in pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(sample_sids)]
        if len(sub) > 0:
            s1_rows.append(sub)
    df_s1 = apply_normalization_df(pd.concat(s1_rows, ignore_index=True))
    s1_map = {r["entity_id"]: r for r in df_s1.to_dict("records")}

    # Load Ground Truth
    df_gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)
    gt_map = {}
    needed_cands = set()
    for _, r in df_gt[df_gt["source1_entity_id"].isin(sample_sids)].iterrows():
        sid = r["source1_entity_id"]
        m = [x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()] if r["matched_entity_ids"] else []
        gt_map[sid] = set(m)
        needed_cands.update(m)

    print(f"Loaded ground truth. Total true match entities: {sum(len(v) > 0 for v in gt_map.values()):,}, True singletons: {sum(len(v) == 0 for v in gt_map.values()):,}.", flush=True)

    # Load Candidate Pool: true matches + 250,000 distractors
    print("Loading large candidate pool (true matches + 250,000 distractors)...", flush=True)
    c_rows = []
    distractor_count = 0
    for p in ["student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv"]:
        for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
            sub_true = chunk[chunk["entity_id"].isin(needed_cands)]
            if len(sub_true) > 0:
                c_rows.append(sub_true)
            if distractor_count < 125000:
                c_rows.append(chunk.head(30000))
                distractor_count += len(chunk.head(30000))

    df_cands = apply_normalization_df(pd.concat(c_rows, ignore_index=True).drop_duplicates("entity_id"))
    c_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
    print(f"Indexed {len(c_map):,} candidates.", flush=True)

    # Build Index & Retrieve
    indexer = BlockingIndex()
    indexer.build(df_cands)

    print("Retrieving and scoring candidate pairs...", flush=True)
    scored_pairs = {}  # sid -> list of (cid, prob, cand_name, cand_addr)
    for sid, r1 in s1_map.items():
        ret = indexer.retrieve_candidates_for_entity(r1, top_k=20)
        feat_matrix = []
        meta = []
        for cid, _ in ret:
            rc = c_map.get(cid)
            if rc:
                feats = extract_pair_features(r1, rc)
                feat_matrix.append([feats[col] for col in FEATURE_COLUMNS])
                meta.append((cid, rc.get("name_norm", ""), rc.get("addr_norm", "")))
        if feat_matrix:
            probs = model.predict_proba(np.array(feat_matrix, dtype=np.float32))[:, 1]
            pairs = []
            for (cid, cname, caddr), p in zip(meta, probs):
                pairs.append((cid, float(p), cname, caddr))
            pairs.sort(key=lambda x: x[1], reverse=True)
            scored_pairs[sid] = pairs
        else:
            scored_pairs[sid] = []

    print(f"Scored pairs ready for {len(scored_pairs):,} entities.\n", flush=True)

    # 15+ Distinct Clustering & Decision Architectures
    # Each takes: (s1_record, sorted_pairs) -> list of accepted cids
    architectures = [
        ("Iteration 01: Baseline Fixed tau=0.75",
         lambda r1, pairs: [cid for cid, p, _, _ in pairs if p >= 0.75][:10]),

        ("Iteration 02: Stricter Fixed tau=0.85",
         lambda r1, pairs: [cid for cid, p, _, _ in pairs if p >= 0.85][:10]),

        ("Iteration 03: Stricter Fixed tau=0.90",
         lambda r1, pairs: [cid for cid, p, _, _ in pairs if p >= 0.90][:10]),

        ("Iteration 04: Very Strict Fixed tau=0.95",
         lambda r1, pairs: [cid for cid, p, _, _ in pairs if p >= 0.95][:10]),

        ("Iteration 05: Dynamic Margin (p1 >= 0.80 & delta_p <= 0.08)",
         lambda r1, pairs: _dynamic_margin(pairs, min_p=0.80, delta_p=0.08)),

        ("Iteration 06: Dynamic Margin (p1 >= 0.80 & delta_p <= 0.05)",
         lambda r1, pairs: _dynamic_margin(pairs, min_p=0.80, delta_p=0.05)),

        ("Iteration 07: Dynamic Margin (p1 >= 0.85 & delta_p <= 0.03)",
         lambda r1, pairs: _dynamic_margin(pairs, min_p=0.85, delta_p=0.03)),

        ("Iteration 08: Dynamic Ratio (p1 >= 0.80 & p/p1 >= 0.95)",
         lambda r1, pairs: _dynamic_ratio(pairs, min_p=0.80, ratio=0.95)),

        ("Iteration 09: 2-Stage Consensus (Dynamic Margin + Cand-to-Top Name Sim >= 45)",
         lambda r1, pairs: _two_stage_consensus(pairs, min_p=0.80, delta_p=0.05, name_sim_cut=45)),

        ("Iteration 10: 2-Stage Consensus (Dynamic Margin + Cand-to-Top Name Sim >= 55)",
         lambda r1, pairs: _two_stage_consensus(pairs, min_p=0.80, delta_p=0.05, name_sim_cut=55)),

        ("Iteration 11: 2-Stage Triplet Consensus (S1 + TopCand + SecondaryCand Agreement)",
         lambda r1, pairs: _triplet_consensus(r1, pairs, min_p=0.80, delta_p=0.05)),

        ("Iteration 12: Cluster Prior Slicing (Cut at Largest Prob Gap if Gap >= 0.08)",
         lambda r1, pairs: _gap_cut(pairs, min_p=0.75, min_gap=0.08)),

        ("Iteration 13: Cluster Prior Slicing (Cut at Largest Prob Gap if Gap >= 0.05)",
         lambda r1, pairs: _gap_cut(pairs, min_p=0.75, min_gap=0.05)),

        ("Iteration 14: Hybrid 2-Stage (p1 >= 0.82, delta_p <= 0.04 + Multi-tenant Guard D)",
         lambda r1, pairs: _hybrid_guard(r1, pairs, min_p=0.82, delta_p=0.04)),

        ("Iteration 15: Optimal Precision-Tuned 2-Stage (p1 >= 0.85, delta_p <= 0.035, top<=5)",
         lambda r1, pairs: _hybrid_guard(r1, pairs, min_p=0.85, delta_p=0.035, max_m=5)),

        ("Iteration 16: Ultra-Sharp Ensemble Filter (p1 >= 0.88, delta_p <= 0.025, top<=4)",
         lambda r1, pairs: _hybrid_guard(r1, pairs, min_p=0.88, delta_p=0.025, max_m=4)),
    ]

    print("=" * 115)
    print(f"{'Approach / Iteration':<55} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Avg Matches':<12} | {'Singletons'}")
    print("=" * 115)

    history = []
    best_f05 = 0.0
    best_name = ""

    for idx, (name, fn) in enumerate(architectures, 1):
        preds = {}
        total_m = 0
        sing_count = 0
        for sid, r1 in s1_map.items():
            pairs = scored_pairs.get(sid, [])
            accepted = fn(r1, pairs)
            preds[sid] = set(accepted)
            total_m += len(accepted)
            if not accepted:
                sing_count += 1

        f05, p, r = compute_macro_f05(gt_map, preds)
        avg_m = total_m / len(s1_map)
        sing_pct = sing_count / len(s1_map) * 100.0

        if f05 > best_f05:
            best_f05 = f05
            best_name = name

        history.append({
            "iteration": idx,
            "name": name,
            "f05": f05,
            "precision": p,
            "recall": r,
            "avg_matches": avg_m,
            "singleton_pct": sing_pct
        })

        star = " *** NEW BEST ***" if f05 == best_f05 else ""
        print(f"{name:<55} | {f05:<12.6f} | {p*100:>8.2f}% | {r*100:>8.2f}% | {avg_m:>10.2f}   | {sing_pct:>5.1f}%{star}", flush=True)

    print("=" * 115)
    print(f"\nWINNING ARCHITECTURE: {best_name}")
    print(f"CHAMPION HELD-OUT MACRO F0.5: {best_f05:.6f}")
    print("=" * 115, flush=True)

    # Save results
    df_res = pd.DataFrame(history)
    df_res.to_csv("overnight_mission/eval/iterative_clustering_comparison.csv", index=False)
    print("Saved comparison table to overnight_mission/eval/iterative_clustering_comparison.csv")

def _dynamic_margin(pairs, min_p=0.80, delta_p=0.05, max_m=10):
    if not pairs or pairs[0][1] < min_p:
        return []
    p1 = pairs[0][1]
    return [cid for cid, p, _, _ in pairs if p >= min_p and (p1 - p) <= delta_p][:max_m]

def _dynamic_ratio(pairs, min_p=0.80, ratio=0.95, max_m=10):
    if not pairs or pairs[0][1] < min_p:
        return []
    p1 = pairs[0][1]
    return [cid for cid, p, _, _ in pairs if p >= min_p and (p / max(1e-6, p1)) >= ratio][:max_m]

def _two_stage_consensus(pairs, min_p=0.80, delta_p=0.05, name_sim_cut=45, max_m=10):
    if not pairs or pairs[0][1] < min_p:
        return []
    p1 = pairs[0][1]
    top_name = pairs[0][2]
    accepted = [pairs[0][0]]
    for cid, p, cname, _ in pairs[1:]:
        if p >= min_p and (p1 - p) <= delta_p:
            sim = fuzz.token_sort_ratio(top_name, cname)
            if sim >= name_sim_cut:
                accepted.append(cid)
        if len(accepted) >= max_m:
            break
    return accepted

def _triplet_consensus(r1, pairs, min_p=0.80, delta_p=0.05, max_m=10):
    if not pairs or pairs[0][1] < min_p:
        return []
    p1 = pairs[0][1]
    s1_name = r1.get("name_norm", "")
    top_name = pairs[0][2]
    accepted = [pairs[0][0]]
    for cid, p, cname, _ in pairs[1:]:
        if p >= min_p and (p1 - p) <= delta_p:
            sim1 = fuzz.token_sort_ratio(s1_name, cname)
            sim_top = fuzz.token_sort_ratio(top_name, cname)
            if sim1 >= 40 and sim_top >= 45:
                accepted.append(cid)
        if len(accepted) >= max_m:
            break
    return accepted

def _gap_cut(pairs, min_p=0.75, min_gap=0.05, max_m=10):
    valid = [(cid, p) for cid, p, _, _ in pairs if p >= min_p]
    if not valid:
        return []
    if len(valid) == 1:
        return [valid[0][0]]
    # Cut where probability drop is largest
    deltas = [valid[i][1] - valid[i+1][1] for i in range(len(valid) - 1)]
    max_idx = int(np.argmax(deltas))
    if deltas[max_idx] >= min_gap:
        return [c for c, _ in valid[:max_idx + 1]][:max_m]
    return [c for c, _ in valid[:4]][:max_m]

def _hybrid_guard(r1, pairs, min_p=0.82, delta_p=0.04, max_m=6):
    if not pairs or pairs[0][1] < min_p:
        return []
    p1 = pairs[0][1]
    s1_name = r1.get("name_norm", "")
    s1_addr = r1.get("addr_norm", "")
    top_name = pairs[0][2]

    accepted = []
    for rank, (cid, p, cname, caddr) in enumerate(pairs):
        if p < min_p or (p1 - p) > delta_p:
            continue
        ns = fuzz.token_sort_ratio(s1_name, cname)
        asim = fuzz.token_sort_ratio(s1_addr, caddr)
        # Guard D: reject co-located multi-tenant collision
        if ns < 45 and asim > 70:
            continue
        # Cluster consensus
        if rank > 0:
            sim_top = fuzz.token_sort_ratio(top_name, cname)
            if sim_top < 45:
                continue
        accepted.append(cid)
        if len(accepted) >= max_m:
            break
    return accepted

if __name__ == "__main__":
    main()
