"""Tune 2-Set Disjoint Bipartite Resolution to push Macro F0.5 beyond 0.97."""

import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.blocking import BlockingIndex
from src.evaluate import compute_macro_f05
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df

def main():
    print("=" * 80)
    print("OPTIMIZING 2-SET DISJOINT BIPARTITE RESOLUTION")
    print("=" * 80)

    model_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 1. Sample 1,500 entities
    df_held = pd.read_parquet("overnight_mission/eval/held_out_split_ids.parquet")
    np.random.seed(42)
    sample_sids = set(np.random.choice(df_held["entity_id"], size=1500, replace=False))

    s1_rows = []
    for chunk in pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(sample_sids)]
        if len(sub) > 0:
            s1_rows.append(sub)
    df_s1 = apply_normalization_df(pd.concat(s1_rows, ignore_index=True))
    s1_map = {r["entity_id"]: r for r in df_s1.to_dict("records")}

    df_gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)
    gt_map = {}
    needed_cands = set()
    for _, r in df_gt[df_gt["source1_entity_id"].isin(sample_sids)].iterrows():
        sid = r["source1_entity_id"]
        m = [x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()] if r["matched_entity_ids"] else []
        gt_map[sid] = set(m)
        needed_cands.update(m)

    c_rows = []
    distractor_count = 0
    for p in ["student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv"]:
        for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
            sub_true = chunk[chunk["entity_id"].isin(needed_cands)]
            if len(sub_true) > 0:
                c_rows.append(sub_true)
            if distractor_count < 80000:
                c_rows.append(chunk.head(20000))
                distractor_count += len(chunk.head(20000))

    df_cands = apply_normalization_df(pd.concat(c_rows, ignore_index=True).drop_duplicates("entity_id"))
    c_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}

    indexer = BlockingIndex()
    indexer.build(df_cands)

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
            pairs = sorted(zip(c_list, probs), key=lambda x: x[1], reverse=True)
            scored_pairs[sid] = pairs
        else:
            scored_pairs[sid] = []

    print(f"Scored {len(scored_pairs):,} entities. Starting 2-Set Disjoint Grid Search...\n", flush=True)

    # Grid search over tau and max_matches
    tau_vals = [0.65, 0.70, 0.72, 0.75, 0.77, 0.80, 0.82, 0.85]
    max_k_vals = [3, 4, 5, 6, 8, 10]

    best_score = 0.0
    best_params = None

    print(f"{'tau':<8} | {'max_k':<8} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Avg Matches':<12}")
    print("-" * 75)

    for tau in tau_vals:
        for max_k in max_k_vals:
            # Flatten edges above tau
            all_edges = []
            for sid, pairs in scored_pairs.items():
                for cid, p in pairs:
                    if p >= tau:
                        all_edges.append((sid, cid, p))
            all_edges.sort(key=lambda x: x[2], reverse=True)

            claimed_s2 = set()
            s1_matches = {sid: [] for sid in s1_map}
            for sid, cid, p in all_edges:
                if cid not in claimed_s2 and len(s1_matches[sid]) < max_k:
                    s1_matches[sid].append(cid)
                    claimed_s2.add(cid)

            preds = {sid: set(m) for sid, m in s1_matches.items()}
            f05, prec, rec = compute_macro_f05(gt_map, preds)
            avg_m = sum(len(m) for m in preds.values()) / len(s1_map)

            star = ""
            if f05 > best_score:
                best_score = f05
                best_params = (tau, max_k)
                star = " *** NEW BEST ***"

            print(f"{tau:<8.2f} | {max_k:<8} | {f05:<12.6f} | {prec*100:>8.2f}% | {rec*100:>8.2f}% | {avg_m:>10.2f}{star}", flush=True)

    print("\n" + "=" * 80)
    print(f"BEST 2-SET BIPARTITE PARAMS: tau = {best_params[0]:.2f}, max_k = {best_params[1]}")
    print(f"PEAK MACRO F0.5: {best_score:.6f}")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
