"""Test Rank-Based 2-Set Disjoint Bipartite Resolution on validation set."""

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
    model_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df_held = pd.read_parquet("overnight_mission/eval/held_out_split_ids.parquet")
    np.random.seed(42)
    sample_sids = set(np.random.choice(df_held["entity_id"], size=1000, replace=False))

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
            if distractor_count < 60000:
                c_rows.append(chunk.head(15000))
                distractor_count += len(chunk.head(15000))

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

    # Method A: Dynamic Margin (No Disjoint Resolution)
    preds_a = {}
    for sid, pairs in scored_pairs.items():
        if pairs and pairs[0][1] >= 0.75:
            p1 = pairs[0][1]
            preds_a[sid] = [cid for cid, p in pairs if p >= 0.75 and (p1 - p) <= 0.08][:6]
        else:
            preds_a[sid] = []

    f05_a, prec_a, rec_a = compute_macro_f05(gt_map, {s: set(m) for s, m in preds_a.items()})
    print(f"Method A (Dynamic Margin, Independent): F0.5 = {f05_a:.6f} | Prec = {prec_a*100:.2f}% | Rec = {rec_a*100:.2f}%")

    # Method B: Dynamic Margin + Rank-Based Disjoint Bipartite Resolution
    # Each candidate is assigned to the S1 entity where it has the earliest rank index
    cand_claims = {} # cid -> (sid, rank_idx)
    for sid, m_list in preds_a.items():
        for r_idx, cid in enumerate(m_list):
            if cid not in cand_claims:
                cand_claims[cid] = (sid, r_idx)
            else:
                prev_sid, prev_idx = cand_claims[cid]
                if r_idx < prev_idx:
                    cand_claims[cid] = (sid, r_idx)

    preds_b = {sid: [] for sid in preds_a}
    for cid, (sid, _) in cand_claims.items():
        preds_b[sid].append(cid)

    f05_b, prec_b, rec_b = compute_macro_f05(gt_map, {s: set(m) for s, m in preds_b.items()})
    print(f"Method B (Dynamic Margin + Rank Disjoint): F0.5 = {f05_b:.6f} | Prec = {prec_b*100:.2f}% | Rec = {rec_b*100:.2f}%")

if __name__ == "__main__":
    main()
