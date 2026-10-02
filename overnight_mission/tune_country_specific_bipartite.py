"""Country-Specific 2-Set Disjoint Calibration (Tuning India vs US separately)."""

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
    print("COUNTRY-SPECIFIC 2-SET DISJOINT CALIBRATION (INDIA RECALL RECOVERY)")
    print("=" * 80)

    model_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load 1,500 validation entities with country annotations
    df_held = pd.read_parquet("overnight_mission/eval/held_out_split_ids.parquet")
    sample_sids = set(df_held["entity_id"].head(2000))

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
    dist_count = 0
    for p in ["student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv"]:
        for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
            sub_true = chunk[chunk["entity_id"].isin(needed_cands)]
            if len(sub_true) > 0:
                c_rows.append(sub_true)
            if dist_count < 80000:
                c_rows.append(chunk.head(20000))
                dist_count += len(chunk.head(20000))

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

    # Separate India and US entities
    india_sids = set([sid for sid, r in s1_map.items() if r.get("country", "").lower() == "india"])
    us_sids = set([sid for sid, r in s1_map.items() if r.get("country", "").lower() in ["us", "usa", "united states"]])
    print(f"Entities: Total={len(s1_map):,}, India={len(india_sids):,}, US={len(us_sids):,}", flush=True)

    # Grid search: tau_us in [0.75, 0.78, 0.80], tau_india in [0.60, 0.65, 0.70, 0.75, 0.80]
    print("\n" + "=" * 95)
    print(f"{'tau_US':<10} | {'tau_India':<10} | {'Overall F0.5':<14} | {'India F0.5':<12} | {'US F0.5':<12} | {'Overall Prec':<12} | {'Overall Rec'}")
    print("=" * 95)

    best_score = 0.0
    best_config = None

    for tau_us in [0.75, 0.78, 0.80]:
        for tau_ind in [0.60, 0.65, 0.70, 0.75, 0.78, 0.80]:
            # Disjoint Bipartite matching with country-specific thresholds
            all_edges = []
            for sid, pairs in scored_pairs.items():
                is_ind = sid in india_sids
                thresh = tau_ind if is_ind else tau_us
                for cid, p in pairs:
                    if p >= thresh:
                        all_edges.append((sid, cid, p))
            all_edges.sort(key=lambda x: x[2], reverse=True)

            claimed = set()
            s1_matches = {sid: [] for sid in s1_map}
            for sid, cid, p in all_edges:
                if cid not in claimed and len(s1_matches[sid]) < 6:
                    s1_matches[sid].append(cid)
                    claimed.add(cid)

            preds = {sid: set(m) for sid, m in s1_matches.items()}
            f05, prec, rec = compute_macro_f05(gt_map, preds)

            preds_ind = {sid: preds[sid] for sid in india_sids if sid in preds}
            gt_ind = {sid: gt_map[sid] for sid in india_sids if sid in gt_map}
            f_ind, p_ind, r_ind = compute_macro_f05(gt_ind, preds_ind)

            preds_us = {sid: preds[sid] for sid in us_sids if sid in preds}
            gt_us = {sid: gt_map[sid] for sid in us_sids if sid in gt_map}
            f_us, p_us, r_us = compute_macro_f05(gt_us, preds_us)

            star = ""
            if f05 > best_score:
                best_score = f05
                best_config = (tau_us, tau_ind)
                star = " *** NEW BEST ***"

            print(f"{tau_us:<10.2f} | {tau_ind:<10.2f} | {f05:<14.6f} | {f_ind:<12.6f} | {f_us:<12.6f} | {prec*100:>10.2f}% | {rec*100:>10.2f}%{star}", flush=True)

    print("=" * 95)
    print(f"OPTIMAL COUNTRY THRESHOLDS: tau_US = {best_config[0]:.2f}, tau_India = {best_config[1]:.2f}")
    print(f"PEAK MACRO F0.5: {best_score:.6f}")

if __name__ == "__main__":
    main()
