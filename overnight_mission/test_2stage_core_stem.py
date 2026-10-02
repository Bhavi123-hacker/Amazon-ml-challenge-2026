"""Test Core-Stem 2-Stage Clustering on Ground Truth."""

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

STOPWORDS = {
    'sarl', 'eurl', 'sas', 'sasu', 'sa', 'sci', 'snc', 'gie',
    'inc', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'club', 'centre', 'ecole', 'association',
    'services', 'enterprises', 'trading', 'solutions', 'technologies', 'group', 'holdings'
}

def get_core_stem(name):
    if not name:
        return ""
    words = [w for w in name.lower().split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words) if words else name.lower()

def main():
    print("=" * 80)
    print("TESTING 2-STAGE CORE-STEM CLUSTERING ON HELD-OUT GROUND TRUTH")
    print("=" * 80)

    # 1. Load model
    with open("overnight_mission/models/champion_step3_model.pkl", "rb") as f:
        model = pickle.load(f)

    # 2. Sample 2,000 entities
    df_held = pd.read_parquet("overnight_mission/eval/held_out_split_ids.parquet")
    np.random.seed(42)
    sample_sids = set(np.random.choice(df_held["entity_id"], size=2000, replace=False))

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
            if distractor_count < 125000:
                c_rows.append(chunk.head(30000))
                distractor_count += len(chunk.head(30000))

    df_cands = apply_normalization_df(pd.concat(c_rows, ignore_index=True).drop_duplicates("entity_id"))
    c_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}

    indexer = BlockingIndex()
    indexer.build(df_cands)

    scored_pairs = {}
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
            pairs = sorted(zip(meta, probs), key=lambda x: x[1], reverse=True)
            scored_pairs[sid] = pairs
        else:
            scored_pairs[sid] = []

    # Test baseline vs 2-Stage Core-Stem Clustering
    print("\nEvaluating Baseline (fixed tau=0.75 without clustering)...")
    preds_base = {}
    for sid in s1_map:
        pairs = scored_pairs.get(sid, [])
        preds_base[sid] = set([cid for (cid, _, _), p in pairs if p >= 0.75][:10])
    f05_b, p_b, r_b = compute_macro_f05(gt_map, preds_base)
    print(f"Baseline: Macro F0.5 = {f05_b:.6f} | Prec: {p_b*100:.2f}% | Rec: {r_b*100:.2f}%")

    print("\nEvaluating 2-Stage Core-Stem Clustering across parameter grid:")
    for min_p in [0.75, 0.80]:
        for delta_p in [0.05, 0.08, 0.10]:
            for stem_sim_cut in [35, 40, 45, 50]:
                preds_2stage = {}
                total_m = 0
                for sid in s1_map:
                    pairs = scored_pairs.get(sid, [])
                    if not pairs or pairs[0][1] < min_p:
                        preds_2stage[sid] = set()
                        continue

                    p1 = pairs[0][1]
                    top_stem = get_core_stem(pairs[0][0][1])
                    accepted = [pairs[0][0][0]]

                    for (cid, cname, caddr), p in pairs[1:]:
                        if p < min_p or (p1 - p) > delta_p:
                            continue
                        c_stem = get_core_stem(cname)
                        if top_stem and c_stem:
                            sim = fuzz.token_sort_ratio(top_stem, c_stem)
                            if sim < stem_sim_cut:
                                continue
                        accepted.append(cid)
                        if len(accepted) >= 6:
                            break

                    preds_2stage[sid] = set(accepted)
                    total_m += len(accepted)

                f05, p, r = compute_macro_f05(gt_map, preds_2stage)
                avg_m = total_m / len(s1_map)
                print(f"min_p={min_p:.2f}, delta_p={delta_p:.2f}, stem_cut={stem_sim_cut} -> F0.5 = {f05:.6f} | Prec: {p*100:.2f}% | Rec: {r*100:.2f}% | AvgM: {avg_m:.2f}", flush=True)

if __name__ == "__main__":
    main()
