"""Error Analysis & Upper Bound Diagnostic for Macro F0.5 = 0.99."""

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
    print("ERROR DECOMPOSITION & THE ROADMAP TO 0.99 MACRO F0.5")
    print("=" * 80)

    model_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 1. Sample 1,000 entities from held-out split
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
    total_true_matches = 0
    for _, r in df_gt[df_gt["source1_entity_id"].isin(sample_sids)].iterrows():
        sid = r["source1_entity_id"]
        m = [x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()] if r["matched_entity_ids"] else []
        gt_map[sid] = set(m)
        needed_cands.update(m)
        total_true_matches += len(m)

    # Load candidate pool
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

    # Build Index
    indexer = BlockingIndex()
    indexer.build(df_cands)

    # 1. Evaluate Candidate Retrieval Upper Bound (Oracle Recall)
    retrieved_cands = {}
    retrieved_true_matches = 0
    for sid, r1 in s1_map.items():
        ret = indexer.retrieve_candidates_for_entity(r1, top_k=20)
        c_set = set([cid for cid, _ in ret])
        retrieved_cands[sid] = c_set
        true_m = gt_map.get(sid, set())
        retrieved_true_matches += len(true_m.intersection(c_set))

    retrieval_recall = retrieved_true_matches / max(1, total_true_matches)
    print(f"Total True Ground Truth Matches : {total_true_matches:,}")
    print(f"True Matches Retrieved in Top 20: {retrieved_true_matches:,} ({retrieval_recall*100:.2f}%)")
    print(f"Retrieval Recall Ceiling (Max Possible Recall): {retrieval_recall*100:.2f}%")

    # Oracle F0.5 if classifier was 100% precision on retrieved candidates:
    oracle_preds = {sid: gt_map[sid].intersection(retrieved_cands[sid]) for sid in s1_map}
    oracle_f05, oracle_p, oracle_r = compute_macro_f05(gt_map, oracle_preds)
    print(f"ORACLE CEILING (Perfect Classifier on Retrieved Candidates):")
    print(f"  Macro F0.5 : {oracle_f05:.6f}")
    print(f"  Precision  : {oracle_p*100:.2f}%")
    print(f"  Recall     : {oracle_r*100:.2f}%\n")

if __name__ == "__main__":
    main()
