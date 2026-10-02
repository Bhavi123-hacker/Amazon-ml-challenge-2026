"""Fine-grained grid search over Dynamic Margin parameters (p1, delta_p)."""

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
    print("FINE-GRAINED GRID SEARCH OVER DYNAMIC MARGIN PARAMETERS")
    print("=" * 80)

    model_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load the same 2,000 entities
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

    # Grid search
    min_p_vals = [0.70, 0.75, 0.78, 0.80, 0.82, 0.85]
    delta_p_vals = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]

    best_grid_f05 = 0.0
    best_params = None

    print(f"{'min_p':<8} | {'delta_p':<8} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Avg Matches':<12} | {'Singletons'}")
    print("-" * 80)

    for min_p in min_p_vals:
        for delta_p in delta_p_vals:
            preds = {}
            total_m = 0
            sing = 0
            for sid in s1_map:
                pairs = scored_pairs.get(sid, [])
                if not pairs or pairs[0][1] < min_p:
                    preds[sid] = set()
                    sing += 1
                else:
                    p1 = pairs[0][1]
                    accepted = [cid for cid, p in pairs if p >= min_p and (p1 - p) <= delta_p][:10]
                    preds[sid] = set(accepted)
                    total_m += len(accepted)
                    if not accepted:
                        sing += 1

            f05, p, r = compute_macro_f05(gt_map, preds)
            avg_m = total_m / len(s1_map)
            sing_pct = sing / len(s1_map) * 100.0

            if f05 > best_grid_f05:
                best_grid_f05 = f05
                best_params = (min_p, delta_p)
                star = " *** NEW BEST ***"
            else:
                star = ""

            print(f"{min_p:<8.2f} | {delta_p:<8.2f} | {f05:<12.6f} | {p*100:>8.2f}% | {r*100:>8.2f}% | {avg_m:>10.2f}   | {sing_pct:>5.1f}%{star}", flush=True)

    print("\n" + "=" * 80)
    print(f"ABSOLUTE BEST PARAMS: min_p = {best_params[0]:.2f}, delta_p = {best_params[1]:.2f}")
    print(f"PEAK HELD-OUT MACRO F0.5: {best_grid_f05:.6f}")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
