"""Train the winning LightGBM model on 250k ground truth pairs and save it.
"""

import os, sys, time, pickle
import pandas as pd
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df
from src.blocking import BlockingIndex

def main():
    print("=== Training Champion LightGBM Model ===", flush=True)
    t0 = time.time()

    # 1. Load ground truth (15,000 entities = ~350,000 pairs)
    print("Loading 15,000 ground truth training entities...", flush=True)
    df_gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=15000, keep_default_na=False)
    gt_map = {}
    s1_set = set()
    cand_set = set()
    for _, r in df_gt.iterrows():
        s1 = r['source1_entity_id'].strip()
        m = r['matched_entity_ids'].strip()
        s1_set.add(s1)
        if m:
            cands = [x.strip() for x in m.split(',') if x.strip()]
            gt_map[s1] = set(cands)
            for c in cands: cand_set.add(c)
        else:
            gt_map[s1] = set()

    def load_recs(path, ids):
        res = {}
        for chunk in pd.read_csv(path, sep='\t', chunksize=50000, keep_default_na=False):
            sub = chunk[chunk['entity_id'].isin(ids)]
            for _, r in sub.iterrows():
                res[r['entity_id']] = r.to_dict()
            if len(res) == len(ids): break
        return res

    print("Loading source records...", flush=True)
    s1_recs = load_recs('student_resource/dataset/train/train_source1.tsv', s1_set)
    s2_recs = load_recs('student_resource/dataset/train/train_source2.tsv', cand_set)
    s3_recs = load_recs('student_resource/dataset/train/train_source3.tsv', cand_set)

    all_cand_list = list(s2_recs.values()) + list(s3_recs.values())
    df_cand_pool = apply_normalization_df(pd.DataFrame(all_cand_list))
    b_idx = BlockingIndex()
    b_idx.build(df_cand_pool)

    df_s1_sub = apply_normalization_df(pd.DataFrame(list(s1_recs.values())))

    print("Building feature matrix...", flush=True)
    X_list = []
    y_list = []

    for _, r1 in df_s1_sub.iterrows():
        s1 = r1['entity_id']
        true_set = gt_map.get(s1, set())
        cands = b_idx.retrieve_candidates_for_entity(r1.to_dict(), top_k=25)
        
        cand_ids = [c[0] for c in cands]
        for cid in true_set:
            if cid not in cand_ids:
                cands.append((cid, 50.0))
                
        for rank, (cid, b_score) in enumerate(cands):
            rc = s2_recs.get(cid) or s3_recs.get(cid)
            if not rc: continue
            label = 1 if cid in true_set else 0
            feats = extract_pair_features(r1.to_dict(), rc, rank=rank, blocking_score=b_score)
            X_list.append([feats[col] for col in FEATURE_COLUMNS])
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"Training LightGBM on {len(X):,} pairs ({np.mean(y)*100:.1f}% positive) across {len(FEATURE_COLUMNS)} features...", flush=True)
    clf = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X, y)

    os.makedirs("models", exist_ok=True)
    model_path = "models/lgbm_champion.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"Model trained and saved to {model_path} in {time.time() - t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
