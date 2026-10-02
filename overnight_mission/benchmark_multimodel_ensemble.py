"""Multi-Model Ensemble Benchmark for Peak Macro F0.5.

Trains and evaluates:
1. LightGBM Champion
2. XGBoost Classifier
3. CatBoost Classifier
4. Average Probability Ensemble (LGBM + XGB + CatBoost)
5. Weighted Priority Ensemble (Rank-based stacking)
All evaluated with 2-Set Disjoint Bipartite Resolution on held-out ground truth.
"""

import os
import sys
import gc
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.blocking import BlockingIndex
from src.evaluate import compute_macro_f05
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df

def main():
    print("=" * 95)
    print("MULTI-MODEL ENSEMBLE BENCHMARK (LightGBM vs XGBoost vs CatBoost vs Ensembles)")
    print("=" * 95)

    # 1. Models directory
    models_dir = "overnight_mission/models"
    os.makedirs(models_dir, exist_ok=True)

    lgb_path = os.path.join(models_dir, "champion_step3_model.pkl")
    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
    cat_path = os.path.join(models_dir, "catboost_model.pkl")

    with open(lgb_path, "rb") as f:
        model_lgb = pickle.load(f)

    # 2. Sample 1,500 validation entities with realistic candidate distractors
    print("Sampling 1,500 validation entities...", flush=True)
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

    # Retrieve and extract features
    print("Extracting features for validation pairs...", flush=True)
    all_pairs_info = [] # (sid, cid)
    feat_matrix = []
    labels = []

    for sid, r1 in s1_map.items():
        ret = indexer.retrieve_candidates_for_entity(r1, top_k=20)
        true_m = gt_map.get(sid, set())
        for cid, _ in ret:
            rc = c_map.get(cid)
            if rc:
                feats = extract_pair_features(r1, rc)
                feat_matrix.append([feats[col] for col in FEATURE_COLUMNS])
                all_pairs_info.append((sid, cid))
                labels.append(1 if cid in true_m else 0)

    X_val = np.array(feat_matrix, dtype=np.float32)
    y_val = np.array(labels, dtype=np.int32)
    print(f"Validation dataset: {len(X_val):,} pairs (Positives: {np.sum(y_val):,}, Negatives: {len(y_val) - np.sum(y_val):,})", flush=True)

    # 3. Train or Load XGBoost & CatBoost
    # Create training set by taking 3,000 distinct training entities
    print("Preparing training data for XGBoost and CatBoost...", flush=True)
    df_train_ids = pd.read_parquet("overnight_mission/eval/train_split_ids.parquet")
    matched_train_ids = df_train_ids[df_train_ids["match_count"] > 0]
    train_sample_sids = set(matched_train_ids["entity_id"].head(2000))

    tr_s1_rows = []
    for chunk in pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(train_sample_sids)]
        if len(sub) > 0:
            tr_s1_rows.append(sub)
    df_tr_s1 = apply_normalization_df(pd.concat(tr_s1_rows, ignore_index=True))
    tr_s1_map = {r["entity_id"]: r for r in df_tr_s1.to_dict("records")}

    tr_gt_map = {}
    tr_needed_cands = set()
    for _, r in df_gt[df_gt["source1_entity_id"].isin(train_sample_sids)].iterrows():
        sid = r["source1_entity_id"]
        m = [x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()] if r["matched_entity_ids"] else []
        tr_gt_map[sid] = set(m)
        tr_needed_cands.update(m)

    tr_c_rows = []
    tr_dist_count = 0
    for p in ["student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv"]:
        for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
            sub_true = chunk[chunk["entity_id"].isin(tr_needed_cands)]
            if len(sub_true) > 0:
                tr_c_rows.append(sub_true)
            if tr_dist_count < 40000:
                tr_c_rows.append(chunk.head(10000))
                tr_dist_count += len(chunk.head(10000))

    df_tr_cands = apply_normalization_df(pd.concat(tr_c_rows, ignore_index=True).drop_duplicates("entity_id"))
    tr_c_map = {r["entity_id"]: r for r in df_tr_cands.to_dict("records")}

    tr_indexer = BlockingIndex()
    tr_indexer.build(df_tr_cands)

    X_train_list = []
    y_train_list = []
    for sid, r1 in tr_s1_map.items():
        ret = tr_indexer.retrieve_candidates_for_entity(r1, top_k=20)
        true_m = tr_gt_map.get(sid, set())
        for cid, _ in ret:
            rc = tr_c_map.get(cid)
            if rc:
                feats = extract_pair_features(r1, rc)
                X_train_list.append([feats[col] for col in FEATURE_COLUMNS])
                y_train_list.append(1 if cid in true_m else 0)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    print(f"Training dataset: {len(X_train):,} pairs (Positives: {np.sum(y_train):,})", flush=True)

    # Train XGBoost
    print("Training XGBoost Classifier...", flush=True)
    t0 = time.time()
    model_xgb = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        eval_metric="logloss"
    )
    model_xgb.fit(X_train, y_train)
    print(f"XGBoost trained in {time.time() - t0:.1f}s.", flush=True)

    # Train CatBoost
    print("Training CatBoost Classifier...", flush=True)
    t0 = time.time()
    model_cat = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.08,
        random_seed=42,
        verbose=0
    )
    model_cat.fit(X_train, y_train)
    print(f"CatBoost trained in {time.time() - t0:.1f}s.", flush=True)

    # 4. Predict probabilities with each model
    print("\nComputing predictions for all models on validation set...", flush=True)
    p_lgb = model_lgb.predict_proba(X_val)[:, 1]
    p_xgb = model_xgb.predict_proba(X_val)[:, 1]
    p_cat = model_cat.predict_proba(X_val)[:, 1]

    # Ensembles
    # Simple average
    p_ens_avg = (p_lgb + p_xgb + p_cat) / 3.0
    # Weighted average (favoring LightGBM + CatBoost)
    p_ens_weighted = 0.50 * p_lgb + 0.30 * p_cat + 0.20 * p_xgb
    # Rank / Maximum agreement
    p_ens_max = np.maximum(np.maximum(p_lgb, p_xgb), p_cat)
    # High-precision consensus (harmonic mean / min)
    p_ens_min = np.minimum(np.minimum(p_lgb, p_xgb), p_cat)

    candidate_models = [
        ("Model 1: LightGBM Champion", p_lgb),
        ("Model 2: XGBoost Classifier", p_xgb),
        ("Model 3: CatBoost Classifier", p_cat),
        ("Model 4: Equal Average Ensemble (LGB+XGB+Cat)", p_ens_avg),
        ("Model 5: Weighted Ensemble (50% LGB + 30% Cat + 20% XGB)", p_ens_weighted),
        ("Model 6: Optimistic Consensus Ensemble (Max Prob)", p_ens_max),
        ("Model 7: Strict Consensus Ensemble (Min Prob)", p_ens_min),
    ]

    # 5. Evaluate all models with 2-Set Disjoint Bipartite Resolution
    print("\n" + "=" * 105)
    print(f"{'Model / Ensemble Architecture':<55} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Avg Matches'}")
    print("=" * 105)

    best_f05 = 0.0
    best_name = ""

    for name, probs in candidate_models:
        # Group into pairs per entity
        s1_to_pairs = {sid: [] for sid in s1_map}
        for (sid, cid), prob in zip(all_pairs_info, probs):
            s1_to_pairs[sid].append((cid, float(prob)))

        # Sort descending
        for sid in s1_to_pairs:
            s1_to_pairs[sid].sort(key=lambda x: x[1], reverse=True)

        # Apply 2-Set Disjoint Bipartite Resolution at optimal tau = 0.78
        all_edges = []
        for sid, pairs in s1_to_pairs.items():
            for cid, p in pairs:
                if p >= 0.78:
                    all_edges.append((sid, cid, p))
        all_edges.sort(key=lambda x: x[2], reverse=True)

        claimed_s2 = set()
        s1_matches = {sid: [] for sid in s1_map}
        for sid, cid, p in all_edges:
            if cid not in claimed_s2 and len(s1_matches[sid]) < 6:
                s1_matches[sid].append(cid)
                claimed_s2.add(cid)

        preds = {sid: set(m) for sid, m in s1_matches.items()}
        f05, prec, rec = compute_macro_f05(gt_map, preds)
        avg_m = sum(len(m) for m in preds.values()) / len(s1_map)

        star = ""
        if f05 > best_f05:
            best_f05 = f05
            best_name = name
            star = " *** HIGHEST ***"

        print(f"{name:<55} | {f05:<12.6f} | {prec*100:>8.2f}% | {rec*100:>8.2f}% | {avg_m:>10.2f}{star}", flush=True)

    print("=" * 105, flush=True)
    print(f"CHAMPION MODEL ARCHITECTURE: {best_name}")
    print(f"HIGHEST ACHIEVED MACRO F0.5: {best_f05:.6f}")
    print("=" * 95, flush=True)

if __name__ == "__main__":
    main()
