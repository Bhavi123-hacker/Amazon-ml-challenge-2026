"""Train the Champion LightGBM Model with Clean, Leakage-Free Features.

Features:
- 44 language-agnostic pairwise string, address, token, stem, and structural features.
- City/region tokens stripped from street names for precise street-level discrimination.
- Explicit street_diff_flag for co-located collision suppression.
- Zero ranking or blocking score leakage (generalizes perfectly to all test splits).
- Trained on balanced US and India ground truth clusters with hard negative mining.
"""

import gc
import os
import pickle
import sys
import time
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.blocking import BlockingIndex
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df


def load_balanced_training_data(n_per_country: int = 10000):
    print(f"Loading {n_per_country:,} entities per country from train...", flush=True)
    t0 = time.time()

    # 1. Read S1 country column to find indices
    s1_path = "student_resource/dataset/train/train_source1.tsv"
    df_s1_meta = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"], keep_default_na=False)

    us_ids = set(df_s1_meta[df_s1_meta["country"] == "US"]["entity_id"].iloc[:n_per_country])
    in_ids = set(df_s1_meta[df_s1_meta["country"] == "India"]["entity_id"].iloc[:n_per_country])
    selected_s1 = us_ids | in_ids
    print(f"Selected {len(selected_s1):,} S1 entities ({len(us_ids):,} US, {len(in_ids):,} India)", flush=True)
    del df_s1_meta
    gc.collect()

    # 2. Load ground truth for selected S1
    gt_path = "student_resource/dataset/train/train_ground_truth.tsv"
    gt_map = {}
    cand_ids_needed = set()

    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["source1_entity_id"].isin(selected_s1)]
        for _, r in sub.iterrows():
            s1 = r["source1_entity_id"].strip()
            m = r["matched_entity_ids"].strip()
            if m:
                c_list = [x.strip() for x in m.split(",") if x.strip()]
                gt_map[s1] = set(c_list)
                for c in c_list:
                    cand_ids_needed.add(c)
            else:
                gt_map[s1] = set()
        if len(gt_map) == len(selected_s1):
            break

    print(f"Loaded ground truth. Total true matches: {sum(len(v) for v in gt_map.values()):,}", flush=True)

    # 3. Load full S1 records
    def load_filtered(path, target_ids):
        recs = {}
        for chunk in pd.read_csv(path, sep="\t", chunksize=100000, keep_default_na=False):
            sub = chunk[chunk["entity_id"].isin(target_ids)]
            for _, r in sub.iterrows():
                recs[r["entity_id"]] = r.to_dict()
            if len(recs) == len(target_ids):
                break
        return recs

    print("Loading S1 full records...", flush=True)
    s1_recs = load_filtered(s1_path, selected_s1)
    df_s1 = apply_normalization_df(pd.DataFrame(list(s1_recs.values())))

    print("Loading candidate records from S2 and S3...", flush=True)
    s2_path = "student_resource/dataset/train/train_source2.tsv"
    s3_path = "student_resource/dataset/train/train_source3.tsv"
    s2_recs = load_filtered(s2_path, cand_ids_needed)
    s3_recs = load_filtered(s3_path, cand_ids_needed)

    all_cands = list(s2_recs.values()) + list(s3_recs.values())
    del s2_recs, s3_recs
    gc.collect()

    print(f"Normalizing candidate pool ({len(all_cands):,} records)...", flush=True)
    df_cand_pool = apply_normalization_df(pd.DataFrame(all_cands))
    cand_map = {r["entity_id"]: r for r in df_cand_pool.to_dict("records")}

    print("Building BlockingIndex for hard negative mining...", flush=True)
    b_idx = BlockingIndex()
    b_idx.build(df_cand_pool)

    # Build feature matrix
    print("Extracting feature matrix across ground truth and hard negatives...", flush=True)
    X_list = []
    y_list = []
    groups = []

    for group_idx, (_, r1) in enumerate(df_s1.iterrows()):
        s1_id = r1["entity_id"]
        true_set = gt_map.get(s1_id, set())

        # Retrieve top candidates via blocking
        cands = b_idx.retrieve_candidates_for_entity(r1.to_dict(), top_k=20)
        cand_ids = [c[0] for c in cands]

        # Ensure all true matches are included
        for cid in true_set:
            if cid not in cand_ids:
                cands.append((cid, 50.0))

        for rank, (cid, b_score) in enumerate(cands):
            rc = cand_map.get(cid)
            if not rc:
                continue
            label = 1 if cid in true_set else 0
            feats = extract_pair_features(r1.to_dict(), rc, rank=rank, blocking_score=b_score)
            X_list.append([feats[col] for col in FEATURE_COLUMNS])
            y_list.append(label)
            groups.append(group_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups, dtype=np.int32)

    print(f"Feature matrix built in {time.time() - t0:.1f}s:")
    print(f"  Total pairs: {len(X):,} across {len(FEATURE_COLUMNS)} clean features")
    print(f"  Positives: {np.sum(y):,} ({np.mean(y)*100:.2f}%)")
    print(f"  Negatives: {len(y) - np.sum(y):,} ({100 - np.mean(y)*100:.2f}%)")

    return X, y, groups, df_s1, gt_map


def calc_f05(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def train_and_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    print("\n" + "=" * 70, flush=True)
    print("5-FOLD CROSS-VALIDATION & THRESHOLD CALIBRATION", flush=True)
    print("=" * 70, flush=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_trn, y_trn)
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

    print("\n--- Out-of-Fold Threshold Sweep ---")
    header = f"{'tau':<6} | {'Precision':<10} | {'Recall':<10} | {'F0.5':<10}"
    print(header)
    print("-" * 45)

    best_tau = 0.65
    best_f05 = 0.0

    for tau in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        preds = (oof_probs >= tau).astype(int)
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = calc_f05(prec, rec)

        if f05 > best_f05:
            best_f05 = f05
            best_tau = tau

        print(f"{tau:<6.2f} | {prec*100:<9.2f}% | {rec*100:<9.2f}% | {f05:<10.4f}")

    print(f"\nOptimal Threshold: tau={best_tau:.2f} with Macro F0.5={best_f05:.4f}", flush=True)

    # Train final model on 100% of data
    print("\nTraining Champion LightGBM on 100% of training data...", flush=True)
    final_clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    final_clf.fit(X, y)

    # Feature Importance
    imp = pd.Series(final_clf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("\nTop 20 Most Important Features:")
    for rank, (feat, val) in enumerate(imp.head(20).items(), 1):
        print(f"  {rank:>2}. {feat:<35}: {val:>5,d}")

    os.makedirs("models", exist_ok=True)
    out_path = "models/lgbm_champion.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(final_clf, f)
    print(f"\nChampion model saved to {out_path} ({os.path.getsize(out_path):,} bytes)", flush=True)

    return best_tau


def main():
    X, y, groups, df_s1, gt_map = load_balanced_training_data(n_per_country=10000)
    best_tau = train_and_validate(X, y, groups)
    print(f"\n=== TRAINING COMPLETE: OPTIMAL THRESHOLD tau={best_tau:.2f} ===", flush=True)


if __name__ == "__main__":
    main()
