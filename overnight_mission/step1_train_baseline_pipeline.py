"""Step 1: Build the Corrected Baseline Pipeline (Part B fixes included from the start).

1. Builds clean training pair dataset sampled strictly from the 80% Train Split.
2. Extracts all 49 clean features (including transliteration, missing address flags, street tokens).
3. Evaluates 5-Fold GroupKFold CV (strictly grouped by source1_entity_id, zero leakage).
4. Fits final LightGBM model and saves to overnight_mission/models/lgbm_step1_baseline.pkl.
5. Evaluates on the held-out benchmark split (overnight_mission/eval/val_10k_benchmark.tsv).
6. Reports raw empirical Precision, Recall, Macro F0.5 per country and overall.
"""

import gc
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.blocking import BlockingIndex
from src.evaluate import compute_macro_f05
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df

def main():
    print("=" * 80, flush=True)
    print("STEP 1: BUILDING AND TRAINING CORRECTED BASELINE PIPELINE (PART B INCLUDED)", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    # 1. Load Step 0 Train Split IDs
    train_split_path = "overnight_mission/eval/train_split_ids.parquet"
    print(f"Loading train split IDs from {train_split_path} ...", flush=True)
    df_train_split = pd.read_parquet(train_split_path)
    train_s1_pool = set(df_train_split["entity_id"])
    print(f"Available training S1 pool: {len(train_s1_pool):,} entities.", flush=True)

    # 2. Load ground truth
    gt_path = "student_resource/dataset/train/train_ground_truth.tsv"
    print(f"Loading ground truth from {gt_path} ...", flush=True)
    df_gt = pd.read_csv(gt_path, sep="\t", keep_default_na=False)
    gt_map = {}
    for _, r in df_gt.iterrows():
        sid = r["source1_entity_id"].strip()
        m_raw = r["matched_entity_ids"].strip()
        m_set = {x.strip() for x in m_raw.split(",") if x.strip()} if m_raw else set()
        gt_map[sid] = m_set

    # 3. Select balanced sample of 6,000 S1 training entities (strictly from train split)
    np.random.seed(42)
    s1_train_selected = []
    for strata, grp in df_train_split.groupby(["country", "match_count"]):
        sub_pool = list(grp["entity_id"])
        n_sample = max(1, int(len(sub_pool) / len(df_train_split) * 6000))
        shuffled = np.random.choice(sub_pool, size=min(n_sample, len(sub_pool)), replace=False)
        s1_train_selected.extend(shuffled)

    s1_train_set = set(s1_train_selected[:6000])
    print(f"Selected {len(s1_train_set):,} S1 training entities strictly from train split.", flush=True)

    # Check zero overlap with held-out split
    df_held = pd.read_parquet("overnight_mission/eval/held_out_split_ids.parquet")
    overlap_check = s1_train_set & set(df_held["entity_id"])
    assert len(overlap_check) == 0, f"FATAL LEAKAGE: {len(overlap_check)} training entities are in held-out split!"
    print(f"Leakage pre-flight check PASSED: 0 entities overlap with held-out split.", flush=True)

    # 4. Load S1 records for training
    print("Loading Source 1 records...", flush=True)
    s1_rows = []
    for chunk in pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(s1_train_set)]
        if len(sub) > 0:
            s1_rows.append(sub)
        if sum(len(x) for x in s1_rows) >= len(s1_train_set):
            break
    df_s1_train = apply_normalization_df(pd.concat(s1_rows, ignore_index=True))
    s1_dict = {r["entity_id"]: r for r in df_s1_train.to_dict("records")}

    # 5. Build candidate pool for training (all true positive matches + retrieved hard negative candidates)
    needed_positives = set()
    for sid in s1_train_set:
        needed_positives.update(gt_map.get(sid, set()))

    print(f"Total true positive candidates needed: {len(needed_positives):,}", flush=True)
    cand_rows = []
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("student_resource/dataset/train", src)
        for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
            sub = chunk[chunk["entity_id"].isin(needed_positives)]
            cand_rows.append(sub)
            if len(cand_rows) <= 2:
                # Add distractor negatives from first chunk
                cand_rows.append(chunk.head(15000))
            if sum(len(x[x["entity_id"].isin(needed_positives)]) for x in cand_rows) >= len(needed_positives):
                break

    df_cands_train = apply_normalization_df(pd.concat(cand_rows, ignore_index=True).drop_duplicates("entity_id"))
    cand_dict = {r["entity_id"]: r for r in df_cands_train.to_dict("records")}
    print(f"Candidate pool built: {len(cand_dict):,} records.", flush=True)

    # Build BlockingIndex on candidate pool
    print("Building BlockingIndex for candidate retrieval...", flush=True)
    indexer = BlockingIndex()
    indexer.build(df_cands_train)

    # 6. Extract features and build training matrix
    print("Extracting pairwise features for training pairs...", flush=True)
    pair_rows = []
    labels = []
    groups = []

    for sid, r1 in s1_dict.items():
        true_m = gt_map.get(sid, set())
        # Retrieve candidates via blocking
        ret = indexer.retrieve_candidates_for_entity(r1, top_k=20)
        c_cand_ids = [c[0] for c in ret]

        # Always include true positive matches even if missed by blocking to ensure positive representation
        candidate_ids = list(dict.fromkeys(list(true_m) + c_cand_ids))

        for cid in candidate_ids:
            rc = cand_dict.get(cid)
            if not rc:
                continue
            feats = extract_pair_features(r1, rc)
            pair_rows.append([feats[col] for col in FEATURE_COLUMNS])
            label = 1 if cid in true_m else 0
            labels.append(label)
            groups.append(sid)

    X = np.array(pair_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    groups = np.array(groups)

    print(f"Training dataset ready: {len(X):,} pairs | Positives: {y.sum():,} ({y.mean()*100:.2f}%) | Groups: {len(np.unique(groups)):,}")

    # 7. Run 5-Fold Stratified Group CV (zero leakage)
    print("\n--- Running 5-Fold GroupKFold Cross-Validation ---", flush=True)
    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    cv_precisions = []
    cv_recalls = []
    cv_f05s = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_tr, y_tr)
        val_probs = clf.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = val_probs

        pred_binary = (val_probs >= 0.50).astype(int)
        tp = np.logical_and(pred_binary == 1, y_va == 1).sum()
        prec = tp / max(1, (pred_binary == 1).sum())
        rec = tp / max(1, (y_va == 1).sum())
        denom = 0.25 * prec + rec
        f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0

        cv_precisions.append(prec)
        cv_recalls.append(rec)
        cv_f05s.append(f05)
        print(f"  Fold {fold+1}: Precision={prec*100:.2f}%, Recall={rec*100:.2f}%, F0.5={f05*100:.2f}%")

    print(f"5-Fold CV Mean: Precision={np.mean(cv_precisions)*100:.2f}%, Recall={np.mean(cv_recalls)*100:.2f}%, F0.5={np.mean(cv_f05s)*100:.2f}%")

    # 8. Train Final Baseline Model on full training split pairs
    print("\nFitting final Baseline LightGBM model on all training pairs...", flush=True)
    final_clf = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    final_clf.fit(X, y)

    # Save model
    model_save_path = "overnight_mission/models/lgbm_step1_baseline.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(final_clf, f)
    print(f"Saved baseline model to {model_save_path} ({os.path.getsize(model_save_path):,} bytes).")

    # Feature Importance Report
    importances = final_clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 15 Feature Importances:")
    for rank, idx in enumerate(sorted_idx[:15], 1):
        print(f"  {rank:>2}. {FEATURE_COLUMNS[idx]:<32} : {importances[idx]:>6d}")

    # 9. Evaluate on Held-out Benchmark Split (val_10k_benchmark.tsv)
    print("\n" + "=" * 80)
    print("STEP 1: HELD-OUT BENCHMARK EVALUATION (10,000 UNSEEN S1 ENTITIES)")
    print("=" * 80, flush=True)

    val_bench_path = "overnight_mission/eval/val_10k_benchmark.tsv"
    df_val_bench = pd.read_csv(val_bench_path, sep="\t", keep_default_na=False)
    val_s1_set = set(df_val_bench["entity_id"])
    val_gt_map = {sid: gt_map.get(sid, set()) for sid in val_s1_set}

    # Load S1 records for benchmark
    val_s1_rows = []
    for chunk in pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(val_s1_set)]
        if len(sub) > 0:
            val_s1_rows.append(sub)
        if sum(len(x) for x in val_s1_rows) >= len(val_s1_set):
            break
    df_s1_val = apply_normalization_df(pd.concat(val_s1_rows, ignore_index=True))
    val_s1_dict = {r["entity_id"]: r for r in df_s1_val.to_dict("records")}

    # Build candidate pool for benchmark
    needed_val_cands = set().union(*val_gt_map.values())
    val_cand_rows = []
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("student_resource/dataset/train", src)
        for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
            sub = chunk[chunk["entity_id"].isin(needed_val_cands)]
            val_cand_rows.append(sub)
            if len(val_cand_rows) <= 2:
                val_cand_rows.append(chunk.head(15000))
            if sum(len(x[x["entity_id"].isin(needed_val_cands)]) for x in val_cand_rows) >= len(needed_val_cands):
                break

    df_val_cands = apply_normalization_df(pd.concat(val_cand_rows, ignore_index=True).drop_duplicates("entity_id"))
    val_cand_dict = {r["entity_id"]: r for r in df_val_cands.to_dict("records")}

    val_indexer = BlockingIndex()
    val_indexer.build(df_val_cands)

    # Candidate retrieval and scoring
    scored_val_pairs = {}
    for sid, r1 in val_s1_dict.items():
        ret = val_indexer.retrieve_candidates_for_entity(r1, top_k=15)
        feat_matrix = []
        c_ids = []
        for cid, _ in ret:
            rc = val_cand_dict.get(cid)
            if rc:
                feats = extract_pair_features(r1, rc)
                feat_matrix.append([feats[col] for col in FEATURE_COLUMNS])
                c_ids.append(cid)
        if feat_matrix:
            probs = final_clf.predict_proba(np.array(feat_matrix, dtype=np.float32))[:, 1]
            scored_val_pairs[sid] = list(zip(c_ids, probs))
        else:
            scored_val_pairs[sid] = []

    # Evaluate by country on held-out split
    s1_by_country = {}
    for r in df_s1_val.to_dict("records"):
        s1_by_country.setdefault(r["country"], []).append(r["entity_id"])

    print("\nHeld-Out Results Across Decision Thresholds (tau in [0.50, 0.60, 0.70, 0.75, 0.85]):")
    for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        preds_all = {}
        for sid in val_s1_set:
            pairs = scored_val_pairs.get(sid, [])
            m = [cid for cid, p in pairs if p >= tau][:10]
            preds_all[sid] = set(m)

        f05_all, prec_all, rec_all = compute_macro_f05(val_gt_map, preds_all)
        sing_pct = sum(1 for v in preds_all.values() if len(v) == 0) / len(preds_all) * 100

        # Per country breakdown
        country_reports = []
        for c in ["US", "India"]:
            c_sids = set(s1_by_country.get(c, []))
            c_gt = {sid: val_gt_map[sid] for sid in c_sids}
            c_pred = {sid: preds_all[sid] for sid in c_sids}
            c_f05, c_p, c_r = compute_macro_f05(c_gt, c_pred)
            country_reports.append(f"{c}: F0.5={c_f05:.4f} (P={c_p:.3f}, R={c_r:.3f})")

        print(f"tau={tau:.2f} -> OVERALL Macro F0.5: {f05_all:.4f} | Prec: {prec_all:.4f} | Rec: {rec_all:.4f} | Sing: {sing_pct:.1f}% | {' | '.join(country_reports)}")

    print(f"\nSTEP 1 COMPLETE in {time.time() - t0:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
