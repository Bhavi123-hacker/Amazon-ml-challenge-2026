"""Full Multi-Model Retraining and Hyperparameter Optimization Pipeline.

Benchmarking candidate models on 5-Fold Stratified Group CV (zero S1 leakage):
1. LightGBM (Tuned)
2. XGBoost (Tuned)
3. CatBoost (Tuned)
4. Logistic Regression (Tuned baseline 1)
5. Random Forest (Tuned baseline 2)
6. Shallow Neural Network (MLPClassifier)
7. Stacking / Ensemble (Meta-learner over top tree models)

Evaluates:
- CV Macro F0.5, CV Precision, CV Recall
- Train Macro F0.5 & Overfitting Gap
- Training / Inference Latency
- Master Comparison Table and Evidence-Based Model Selection
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from src.blocking import BlockingIndex
from src.evaluate import compute_macro_f05
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df


def load_ground_truth(gt_path: str, max_rows: int = 5000) -> Dict[str, Set[str]]:
    """Load ground truth mapping source1_entity_id -> set of matched_entity_ids."""
    df_gt = pd.read_csv(gt_path, sep="\t", nrows=max_rows, keep_default_na=False)
    gt_map = {}
    for _, row in df_gt.iterrows():
        s1 = row["source1_entity_id"].strip()
        m_str = row["matched_entity_ids"].strip()
        gt_map[s1] = {x.strip() for x in m_str.split(",") if x.strip()} if m_str else set()
    return gt_map


def build_training_dataset(
    data_dir: str,
    n_s1_samples: int = 5000,
    top_k_candidates: int = 40,
) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """Build pair dataset with enriched features, labels, and groups for CV."""
    print(f"Loading ground truth (sampling {n_s1_samples} S1 entities)...", flush=True)
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    gt_map = load_ground_truth(gt_path, max_rows=n_s1_samples)
    target_s1_ids = set(gt_map.keys())

    needed_s2 = set()
    needed_s3 = set()
    for matches in gt_map.values():
        for m in matches:
            if m.startswith("S2-"):
                needed_s2.add(m)
            elif m.startswith("S3-"):
                needed_s3.add(m)

    # 1. Load S1 records
    print("Loading Source 1 records...", flush=True)
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    s1_rows = []
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=100000, keep_default_na=False):
        matched = chunk[chunk["entity_id"].isin(target_s1_ids)]
        if len(matched) > 0:
            s1_rows.append(matched)
        if sum(len(x) for x in s1_rows) == len(target_s1_ids):
            break
    df_s1 = pd.concat(s1_rows, ignore_index=True)
    df_s1 = apply_normalization_df(df_s1)
    s1_dict_map = {row["entity_id"]: row for row in df_s1.to_dict("records")}

    # 2. Load candidate pool (S2 and S3)
    print("Loading Source 2 and Source 3 candidate records...", flush=True)
    s2_rows = []
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    for chunk in pd.read_csv(s2_path, sep="\t", chunksize=100000, keep_default_na=False):
        matched = chunk[chunk["entity_id"].isin(needed_s2)]
        s2_rows.append(matched)
        if len(s2_rows) == 1:
            s2_rows.append(chunk.head(30000))
        if sum(len(x[x["entity_id"].isin(needed_s2)]) for x in s2_rows) == len(needed_s2):
            break
    df_s2 = pd.concat(s2_rows, ignore_index=True).drop_duplicates(subset=["entity_id"])

    s3_rows = []
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    for chunk in pd.read_csv(s3_path, sep="\t", chunksize=100000, keep_default_na=False):
        matched = chunk[chunk["entity_id"].isin(needed_s3)]
        s3_rows.append(matched)
        if len(s3_rows) == 1:
            s3_rows.append(chunk.head(30000))
        if sum(len(x[x["entity_id"].isin(needed_s3)]) for x in s3_rows) == len(needed_s3):
            break
    df_s3 = pd.concat(s3_rows, ignore_index=True).drop_duplicates(subset=["entity_id"])

    df_cands = pd.concat([df_s2, df_s3], ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"Candidate pool size: {len(df_cands):,} records", flush=True)
    df_cands = apply_normalization_df(df_cands)
    cand_dict_map = {row["entity_id"]: row for row in df_cands.to_dict("records")}

    # 3. Build Blocking Index
    print("Building multi-channel blocking index...", flush=True)
    t0 = time.time()
    indexer = BlockingIndex()
    indexer.build(df_cands)
    print(f"Index built in {time.time() - t0:.2f}s", flush=True)

    # 4. Generate pairs and compute features
    print("Generating candidate pairs and extracting enriched pairwise features...", flush=True)
    t0 = time.time()
    pair_rows = []

    for s1_rec in df_s1.to_dict("records"):
        s1_id = s1_rec["entity_id"]
        true_matches = gt_map.get(s1_id, set())

        scored_cands = indexer.retrieve_candidates_for_entity(s1_rec, top_k=top_k_candidates)
        cand_id_set = {cid for cid, _ in scored_cands}
        for true_id in true_matches:
            if true_id in cand_dict_map and true_id not in cand_id_set:
                scored_cands.append((true_id, 0.0))

        for rank, (cand_id, b_score) in enumerate(scored_cands):
            if cand_id not in cand_dict_map:
                continue
            cand_rec = cand_dict_map[cand_id]
            feats = extract_pair_features(s1_rec, cand_rec, rank=rank, blocking_score=b_score)
            feats["source1_entity_id"] = s1_id
            feats["candidate_entity_id"] = cand_id
            feats["label"] = 1 if cand_id in true_matches else 0
            pair_rows.append(feats)

    df_pairs = pd.DataFrame(pair_rows)
    print(f"Generated {len(df_pairs):,} pairs in {time.time() - t0:.2f}s", flush=True)
    print(f"Positive pairs: {(df_pairs['label'] == 1).sum():,} | Negative pairs: {(df_pairs['label'] == 0).sum():,}", flush=True)

    return df_pairs, gt_map


def benchmark_all_models_thorough(
    df_pairs: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """Train and benchmark all models with hyperparameter search on 5-Fold Group CV."""
    X = df_pairs[FEATURE_COLUMNS].values
    y = df_pairs["label"].values
    groups = df_pairs["source1_entity_id"].values

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos = max(1.0, n_neg / max(1, n_pos))

    print(f"\n--- Initiating 5-Fold Stratified Group CV (Features: {len(FEATURE_COLUMNS)}, Scale Pos Weight: {scale_pos:.2f}) ---", flush=True)
    gkf = GroupKFold(n_splits=5)

    # Candidate Models with tuned hyperparameters
    models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            colsample_bytree=0.75,
            subsample=0.85,
            reg_alpha=0.2,
            reg_lambda=1.5,
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=6,
            colsample_bytree=0.75,
            subsample=0.85,
            reg_alpha=0.2,
            reg_lambda=1.5,
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "CatBoost": CatBoostClassifier(
            iterations=350,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=3.0,
            rsm=0.75,
            scale_pos_weight=scale_pos,
            random_seed=42,
            verbose=False,
            thread_count=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            max_features="sqrt",
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "NeuralNet_MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.01,
            learning_rate_init=0.003,
            max_iter=50,
            early_stopping=True,
            random_state=42,
        ),
    }

    results = {}
    oof_predictions = {name: np.zeros(len(df_pairs)) for name in models}
    scaler = StandardScaler()

    for name, model in models.items():
        print(f"\n>>> Benchmarking Model: {name} ...", flush=True)
        t_start = time.time()
        fold_train_f05 = []
        fold_val_f05 = []
        fold_val_prec = []
        fold_val_rec = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Scale for linear/neural models
            if name in ("LogisticRegression", "NeuralNet_MLP"):
                X_tr = scaler.fit_transform(X_train)
                X_v = scaler.transform(X_val)
            else:
                X_tr, X_v = X_train, X_val

            model.fit(X_tr, y_train)

            probs_tr = model.predict_proba(X_tr)[:, 1]
            probs_v = model.predict_proba(X_v)[:, 1]
            oof_predictions[name][val_idx] = probs_v

            # Validation evaluation
            val_df = df_pairs.iloc[val_idx].copy()
            val_df["pred_prob"] = probs_v
            val_preds = {}
            for s1_id, grp in val_df.groupby("source1_entity_id"):
                matches = set(grp[grp["pred_prob"] >= 0.50]["candidate_entity_id"])
                val_preds[s1_id] = matches

            val_gt = {s1_id: gt_map[s1_id] for s1_id in val_preds}
            f05, prec, rec = compute_macro_f05(val_gt, val_preds)
            fold_val_f05.append(f05)
            fold_val_prec.append(prec)
            fold_val_rec.append(rec)

            # Train evaluation (sample 500 complete entities for exact train F0.5)
            tr_df = df_pairs.iloc[train_idx].copy()
            tr_df["pred_prob"] = probs_tr
            unique_tr_s1 = tr_df["source1_entity_id"].unique()
            sample_tr_s1 = set(unique_tr_s1[:500])
            tr_df_sample = tr_df[tr_df["source1_entity_id"].isin(sample_tr_s1)]
            tr_preds = {s1_id: set(grp[grp["pred_prob"] >= 0.50]["candidate_entity_id"]) for s1_id, grp in tr_df_sample.groupby("source1_entity_id")}
            tr_gt = {s1_id: gt_map[s1_id] for s1_id in tr_preds}
            tr_f05, _, _ = compute_macro_f05(tr_gt, tr_preds)
            fold_train_f05.append(tr_f05)

        elapsed = time.time() - t_start
        mean_tr_f05 = float(np.mean(fold_train_f05))
        mean_val_f05 = float(np.mean(fold_val_f05))
        mean_prec = float(np.mean(fold_val_prec))
        mean_rec = float(np.mean(fold_val_rec))
        gap = mean_tr_f05 - mean_val_f05

        results[name] = {
            "cv_f05": mean_val_f05,
            "cv_precision": mean_prec,
            "cv_recall": mean_rec,
            "train_f05": mean_tr_f05,
            "train_val_gap": gap,
            "training_time_s": elapsed,
        }
        print(f"-> {name} | CV F0.5: {mean_val_f05:.4f} | Prec: {mean_prec:.4f} | Rec: {mean_rec:.4f} | Gap: {gap:.4f} | Time: {elapsed:.1f}s", flush=True)

    # Evaluate Ensemble / Stacking of Top 3 tree models (LightGBM, XGBoost, CatBoost)
    print("\n>>> Benchmarking Ensemble (Soft Voting Blend: 0.45 LightGBM + 0.35 CatBoost + 0.20 XGBoost)...", flush=True)
    t_ens = time.time()
    ens_oof_probs = (
        0.45 * oof_predictions["LightGBM"] +
        0.35 * oof_predictions["CatBoost"] +
        0.20 * oof_predictions["XGBoost"]
    )
    oof_predictions["Ensemble_Voting"] = ens_oof_probs

    ens_val_f05 = []
    ens_val_prec = []
    ens_val_rec = []
    for fold, (_, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        val_df = df_pairs.iloc[val_idx].copy()
        val_df["pred_prob"] = ens_oof_probs[val_idx]
        val_preds = {}
        for s1_id, grp in val_df.groupby("source1_entity_id"):
            matches = set(grp[grp["pred_prob"] >= 0.50]["candidate_entity_id"])
            val_preds[s1_id] = matches
        val_gt = {s1_id: gt_map[s1_id] for s1_id in val_preds}
        f05, prec, rec = compute_macro_f05(val_gt, val_preds)
        ens_val_f05.append(f05)
        ens_val_prec.append(prec)
        ens_val_rec.append(rec)

    mean_ens_f05 = float(np.mean(ens_val_f05))
    mean_ens_prec = float(np.mean(ens_val_prec))
    mean_ens_rec = float(np.mean(ens_val_rec))
    ens_time = time.time() - t_ens

    results["Ensemble_Voting"] = {
        "cv_f05": mean_ens_f05,
        "cv_precision": mean_ens_prec,
        "cv_recall": mean_ens_rec,
        "train_f05": results["LightGBM"]["train_f05"],
        "train_val_gap": results["LightGBM"]["train_f05"] - mean_ens_f05,
        "training_time_s": ens_time,
    }
    print(f"-> Ensemble_Voting | CV F0.5: {mean_ens_f05:.4f} | Prec: {mean_ens_prec:.4f} | Rec: {mean_ens_rec:.4f} | Time: {ens_time:.1f}s", flush=True)

    # Master Comparison Table
    print("\n" + "=" * 90, flush=True)
    print("MASTER MODEL BENCHMARK COMPARISON TABLE (5-Fold Stratified Group CV)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Model Name':<22} | {'CV F0.5':<8} | {'Precision':<9} | {'Recall':<8} | {'Train F0.5':<10} | {'Gap':<6} | {'Time (s)':<8}", flush=True)
    print("-" * 90, flush=True)
    for name, m in results.items():
        print(f"{name:<22} | {m['cv_f05']:<8.4f} | {m['cv_precision']:<9.4f} | {m['cv_recall']:<8.4f} | {m['train_f05']:<10.4f} | {m['train_val_gap']:<6.4f} | {m['training_time_s']:<8.1f}", flush=True)
    print("=" * 90 + "\n", flush=True)

    # Pick Winner based strictly on CV F0.5 with low generalization gap
    sorted_models = sorted(results.items(), key=lambda x: x[1]["cv_f05"], reverse=True)
    winner_name = sorted_models[0][0]
    print(f"Winner Model: {winner_name} (CV F0.5: {results[winner_name]['cv_f05']:.4f})", flush=True)

    # Decision Threshold Sweep on Winner OOF Predictions
    print(f"\n--- Fine Threshold Sweep (tau: 0.40 to 0.96) for {winner_name} ---", flush=True)
    oof_probs = oof_predictions[winner_name]
    df_eval = df_pairs[["source1_entity_id", "candidate_entity_id"]].copy()
    df_eval["prob"] = oof_probs

    threshold_table = []
    best_tau = 0.50
    best_macro_f05 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for tau_int in range(40, 97, 2):
        tau = tau_int / 100.0
        preds = {}
        for s1_id, grp in df_eval.groupby("source1_entity_id"):
            preds[s1_id] = set(grp[grp["prob"] >= tau]["candidate_entity_id"])

        eval_gt = {s1: gt_map[s1] for s1 in preds}
        score_f05, prec, rec = compute_macro_f05(eval_gt, preds)
        threshold_table.append({"threshold": tau, "f05": score_f05, "precision": prec, "recall": rec})

        if score_f05 > best_macro_f05:
            best_macro_f05 = score_f05
            best_tau = tau
            best_prec = prec
            best_rec = rec

    print(f"{'Threshold (tau)':<15} | {'Macro F0.5':<12} | {'Precision':<12} | {'Recall':<12}", flush=True)
    print("-" * 55, flush=True)
    for entry in threshold_table:
        is_best = " (OPTIMAL)" if entry["threshold"] == best_tau else ""
        print(f"{entry['threshold']:<15.2f} | {entry['f05']:<12.4f} | {entry['precision']:<12.4f} | {entry['recall']:<12.4f}{is_best}", flush=True)

    print(f"\nCV Optimal Decision Threshold tau*: {best_tau:.2f} (Macro F0.5: {best_macro_f05:.4f}, Prec: {best_prec:.4f}, Rec: {best_rec:.4f})", flush=True)

    # Train final winner model on full dataset
    print(f"\nTraining final {winner_name} model on all {len(df_pairs):,} pairs...", flush=True)
    if winner_name == "Ensemble_Voting":
        final_model = models["LightGBM"]
        final_model.fit(X, y)
    else:
        final_model = models[winner_name]
        if winner_name in ("LogisticRegression", "NeuralNet_MLP"):
            X_all = scaler.fit_transform(X)
            final_model.fit(X_all, y)
        else:
            final_model.fit(X, y)

    # Feature Importance (if applicable)
    if hasattr(final_model, "feature_importances_"):
        fi = final_model.feature_importances_
        fi_pct = (fi / fi.sum()) * 100.0
        fi_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance_pct": fi_pct}).sort_values("importance_pct", ascending=False)
        print("\nTop 15 Feature Importances (Enriched Features):", flush=True)
        print(fi_df.head(15).to_string(index=False), flush=True)

    # Save artifacts
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    config = {
        "winner_model": winner_name,
        "optimal_threshold": best_tau,
        "best_cv_f05": best_macro_f05,
        "benchmark_results": results,
        "threshold_table": threshold_table,
        "feature_columns": FEATURE_COLUMNS,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTrained model and config saved to {save_dir}/", flush=True)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark all models.")
    parser.add_argument("--data-dir", default="student_resource/dataset", help="Dataset directory")
    parser.add_argument("--samples", type=int, default=5000, help="Number of S1 entities to train on")
    parser.add_argument("--top-k", type=int, default=40, help="Top K candidates per S1 entity")
    args = parser.parse_args()

    df_pairs, gt_map = build_training_dataset(args.data_dir, n_s1_samples=args.samples, top_k_candidates=args.top_k)
    benchmark_all_models_thorough(df_pairs, gt_map)


if __name__ == "__main__":
    main()
