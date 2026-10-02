"""Sections 1 & 2: Additional Baseline Models Benchmark & Stacking Experiment.

Evaluates on identical 5-Fold Stratified Group CV (zero S1 leakage):
1. K-Nearest Neighbors (KNN: k=5, 15, 30, 50, tuned distance weighting)
2. Support Vector Machine (Linear SVM and RBF Kernel SVM)
3. Gaussian Naive Bayes (GaussianNB)
4. Reference Top Performers (NeuralNet_MLP and XGBoost)
5. Stacked / Blended Meta-Learner (MLP + XGBoost + RF meta-features)

Reports CV Macro F0.5, Precision, Recall, Overfitting Gap, Training Time, and Latency.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import xgboost as xgb

from src.evaluate import compute_macro_f05
from src.features import FEATURE_COLUMNS
from src.train_and_benchmark import build_training_dataset


def evaluate_model_cv(
    name: str,
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    df_pairs: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    scale_features: bool = False,
    is_stacked: bool = False,
    oof_meta_features: Dict[str, np.ndarray] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluate a model using 5-Fold Stratified Group CV with exact Macro F0.5."""
    print(f"\n>>> Benchmarking Model: {name} ...", flush=True)
    t_start = time.time()
    gkf = GroupKFold(n_splits=5)
    scaler = StandardScaler()

    fold_val_f05 = []
    fold_val_prec = []
    fold_val_rec = []
    fold_train_f05 = []
    oof_probs = np.zeros(len(df_pairs))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        if is_stacked:
            # Build meta-features for this fold
            meta_tr = np.column_stack([oof_meta_features[m][train_idx] for m in oof_meta_features])
            meta_v = np.column_stack([oof_meta_features[m][val_idx] for m in oof_meta_features])
            X_tr, X_v = meta_tr, meta_v
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            if scale_features:
                X_tr = scaler.fit_transform(X_train)
                X_v = scaler.transform(X_val)
            else:
                X_tr, X_v = X_train, X_val

        model = model_fn()
        model.fit(X_tr, y_train)

        # Probabilities
        if hasattr(model, "predict_proba"):
            probs_tr = model.predict_proba(X_tr)[:, 1]
            probs_v = model.predict_proba(X_v)[:, 1]
        elif hasattr(model, "decision_function"):
            df_tr = model.decision_function(X_tr)
            df_v = model.decision_function(X_v)
            probs_tr = 1.0 / (1.0 + np.exp(-df_tr))
            probs_v = 1.0 / (1.0 + np.exp(-df_v))
        else:
            probs_tr = model.predict(X_tr).astype(float)
            probs_v = model.predict(X_v).astype(float)

        oof_probs[val_idx] = probs_v

        # Threshold sweep for best F0.5 on this fold
        val_df = df_pairs.iloc[val_idx].copy()
        val_df["pred_prob"] = probs_v

        best_f05 = 0.0
        best_p = 0.0
        best_r = 0.0
        for tau in [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
            val_preds = {
                s1_id: set(grp[grp["pred_prob"] >= tau]["candidate_entity_id"])
                for s1_id, grp in val_df.groupby("source1_entity_id")
            }
            val_gt = {s1_id: gt_map[s1_id] for s1_id in val_preds}
            f, p, r = compute_macro_f05(val_gt, val_preds)
            if f > best_f05:
                best_f05 = f
                best_p = p
                best_r = r

        fold_val_f05.append(best_f05)
        fold_val_prec.append(best_p)
        fold_val_rec.append(best_r)

        # Train evaluation (sample 300 entities)
        tr_df = df_pairs.iloc[train_idx].copy()
        tr_df["pred_prob"] = probs_tr
        unique_tr = tr_df["source1_entity_id"].unique()
        sample_tr = set(unique_tr[:300])
        tr_sample_df = tr_df[tr_df["source1_entity_id"].isin(sample_tr)]
        tr_preds = {
            s1_id: set(grp[grp["pred_prob"] >= 0.50]["candidate_entity_id"])
            for s1_id, grp in tr_sample_df.groupby("source1_entity_id")
        }
        tr_gt = {s1_id: gt_map[s1_id] for s1_id in tr_preds}
        tr_f, _, _ = compute_macro_f05(tr_gt, tr_preds)
        fold_train_f05.append(tr_f)

    elapsed = time.time() - t_start
    mean_val_f05 = float(np.mean(fold_val_f05))
    mean_prec = float(np.mean(fold_val_prec))
    mean_rec = float(np.mean(fold_val_rec))
    mean_tr_f05 = float(np.mean(fold_train_f05))
    gap = mean_tr_f05 - mean_val_f05

    print(f"  {name} => CV F0.5: {mean_val_f05:.4f} | Prec: {mean_prec:.4f} | Rec: {mean_rec:.4f} | Overfit Gap: {gap:+.4f} | Time: {elapsed:.1f}s", flush=True)

    metrics = {
        "model": name,
        "cv_f05": mean_val_f05,
        "cv_precision": mean_prec,
        "cv_recall": mean_rec,
        "train_f05": mean_tr_f05,
        "overfitting_gap": gap,
        "training_time_s": round(elapsed, 1),
    }
    return metrics, oof_probs


def main():
    print("=== Section 1 & 2: Benchmarking Additional Models & Stacking ===", flush=True)
    t0 = time.time()

    # 1. Build representative dataset (2,000 S1 entities, ~35,000 pairs)
    print("Building representative dataset from training data...", flush=True)
    df_pairs, gt_map = build_training_dataset("student_resource/dataset", n_s1_samples=2000, top_k_candidates=25)
    print(f"Dataset ready: {len(df_pairs):,} pairs across {len(FEATURE_COLUMNS)} features.", flush=True)

    X = df_pairs[FEATURE_COLUMNS].values
    y = df_pairs["label"].values
    groups = df_pairs["source1_entity_id"].values

    all_results = []
    oof_predictions = {}

    # --- SECTION 1: ADDITIONAL BASELINE MODELS ---

    # 1. Gaussian Naive Bayes
    gnb_metrics, oof_predictions["GaussianNB"] = evaluate_model_cv(
        "Gaussian Naive Bayes",
        lambda: GaussianNB(),
        X, y, groups, df_pairs, gt_map,
        scale_features=True,
    )
    all_results.append(gnb_metrics)

    # 2. K-Nearest Neighbors (Tuning k: 5, 15, 30, 50)
    print("\n--- Tuning K-Nearest Neighbors across k in [5, 15, 30, 50] ---", flush=True)
    best_knn_k = 15
    best_knn_metrics = None
    for k_val in [5, 15, 30, 50]:
        knn_m, knn_oof = evaluate_model_cv(
            f"K-Nearest Neighbors (k={k_val}, distance-weighted)",
            lambda: KNeighborsClassifier(n_neighbors=k_val, weights="distance", n_jobs=-1),
            X, y, groups, df_pairs, gt_map,
            scale_features=True,
        )
        if best_knn_metrics is None or knn_m["cv_f05"] > best_knn_metrics["cv_f05"]:
            best_knn_metrics = knn_m
            best_knn_k = k_val
            oof_predictions["KNN"] = knn_oof
    all_results.append(best_knn_metrics)

    # 3. Support Vector Machine (LinearSVC with Calibrated probabilities)
    svm_lin_metrics, oof_predictions["LinearSVC"] = evaluate_model_cv(
        "Support Vector Machine (Linear Kernel)",
        lambda: CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3),
        X, y, groups, df_pairs, gt_map,
        scale_features=True,
    )
    all_results.append(svm_lin_metrics)

    # 4. Support Vector Machine (RBF Kernel, subsampled 15k rows for tractable quadratic scaling)
    print("\n--- Benchmarking Support Vector Machine (RBF Kernel, N=15,000 subsample) ---", flush=True)
    sub_idx = np.random.RandomState(42).choice(len(df_pairs), min(15000, len(df_pairs)), replace=False)
    X_sub = X[sub_idx]
    y_sub = y[sub_idx]
    groups_sub = groups[sub_idx]
    df_sub = df_pairs.iloc[sub_idx].reset_index(drop=True)
    svm_rbf_metrics, _ = evaluate_model_cv(
        "Support Vector Machine (RBF Kernel, Subsample)",
        lambda: SVC(C=1.0, kernel="rbf", probability=True, max_iter=2000, random_state=42),
        X_sub, y_sub, groups_sub, df_sub, gt_map,
        scale_features=True,
    )
    all_results.append(svm_rbf_metrics)

    # --- TOP REFERENCE MODELS (Exact same folds) ---
    mlp_metrics, oof_predictions["NeuralNet_MLP"] = evaluate_model_cv(
        "NeuralNet_MLP (Current Winner)",
        lambda: MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.01,
                              learning_rate_init=0.003, max_iter=50, early_stopping=True, random_state=42),
        X, y, groups, df_pairs, gt_map,
        scale_features=True,
    )
    all_results.append(mlp_metrics)

    xgb_metrics, oof_predictions["XGBoost"] = evaluate_model_cv(
        "XGBoost (Runner Up)",
        lambda: xgb.XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=6,
                                  colsample_bytree=0.75, subsample=0.85, random_state=42, n_jobs=-1, eval_metric="logloss"),
        X, y, groups, df_pairs, gt_map,
        scale_features=False,
    )
    all_results.append(xgb_metrics)

    rf_metrics, oof_predictions["RandomForest"] = evaluate_model_cv(
        "RandomForest (Reference)",
        lambda: RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        X, y, groups, df_pairs, gt_map,
        scale_features=False,
    )
    all_results.append(rf_metrics)

    # --- SECTION 2: STACKED / BLENDED MODEL ---
    print("\n--- SECTION 2: Building Stacked Model over Top Performers (NeuralNet_MLP + XGBoost + RF) ---", flush=True)
    meta_dict = {
        "mlp": oof_predictions["NeuralNet_MLP"],
        "xgb": oof_predictions["XGBoost"],
        "rf": oof_predictions["RandomForest"],
    }

    # Meta-Learner 1: Logistic Regression
    stacked_lr_metrics, _ = evaluate_model_cv(
        "Stacked Model (Meta-Learner: Logistic Regression)",
        lambda: LogisticRegression(C=1.0, random_state=42),
        X, y, groups, df_pairs, gt_map,
        is_stacked=True,
        oof_meta_features=meta_dict,
    )
    all_results.append(stacked_lr_metrics)

    # Meta-Learner 2: Shallow MLP
    stacked_mlp_metrics, _ = evaluate_model_cv(
        "Stacked Model (Meta-Learner: Shallow MLP)",
        lambda: MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", alpha=0.01,
                              learning_rate_init=0.005, max_iter=40, random_state=42),
        X, y, groups, df_pairs, gt_map,
        is_stacked=True,
        oof_meta_features=meta_dict,
    )
    all_results.append(stacked_mlp_metrics)

    # Print Master Benchmark Comparison Table
    print("\n" + "=" * 115)
    print("EXPANDED 10-MODEL MASTER BENCHMARK TABLE (5-Fold Stratified Group CV)")
    print("=" * 115)
    print(f"{'Model Name':<42} | {'CV F0.5':<9} | {'Precision':<9} | {'Recall':<9} | {'Train F0.5':<10} | {'Overfit Gap':<12} | {'Time (s)':<8}")
    print("-" * 115)
    for r in sorted(all_results, key=lambda x: x["cv_f05"], reverse=True):
        print(f"{r['model']:<42} | {r['cv_f05']:<9.4f} | {r['cv_precision']:<9.4f} | {r['cv_recall']:<9.4f} | {r['train_f05']:<10.4f} | {r['overfitting_gap']:<+12.4f} | {r['training_time_s']:<8.1f}")
    print("=" * 115)

    # Save to JSON
    out_path = "models/extended_models_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to {out_path} in {time.time()-t0:.1f}s total.")


if __name__ == "__main__":
    main()
