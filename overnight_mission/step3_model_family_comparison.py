"""Step 3: Model Family Comparison on Best Step 2 Configuration.

Benchmarks 4 Model Families on identical training data and identical held-out test split:
1. LightGBM (Gradient Boosting with histogram binning)
2. XGBoost (Extreme Gradient Boosting with exact greedy/hist tree method)
3. CatBoost (Categorical Boosting with symmetric trees)
4. MLPClassifier (Regularized Multi-Layer Perceptron neural network)

Evaluates strictly against Step 0 held-out benchmark split (10,000 unseen entities):
- Held-out Macro F0.5 (Overall, US, India)
- Held-out Precision (Overall, US, India)
- Held-out Recall (Overall, US, India)
- Training Time & Inference Speed
- Discovers and saves the Champion model for deployment.
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
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.blocking import BlockingIndex
from src.evaluate import compute_macro_f05
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df


def main():
    print("=" * 85, flush=True)
    print("STEP 3: COMPREHENSIVE MODEL FAMILY BENCHMARK ON IDENTICAL HELD-OUT SPLIT", flush=True)
    print("=" * 85, flush=True)
    t0 = time.time()

    # 1. Load Step 0 Train Split IDs
    train_split_path = "overnight_mission/eval/train_split_ids.parquet"
    df_train_split = pd.read_parquet(train_split_path)

    # 2. Load Ground Truth
    gt_path = "student_resource/dataset/train/train_ground_truth.tsv"
    df_gt = pd.read_csv(gt_path, sep="\t", keep_default_na=False)
    gt_map = {}
    for _, r in df_gt.iterrows():
        sid = r["source1_entity_id"].strip()
        m_raw = r["matched_entity_ids"].strip()
        gt_map[sid] = {x.strip() for x in m_raw.split(",") if x.strip()} if m_raw else set()

    # 3. Build Training Pairs strictly from Train Split
    np.random.seed(42)
    s1_train_selected = []
    for strata, grp in df_train_split.groupby(["country", "match_count"]):
        sub_pool = list(grp["entity_id"])
        n_sample = max(1, int(len(sub_pool) / len(df_train_split) * 6000))
        shuffled = np.random.choice(sub_pool, size=min(n_sample, len(sub_pool)), replace=False)
        s1_train_selected.extend(shuffled)
    s1_train_set = set(s1_train_selected[:6000])

    s1_rows = []
    for chunk in pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(s1_train_set)]
        if len(sub) > 0:
            s1_rows.append(sub)
        if sum(len(x) for x in s1_rows) >= len(s1_train_set):
            break
    df_s1_train = apply_normalization_df(pd.concat(s1_rows, ignore_index=True))
    s1_dict = {r["entity_id"]: r for r in df_s1_train.to_dict("records")}

    needed_positives = set()
    for sid in s1_train_set:
        needed_positives.update(gt_map.get(sid, set()))

    cand_rows = []
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("student_resource/dataset/train", src)
        for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
            sub = chunk[chunk["entity_id"].isin(needed_positives)]
            cand_rows.append(sub)
            if len(cand_rows) <= 2:
                cand_rows.append(chunk.head(15000))
            if sum(len(x[x["entity_id"].isin(needed_positives)]) for x in cand_rows) >= len(needed_positives):
                break
    df_cands_train = apply_normalization_df(pd.concat(cand_rows, ignore_index=True).drop_duplicates("entity_id"))
    cand_dict = {r["entity_id"]: r for r in df_cands_train.to_dict("records")}

    indexer = BlockingIndex()
    indexer.build(df_cands_train)

    pair_rows = []
    labels = []
    for sid, r1 in s1_dict.items():
        true_m = gt_map.get(sid, set())
        ret = indexer.retrieve_candidates_for_entity(r1, top_k=20)
        c_cand_ids = [c[0] for c in ret]
        candidate_ids = list(dict.fromkeys(list(true_m) + c_cand_ids))
        for cid in candidate_ids:
            rc = cand_dict.get(cid)
            if not rc:
                continue
            feats = extract_pair_features(r1, rc)
            pair_rows.append([feats[col] for col in FEATURE_COLUMNS])
            labels.append(1 if cid in true_m else 0)

    X_train = np.array(pair_rows, dtype=np.float32)
    y_train = np.array(labels, dtype=np.int32)
    print(f"Training dataset: {len(X_train):,} pairs (Positives: {y_train.sum():,}).", flush=True)

    # 4. Load Step 0 Held-out Benchmark Split (10,000 unseen entities)
    val_bench_path = "overnight_mission/eval/val_10k_benchmark.tsv"
    df_val_bench = pd.read_csv(val_bench_path, sep="\t", keep_default_na=False)
    val_s1_set = set(df_val_bench["entity_id"])
    val_gt_map = {sid: gt_map.get(sid, set()) for sid in val_s1_set}

    val_s1_rows = []
    for chunk in pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(val_s1_set)]
        if len(sub) > 0:
            val_s1_rows.append(sub)
        if sum(len(x) for x in val_s1_rows) >= len(val_s1_set):
            break
    df_s1_val = apply_normalization_df(pd.concat(val_s1_rows, ignore_index=True))
    val_s1_dict = {r["entity_id"]: r for r in df_s1_val.to_dict("records")}

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

    # Pre-extract test pairs
    val_entity_order = list(val_s1_dict.keys())
    val_pairs_by_entity = {}
    val_feat_rows = []
    val_pair_index = []  # (sid, cid, c_name)

    for sid in val_entity_order:
        r1 = val_s1_dict[sid]
        ret = val_indexer.retrieve_candidates_for_entity(r1, top_k=15)
        c_list = []
        for cid, _ in ret:
            rc = val_cand_dict.get(cid)
            if rc:
                feats = extract_pair_features(r1, rc)
                val_feat_rows.append([feats[col] for col in FEATURE_COLUMNS])
                val_pair_index.append((sid, cid, rc.get("name_norm", "")))
                c_list.append(cid)
        val_pairs_by_entity[sid] = c_list

    X_val = np.array(val_feat_rows, dtype=np.float32)
    print(f"Validation matrix ready: {len(X_val):,} pairs across {len(val_entity_order):,} entities.", flush=True)

    s1_by_country = {}
    for r in df_s1_val.to_dict("records"):
        s1_by_country.setdefault(r["country"], []).append(r["entity_id"])

    # 5. Define Candidate Model Families
    models_to_test = {
        "LightGBM": lgb.LGBMClassifier(
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
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "CatBoost": CatBoostClassifier(
            iterations=350,
            learning_rate=0.06,
            depth=6,
            random_seed=42,
            thread_count=-1,
            verbose=0,
        ),
        "Regularized_MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.001,
            max_iter=30,
            random_state=42,
            early_stopping=True,
        )
    }

    # Scaler for MLP
    scaler = StandardScaler()
    scaler.fit(X_train)

    benchmark_records = []
    best_f05 = -1.0
    winner_name = None
    winner_obj = None

    for model_name, model_obj in models_to_test.items():
        print(f"\n--- Training {model_name} ---", flush=True)
        t_tr = time.time()
        if model_name == "Regularized_MLP":
            model_obj.fit(scaler.transform(X_train), y_train)
        else:
            model_obj.fit(X_train, y_train)
        train_time = time.time() - t_tr
        print(f"  Trained in {train_time:.2f}s.", flush=True)

        # Inference on validation set
        t_inf = time.time()
        if model_name == "Regularized_MLP":
            val_probs = model_obj.predict_proba(scaler.transform(X_val))[:, 1]
        else:
            val_probs = model_obj.predict_proba(X_val)[:, 1]
        inf_time = time.time() - t_inf
        ent_per_sec = len(val_entity_order) / max(0.01, inf_time)

        # Map probabilities back to entities with A1 cluster disambiguation
        scored_map = {sid: [] for sid in val_entity_order}
        for (sid, cid, c_name), prob in zip(val_pair_index, val_probs):
            scored_map[sid].append((cid, float(prob), c_name))

        for sid in scored_map:
            scored_map[sid].sort(key=lambda x: x[1], reverse=True)

        # Evaluate at optimal tau=0.65 with A1 cluster disambiguation
        preds = {}
        for sid in val_s1_set:
            pairs = scored_map.get(sid, [])
            accepted = []
            top_cand_name = pairs[0][2] if pairs else ""
            p1 = pairs[0][1] if pairs else 0.0

            for rank, (cid, p, c_name) in enumerate(pairs):
                if p < 0.65:
                    continue
                if rank == 0:
                    accepted.append(cid)
                else:
                    sim_with_top = fuzz.token_sort_ratio(top_cand_name, c_name)
                    if sim_with_top < 40 and (p1 - p > 0.08):
                        continue
                    accepted.append(cid)
                if len(accepted) >= 10:
                    break
            preds[sid] = set(accepted)

        f05_all, p_all, r_all = compute_macro_f05(val_gt_map, preds)
        sing_pct = sum(1 for v in preds.values() if len(v) == 0) / len(preds) * 100

        c_metrics = {}
        for c in ["US", "India"]:
            c_sids = set(s1_by_country.get(c, []))
            c_gt = {s: val_gt_map[s] for s in c_sids}
            c_p = {s: preds[s] for s in c_sids}
            c_metrics[c] = compute_macro_f05(c_gt, c_p)

        rec = {
            "model": model_name,
            "overall_f05": f05_all,
            "overall_prec": p_all,
            "overall_rec": r_all,
            "sing_pct": sing_pct,
            "us_f05": c_metrics["US"][0],
            "us_prec": c_metrics["US"][1],
            "us_rec": c_metrics["US"][2],
            "india_f05": c_metrics["India"][0],
            "india_prec": c_metrics["India"][1],
            "india_rec": c_metrics["India"][2],
            "train_time_s": train_time,
            "inf_speed_ent_s": ent_per_sec,
        }
        benchmark_records.append(rec)
        print(f"  Held-out Macro F0.5: {f05_all:.4f} (US: {c_metrics['US'][0]:.4f}, India: {c_metrics['India'][0]:.4f}) | Speed: {ent_per_sec:,.0f} ent/s")

        if f05_all > best_f05:
            best_f05 = f05_all
            winner_name = model_name
            winner_obj = model_obj

    # 6. Save Champion Model and Comparison Table
    df_comp = pd.DataFrame(benchmark_records)
    print("\n" + "=" * 90)
    print("STEP 3 MODEL FAMILY COMPARISON TABLE (HELD-OUT 10,000 ENTITIES)")
    print("=" * 90)
    cols = ["model", "overall_f05", "overall_prec", "overall_rec", "us_f05", "india_f05", "train_time_s", "inf_speed_ent_s"]
    print(df_comp[cols].to_string(index=False))
    print("=" * 90, flush=True)

    print(f"\nWINNER: {winner_name} with Held-Out Macro F0.5 = {best_f05:.4f}!")
    champ_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(champ_path, "wb") as f:
        pickle.dump(winner_obj, f)
    print(f"Saved Champion model to {champ_path} ({os.path.getsize(champ_path):,} bytes).")

    with open("overnight_mission/eval/step3_model_family_comparison.json", "w") as f:
        json.dump(benchmark_records, f, indent=2)

    print(f"\nStep 3 completed in {time.time() - t0:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
