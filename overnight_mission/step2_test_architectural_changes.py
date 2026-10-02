"""Step 2: Test Part A Architectural Changes Against Step 1 Baseline.

Architectures Evaluated on the SAME Held-Out Benchmark Split (10,000 unseen entities):
- Baseline: Flat thresholding (tau = 0.65).
- A1: Cluster-level disambiguation (candidate-candidate name dissimilarity suppression).
- A3: Ranking-based adaptive margin rule (differential bar for rank 1 vs rank 2+).
- A1 + A3 Combined: Both cluster disambiguation and adaptive margin ranking.

Reports exact Macro F0.5, Precision, Recall, Singleton rate for US, India, and Overall.
"""

import os
import sys
import time
import json
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
    print("=" * 80, flush=True)
    print("STEP 2: TESTING ARCHITECTURAL CHANGES AGAINST STEP 1 BASELINE", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    # 1. Load Step 1 Baseline Model
    model_path = "overnight_mission/models/lgbm_step1_baseline.pkl"
    print(f"Loading trained LightGBM model from {model_path} ...", flush=True)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Load Step 0 Held-out Benchmark Split (10,000 entities)
    val_bench_path = "overnight_mission/eval/val_10k_benchmark.tsv"
    print(f"Loading held-out benchmark split from {val_bench_path} ...", flush=True)
    df_val_bench = pd.read_csv(val_bench_path, sep="\t", keep_default_na=False)
    val_s1_set = set(df_val_bench["entity_id"])

    # Load ground truth for benchmark
    gt_path = "student_resource/dataset/train/train_ground_truth.tsv"
    df_gt = pd.read_csv(gt_path, sep="\t", keep_default_na=False)
    val_gt_map = {}
    for _, r in df_gt.iterrows():
        sid = r["source1_entity_id"].strip()
        if sid in val_s1_set:
            m_raw = r["matched_entity_ids"].strip()
            val_gt_map[sid] = {x.strip() for x in m_raw.split(",") if x.strip()} if m_raw else set()

    # 3. Load S1 and Candidates
    print("Loading normalized S1 and Candidate records...", flush=True)
    val_s1_rows = []
    for chunk in pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(val_s1_set)]
        if len(sub) > 0:
            val_s1_rows.append(sub)
        if sum(len(x) for x in val_s1_rows) >= len(val_s1_set):
            break
    df_s1_val = apply_normalization_df(pd.concat(val_s1_rows, ignore_index=True))
    val_s1_dict = {r["entity_id"]: r for r in df_s1_val.to_dict("records")}

    needed_cands = set().union(*val_gt_map.values())
    val_cand_rows = []
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("student_resource/dataset/train", src)
        for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
            sub = chunk[chunk["entity_id"].isin(needed_cands)]
            val_cand_rows.append(sub)
            if len(val_cand_rows) <= 2:
                val_cand_rows.append(chunk.head(15000))
            if sum(len(x[x["entity_id"].isin(needed_cands)]) for x in val_cand_rows) >= len(needed_cands):
                break

    df_val_cands = apply_normalization_df(pd.concat(val_cand_rows, ignore_index=True).drop_duplicates("entity_id"))
    val_cand_dict = {r["entity_id"]: r for r in df_val_cands.to_dict("records")}

    # Build index & score candidates
    print("Indexing and scoring candidate pairs...", flush=True)
    val_indexer = BlockingIndex()
    val_indexer.build(df_val_cands)

    scored_candidates = {}  # sid -> [(cid, prob, name_norm, addr_norm)]
    for sid, r1 in val_s1_dict.items():
        ret = val_indexer.retrieve_candidates_for_entity(r1, top_k=15)
        feat_matrix = []
        c_meta = []
        for cid, _ in ret:
            rc = val_cand_dict.get(cid)
            if rc:
                feats = extract_pair_features(r1, rc)
                feat_matrix.append([feats[col] for col in FEATURE_COLUMNS])
                c_meta.append((cid, rc.get("name_norm", ""), rc.get("addr_norm", "")))

        if feat_matrix:
            probs = model.predict_proba(np.array(feat_matrix, dtype=np.float32))[:, 1]
            scored_pairs = []
            for (cid, c_name, c_addr), p in zip(c_meta, probs):
                scored_pairs.append((cid, float(p), c_name, c_addr))
            # Sort descending by probability
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            scored_candidates[sid] = scored_pairs
        else:
            scored_candidates[sid] = []

    s1_by_country = {}
    for r in df_s1_val.to_dict("records"):
        s1_by_country.setdefault(r["country"], []).append(r["entity_id"])

    def evaluate_decision_rule(preds_dict, rule_name):
        f05_all, p_all, r_all = compute_macro_f05(val_gt_map, preds_dict)
        sing_pct = sum(1 for v in preds_dict.values() if len(v) == 0) / len(preds_dict) * 100

        c_reports = {}
        for c in ["US", "India"]:
            c_sids = set(s1_by_country.get(c, []))
            c_gt = {sid: val_gt_map[sid] for sid in c_sids}
            c_pred = {sid: preds_dict[sid] for sid in c_sids}
            c_f05, c_p, c_r = compute_macro_f05(c_gt, c_pred)
            c_reports[c] = (c_f05, c_p, c_r)

        return {
            "rule": rule_name,
            "overall_f05": f05_all,
            "overall_prec": p_all,
            "overall_rec": r_all,
            "singleton_pct": sing_pct,
            "us_f05": c_reports["US"][0],
            "us_prec": c_reports["US"][1],
            "us_rec": c_reports["US"][2],
            "india_f05": c_reports["India"][0],
            "india_prec": c_reports["India"][1],
            "india_rec": c_reports["India"][2],
        }

    results_table = []

    # --- 1. Baseline: Pure thresholding at tau = 0.65 ---
    preds_base = {}
    for sid in val_s1_set:
        pairs = scored_candidates.get(sid, [])
        m = [cid for cid, p, _, _ in pairs if p >= 0.65][:10]
        preds_base[sid] = set(m)
    results_table.append(evaluate_decision_rule(preds_base, "Baseline (Pure tau=0.65)"))

    # --- 2. Architecture A1: Cluster-level Disambiguation ---
    # If multiple candidates pass threshold, check mutual candidate-to-candidate name similarity.
    # Suppress secondary candidate if it has high prob but completely different name (< 35 sim) from top candidate
    for cand_sim_threshold in [30, 40, 50]:
        preds_a1 = {}
        for sid in val_s1_set:
            pairs = scored_candidates.get(sid, [])
            accepted = []
            top_cand_name = pairs[0][2] if pairs else ""

            for rank, (cid, p, c_name, _) in enumerate(pairs):
                if p < 0.65:
                    continue
                if rank == 0:
                    accepted.append(cid)
                else:
                    # Check candidate-to-candidate name similarity
                    sim_with_top = fuzz.token_sort_ratio(top_cand_name, c_name)
                    # Suppress only if severe name clash (co-located different entity) AND prob margin
                    if sim_with_top < cand_sim_threshold and (pairs[0][1] - p > 0.08):
                        continue
                    accepted.append(cid)
                if len(accepted) >= 10:
                    break
            preds_a1[sid] = set(accepted)
        results_table.append(evaluate_decision_rule(preds_a1, f"A1 Cluster Disambig (sim<{cand_sim_threshold})"))

    # --- 3. Architecture A3: Ranking-Based Adaptive Decision Rule ---
    # Rank 1 accepts at lower bar (p >= 0.50); Rank 2+ requires higher bar (p >= 0.70) AND margin constraint
    for tau1 in [0.45, 0.50, 0.55]:
        for tau2 in [0.65, 0.70, 0.75]:
            preds_a3 = {}
            for sid in val_s1_set:
                pairs = scored_candidates.get(sid, [])
                accepted = []
                p1 = pairs[0][1] if pairs else 0.0
                for rank, (cid, p, _, _) in enumerate(pairs):
                    if rank == 0:
                        if p >= tau1:
                            accepted.append(cid)
                    else:
                        # Rank 2+ must clear higher bar AND margin
                        if p >= tau2 and (p1 - p <= 0.35):
                            accepted.append(cid)
                    if len(accepted) >= 10:
                        break
                preds_a3[sid] = set(accepted)
            results_table.append(evaluate_decision_rule(preds_a3, f"A3 Ranking Rule (tau1={tau1}, tau2={tau2})"))

    # --- 4. Architecture A1 + A3 Combined ---
    for tau1, tau2 in [(0.50, 0.70), (0.55, 0.70)]:
        preds_combo = {}
        for sid in val_s1_set:
            pairs = scored_candidates.get(sid, [])
            accepted = []
            top_cand_name = pairs[0][2] if pairs else ""
            p1 = pairs[0][1] if pairs else 0.0

            for rank, (cid, p, c_name, _) in enumerate(pairs):
                if rank == 0:
                    if p >= tau1:
                        accepted.append(cid)
                else:
                    if p >= tau2 and (p1 - p <= 0.35):
                        sim_with_top = fuzz.token_sort_ratio(top_cand_name, c_name)
                        if sim_with_top < 35 and (p1 - p > 0.08):
                            continue
                        accepted.append(cid)
                if len(accepted) >= 10:
                    break
            preds_combo[sid] = set(accepted)
        results_table.append(evaluate_decision_rule(preds_combo, f"A1+A3 Combined (tau1={tau1}, tau2={tau2})"))

    # Print comprehensive comparison table
    df_res = pd.DataFrame(results_table)
    print("\n" + "=" * 90)
    print("STEP 2 ARCHITECTURAL COMPARISON ON 10,000 HELD-OUT ENTITIES")
    print("=" * 90)
    fmt_cols = ["rule", "overall_f05", "overall_prec", "overall_rec", "singleton_pct", "us_f05", "india_f05"]
    print(df_res[fmt_cols].to_string(index=False))
    print("=" * 90, flush=True)

    # Save results to json
    with open("overnight_mission/eval/step2_architectural_comparison.json", "w") as f:
        json.dump(results_table, f, indent=2)

    print(f"\nStep 2 completed in {time.time() - t0:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
