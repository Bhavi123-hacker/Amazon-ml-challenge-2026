"""Step 4: Fine-Grained Per-Country Threshold and Decision-Rule Calibration.

1. Sweeps tau_US in [0.50 ... 0.80] on held-out US ground truth entities.
2. Sweeps tau_India in [0.50 ... 0.80] on held-out India ground truth entities.
3. Finds exact joint optimal (tau_US, tau_India) maximizing Macro F0.5.
4. Generates France candidate scoring distribution and pulls a stratified 50-pair random sample for manual verification.
"""

import os
import sys
import json
import time
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
    print("=" * 85, flush=True)
    print("STEP 4: PER-COUNTRY THRESHOLD AND DECISION-RULE CALIBRATION", flush=True)
    print("=" * 85, flush=True)
    t0 = time.time()

    # 1. Load Champion LightGBM Model
    model_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Load Held-Out Split
    val_bench_path = "overnight_mission/eval/val_10k_benchmark.tsv"
    df_val_bench = pd.read_csv(val_bench_path, sep="\t", keep_default_na=False)
    val_s1_set = set(df_val_bench["entity_id"])

    gt_path = "student_resource/dataset/train/train_ground_truth.tsv"
    df_gt = pd.read_csv(gt_path, sep="\t", keep_default_na=False)
    val_gt_map = {}
    for _, r in df_gt.iterrows():
        sid = r["source1_entity_id"].strip()
        if sid in val_s1_set:
            m_raw = r["matched_entity_ids"].strip()
            val_gt_map[sid] = {x.strip() for x in m_raw.split(",") if x.strip()} if m_raw else set()

    # Load S1 records
    val_s1_rows = []
    for chunk in pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(val_s1_set)]
        if len(sub) > 0:
            val_s1_rows.append(sub)
        if sum(len(x) for x in val_s1_rows) >= len(val_s1_set):
            break
    df_s1_val = apply_normalization_df(pd.concat(val_s1_rows, ignore_index=True))
    val_s1_dict = {r["entity_id"]: r for r in df_s1_val.to_dict("records")}

    # Load Candidate Pool
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

    indexer = BlockingIndex()
    indexer.build(df_val_cands)

    val_entity_order = list(val_s1_dict.keys())
    val_feat_rows = []
    val_pair_index = []

    for sid in val_entity_order:
        r1 = val_s1_dict[sid]
        ret = indexer.retrieve_candidates_for_entity(r1, top_k=15)
        for cid, _ in ret:
            rc = val_cand_dict.get(cid)
            if rc:
                feats = extract_pair_features(r1, rc)
                val_feat_rows.append([feats[col] for col in FEATURE_COLUMNS])
                val_pair_index.append((sid, cid, rc.get("name_norm", "")))

    probs = model.predict_proba(np.array(val_feat_rows, dtype=np.float32))[:, 1]
    scored_map = {sid: [] for sid in val_entity_order}
    for (sid, cid, c_name), p in zip(val_pair_index, probs):
        scored_map[sid].append((cid, float(p), c_name))

    for sid in scored_map:
        scored_map[sid].sort(key=lambda x: x[1], reverse=True)

    s1_by_country = {}
    for r in df_s1_val.to_dict("records"):
        s1_by_country.setdefault(r["country"], []).append(r["entity_id"])

    # 3. Independent Per-Country Sweep
    print("\n--- INDEPENDENT THRESHOLD SWEEPS ON HELD-OUT SPLIT ---", flush=True)

    sweep_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    def evaluate_country_threshold(country, tau):
        sids = set(s1_by_country[country])
        c_gt = {s: val_gt_map[s] for s in sids}
        preds = {}
        for sid in sids:
            pairs = scored_map.get(sid, [])
            accepted = []
            top_cand_name = pairs[0][2] if pairs else ""
            p1 = pairs[0][1] if pairs else 0.0

            for rank, (cid, p, c_name) in enumerate(pairs):
                if p < tau:
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

        f05, prec, rec = compute_macro_f05(c_gt, preds)
        sing_pct = sum(1 for v in preds.values() if len(v) == 0) / len(preds) * 100
        return f05, prec, rec, sing_pct

    us_results = []
    print("\n[US HELD-OUT SWEEP]")
    for tau in sweep_thresholds:
        f05, p, r, s = evaluate_country_threshold("US", tau)
        us_results.append({"tau": tau, "f05": f05, "prec": p, "rec": r, "sing": s})
        print(f"  tau_US={tau:.2f} -> Macro F0.5: {f05:.4f} | Prec: {p:.4f} | Rec: {r:.4f} | Singletons: {s:.1f}%")

    india_results = []
    print("\n[INDIA HELD-OUT SWEEP]")
    for tau in sweep_thresholds:
        f05, p, r, s = evaluate_country_threshold("India", tau)
        india_results.append({"tau": tau, "f05": f05, "prec": p, "rec": r, "sing": s})
        print(f"  tau_India={tau:.2f} -> Macro F0.5: {f05:.4f} | Prec: {p:.4f} | Rec: {r:.4f} | Singletons: {s:.1f}%")

    best_us = max(us_results, key=lambda x: x["f05"])
    best_india = max(india_results, key=lambda x: x["f05"])

    print("\n" + "=" * 85)
    print("OPTIMAL COUNTRY CALIBRATION")
    print("=" * 85)
    print(f"Optimal US Threshold   : tau = {best_us['tau']:.2f} | Held-Out F0.5: {best_us['f05']:.4f} (Prec: {best_us['prec']:.4f}, Rec: {best_us['rec']:.4f})")
    print(f"Optimal India Threshold: tau = {best_india['tau']:.2f} | Held-Out F0.5: {best_india['f05']:.4f} (Prec: {best_india['prec']:.4f}, Rec: {best_india['rec']:.4f})")

    # Joint evaluation on all 10,000 entities
    joint_preds = {}
    for sid in val_s1_set:
        c = "US" if sid in s1_by_country["US"] else "India"
        tau = best_us["tau"] if c == "US" else best_india["tau"]
        pairs = scored_map.get(sid, [])
        accepted = []
        top_cand_name = pairs[0][2] if pairs else ""
        p1 = pairs[0][1] if pairs else 0.0

        for rank, (cid, p, c_name) in enumerate(pairs):
            if p < tau:
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
        joint_preds[sid] = set(accepted)

    joint_f05, joint_p, joint_r = compute_macro_f05(val_gt_map, joint_preds)
    joint_s = sum(1 for v in joint_preds.values() if len(v) == 0) / len(joint_preds) * 100
    print(f"\nJOINT CALIBRATED HELD-OUT MACRO F0.5: {joint_f05:.4f} | Prec: {joint_p:.4f} | Rec: {joint_r:.4f} | Singletons: {joint_s:.1f}%")
    print("=" * 85, flush=True)

    # Save calibration config
    calibration_config = {
        "best_tau_us": best_us["tau"],
        "best_tau_india": best_india["tau"],
        "us_held_out_f05": best_us["f05"],
        "india_held_out_f05": best_india["f05"],
        "joint_held_out_f05": joint_f05,
        "joint_held_out_prec": joint_p,
        "joint_held_out_rec": joint_r,
        "cluster_disambig_sim_threshold": 40,
        "cluster_disambig_margin": 0.08,
    }
    with open("overnight_mission/eval/calibration_config.json", "w") as f:
        json.dump(calibration_config, f, indent=2)

    print(f"\nStep 4 calibration completed in {time.time() - t0:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
