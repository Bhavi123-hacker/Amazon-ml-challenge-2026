"""15-Iteration Architectural Tournament for Business Entity Resolution.

Systematically evaluates 15 structurally distinct clustering and decision architectures
on held-out validation data to discover the approach with the highest Macro F0.5.
"""

import os
import sys
import gc
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

def run_tournament():
    print("=" * 100, flush=True)
    print("15-ITERATION ARCHITECTURAL TOURNAMENT: SEARCH FOR PEAK MACRO F0.5", flush=True)
    print("=" * 100, flush=True)

    # 1. Load Champion Model
    model_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Sample 1,000 entities from held-out split
    print("Sampling 1,000 held-out entities...", flush=True)
    df_held = pd.read_parquet("overnight_mission/eval/held_out_split_ids.parquet")
    np.random.seed(42)
    sample_sids = set(np.random.choice(df_held["entity_id"], size=1000, replace=False))

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

    # 3. Load realistic candidate pool
    print("Loading candidate pool (true matches + 60,000 distractors)...", flush=True)
    c_rows = []
    distractor_count = 0
    for p in ["student_resource/dataset/train/train_source2.tsv", "student_resource/dataset/train/train_source3.tsv"]:
        for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
            sub_true = chunk[chunk["entity_id"].isin(needed_cands)]
            if len(sub_true) > 0:
                c_rows.append(sub_true)
            if distractor_count < 60000:
                c_rows.append(chunk.head(15000))
                distractor_count += len(chunk.head(15000))

    df_cands = apply_normalization_df(pd.concat(c_rows, ignore_index=True).drop_duplicates("entity_id"))
    c_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
    print(f"Candidate pool built: {len(c_map):,} candidates.", flush=True)

    # 4. Build index and retrieve top 20 candidates per entity
    indexer = BlockingIndex()
    indexer.build(df_cands)

    print("Scoring candidates for 1,000 entities...", flush=True)
    scored_pairs = {}
    pair_features = {}
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
                pair_features[(sid, cid)] = feats
        if feat_matrix:
            probs = model.predict_proba(np.array(feat_matrix, dtype=np.float32))[:, 1]
            pairs = sorted(zip(c_list, probs), key=lambda x: x[1], reverse=True)
            scored_pairs[sid] = pairs
        else:
            scored_pairs[sid] = []

    print(f"Scoring complete. Evaluating 15 Iterations...", flush=True)

    # -------------------------------------------------------------
    # DEFINE THE 15 DISTINCT ARCHITECTURAL ITERATIONS
    # -------------------------------------------------------------
    tournament_results = []

    # Iteration 1: Baseline Hard Threshold (tau = 0.75)
    def it1_baseline(sid_pairs_dict):
        preds = {}
        for sid, pairs in sid_pairs_dict.items():
            preds[sid] = set([cid for cid, p in pairs if p >= 0.75][:10])
        return preds

    # Iteration 2: High-Precision Hard Threshold (tau = 0.90)
    def it2_high_tau(sid_pairs_dict):
        preds = {}
        for sid, pairs in sid_pairs_dict.items():
            preds[sid] = set([cid for cid, p in pairs if p >= 0.90][:10])
        return preds

    # Iteration 3: 2-Stage Dynamic Margin (tau = 0.75, delta = 0.08, max 6)
    def it3_dynamic_margin(sid_pairs_dict):
        preds = {}
        for sid, pairs in sid_pairs_dict.items():
            if not pairs or pairs[0][1] < 0.75:
                preds[sid] = set()
            else:
                p1 = pairs[0][1]
                preds[sid] = set([cid for cid, p in pairs if p >= 0.75 and (p1 - p) <= 0.08][:6])
        return preds

    # Iteration 4: Ultra-Tight Dynamic Margin (tau = 0.80, delta = 0.04, max 5)
    def it4_tight_margin(sid_pairs_dict):
        preds = {}
        for sid, pairs in sid_pairs_dict.items():
            if not pairs or pairs[0][1] < 0.80:
                preds[sid] = set()
            else:
                p1 = pairs[0][1]
                preds[sid] = set([cid for cid, p in pairs if p >= 0.80 and (p1 - p) <= 0.04][:5])
        return preds

    # Iteration 5: Adaptive Scaled Margin (delta = 0.06 * p1)
    def it5_adaptive_margin(sid_pairs_dict):
        preds = {}
        for sid, pairs in sid_pairs_dict.items():
            if not pairs or pairs[0][1] < 0.75:
                preds[sid] = set()
            else:
                p1 = pairs[0][1]
                delta = 0.06 * p1
                preds[sid] = set([cid for cid, p in pairs if p >= 0.75 and (p1 - p) <= delta][:6])
        return preds

    # Iteration 6: Relative Probability Ratio (p / p1 >= 0.92)
    def it6_prob_ratio(sid_pairs_dict):
        preds = {}
        for sid, pairs in sid_pairs_dict.items():
            if not pairs or pairs[0][1] < 0.75:
                preds[sid] = set()
            else:
                p1 = pairs[0][1]
                preds[sid] = set([cid for cid, p in pairs if p >= 0.75 and (p / max(1e-6, p1)) >= 0.92][:6])
        return preds

    # Iteration 7: Elbow / Cliff Detection (Cut at first delta > 0.08)
    def it7_cliff_detection(sid_pairs_dict):
        preds = {}
        for sid, pairs in sid_pairs_dict.items():
            valid = [(cid, p) for cid, p in pairs if p >= 0.75]
            if not valid:
                preds[sid] = set()
            elif len(valid) == 1:
                preds[sid] = {valid[0][0]}
            else:
                chosen = [valid[0][0]]
                for i in range(len(valid) - 1):
                    if (valid[i][1] - valid[i+1][1]) > 0.08:
                        break
                    chosen.append(valid[i+1][0])
                    if len(chosen) >= 6:
                        break
                preds[sid] = set(chosen)
        return preds

    # Iteration 8: 2-Set Disjoint Bipartite Resolution (No S2 can belong to > 1 S1)
    def it8_bipartite_disjoint(sid_pairs_dict):
        # Flatten all proposed pairs above 0.75, sort by probability descending
        all_edges = []
        for sid, pairs in sid_pairs_dict.items():
            for cid, p in pairs:
                if p >= 0.75:
                    all_edges.append((sid, cid, p))
        all_edges.sort(key=lambda x: x[2], reverse=True)

        claimed_s2 = set()
        s1_to_matches = {sid: [] for sid in sid_pairs_dict}
        for sid, cid, p in all_edges:
            if cid not in claimed_s2 and len(s1_to_matches[sid]) < 10:
                s1_to_matches[sid].append(cid)
                claimed_s2.add(cid)
        return {sid: set(m) for sid, m in s1_to_matches.items()}

    # Iteration 9: 2-Stage Dynamic Margin + 2-Set Disjoint Bipartite Resolution
    def it9_dynamic_margin_plus_bipartite(sid_pairs_dict):
        # Step A: Dynamic margin filtering
        candidate_edges = []
        for sid, pairs in sid_pairs_dict.items():
            if pairs and pairs[0][1] >= 0.75:
                p1 = pairs[0][1]
                for cid, p in pairs:
                    if p >= 0.75 and (p1 - p) <= 0.08:
                        candidate_edges.append((sid, cid, p))
        candidate_edges.sort(key=lambda x: x[2], reverse=True)

        # Step B: Disjoint bipartite conflict resolution
        claimed_s2 = set()
        s1_matches = {sid: [] for sid in sid_pairs_dict}
        for sid, cid, p in candidate_edges:
            if cid not in claimed_s2 and len(s1_matches[sid]) < 6:
                s1_matches[sid].append(cid)
                claimed_s2.add(cid)
        return {sid: set(m) for sid, m in s1_matches.items()}

    # Iteration 10: Ground Truth Prior Truncation (Max matches = 4, tau = 0.78)
    def it10_prior_truncation(sid_pairs_dict):
        preds = {}
        for sid, pairs in sid_pairs_dict.items():
            preds[sid] = set([cid for cid, p in pairs if p >= 0.78][:4])
        return preds

    # Iteration 11: 1D Density Clustering (Cluster around top candidate within radius eps=0.06)
    def it11_density_clustering(sid_pairs_dict):
        preds = {}
        for sid, pairs in sid_pairs_dict.items():
            valid = [(cid, p) for cid, p in pairs if p >= 0.75]
            if not valid:
                preds[sid] = set()
            else:
                cluster = [valid[0][0]]
                prev_p = valid[0][1]
                for cid, p in valid[1:]:
                    if (prev_p - p) <= 0.06:
                        cluster.append(cid)
                        prev_p = p
                    else:
                        break
                    if len(cluster) >= 6:
                        break
                preds[sid] = set(cluster)
        return preds

    # Iteration 12: Feature-Augmented 2-Set (Probability >= 0.75 AND name_token_sort >= 45%)
    def it12_feature_augmented(sid_pairs_dict):
        preds = {}
        for sid, pairs in sid_pairs_dict.items():
            accepted = []
            for cid, p in pairs:
                if p >= 0.75:
                    feats = pair_features.get((sid, cid), {})
                    if feats.get("feat_name_token_sort", 0) >= 45.0:
                        accepted.append(cid)
                if len(accepted) >= 6:
                    break
            preds[sid] = set(accepted)
        return preds

    # Iteration 13: Softmax Temperature Top-Cumulative (Cumulative mass <= 0.85)
    def it13_softmax_mass(sid_pairs_dict):
        preds = {}
        temperature = 0.15
        for sid, pairs in sid_pairs_dict.items():
            valid = [(cid, p) for cid, p in pairs if p >= 0.75]
            if not valid:
                preds[sid] = set()
            else:
                p_arr = np.array([p for _, p in valid])
                exp_p = np.exp((p_arr - np.max(p_arr)) / temperature)
                weights = exp_p / np.sum(exp_p)
                cum_weights = np.cumsum(weights)
                selected = [valid[0][0]]
                for i in range(1, len(valid)):
                    if cum_weights[i] <= 0.88:
                        selected.append(valid[i][0])
                    else:
                        break
                    if len(selected) >= 6:
                        break
                preds[sid] = set(selected)
        return preds

    # Iteration 14: Guard-D Enhanced Dynamic Margin (Margin + Multi-Tenant Address Guard)
    def it14_guard_d_dynamic(sid_pairs_dict):
        preds = {}
        for sid, pairs in sid_pairs_dict.items():
            if not pairs or pairs[0][1] < 0.75:
                preds[sid] = set()
            else:
                p1 = pairs[0][1]
                accepted = []
                for cid, p in pairs:
                    if p < 0.75 or (p1 - p) > 0.08:
                        break
                    feats = pair_features.get((sid, cid), {})
                    ns = feats.get("feat_name_token_sort", 0)
                    asim = feats.get("feat_addr_token_sort", 0)
                    # Guard D: reject if name < 45% and address > 70%
                    if ns < 45.0 and asim > 70.0:
                        continue
                    accepted.append(cid)
                    if len(accepted) >= 6:
                        break
                preds[sid] = set(accepted)
        return preds

    # Iteration 15: The Grand Synthesis (Dynamic Margin + Cliff Cut + Guard D + Disjoint Bipartite)
    def it15_grand_synthesis(sid_pairs_dict):
        # Stage 1: Candidate edge generation with Guard D + Adaptive Margin + Cliff Cut
        candidate_edges = []
        for sid, pairs in sid_pairs_dict.items():
            if pairs and pairs[0][1] >= 0.75:
                p1 = pairs[0][1]
                cur_edges = []
                prev_p = p1
                for cid, p in pairs:
                    if p < 0.75 or (p1 - p) > 0.08:
                        break
                    # Cliff check
                    if (prev_p - p) > 0.07:
                        break
                    feats = pair_features.get((sid, cid), {})
                    ns = feats.get("feat_name_token_sort", 0)
                    asim = feats.get("feat_addr_token_sort", 0)
                    if ns < 45.0 and asim > 70.0:
                        continue
                    cur_edges.append((sid, cid, p))
                    prev_p = p
                    if len(cur_edges) >= 5:
                        break
                candidate_edges.extend(cur_edges)

        # Stage 2: 2-Set Disjoint Bipartite Conflict Resolution
        candidate_edges.sort(key=lambda x: x[2], reverse=True)
        claimed_s2 = set()
        s1_matches = {sid: [] for sid in sid_pairs_dict}
        for sid, cid, p in candidate_edges:
            if cid not in claimed_s2 and len(s1_matches[sid]) < 5:
                s1_matches[sid].append(cid)
                claimed_s2.add(cid)
        return {sid: set(m) for sid, m in s1_matches.items()}

    # Run all 15 Iterations
    iterations = [
        ("Iteration 1: Baseline Hard Threshold (tau=0.75)", it1_baseline),
        ("Iteration 2: High-Precision Threshold (tau=0.90)", it2_high_tau),
        ("Iteration 3: 2-Stage Dynamic Margin (tau=0.75, dp=0.08, max 6)", it3_dynamic_margin),
        ("Iteration 4: Ultra-Tight Dynamic Margin (tau=0.80, dp=0.04, max 5)", it4_tight_margin),
        ("Iteration 5: Adaptive Scaled Margin (dp=0.06*p1, max 6)", it5_adaptive_margin),
        ("Iteration 6: Relative Probability Ratio (p/p1 >= 0.92, max 6)", it6_prob_ratio),
        ("Iteration 7: Cliff / Elbow Dropoff Detection (drop > 0.08, max 6)", it7_cliff_detection),
        ("Iteration 8: 2-Set Disjoint Bipartite Resolution (No S2 > 1 S1)", it8_bipartite_disjoint),
        ("Iteration 9: 2-Stage Dynamic Margin + Disjoint Bipartite", it9_dynamic_margin_plus_bipartite),
        ("Iteration 10: Ground Truth Prior Truncation (tau=0.78, max 4)", it10_prior_truncation),
        ("Iteration 11: 1D Density Clustering (Radius eps=0.06)", it11_density_clustering),
        ("Iteration 12: Feature-Augmented 2-Set (tau=0.75 + name >= 45%)", it12_feature_augmented),
        ("Iteration 13: Softmax Temperature Top-Cumulative (tau=0.75, mass <= 0.88)", it13_softmax_mass),
        ("Iteration 14: Guard-D Enhanced Dynamic Margin (Margin + Guard D)", it14_guard_d_dynamic),
        ("Iteration 15: Grand Synthesis (Margin + Cliff + Guard D + Disjoint)", it15_grand_synthesis),
    ]

    print("\n" + "=" * 105)
    print(f"{'Iteration / Architecture Description':<55} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Avg Match'}")
    print("=" * 105)

    best_score = 0.0
    best_name = ""

    for name, fn in iterations:
        preds = fn(scored_pairs)
        f05, prec, rec = compute_macro_f05(gt_map, preds)
        total_m = sum(len(m) for m in preds.values())
        avg_m = total_m / len(s1_map)
        star = ""
        if f05 > best_score:
            best_score = f05
            best_name = name
            star = "  <-- BEST"
        print(f"{name:<55} | {f05:<12.6f} | {prec*100:>8.2f}% | {rec*100:>8.2f}% | {avg_m:>7.2f}{star}", flush=True)

    print("=" * 105, flush=True)
    print(f"TOURNAMENT WINNER: {best_name}")
    print(f"PEAK MACRO F0.5 ACHIEVED: {best_score:.6f}", flush=True)

if __name__ == "__main__":
    run_tournament()
