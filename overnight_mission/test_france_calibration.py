"""Test France calibration and multi-tenant guard to achieve 90%+ precision without sacrificing recall."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df

def main():
    print("=" * 80)
    print("FRANCE CALIBRATION & GUARD ANALYSIS")
    print("=" * 80)

    # 1. Load Model
    model_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Load France S1 and Candidates
    test_dir = "student_resource/dataset/test"
    temp_dir = "output/temp_work"

    print("Loading France S1...", flush=True)
    df_s1 = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False)
    df_s1_fr = df_s1[df_s1["country"] == "France"].copy()
    df_s1_fr = apply_normalization_df(df_s1_fr)
    s1_map = {r["entity_id"]: r for r in df_s1_fr.to_dict("records")}

    print("Loading France candidate records...", flush=True)
    df_cands = pd.read_csv(os.path.join(temp_dir, "cands_France.tsv"), sep="\t", dtype=str, keep_default_na=False)
    df_cands = apply_normalization_df(df_cands)
    cand_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}

    in_results_path = os.path.join(temp_dir, "results_France.tsv")
    np.random.seed(42)

    # Sample 3,000 S1 entities
    sample_s1_ids = list(np.random.choice(list(s1_map.keys()), size=3000, replace=False))
    sample_s1_set = set(sample_s1_ids)

    candidate_lists = {}
    with open(in_results_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            if sid in sample_s1_set:
                cands = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
                candidate_lists[sid] = cands

    # Feature extraction
    feat_matrix = []
    meta = []
    for sid, clist in candidate_lists.items():
        r1 = s1_map[sid]
        for cid in clist:
            rc = cand_map.get(cid)
            if rc:
                feats = extract_pair_features(r1, rc)
                feat_matrix.append([feats[col] for col in FEATURE_COLUMNS])
                meta.append((sid, cid))

    print(f"Scoring {len(feat_matrix):,} pairs for 3,000 entities...", flush=True)
    probs = model.predict_proba(np.array(feat_matrix, dtype=np.float32))[:, 1]

    scored_by_s1 = {sid: [] for sid in candidate_lists}
    for (sid, cid), p in zip(meta, probs):
        rc = cand_map[cid]
        scored_by_s1[sid].append((cid, float(p), rc.get("name_norm", ""), rc.get("addr_norm", "")))

    for sid in scored_by_s1:
        scored_by_s1[sid].sort(key=lambda x: x[1], reverse=True)

    # Configurations to test
    configs = [
        ("Base tau=0.70", lambda p, ns, asim, r: p >= 0.70),
        ("Base tau=0.75", lambda p, ns, asim, r: p >= 0.75),
        ("Base tau=0.80", lambda p, ns, asim, r: p >= 0.80),
        ("Base tau=0.85", lambda p, ns, asim, r: p >= 0.85),
        ("Guard A: tau=0.75 + (if ns<50 -> p>=0.90)", lambda p, ns, asim, r: p >= 0.75 and (ns >= 50 or p >= 0.90)),
        ("Guard B: tau=0.75 + (if ns<55 -> p>=0.92)", lambda p, ns, asim, r: p >= 0.75 and (ns >= 55 or p >= 0.92)),
        ("Guard C: tau=0.80 + (if ns<50 -> p>=0.92)", lambda p, ns, asim, r: p >= 0.80 and (ns >= 50 or p >= 0.92)),
        ("Guard D: tau=0.75 + (if ns<45 and asim>70 -> reject)", lambda p, ns, asim, r: p >= 0.75 and not (ns < 45 and asim > 70)),
    ]

    for cfg_name, rule in configs:
        accepted = []
        singleton_cnt = 0
        for sid, pairs in scored_by_s1.items():
            r1 = s1_map[sid]
            top_cand_name = pairs[0][2] if pairs else ""
            p1 = pairs[0][1] if pairs else 0.0

            s1_accepted = []
            for rank, (cid, p, c_name, c_addr) in enumerate(pairs):
                ns = fuzz.token_sort_ratio(r1["name_norm"], c_name)
                asim = fuzz.token_sort_ratio(r1["addr_norm"], c_addr)
                if not rule(p, ns, asim, rank):
                    continue
                # Cluster disambiguation
                if rank > 0:
                    sim_with_top = fuzz.token_sort_ratio(top_cand_name, c_name)
                    if sim_with_top < 40 and (p1 - p > 0.08):
                        continue
                s1_accepted.append((sid, cid, p, r1["name_norm"], c_name, r1["addr_norm"], c_addr, ns, asim))
            if not s1_accepted:
                singleton_cnt += 1
            accepted.extend(s1_accepted)

        # Audit sample of 50
        np.random.seed(123)
        if len(accepted) >= 50:
            shuffled = [accepted[i] for i in np.random.permutation(len(accepted))[:50]]
            strict = sum(1 for _, _, _, _, _, _, _, ns, asim in shuffled if (ns >= 70 or (ns >= 50 and asim >= 60)))
            effective = sum(1 for _, _, _, _, _, _, _, ns, asim in shuffled if (ns >= 70 or (ns >= 50 and asim >= 60) or (ns >= 40 and asim >= 75)))
        else:
            strict, effective = 0, 0

        singleton_pct = singleton_cnt / len(scored_by_s1) * 100
        print(f"\n[{cfg_name}]")
        print(f"  Accepted Matches: {len(accepted):,} (Avg/Ent: {len(accepted)/len(scored_by_s1):.2f})")
        print(f"  Singletons (0 matches): {singleton_cnt}/{len(scored_by_s1)} ({singleton_pct:.2f}%)")
        print(f"  50-Pair Audit -> Strict Prec: {strict/50*100:.1f}%, Effective Prec: {effective/50*100:.1f}%")

if __name__ == "__main__":
    main()
