"""Audit 40 predicted match pairs for France at tau = 0.98.
"""

import os
import sys
import pickle
import random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df


def main():
    random.seed(42)
    np.random.seed(42)

    # 1. Load model
    model_path = "models/lgbm_champion.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Load France S1 records (raw)
    test_dir = "student_resource/dataset/test"
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Loading France S1 from {s1_path}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1_fr = df_s1[df_s1["country"] == "France"].copy()
    s1_raw_map = {r["entity_id"]: r for r in df_s1_fr.to_dict("records")}

    # Also normalized
    df_s1_fr_norm = apply_normalization_df(df_s1_fr)
    s1_norm_map = {r["entity_id"]: r for r in df_s1_fr_norm.to_dict("records")}

    # 3. Load candidates
    temp_dir = "output/temp_work"
    cands_path = os.path.join(temp_dir, "cands_France.tsv")
    print(f"Loading candidate records from {cands_path}...")
    df_cands = pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False)
    cand_raw_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}

    df_cands_norm = apply_normalization_df(df_cands)
    cand_norm_map = {r["entity_id"]: r for r in df_cands_norm.to_dict("records")}

    # 4. Read France tau 0.98 results
    results_path = os.path.join(temp_dir, "lgbm_clean_results_France_tau_98.tsv")
    print(f"Reading results from {results_path}...")

    buckets = {
        "low (1-2)": [],
        "mid (3-6)": [],
        "high (7-9)": [],
        "cap (10)": []
    }

    with open(results_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            if len(parts) < 3 or not parts[2].strip():
                continue
            matches = [m.strip() for m in parts[2].split(",") if m.strip()]
            n_m = len(matches)
            if n_m == 0:
                continue

            for m in matches:
                item = (sid, m, n_m)
                if 1 <= n_m <= 2:
                    buckets["low (1-2)"].append(item)
                elif 3 <= n_m <= 6:
                    buckets["mid (3-6)"].append(item)
                elif 7 <= n_m <= 9:
                    buckets["high (7-9)"].append(item)
                elif n_m == 10:
                    buckets["cap (10)"].append(item)

    print("Bucket sizes (pairs):")
    for b, items in buckets.items():
        print(f"  {b}: {len(items):,} pairs")

    # Sample exactly:
    # 10 from low (1-2)
    # 15 from mid (3-6)
    # 5 from high (7-9)
    # 10 from cap (10)
    # Total = 40
    sample_targets = [
        ("low (1-2)", 10),
        ("mid (3-6)", 15),
        ("high (7-9)", 5),
        ("cap (10)", 10)
    ]

    selected_pairs = []
    for b, count in sample_targets:
        sampled = random.sample(buckets[b], count)
        for sid, cid, n_m in sampled:
            selected_pairs.append((b, sid, cid, n_m))

    # Shuffle to interleave buckets
    random.shuffle(selected_pairs)

    print(f"\nScoring and printing {len(selected_pairs)} sampled pairs:\n")

    for i, (bucket_name, sid, cid, n_m) in enumerate(selected_pairs, 1):
        r1_raw = s1_raw_map[sid]
        rc_raw = cand_raw_map.get(cid, {})

        r1_norm = s1_norm_map[sid]
        rc_norm = cand_norm_map.get(cid, {})

        feats = extract_pair_features(r1_norm, rc_norm)
        X = np.array([[feats[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
        prob = model.predict_proba(X)[0, 1]

        print(f"--- [Pair {i:02d}/40] [Bucket: {bucket_name}, Total Matches: {n_m}] ---")
        print(f"S1 ID : {sid}")
        print(f"S1 Raw: Name='{r1_raw.get('name', '')}' | Addr='{r1_raw.get('address', '')}' | City='{r1_raw.get('city', '')}'")
        print(f"Cand ID: {cid}")
        print(f"Cand  : Name='{rc_raw.get('name', '')}' | Addr='{rc_raw.get('address', '')}' | City='{rc_raw.get('city', '')}'")
        print(f"Model Prob: {prob:.6f}")
        print()


if __name__ == "__main__":
    main()
